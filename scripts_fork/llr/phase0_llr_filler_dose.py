# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Filler dose-response: does the trajectory likelihood rise with the NUMBER of reasoning
positions, when those positions carry no information at all?

WHY THIS EXISTS
---------------
The presence term (gold CoC vs empty slot) is positive, the content term (gold CoC vs another
event's CoC) is zero, and the two are reconciled by a length story: what an occupied reasoning
block buys is positions to compute over, not meaning. The evidence for that story is a
regression of LLR on CoC token count ACROSS events (+0.00122 nats/token on 1.5), which
``phase0_llr_donor_sweep.py`` then showed does NOT reproduce within a fixed scene -- the
per-scene slopes disagree in sign and average to roughly nothing.

So the length story is currently unsupported where it matters. This tests it directly, in the
one design where content is zero BY CONSTRUCTION rather than by argument: hold the scene and the
target trajectory fixed, put n filler tokens in the reasoning block, and sweep n.

    y(n) = log p(a* | v, filler^n)        n = 0, 1, 2, 4, ... 128

Filler cannot "say more" as it gets longer, so any slope in y(n) is positions-to-compute-over --
the pause-token effect, isolated. A flat y(n) would mean the presence term is not positional
either, which would leave "being real text at all" as the live explanation and would reframe the
content null rather than support it.

The existing ``--denominator pad`` arm of ``phase0_llr_per_token.py`` is the n = n_gold slice of
this curve and nothing else: it overwrites the prose in place, holding length fixed. It was run
on four events. This sweeps n.

TWO FILLER KINDS, because they fail differently:

  ``pad``     -- the tokenizer's pad token, as the existing pad arm uses. Maximally
                 information-free, but padding inside a reasoning block is a sequence training
                 never produces, so it shares the empty control's off-distribution problem.
  ``natural`` -- a repeated ordinary word. Still contentless with respect to the scene, but
                 well-formed text in the register the model was tuned on.

Running both separates "more positions" from "more positions the model finds plausible". If
only ``natural`` shows a slope, the effect needs the filler to be in-distribution; if both do,
it is positional in the crude sense.

ALSO SCORED per scene, for scale: the gold CoC, and the empty block (n = 0 is exactly the empty
block, so the curve starts at the presence term's denominator).

Everything is within-scene and teacher-forced on the SAME 128 gold trajectory tokens, so the
comparison never crosses scenes and never changes the target. See the invariants in
``phase0_llr_per_token.py``'s docstring; the future-token selection here is identical.

Run from a recipe directory with the _b300 venv, and use the SAME venv as every other LLR run
(``recipes/alpamayo1_x_rl/a1x_rl_b300``) -- the two venvs differ by ~0.01 nats on identical
inputs (different fp32 reduction kernel in RMSNorm between torch 2.8 and 2.11)::

    CUDA_VISIBLE_DEVICES=7 a1x_rl_b300/bin/python \\
      ../../scripts_fork/llr/phase0_llr_filler_dose.py --n-scenes 30 --out /tmp/filler.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

DEFAULT_MODEL_PATH = "/mnt/efs/users/rod/ckpts/Alpamayo-1.5-10B-rl-training"
DEFAULT_PARQUET = "/opt/dlami/nvme/rod/datasets/pai_av_dataset/reasoning/ood_reasoning.parquet"
MIN_T0_US = 1_700_000
DOSES = [0, 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--n-scenes", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="both", choices=["train", "val", "both"])
    p.add_argument(
        "--natural-word", default=" the",
        help="The repeated word for the 'natural' filler. Must be ONE token; asserted at startup.",
    )
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_events(parquet_path: str, split: str) -> list[dict]:
    """One row per annotated event. Kept in step with phase0_llr_per_token.py's loader."""
    df = pd.read_parquet(parquet_path)
    if split != "both":
        df = df[df["split"] == split].copy()
    df["events"] = df["events"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    df = df.dropna(subset=["events"])
    rows = []
    for clip_id, row in df.iterrows():
        for ei, ev in enumerate(row["events"]):
            rows.append(
                {
                    "clip_id": clip_id,
                    "event_idx": ei,
                    "t0_us": max(int(ev["event_start_timestamp"]), MIN_T0_US),
                    "gold_coc": ev["coc"],
                    "event_cluster": row["event_cluster"],
                    "split": row["split"],
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    sys.path[:0] = ["../../../src", "../.."]
    import physical_ai_av
    from alpamayo_r1.helper import to_device
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo.processor.qwen_processor import (
        collate_fn_from_model_config,
        get_preprocess_data_fn_from_model_config,
        build_processor,
    )

    from alpamayo.utils.get_label_mask import get_label_mask

    from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA

    pool = load_events(args.parquet, args.split)
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(pool), size=min(args.n_scenes, len(pool)), replace=False)
    scenes = [pool[int(i)] for i in sorted(pick)]
    scenes = [s for i, s in enumerate(scenes) if i % args.num_shards == args.shard]
    print(f"[filler] pool {len(pool)}; {len(scenes)} scenes on shard "
          f"{args.shard}/{args.num_shards}; doses {DOSES}", flush=True)

    model = (
        TrainableReasoningVLA.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
        .cuda()
        .eval()
    )
    kw = dict(include_camera_ids=True, include_frame_nums=True, chat_template_version="r1")
    preprocess_fn = get_preprocess_data_fn_from_model_config(
        components_order=["image", "traj_history", "prompt", "cot", "traj_future"],
        components_prompt=["cot", "traj_future"],
        label_components=["cot"],
        generation_mode=False,
        model_config=model.config,
        **kw,
    )
    processor = build_processor(
        vlm_name_or_path=model.config.vlm_name_or_path,
        traj_vocab_size=model.config.traj_vocab_size,
        min_pixels=model.config.min_pixels,
        max_pixels=model.config.max_pixels,
        **kw,
    )
    tokenizer = processor.tokenizer
    traj_lo = model.future_token_start_idx
    traj_hi = traj_lo + model.config.traj_vocab_size
    sid = model.special_token_ids
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    # The natural filler must be exactly one token, or "n tokens of filler" is a lie.
    nat_ids = tokenizer(args.natural_word, add_special_tokens=False)["input_ids"]
    assert len(nat_ids) == 1, f"--natural-word {args.natural_word!r} is {len(nat_ids)} tokens"
    nat_id = nat_ids[0]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    print(f"[filler] natural={args.natural_word!r} id={nat_id}  pad id={pad_id}", flush=True)

    def tokenize(data: dict, cot_text: str) -> dict:
        sample = {
            "image_frames": data["image_frames"],
            "camera_indices": data["camera_indices"],
            "relative_timestamps": data["relative_timestamps"],
            "ego_history_xyz": data["ego_history_xyz"].squeeze(0),
            "ego_history_rot": data["ego_history_rot"].squeeze(0),
            "cot": cot_text,
        }
        sample["tokenized_data"] = preprocess_fn(sample)
        return collate_fn_from_model_config([sample], model_config=model.config, **kw)[
            "tokenized_data"
        ]

    def score(td: dict, data: dict) -> tuple[float, int]:
        """Mean log p over the 128 FUTURE trajectory tokens only.

        History trajectory tokens share the same id block as future ones, so they are excluded
        by sequence position relative to the last traj_future_start, not by id.
        """
        t = {
            k: data[k].cuda()
            for k in ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot")
        }
        fused = model.fuse_traj_tokens(td["input_ids"].clone().cuda(), t)
        m_all = (
            ((fused >= traj_lo) & (fused < traj_hi))
            | (fused == sid["traj_future_start"])
            | (fused == sid["traj_future_end"])
        )
        fsp = int((fused[0] == sid["traj_future_start"]).nonzero().flatten()[-1].item())
        mi = to_device({"tokenized_data": dict(td)}, "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(tokenized_data=mi["tokenized_data"], labels_mask=None, **t)
        m = m_all[:, 1:]
        sl = fused[:, 1:][m].contiguous()
        lg = out.logits[..., :-1, :][m].contiguous().float()
        lp = -F.cross_entropy(lg, sl, reduction="none")
        sp = m[0].nonzero().flatten() + 1
        isf = (sl >= traj_lo) & (sl < traj_hi) & (sp > fsp)
        return float(lp[isf].mean().item()), int(isf.sum().item())

    def filler_td(data: dict, n: int, kind: str) -> dict:
        """Build a sequence whose reasoning block holds exactly n filler tokens.

        For 'natural' the filler is real text, so it goes through the normal tokenizer path and
        the block is built by the template like any other CoC. For 'pad' the pad id never
        survives tokenization of a string, so the ids are written in directly: tokenize a
        natural filler of the same length, then overwrite exactly those n positions. Both
        therefore have identical block structure and identical length -- they differ only in
        which id sits in the n slots.
        """
        if n == 0:
            return tokenize(data, "")
        text = args.natural_word * n
        td = tokenize(data, text)
        if kind == "natural":
            return td
        ids = td["input_ids"].clone()
        # Restrict to the reasoning block: the filler id also occurs in the prompt template, so
        # a whole-sequence search would overwrite real prompt tokens. get_label_mask's cot span
        # is contiguous and INCLUSIVE of <|cot_start|>/<|cot_end|>; never touch those two --
        # overwriting a marker is what produced the retracted delimiter artifact.
        span = get_label_mask(ids, tokenizer, ["cot"])[0].nonzero().flatten()
        assert ids[0, span[0]].item() == sid["cot_start"], "cot span layout changed"
        assert ids[0, span[-1]].item() == sid["cot_end"], "cot span layout changed"
        inner = span[1:-1]
        if len(inner) != n or not bool((ids[0, inner] == nat_id).all()):
            raise RuntimeError(
                f"expected {n} filler tokens inside the cot block, found {len(inner)}")
        ids[0, inner] = pad_id
        td2 = dict(td)
        td2["input_ids"] = ids
        return td2

    rows = []
    t0 = time.time()
    for k, s in enumerate(scenes):
        data = load_physical_aiavdataset(s["clip_id"], t0_us=s["t0_us"], avdi=avdi)
        gold = s["gold_coc"]
        n_gold = len(tokenizer(gold, add_special_tokens=False)["input_ids"])
        lp_gold, nf = score(tokenize(data, gold), data)
        if nf != 128:
            raise RuntimeError(f"{s['clip_id'][:8]}: gold pass scored {nf}")

        base = {
            "clip_id": s["clip_id"], "clip8": s["clip_id"][:8], "event_idx": s["event_idx"],
            "event_cluster": s["event_cluster"], "split": s["split"],
            "gold_coc": gold, "n_tok_gold": n_gold, "logp_gold": lp_gold,
        }
        per_scene = {}
        for kind in ("natural", "pad"):
            for n in DOSES:
                if n == 0 and kind == "pad":
                    continue  # n=0 is the empty block; identical for both kinds
                lp, nf = score(filler_td(data, n, kind), data)
                if nf != 128:
                    raise RuntimeError(f"{s['clip_id'][:8]} {kind} n={n}: scored {nf}")
                rows.append({**base, "filler": kind, "n_filler": n, "logp": lp})
                per_scene[(kind, n)] = lp
        lp0 = per_scene[("natural", 0)]
        for r in rows[-(2 * len(DOSES) - 1):]:
            r["logp_empty"] = lp0
            r["llr_vs_empty"] = r["logp"] - lp0
            r["llr_gold_vs_empty"] = lp_gold - lp0

        parent = os.path.dirname(args.out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        pd.DataFrame(rows).to_parquet(args.out, index=False)
        nat_hi = per_scene[("natural", 128)] - lp0
        pad_hi = per_scene[("pad", 128)] - lp0
        print(
            f"[filler] {k+1}/{len(scenes)} {s['clip8'] if 'clip8' in s else s['clip_id'][:8]} "
            f"gold({n_gold}tok) {lp_gold-lp0:+.4f} | natural n=128 {nat_hi:+.4f} | "
            f"pad n=128 {pad_hi:+.4f}  ({(time.time()-t0)/60:.1f} min)",
            flush=True,
        )

    print(f"[filler] wrote {args.out}: {len(rows)} rows, {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
