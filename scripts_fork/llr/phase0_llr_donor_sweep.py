# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Within-scene donor sweep: hold the scene fixed, vary only the reasoning, and watch what the
trajectory likelihood does as a function of how many reasoning tokens it was given.

WHY THIS EXISTS
---------------
The full-split runs established two facts that sit awkwardly together:

  * ``gold - donor`` (the CONTENT term) is zero on both models -- swapping in another event's
    real prose costs nothing (1.5 trimmed -0.0004, Wilcoxon p=0.99).
  * There is nonetheless a LENGTH dose-response: +0.00122 nats/token on 1.5 (7.0 sigma),
    +0.00158 on A2S (8.8 sigma) -- but with R^2 of only 2.3% / 3.6%.

That regression is ACROSS scenes: each point is a different event, so scene difficulty,
manoeuvre type and CoC length are all confounded, and the tiny R^2 is unsurprising. It cannot
distinguish "longer reasoning raises the likelihood of any trajectory" from "scenes that get
long annotations happen to be scenes the model scores well."

This script removes the confound by regressing WITHIN a single scene. One scene, one fixed
target trajectory, ~100 different donor CoCs of naturally varying length. Everything that makes
a scene easy or hard is held constant by construction, so any residual slope is attributable to
the reasoning text alone -- and mostly to its length, since donor content is by definition
unrelated to this scene.

Run on the tails as well as the middle: the ``min``/``max`` events (most negative / most
positive per-event LLR in the corrected full-split run) are where a reader is most tempted to
say "reasoning worked here", so they are where a length explanation most needs testing.

WHAT IS MEASURED
----------------
Per scene, all passes in natural trained order, scoring the 128 future trajectory tokens only
(never delimiters or the 48 history tokens -- see ``results/phase0_llr_action_direction.md``)::

    logp_gold     = log p(a* | v, ell_gold)        one pass
    logp_none     = log p(a* | v)                  one pass, cot spliced to empty
    logp_donor_j  = log p(a* | v, ell_j)           K passes, ell_j = another event's real prose

and per donor row::

    llr_donor      = logp_donor_j - logp_none      what the donor CoC is worth on its own
    gold_vs_donor  = logp_gold    - logp_donor_j   the CONTENT term, per donor
    d_tokens       = n_tok_donor  - n_tok_gold     signed length difference

The headline is the within-scene OLS slope of ``logp_donor_j`` on ``n_tok_donor``. If reasoning
length is doing the work, that slope is positive and of the same order as the across-scene
+0.0012-0.0016 nats/token. If it is flat within a scene, the across-scene slope was scene
composition, not a token-count effect.

``logp_gold`` is asserted against the corrected run's per-event value (``--tol``) -- the
integrity check that this script scores the same quantity as everything else in the Phase-0 set.

DONOR DRAWS are deterministic per (seed, clip_id, event_idx) via sha256 -- NOT ``hash()``, which
is salted per process and would make shards irreproducible. Donors are drawn from other events
only; a donor whose prose is byte-identical to the gold is skipped.

Run from a recipe directory with the _b300 venv (see README.md)::

    CUDA_VISIBLE_DEVICES=7 a1_5_sft_b300/bin/python ../../scripts_fork/llr/phase0_llr_donor_sweep.py \\
      --events-json <manifest.json> --n-donors 100 --out /tmp/llr_donor_sweep.parquet
"""

from __future__ import annotations

import argparse
import hashlib
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument(
        "--events-json",
        required=True,
        help="Manifest of the scenes to sweep: list of dicts with clip_id and event_idx "
             "(a viz manifest works). Each also MAY carry llr_act / label / rank, which are "
             "copied through and used for the --tol integrity check.",
    )
    p.add_argument("--n-donors", type=int, default=100, help="Donor CoCs per scene.")
    p.add_argument("--donor-seed", type=int, default=0)
    p.add_argument(
        "--tol", type=float, default=2e-3,
        help="Tolerance on reproducing llr_act from the manifest. Checked only for scenes whose "
             "manifest entry carries it.",
    )
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_events(parquet_path: str) -> list[dict]:
    """Flatten ood_reasoning.parquet into one row per annotated event.

    Identical to phase0_llr_per_token.py's loader -- kept in step with it deliberately, so the
    donor POOL here is the same pool the full-split runs drew from.
    """
    df = pd.read_parquet(parquet_path)
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

    from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA

    pool = load_events(args.parquet)
    by_key = {(e["clip_id"], e["event_idx"]): e for e in pool}
    print(f"[sweep] donor pool: {len(pool)} events", flush=True)

    with open(args.events_json) as fh:
        want = json.load(fh)

    print(f"[sweep] loading model {args.model_path} ...", flush=True)
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
        batch = collate_fn_from_model_config([sample], model_config=model.config, **kw)
        return batch["tokenized_data"]

    def score_future(td: dict, data: dict) -> tuple[float, int]:
        """Mean log p over the 128 FUTURE trajectory tokens only.

        The 48 history tokens share the same id block (traj_vocab_size 4000 vs num_bins 3000), so
        they are excluded by sequence position relative to the last traj_future_start, not by id.
        """
        input_ids = td["input_ids"]
        traj_data = {
            "ego_history_xyz": data["ego_history_xyz"].cuda(),
            "ego_history_rot": data["ego_history_rot"].cuda(),
            "ego_future_xyz": data["ego_future_xyz"].cuda(),
            "ego_future_rot": data["ego_future_rot"].cuda(),
        }
        fused = model.fuse_traj_tokens(input_ids.clone().cuda(), traj_data)
        traj_mask = (
            ((fused >= traj_lo) & (fused < traj_hi))
            | (fused == sid["traj_future_start"])
            | (fused == sid["traj_future_end"])
        )
        fsp = int((fused[0] == sid["traj_future_start"]).nonzero().flatten()[-1].item())
        model_inputs = to_device({"tokenized_data": dict(td)}, "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                tokenized_data=model_inputs["tokenized_data"],
                ego_history_xyz=traj_data["ego_history_xyz"],
                ego_history_rot=traj_data["ego_history_rot"],
                ego_future_xyz=traj_data["ego_future_xyz"],
                ego_future_rot=traj_data["ego_future_rot"],
                labels_mask=None,
            )
        m = traj_mask[:, 1:]
        sl = fused[:, 1:][m].contiguous()
        lg = out.logits[..., :-1, :][m].contiguous().float()
        logp = -F.cross_entropy(lg, sl, reduction="none")
        seq_pos = m[0].nonzero().flatten() + 1
        is_fut = (sl >= traj_lo) & (sl < traj_hi) & (seq_pos > fsp)
        return float(logp[is_fut].mean().item()), int(is_fut.sum().item())

    def n_tok(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    rows = []
    t0 = time.time()
    for w in want:
        key = (w["clip_id"], int(w["event_idx"]))
        ent = by_key.get(key)
        if ent is None:
            print(f"[sweep] {key} not in pool -- skipped", flush=True)
            continue

        data = load_physical_aiavdataset(ent["clip_id"], t0_us=ent["t0_us"], avdi=avdi)
        gold = ent["gold_coc"]

        lp_gold, n_g = score_future(tokenize(data, gold), data)
        lp_none, n_n = score_future(tokenize(data, ""), data)
        if not (n_g == n_n == 128):
            raise RuntimeError(f"{key}: future-token count {n_g}/{n_n}, expected 128")

        llr_gold = lp_gold - lp_none
        status = "ok"
        if "llr_act" in w and w["llr_act"] is not None:
            err = abs(llr_gold - float(w["llr_act"]))
            if err > args.tol:
                status = f"MISMATCH (expected {float(w['llr_act']):+.4f}, err {err:.2e})"

        # Deterministic donor draw. sha256, not hash() -- hash() is salted per process.
        seed_key = f"{args.donor_seed}|{ent['clip_id']}|{ent['event_idx']}".encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed_key).digest()[:8], "big"))
        cand = [e for e in pool if (e["clip_id"], e["event_idx"]) != key and e["gold_coc"] != gold]
        pick = rng.choice(len(cand), size=min(args.n_donors, len(cand)), replace=False)

        label = w.get("label", "")
        print(
            f"[sweep] {label}{w.get('rank','')} {ent['clip_id'][:8]} llr_gold={llr_gold:+.4f} "
            f"[{status}] gold={n_tok(gold)}tok -- {len(pick)} donors ({(time.time()-t0)/60:.1f} min)",
            flush=True,
        )

        for j, ci in enumerate(pick):
            d = cand[int(ci)]
            lp_d, n_d = score_future(tokenize(data, d["gold_coc"]), data)
            if n_d != 128:
                raise RuntimeError(f"{key} donor {j}: future-token count {n_d}")
            rows.append(
                {
                    "label": label,
                    "rank": w.get("rank", None),
                    "clip_id": ent["clip_id"],
                    "clip8": ent["clip_id"][:8],
                    "event_idx": ent["event_idx"],
                    "event_cluster": ent["event_cluster"],
                    "split": ent["split"],
                    "repro_status": status,
                    "gold_coc": gold,
                    "n_tok_gold": n_tok(gold),
                    "logp_gold": lp_gold,
                    "logp_none": lp_none,
                    "llr_gold": llr_gold,
                    "donor_j": j,
                    "donor_clip_id": d["clip_id"],
                    "donor_event_idx": d["event_idx"],
                    "donor_coc": d["gold_coc"],
                    "n_tok_donor": n_tok(d["gold_coc"]),
                    "logp_donor": lp_d,
                    "llr_donor": lp_d - lp_none,
                    "gold_vs_donor": lp_gold - lp_d,
                    "d_tokens": n_tok(d["gold_coc"]) - n_tok(gold),
                }
            )

        # Write incrementally: a sweep is many hours and a crash should not lose finished scenes.
        parent = os.path.dirname(args.out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        pd.DataFrame(rows).to_parquet(args.out, index=False)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 92)
    print(
        f"{'scene':14s} {'llr_gold':>9s} {'gold tok':>8s} {'K':>4s} "
        f"{'slope/tok':>10s} {'t':>7s} {'R^2':>6s}  {'mean llr_donor':>14s}"
    )
    print("=" * 92)
    for (lab, c8), g in df.groupby(["label", "clip8"], sort=False):
        x = g["n_tok_donor"].to_numpy(float)
        y = g["logp_donor"].to_numpy(float)
        if len(g) > 2 and x.std() > 0:
            b, a = np.polyfit(x, y, 1)
            yhat = a + b * x
            r2 = 1.0 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            se = np.sqrt(((y - yhat) ** 2).sum() / (len(x) - 2) / ((x - x.mean()) ** 2).sum())
            tstat = b / se if se > 0 else float("nan")
        else:
            b = r2 = tstat = float("nan")
        print(
            f"{lab+' '+c8:14s} {g['llr_gold'].iloc[0]:+9.4f} {g['n_tok_gold'].iloc[0]:8d} "
            f"{len(g):4d} {b:+10.5f} {tstat:+7.2f} {r2:6.3f}  {g['llr_donor'].mean():+14.4f}"
        )
    print("=" * 92)
    bad = df[df["repro_status"] != "ok"]["clip8"].unique()
    if len(bad):
        print(f"WARNING: llr_gold did not reproduce for {list(bad)}")
    print(f"wrote {args.out}  ({len(df)} rows, {(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
