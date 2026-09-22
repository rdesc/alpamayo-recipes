# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Crossed scene x CoC grid: separate "this text is a good prefix for anything" from
"this text is the right reasoning for THIS scene".

WHY THIS EXISTS
---------------
Every earlier measurement compared a scene's gold CoC against a few random donors and called
the difference CONTENT. That comparison cannot separate two very different things:

  * a CoC MAIN EFFECT -- some strings raise the trajectory likelihood of *any* scene, e.g. by
    being long, or fluent, or in the register the model was tuned on;
  * a scene x CoC INTERACTION -- a string helps the scene it actually describes.

Only the second is "the model uses the reasoning". With a handful of donors per event the two
are entangled, and the donor draw's main effect lands in the content estimate as noise.

The within-scene sweep (``phase0_llr_donor_sweep.py``) fixed the scene and varied the text,
which controls the scene main effect but not the CoC one. This script crosses both: N scenes
x the N gold CoCs those scenes come with, ALL N^2 combinations, so the two main effects are
estimable and the diagonal (each scene meeting its own reasoning) can be tested against the
additive prediction.

THE MODEL
---------
For scene s scored under CoC c, with y = mean log p over the 128 future trajectory tokens::

    y[s,c] = mu + alpha[s] + beta[c] + eps[s,c]

``alpha`` is scene difficulty, ``beta`` is the CoC's generic prefix value. Fit by the usual
two-way means (rows, columns, grand mean) and look at the residual::

    r[s,c] = y[s,c] - (mu + alpha[s] + beta[c])

CONTENT, stated sharply: is ``r[s,s]`` (diagonal, matched) greater than ``r[s,c!=s]``? That is
a within-row, within-column contrast, so neither a hard scene nor a flattering CoC can produce
it. Paired over scenes it is a one-sample test on ``r[s,s] - mean_c!=s r[s,c]``.

Note the diagonal is where the gold pairing lives, so the diagonal contrast IS the content
term -- but measured with the CoC main effect removed rather than averaged over.

SCENES are a seeded RANDOM sample of the split, deliberately NOT the llr tails: the sweep
showed the tails are selection artefacts that regress toward the mean under any perturbation,
so an effect estimated there would be measuring the selection, not the model.

Sharding is BY SCENE (rows): each shard loads its own scenes and scores all N CoCs against
them, so every shard writes complete rows and the grid can be analysed from a partial run.

Run from a recipe directory with the _b300 venv. Use the SAME venv as every other LLR run
(``recipes/alpamayo1_x_rl/a1x_rl_b300``): the two venvs differ by ~0.01 nats on identical
inputs (different fp32 reduction kernel in RMSNorm between torch 2.8 and 2.11), which is far
larger than the effects here::

    CUDA_VISIBLE_DEVICES=0 a1x_rl_b300/bin/python ../../scripts_fork/llr/phase0_llr_coc_grid.py \\
      --n-scenes 40 --num-shards 7 --shard 0 --out /tmp/grid.shard0-of-7.parquet
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--n-scenes", type=int, default=40, help="N; the grid is N x N.")
    p.add_argument("--seed", type=int, default=0, help="Seeds the scene sample.")
    p.add_argument("--split", default="both", choices=["train", "val", "both"])
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

    from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA

    pool = load_events(args.parquet, args.split)
    # The N scenes are drawn once, from a seed independent of the shard, so every shard agrees
    # on the CoC set (the columns) even though it only scores its own rows.
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(pool), size=min(args.n_scenes, len(pool)), replace=False)
    scenes = [pool[int(i)] for i in sorted(pick)]
    cocs = [s["gold_coc"] for s in scenes]
    n = len(scenes)
    mine = [i for i in range(n) if i % args.num_shards == args.shard]
    print(f"[grid] pool {len(pool)}; grid {n}x{n}; shard {args.shard}/{args.num_shards} "
          f"scores rows {mine}", flush=True)

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
        return collate_fn_from_model_config([sample], model_config=model.config, **kw)[
            "tokenized_data"
        ]

    def score_future(td: dict, data: dict) -> tuple[float, int]:
        """Mean log p over the 128 FUTURE trajectory tokens only.

        History trajectory tokens share the same id block, so they are excluded by sequence
        position relative to the last traj_future_start rather than by id.
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

    n_tok = [len(tokenizer(c, add_special_tokens=False)["input_ids"]) for c in cocs]

    rows = []
    t0 = time.time()
    for si in mine:
        s = scenes[si]
        data = load_physical_aiavdataset(s["clip_id"], t0_us=s["t0_us"], avdi=avdi)
        lp_none, n_n = score_future(tokenize(data, ""), data)
        if n_n != 128:
            raise RuntimeError(f"scene {si}: empty pass scored {n_n} tokens")
        for ci, coc in enumerate(cocs):
            lp, nf = score_future(tokenize(data, coc), data)
            if nf != 128:
                raise RuntimeError(f"scene {si} coc {ci}: scored {nf} tokens")
            rows.append(
                {
                    "scene_i": si,
                    "coc_j": ci,
                    "matched": si == ci,
                    "clip_id": s["clip_id"],
                    "event_idx": s["event_idx"],
                    "event_cluster": s["event_cluster"],
                    "split": s["split"],
                    "coc_clip_id": scenes[ci]["clip_id"],
                    "n_tok_coc": n_tok[ci],
                    "logp": lp,
                    "logp_none": lp_none,
                    "llr": lp - lp_none,
                }
            )
        parent = os.path.dirname(args.out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        pd.DataFrame(rows).to_parquet(args.out, index=False)
        d = [r for r in rows if r["scene_i"] == si]
        diag = next(r["llr"] for r in d if r["matched"])
        off = float(np.mean([r["llr"] for r in d if not r["matched"]]))
        print(
            f"[grid] row {si} ({s['clip_id'][:8]}) matched llr {diag:+.4f} vs "
            f"off-diagonal mean {off:+.4f}  ({(time.time()-t0)/60:.1f} min)",
            flush=True,
        )

    print(f"[grid] wrote {args.out}: {len(rows)} cells, {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
