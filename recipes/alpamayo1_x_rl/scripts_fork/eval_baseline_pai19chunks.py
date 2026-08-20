# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Baseline (no-RL) batch inference + ADE + CoC-action-consistency over the 19-chunk PAI-AV subset.

Prompt construction (2026-08-19 fix): builds prompts via
``alpamayo.processor.qwen_processor.get_preprocess_data_fn_from_model_config`` -- the SAME
function the actual RL training data pipeline uses (``alpamayo1_x_rl/hydra_configs/
alpamayo1_5_rvla_rl_pai.yaml``'s ``data.train.dataset.vla_preprocess_args``), with that config's
exact ``components_order``/``components_prompt``/``chat_template_version`` values, rather than
the older ``alpamayo_r1.helper.create_message`` (no camera-identity/frame-index text at all).
See ``scripts_fork/results/eval_prompt_format_discrepancy.md`` for the investigation that found
this.

Answers "what does the un-RL'd Alpamayo 1.5 checkpoint score, before any of this fork's RL
post-training?" -- the number every ``CoC_Consistency_*`` / reasoning RL run should be read
against. Two existing pieces of inference code were considered and rejected as-is:

  - ``notebooks/rl_checkpoint_inference.ipynb`` -- the single-clip pattern this script scales
    up (model load, ``load_physical_aiavdataset`` + ``create_message``/``get_processor``,
    ``model.sample_trajectories_from_data``), but it is one clip, one GPU, no reward scoring.
  - ``~/repos/alpamayo1.5/eval_pai_av_val.py`` (+ ``eval_common.py``) -- the existing batch-eval
    harness for Alpamayo 1.5, but it (a) targets the 16-clip CoT-labelled ``ood_reasoning.parquet``
    event set, not the 19-chunk nav subset, and (b) computes ADE only -- it predates this fork's
    ``rewards/coc_action_consistency_reward.py`` (the *validated* matcher/extractor port from
    alpamayo-coc-autolabeler; ``~/repos/alpamayo1.5/coc_action_consistency.py`` is an earlier,
    coarser, offline reconstruction of the same idea, calibrated only to a ~0.82 GT ceiling).

So this script is new, but every piece it's built from is reused, not reimplemented:
  - model + sampling               -> the notebook's pattern, RLWrapperReasoningVLA.from_pretrained
  - ADE / minADE / corner_distance -> rewards/val_metrics.group_distance_metrics (same SFT-parity
    metric this recipe already logs as ``val/min_ade`` etc. during RL, so this baseline is directly
    comparable to those curves)
  - CoC-action-consistency         -> rewards/coc_action_consistency_reward.compute_component,
    with the same in-loop QwenClaimExtractor (rewards/coc_action_consistency_extract.py) the
    CoC_Consistency_Qwen_Mini891 RL run used, batched per-clip via ``extract_batch``

Dataset: the full 19-chunk nav subset (``clip_index_19chunks_valid.parquet`` -- all
``clip_is_valid`` rows across the 19 downloaded PAI chunks listed in the dataset root's
README, train+val+test splits combined: 1877 clips total, 891/688/298 by split; see
``dataset_spec.py`` for how ``clip_index_mini*``/``clip_index_smoke16`` are carved from the
same 19 chunks). One window per clip at the default keyframe (t0=5.1s), matching
``use_default_keyframe=True`` used by every RL profile trained on this root -- so this
baseline's windows are the same windows those runs train/validate on.

Multi-GPU: no torchrun / process groups needed (plain single-device inference per clip), so
parallelism is just N independent OS processes, one per GPU, each taking every Nth clip
(``--num-shards``/--shard-idx``, mirrors ``eval_common.py``'s convention). Launch with:

    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$((4+i)) a1x_rl/bin/python scripts_fork/eval_baseline_pai19chunks.py \\
        --num-shards 4 --shard-idx $i \\
        --out /opt/dlami/nvme/rod/results/baseline_pai19chunks/baseline.parquet &
    done; wait

Output: one row per (clip_id, trajectory_sample) -- ``--out``, sharded as
``<out>.shard{i}-of-{N}.parquet`` -- plus, after merging shards,
``<out-dir>/summary.json`` with dataset-level aggregates. Parquet (not a training-time-only
log) so it's readable by any later analysis without re-running inference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

DEFAULT_MODEL_PATH = "/mnt/efs/users/rod/ckpts/Alpamayo-1.5-10B-rl-training"
DEFAULT_PAI_ROOT = "/opt/dlami/nvme/rod/datasets/pai_nav_subset_19chunks"
DEFAULT_CLIP_INDEX = "clip_index_19chunks_valid.parquet"
DEFAULT_T0_US = 5_100_000  # alpamayo.data.pai.PAIDataset.DEFAULT_T0_US -- single-keyframe mode
# load_physical_aiavdataset asserts t0_us > num_history_steps*time_step*1e6 == 1_600_000 exactly
# (an ABSOLUTE floor, not clip-relative) -- matches eval_common.py's MIN_T0_US in ~/repos/alpamayo1.5.
# Needed because a clip's egomotion.timestamps can start below 0 (padding/pre-roll), so
# `timestamps.min() + start_margin_us` alone can undershoot this floor.
MIN_T0_US = 1_700_000
DEFAULT_QWEN_EXTRACTOR = (
    "/mnt/efs/users/rod/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/"
    "snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Base (non-RL) checkpoint dir.")
    p.add_argument("--pai-root", default=DEFAULT_PAI_ROOT)
    p.add_argument("--clip-index", default=DEFAULT_CLIP_INDEX)
    p.add_argument("--features-metadata", default="features.csv")
    p.add_argument(
        "--coc-extractor-model-path",
        default=DEFAULT_QWEN_EXTRACTOR,
        help="Set to empty string to fall back to the regex extractor (no GPU/model cost).",
    )
    p.add_argument("--num-traj-samples", type=int, default=6, help="K trajectories/window (paper: minADE_6).")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.98)
    p.add_argument("--max-generation-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--t0-stride-s",
        type=float,
        default=None,
        help=(
            "If set, evaluate every window at this stride (seconds) across each clip's full "
            "duration, instead of the single PAIDataset.DEFAULT_T0_US=5.1s keyframe every "
            "use_default_keyframe=True profile trains/validates on. That default is just "
            "alpamayo_r1's single-example demo timestamp (test_inference.py); it is not "
            "tuned to this dataset -- PAI clips here run 20-140s (median 136s), so one 5.1s "
            "slice covers a few percent of a clip's content. Windows need "
            "--start-margin-s history before them and --end-margin-s future after them, so "
            "count = floor((clip_duration - start_margin - end_margin) / stride) + 1. "
            "Produces multiple rows per clip (see --num-traj-samples: cost scales as "
            "n_windows * K, so drop K when stride is small)."
        ),
    )
    p.add_argument("--start-margin-s", type=float, default=1.6, help="Min seconds of history required before t0.")
    p.add_argument("--end-margin-s", type=float, default=6.4, help="Min seconds of future required after t0.")
    p.add_argument("--limit", type=int, default=None, help="Only this many clips (post-shard, post-limit).")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--out", required=True, help="Output .parquet path (sharded when --num-shards > 1).")
    return p.parse_args()


def _cot_text(extra: dict, k: int) -> str:
    """``extra['cot']`` is shaped [B=1, ns=1, K] by sample_trajectories_from_data; pull sample k."""
    val = extra["cot"][0, 0, k]
    return str(val) if val is not None else ""


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    sys.path[:0] = ["../../../src", "../.."]  # matches notebooks/rl_checkpoint_inference.ipynb
    from alpamayo.data.pai_utils import PhysicalAIAVDatasetLocalInterface
    from alpamayo_r1.helper import to_device
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo.processor.qwen_processor import (
        collate_fn_from_model_config,
        get_preprocess_data_fn_from_model_config,
    )

    from alpamayo1_x_rl.models.reasoning_vla.base_model import RLWrapperReasoningVLA
    from alpamayo1_x_rl.rewards.coc_action_consistency_extract import (
        QwenClaimExtractor,
        extract_claims_regex,
    )
    from alpamayo1_x_rl.rewards.coc_action_consistency_reward import compute_component
    from alpamayo1_x_rl.rewards.traj_reward import calculate_ade
    from alpamayo1_x_rl.rewards.val_metrics import group_distance_metrics

    print(f"[eval] loading model {args.model_path} ...", flush=True)
    model = RLWrapperReasoningVLA.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).cuda().eval()

    # Matches alpamayo1_x_rl/hydra_configs/alpamayo1_5_rvla_rl_pai.yaml's
    # data.train.dataset.vla_preprocess_args exactly -- the real RL training prompt format.
    preprocess_fn = get_preprocess_data_fn_from_model_config(
        components_order=["image", "traj_history", "prompt", "cot"],
        components_prompt=["cot", "traj_future"],
        label_components=[],
        generation_mode=True,
        include_camera_ids=True,
        include_frame_nums=True,
        model_config=model.config,
        chat_template_version="r1",  # RL config default; NOT "r1_5" (that's the SFT recipe's default)
    )

    extractor = None
    if args.coc_extractor_model_path:
        print(f"[eval] loading CoC extractor {args.coc_extractor_model_path} ...", flush=True)
        extractor = QwenClaimExtractor(args.coc_extractor_model_path, device="cuda")

    avdi = PhysicalAIAVDatasetLocalInterface(
        local_dir=args.pai_root,
        features_metadata=args.features_metadata,
        clip_index_metadata=args.clip_index,
    )
    clip_ids = avdi.get_all_clip_ids()
    if args.num_shards > 1:
        clip_ids = clip_ids[args.shard_idx :: args.num_shards]
    if args.limit:
        clip_ids = clip_ids[: args.limit]

    # (clip_id, t0_us) candidates. Single-keyframe mode (default) mirrors
    # PAIDataset(use_default_keyframe=True): one window/clip at DEFAULT_T0_US.
    #
    # Stride mode: candidates are generated, NOT precomputed from a duration bound.
    # avdi.get_clip_feature(...).timestamps/.time_range turned out to be unreliable as a
    # bound -- empirically, for the same clip_id, a bare get_clip_feature() call can report
    # a much larger span (e.g. ~140s) than what load_physical_aiavdataset's OWN internal
    # query later enforces (e.g. ~20s) for that identical clip. Rather than guess which
    # number is real, each clip's candidates are generated up to MAX_WINDOWS_PER_CLIP and
    # fed straight to load_physical_aiavdataset (the same call used for real inference); the
    # first failure is treated as "reached the end of this clip's valid range" and the
    # window loop moves on to the next clip. Cheap to detect -- the failure happens inside
    # the CPU-only egomotion interpolation, before any image decode or GPU work.
    MAX_WINDOWS_PER_CLIP = 200  # safety cap (400s of coverage @ 2s stride) against a runaway loop
    stride_us = int(args.t0_stride_s * 1e6) if args.t0_stride_s else None

    print(
        f"[eval] shard {args.shard_idx}/{args.num_shards}: {len(clip_ids)} clips "
        f"(K={args.num_traj_samples}, stride={args.t0_stride_s or 'single-keyframe'}, "
        f"extractor={'qwen' if extractor else 'regex'})",
        flush=True,
    )

    rows: list[dict] = []
    t_start = time.time()
    n_windows_seen = 0
    for clip_id in clip_ids:
        chunk = int(avdi.clip_index.at[clip_id, "chunk"])
        split = str(avdi.clip_index.at[clip_id, "split"])
        t0_candidates = (
            (MIN_T0_US + w * stride_us for w in range(MAX_WINDOWS_PER_CLIP))
            if stride_us
            else iter([DEFAULT_T0_US])
        )

        for t0_us in t0_candidates:
            try:
                data = load_physical_aiavdataset(clip_id, t0_us=t0_us, avdi=avdi)
            except Exception:
                if stride_us:
                    break  # reached the end of this clip's valid range -- next clip
                rows.append({"clip_id": clip_id, "t0_us": t0_us, "sample_idx": -1, "ok": False})
                break

            n_windows_seen += 1
            try:
                # Same leading-batch-dim squeeze as eval_baseline_ood_reasoning.py -- see the
                # note there (mirrors eval_navsim_stage1.py's documented "shape gotcha").
                sample = {
                    "image_frames": data["image_frames"],
                    "camera_indices": data["camera_indices"],
                    "relative_timestamps": data["relative_timestamps"],
                    "ego_history_xyz": data["ego_history_xyz"].squeeze(0),
                    "ego_history_rot": data["ego_history_rot"].squeeze(0),
                }
                sample["tokenized_data"] = preprocess_fn(sample)
                batch = collate_fn_from_model_config(
                    [sample], model_config=model.config,
                    include_camera_ids=True, include_frame_nums=True, chat_template_version="r1",
                )
                model_inputs = to_device(batch, "cuda")

                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_xyz, pred_rot, _, extra = model.sample_trajectories_from_data(
                        data=model_inputs,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_traj_samples=args.num_traj_samples,
                        max_generation_length=args.max_generation_length,
                        return_extra=True,
                    )  # pred_xyz/pred_rot: [1, 1, K, T, ...]

                gt_xyz = data["ego_future_xyz"][0].cuda()  # [1, T, 3] (n_traj_group=1 acts as G=1)
                gt_rot = data["ego_future_rot"][0].cuda()

                pred_xyz_list = [pred_xyz[:, 0, k] for k in range(args.num_traj_samples)]
                pred_rot_list = [pred_rot[:, 0, k] for k in range(args.num_traj_samples)]
                group_metrics = group_distance_metrics(pred_xyz_list, pred_rot_list, gt_xyz, gt_rot)

                cot_texts = [_cot_text(extra, k) for k in range(args.num_traj_samples)]
                if extractor is not None:
                    claims_per_k = extractor.extract_batch(cot_texts)
                else:
                    claims_per_k = [extract_claims_regex(t) for t in cot_texts]

                for k in range(args.num_traj_samples):
                    ade_k = calculate_ade(pred_xyz[0, 0, k], data["ego_future_xyz"][0, 0].cuda())
                    comp = compute_component(
                        cot_texts[k], pred_xyz[0, 0, k], pred_rot[0, 0, k], claims=claims_per_k[k]
                    )
                    rows.append(
                        {
                            "clip_id": clip_id,
                            "chunk": chunk,
                            "split": split,
                            "sample_idx": k,
                            "t0_us": int(data["t0_us"]),
                            "ade": ade_k,
                            "coc_reward_contribution": comp["reward_contribution"],
                            "coc_graded": comp["graded"],
                            "coc_binary": comp["binary"],
                            "coc_abstained": bool(comp["abstained"]),
                            "coc_parsed": bool(comp["parsed"]),
                            "coc_n_scored": comp["n_scored"],
                            "pred_cot": cot_texts[k],
                            # Group-level (same value on every row of the group) -- mirrors
                            # val_metrics.py's own convention so a flat per-row table still
                            # averages correctly by clip or overall.
                            **group_metrics,
                            "ok": True,
                        }
                    )
                if n_windows_seen % 20 == 0:
                    elapsed = time.time() - t_start
                    print(
                        f"[eval] [window {n_windows_seen}] {clip_id}@{t0_us / 1e6:.1f}s split={split} "
                        f"ade_mean={np.mean([r['ade'] for r in rows[-args.num_traj_samples:]]):.3f}m "
                        f"coc_mean={np.mean([r['coc_reward_contribution'] for r in rows[-args.num_traj_samples:]]):.3f} "
                        f"({elapsed / 60:.1f} min elapsed)",
                        flush=True,
                    )
            except Exception as e:  # keep going -- one bad window must not kill an hours-long shard
                rows.append({"clip_id": clip_id, "t0_us": t0_us, "sample_idx": -1, "ok": False, "error": str(e)})
                print(
                    f"[eval] {clip_id}@{t0_us / 1e6:.1f}s FAILED: {type(e).__name__}: {e}",
                    flush=True,
                )

    import pandas as pd

    df = pd.DataFrame(rows)
    out_path = args.out
    if args.num_shards > 1:
        root, _, ext = args.out.rpartition(".")
        out_path = f"{root}.shard{args.shard_idx}-of-{args.num_shards}.{ext}"
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_parquet(out_path, index=False)

    ok = df[df.get("ok", False) == True]  # noqa: E712
    print(
        f"\n[eval] shard {args.shard_idx}/{args.num_shards} done in "
        f"{(time.time() - t_start) / 60:.1f} min. {len(ok)}/{len(df)} sample-rows ok "
        f"({len(ok) // max(args.num_traj_samples, 1)} windows over {len(clip_ids)} clips). Wrote {out_path}",
        flush=True,
    )
    if len(ok):
        print(
            f"[eval] shard summary: ADE mean={ok['ade'].mean():.3f}m  "
            f"min_ade mean={ok['min_ade'].mean():.3f}m  "
            f"coc_consistency mean={ok['coc_reward_contribution'].mean():.3f}  "
            f"abstain_rate={ok['coc_abstained'].mean():.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
