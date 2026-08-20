# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Baseline (no-RL) batch inference + min_ADE + CoC-action-consistency over the PAI-AV OOD
reasoning subset (``reasoning/ood_reasoning.parquet``), train+val splits -- ACTION-HEAD variant.

Prompt construction (2026-08-19 fix): builds prompts via
``alpamayo.processor.qwen_processor.get_preprocess_data_fn_from_model_config`` -- the SAME
function actual Stage-1/Stage-2 SFT training uses (``configs/vla_processor/nav.yaml``, selected
by ``sft_stage1_nav.yaml``/``sft_stage2_nav.yaml``), with that config's exact
``components_order``/``components_prompt``/``label_components``/``chat_template_version``
values, rather than the older ``alpamayo_r1.helper.create_message`` (no camera-identity/frame-
index text at all -- confirmed by camera-annotated ablation to have closed roughly half the
original gap to ``~/repos/alpamayo1.5``'s published numbers; see
``recipes/alpamayo1_x_rl/scripts_fork/results/eval_prompt_format_discrepancy.md``).

Sibling of ``alpamayo1_x_rl/scripts_fork/eval_baseline_ood_reasoning.py``: same dataset, same
split, same K (6, min-ADE_6 paper convention), same ADE / CoC-action-consistency metrics, same
output schema -- so the two parquets/results are directly comparable row-for-row. The ONLY
difference is how the trajectory is generated:

  - ``eval_baseline_ood_reasoning.py`` (alpamayo1_x_rl) loads ``RLWrapperReasoningVLA`` and
    calls its ``sample_trajectories_from_data``, which autoregressively generates DISCRETE
    trajectory tokens from the VLM's own vocabulary and decodes them via the frozen
    ``traj_tokenizer`` -- no diffusion involved. That's the only trajectory pathway
    ``RLWrapperReasoningVLA`` has; the RL recipe never trains or touches an action head
    (confirmed: ``recipes/alpamayo1_x_rl/SKILL.md`` -- "RL here trains the autoregressive-
    trajectory-token pathway of the VLM. The action expert (flow-matching head) is not
    trained").
  - This script instead loads the SAME checkpoint
    (``/mnt/efs/users/rod/ckpts/Alpamayo-1.5-10B-rl-training``) via Stage-2's
    ``TrainableAlpamayoR1`` (``alpamayo1_5_sft/models/sft_alpamayo_r1.py``). That checkpoint
    was verified to carry real action-expert weights (397 ``expert.*`` + 10
    ``action_in_proj.*`` + 2 ``action_out_proj.*`` tensors in its
    ``model.safetensors.index.json`` -- ``RLWrapperReasoningVLA`` just never loads them).
    ``TrainableAlpamayoR1.sample_trajectories_from_data`` generates the VLM's reasoning text
    autoregressively up to ``<traj_future_start>``, then runs a flow-matching diffusion expert
    (cross-attending into the VLM's KV-cache) to produce a CONTINUOUS trajectory in action
    space, decoded via ``self.action_space.action_to_traj``. Called exactly the way this
    recipe's own Stage-2 val eval already calls it -- same signature/kwargs as
    ``callbacks/val_metric_callback.py``'s ``ValMinADECallback`` and ``evaluate_hf.py``'s
    ``metric_runner`` path (per the user: "just follow the eval that's done during training
    for stage 2 of SFT ... since we're already doing this") -- no bespoke handling of
    action-space/traj-tokenizer internals needed; ``pred_xyz``/``pred_rot`` come back in the
    same shape/convention either pathway uses downstream.

CoC-action-consistency is always scored against the model's OWN predicted CoC vs. its own
predicted trajectory -- never against the dataset's gold ``coc`` text (stored per row for
reference only).

Usage::

    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$((4+i)) a1_5_sft/bin/python scripts_fork/eval_baseline_ood_reasoning_actionhead.py \\
        --num-shards 4 --shard-idx $i \\
        --out /opt/dlami/nvme/rod/results/baseline_ood_reasoning_actionhead/baseline.parquet &
    done; wait
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

DEFAULT_MODEL_PATH = "/mnt/efs/users/rod/ckpts/Alpamayo-1.5-10B-rl-training"
DEFAULT_PARQUET = "/opt/dlami/nvme/rod/datasets/pai_av_dataset/reasoning/ood_reasoning.parquet"
DEFAULT_QWEN_EXTRACTOR = (
    "/mnt/efs/users/rod/hf_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/"
    "snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
# t0 must exceed the history window (16 * 0.1s = 1.6s); matches eval_common.py's MIN_T0_US
# and eval_baseline_ood_reasoning.py's own floor.
MIN_T0_US = 1_700_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--split", default="both", choices=["train", "val", "both"])
    p.add_argument("--first-event-only", action="store_true", help="One event/clip instead of all annotated events.")
    p.add_argument(
        "--coc-extractor-model-path",
        default=DEFAULT_QWEN_EXTRACTOR,
        help="Set to empty string to fall back to the regex extractor (no GPU/model cost).",
    )
    p.add_argument("--num-traj-samples", type=int, default=6, help="K trajectories/event (paper: minADE_6).")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.98)
    p.add_argument("--max-generation-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="Only this many events (post-shard).")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--out", required=True, help="Output .parquet path (sharded when --num-shards > 1).")
    return p.parse_args()


def _cot_text(extra: dict, k: int) -> str:
    val = extra["cot"][0, 0, k]
    return str(val) if val is not None else ""


def load_events(parquet_path: str, split: str, all_events: bool) -> list[dict]:
    """Flatten ood_reasoning.parquet into one row per annotated event.

    Identical to eval_baseline_ood_reasoning.py's loader (see that file for provenance).
    """
    df = pd.read_parquet(parquet_path)
    if split != "both":
        df = df[df["split"] == split].copy()
    df["events"] = df["events"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    df = df.dropna(subset=["events"])
    rows = []
    for clip_id, row in df.iterrows():
        events = row["events"]
        idxs = range(len(events)) if all_events else [0]
        for ei in idxs:
            ev = events[ei]
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
    torch.manual_seed(args.seed)

    sys.path[:0] = ["../../../src", "../.."]
    import physical_ai_av
    from alpamayo_r1.helper import to_device
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo.processor.qwen_processor import (
        collate_fn_from_model_config,
        get_preprocess_data_fn_from_model_config,
    )

    from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1
    from alpamayo1_x_rl.rewards.coc_action_consistency_extract import (
        QwenClaimExtractor,
        extract_claims_regex,
    )
    from alpamayo1_x_rl.rewards.coc_action_consistency_reward import compute_component
    from alpamayo1_x_rl.rewards.traj_reward import calculate_ade
    from alpamayo1_x_rl.rewards.val_metrics import group_distance_metrics

    print(f"[eval] loading model {args.model_path} (action-head/diffusion path) ...", flush=True)
    model = TrainableAlpamayoR1.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).cuda().eval()

    # Matches configs/vla_processor/nav.yaml exactly -- the real Stage-1/Stage-2 SFT training
    # prompt format for the nav-conditioned trajectory task.
    preprocess_fn = get_preprocess_data_fn_from_model_config(
        components_order=["image", "traj_history", "route", "prompt", "traj_future"],
        components_prompt=["traj_future"],
        label_components=["traj_future"],
        generation_mode=True,
        include_camera_ids=True,
        include_frame_nums=True,
        model_config=model.config,
        chat_template_version="r1_5",
    )

    extractor = None
    if args.coc_extractor_model_path:
        print(f"[eval] loading CoC extractor {args.coc_extractor_model_path} ...", flush=True)
        extractor = QwenClaimExtractor(args.coc_extractor_model_path, device="cuda")

    events = load_events(args.parquet, args.split, all_events=not args.first_event_only)
    if args.num_shards > 1:
        events = events[args.shard_idx :: args.num_shards]
    if args.limit:
        events = events[: args.limit]
    print(
        f"[eval] shard {args.shard_idx}/{args.num_shards}: {len(events)} events "
        f"(split={args.split}, K={args.num_traj_samples}, extractor={'qwen' if extractor else 'regex'})",
        flush=True,
    )

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()  # streaming -- this root has no local clip media

    rows: list[dict] = []
    t_start = time.time()
    for i, ev in enumerate(events):
        clip_id, t0_us = ev["clip_id"], ev["t0_us"]
        try:
            data = load_physical_aiavdataset(clip_id, t0_us=t0_us, avdi=avdi)
            # Squeeze the leading batch-of-1 dim load_physical_aiavdataset bakes into
            # ego_history_{xyz,rot} ((1,1,16,3) -> (1,16,3)) before collate_fn_from_model_config's
            # torch.stack adds a real batch dim back -- otherwise the result is double-batched.
            # Same gotcha eval_navsim_stage1.py documents for the NAVSIM npz loader.
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
                include_camera_ids=True, include_frame_nums=True, chat_template_version="r1_5",
            )
            model_inputs = to_device(batch, "cuda")

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = model.sample_trajectories_from_data(
                    data=model_inputs,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_traj_samples=args.num_traj_samples,
                    num_traj_sets=1,
                    max_generation_length=args.max_generation_length,
                    return_extra=True,
                )  # [1, 1, K, T, ...]

            gt_xyz = data["ego_future_xyz"][0].cuda()
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
                        "event_idx": ev["event_idx"],
                        "event_cluster": ev["event_cluster"],
                        "split": ev["split"],
                        "sample_idx": k,
                        "t0_us": int(data["t0_us"]),
                        "gold_coc": ev["gold_coc"],
                        "ade": ade_k,
                        "coc_reward_contribution": comp["reward_contribution"],
                        "coc_graded": comp["graded"],
                        "coc_binary": comp["binary"],
                        "coc_abstained": bool(comp["abstained"]),
                        "coc_parsed": bool(comp["parsed"]),
                        "coc_n_scored": comp["n_scored"],
                        "pred_cot": cot_texts[k],
                        **group_metrics,
                        "ok": True,
                    }
                )
            if (i + 1) % 20 == 0 or (i + 1) == len(events):
                elapsed = time.time() - t_start
                print(
                    f"[eval] [{i + 1}/{len(events)}] {clip_id} split={ev['split']} "
                    f"ade_mean={np.mean([r['ade'] for r in rows[-args.num_traj_samples:]]):.3f}m "
                    f"coc_mean={np.mean([r['coc_reward_contribution'] for r in rows[-args.num_traj_samples:]]):.3f} "
                    f"({elapsed / 60:.1f} min elapsed)",
                    flush=True,
                )
        except Exception as e:  # keep going -- one bad event must not kill the shard
            rows.append(
                {
                    "clip_id": clip_id,
                    "event_idx": ev["event_idx"],
                    "split": ev["split"],
                    "sample_idx": -1,
                    "ok": False,
                    "error": str(e),
                }
            )
            print(f"[eval] [{i + 1}/{len(events)}] {clip_id} FAILED: {type(e).__name__}: {e}", flush=True)

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
        f"{(time.time() - t_start) / 60:.1f} min. {len(ok)}/{len(df)} sample-rows ok. Wrote {out_path}",
        flush=True,
    )
    if len(ok):
        print(
            f"[eval] shard summary: min_ade mean={ok['min_ade'].mean():.3f}m  "
            f"coc_consistency mean={ok['coc_reward_contribution'].mean():.3f}  "
            f"abstain_rate={ok['coc_abstained'].mean():.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
