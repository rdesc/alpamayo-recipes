# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Ablation of eval_baseline_ood_reasoning.py: same discrete-token (RLWrapperReasoningVLA)
baseline, same checkpoint, same PAI-AV OOD reasoning subset -- but with the VLM prompt built
WITH camera-identity/frame-index text, to test whether that closes the gap against
~/repos/alpamayo1.5/docs/first_round_eval_results.md's published minADE numbers.

Background (see scripts_fork/results/eval_prompt_format_discrepancy.md for the full writeup):
our eval scripts build prompts via ``alpamayo_r1.helper.create_message(frames)``, which has no
way to include camera identity -- it just lists raw frames. The official
``~/repos/alpamayo1.5/eval_pai_av_val.py`` instead uses
``alpamayo1_5.helper.create_message(frames, camera_indices=..., ...)``, which (per its own
docstring) prepends text like "Front left camera: " / "frame 0 " before each image --
"matching the format used during model training." ``alpamayo_r1.load_physical_aiavdataset``
already returns ``data["camera_indices"]``; our scripts just never passed it anywhere.

This script changes ONE thing relative to eval_baseline_ood_reasoning.py: message construction.
``_create_message_with_cameras`` below is a local copy of alpamayo1_5.helper's image-annotation
logic (``_build_image_content``), grafted onto alpamayo_r1.helper.create_message's own prompt
text/special-token conventions (identical wording either way, so this isn't a confound) --
avoids a cross-repo import of the separate ``~/repos/alpamayo1.5`` checkout/venv. Everything
else (model class, checkpoint, sampling params, metrics, output schema) is unchanged, so any
score delta vs. the original script isolates the prompt-format effect.

Defaults to --split val (349 events) to match the published table directly and keep this a
fast ablation rather than a full 2,077-event rerun.

Usage::

    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$((4+i)) a1x_rl/bin/python scripts_fork/eval_baseline_ood_reasoning_camprompt.py \\
        --num-shards 4 --shard-idx $i \\
        --out /opt/dlami/nvme/rod/results/baseline_ood_reasoning_camprompt/baseline.parquet &
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
MIN_T0_US = 1_700_000

# Mirrors alpamayo1_5.helper.CAMERA_DISPLAY_NAMES.
_CAMERA_DISPLAY_NAMES = {
    0: "Front left camera",
    1: "Front camera",
    2: "Front right camera",
    3: "Rear left camera",
    4: "Rear camera",
    5: "Rear right camera",
    6: "Front telephoto camera",
}


def _build_image_content(frames: torch.Tensor, camera_indices: torch.Tensor, num_frames_per_camera: int = 4):
    """Local copy of alpamayo1_5.helper._build_image_content's annotation logic."""
    expanded_cam_ids = camera_indices.repeat_interleave(num_frames_per_camera)
    content: list[dict] = []
    prev_cam_id = None
    frame_idx = 0
    for i, frame in enumerate(frames):
        cam_id = expanded_cam_ids[i].item()
        if prev_cam_id is not None and cam_id != prev_cam_id:
            frame_idx = 0
        if frame_idx == 0:
            cam_name = _CAMERA_DISPLAY_NAMES.get(cam_id, f"Camera {cam_id}")
            content.append({"type": "text", "text": f"{cam_name}: "})
        content.append({"type": "text", "text": f"frame {frame_idx} "})
        content.append({"type": "image", "image": frame})
        prev_cam_id = cam_id
        frame_idx += 1
    return content


def _create_message_with_cameras(frames: torch.Tensor, camera_indices: torch.Tensor):
    """alpamayo_r1.helper.create_message, but with camera-identity/frame-index image annotations."""
    if frames.ndim != 4:
        raise ValueError(f"{frames.ndim=}, expected 4 (N, C, H, W)")

    num_traj_token = 48
    hist_traj_placeholder = (
        f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
    )
    image_content = _build_image_content(frames, camera_indices)

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a driving assistant that generates safe and accurate actions.",
                }
            ],
        },
        {
            "role": "user",
            "content": image_content
            + [
                {
                    "type": "text",
                    "text": f"{hist_traj_placeholder}output the chain-of-thought reasoning of the driving process, then output the future trajectory.",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "<|cot_start|>"}],
        },
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--split", default="val", choices=["train", "val", "both"])
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
    from alpamayo_r1.helper import get_processor, to_device
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

    from alpamayo1_x_rl.models.reasoning_vla.base_model import RLWrapperReasoningVLA
    from alpamayo1_x_rl.rewards.coc_action_consistency_extract import (
        QwenClaimExtractor,
        extract_claims_regex,
    )
    from alpamayo1_x_rl.rewards.coc_action_consistency_reward import compute_component
    from alpamayo1_x_rl.rewards.traj_reward import calculate_ade
    from alpamayo1_x_rl.rewards.val_metrics import group_distance_metrics

    print(f"[eval] loading model {args.model_path} (camera-annotated prompt) ...", flush=True)
    model = RLWrapperReasoningVLA.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).cuda().eval()
    processor = get_processor(model.tokenizer)

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

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    rows: list[dict] = []
    t_start = time.time()
    for i, ev in enumerate(events):
        clip_id, t0_us = ev["clip_id"], ev["t0_us"]
        try:
            data = load_physical_aiavdataset(clip_id, t0_us=t0_us, avdi=avdi)
            messages = _create_message_with_cameras(
                data["image_frames"].flatten(0, 1), data["camera_indices"]
            )
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            )
            model_inputs = to_device(
                {
                    "tokenized_data": inputs,
                    "ego_history_xyz": data["ego_history_xyz"],
                    "ego_history_rot": data["ego_history_rot"],
                },
                "cuda",
            )

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot, _, extra = model.sample_trajectories_from_data(
                    data=model_inputs,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_traj_samples=args.num_traj_samples,
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
