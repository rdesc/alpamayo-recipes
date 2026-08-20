# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Confirms (rather than just argues) that the minADE selection-convention difference explains
the residual gap between this fork's OOD-reasoning eval and ~/repos/alpamayo1.5's published
numbers, by computing BOTH conventions on the SAME raw predicted trajectories from the SAME
model/checkpoint/prompt.

See scripts_fork/results/eval_prompt_format_discrepancy.md for the full background. Two
conventions, both applied here to identical (pred_xy, gt_xy) data per event:

  - "ours": alpamayo.metrics.distance_metrics.compute_minade -- selects the best-of-K trajectory
    ONCE using the full-length mean error, then reports that same trajectory's error truncated
    to each horizon.
  - "official": ~/repos/alpamayo1.5/eval_common.py's clip_metrics -- re-selects the best-of-K
    trajectory INDEPENDENTLY at each horizon. Imported directly (pure numpy/pandas, no torch,
    per that module's own docstring) rather than reimplemented, so there's no risk of a subtly
    different reimplementation.

Uses the camera-annotated-prompt path (eval_baseline_ood_reasoning_camprompt.py's fix) and
RLWrapperReasoningVLA (discrete-token), same checkpoint. Skips the CoC extractor entirely --
this script is ADE-only, not re-litigating CoC-consistency.

If "official-convention-applied-to-our-own-predictions" comes out close to
~/repos/alpamayo1.5/docs/first_round_eval_results.md's published numbers, that confirms the
convention difference is (most of) the story. If a gap remains even then, something else is
still unexplained.

Usage::

    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$((4+i)) a1x_rl_b300/bin/python scripts_fork/eval_metric_convention_compare.py \\
        --num-shards 4 --shard-idx $i \\
        --out /opt/dlami/nvme/rod/results/metric_convention_compare/results.parquet &
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
MIN_T0_US = 1_700_000

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
                {"type": "text", "text": "You are a driving assistant that generates safe and accurate actions."}
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
        {"role": "assistant", "content": [{"type": "text", "text": "<|cot_start|>"}]},
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--split", default="val", choices=["train", "val", "both"])
    p.add_argument("--first-event-only", action="store_true")
    p.add_argument("--num-traj-samples", type=int, default=6)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.98)
    p.add_argument("--max-generation-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


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
                    "event_cluster": row["event_cluster"],
                    "split": row["split"],
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    sys.path[:0] = ["../../../src", "../..", "/home/rod/repos/alpamayo1.5"]
    import physical_ai_av
    from alpamayo_r1.helper import get_processor, to_device
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

    from alpamayo1_x_rl.models.reasoning_vla.base_model import RLWrapperReasoningVLA
    from alpamayo.metrics import distance_metrics

    import eval_common as ec  # ~/repos/alpamayo1.5's pure-numpy metric module

    print(f"[eval] loading model {args.model_path} (metric-convention comparison) ...", flush=True)
    model = RLWrapperReasoningVLA.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).cuda().eval()
    processor = get_processor(model.tokenizer)

    events = load_events(args.parquet, args.split, all_events=not args.first_event_only)
    if args.num_shards > 1:
        events = events[args.shard_idx :: args.num_shards]
    if args.limit:
        events = events[: args.limit]
    print(f"[eval] shard {args.shard_idx}/{args.num_shards}: {len(events)} events", flush=True)

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
                messages, tokenize=True, add_generation_prompt=False,
                continue_final_message=True, return_dict=True, return_tensors="pt",
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
                pred_xyz, pred_rot, _, _extra = model.sample_trajectories_from_data(
                    data=model_inputs,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_traj_samples=args.num_traj_samples,
                    max_generation_length=args.max_generation_length,
                    return_extra=True,
                )  # pred_xyz: [1, 1, K, T, 3]

            gt_xyz = data["ego_future_xyz"][0].cuda()  # [1, T, 3]

            # --- "ours": alpamayo.metrics.distance_metrics.compute_minade ---
            ours = distance_metrics.compute_minade(
                pred_xyz.float(), gt_xyz.float(),
                disable_summary=True, timestep_horizons=[10, 30], time_step=0.1,
            )
            ours_1s = float(ours["min_ade/by_t=1.0"].mean())
            ours_3s = float(ours["min_ade/by_t=3.0"].mean())

            # --- "official": eval_common.clip_metrics, applied to the SAME predictions ---
            pred_xy = pred_xyz[0, 0, :, :, :2].float().cpu().numpy()  # (K, T, 2)
            gt_xy = data["ego_future_xyz"][0, 0, :, :2].float().numpy()  # (T, 2)
            official_metrics, medoid = ec.clip_metrics(pred_xy, gt_xy)

            rows.append(
                {
                    "clip_id": clip_id,
                    "event_idx": ev["event_idx"],
                    "event_cluster": ev["event_cluster"],
                    "split": ev["split"],
                    "ours_min_ade_1s": ours_1s,
                    "ours_min_ade_3s": ours_3s,
                    "official_min_ade_1s": official_metrics["min_ade_1s"],
                    "official_min_ade_3s": official_metrics["min_ade_3s"],
                    "official_min_ade_6s": official_metrics["min_ade_6s"],
                    "ok": True,
                }
            )
            if (i + 1) % 20 == 0 or (i + 1) == len(events):
                elapsed = time.time() - t_start
                print(
                    f"[eval] [{i + 1}/{len(events)}] {clip_id} "
                    f"ours@1s={ours_1s:.3f} official@1s={official_metrics['min_ade_1s']:.3f} "
                    f"({elapsed / 60:.1f} min elapsed)",
                    flush=True,
                )
        except Exception as e:
            rows.append(
                {"clip_id": clip_id, "event_idx": ev["event_idx"], "split": ev["split"], "ok": False, "error": str(e)}
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
    print(f"\n[eval] shard {args.shard_idx}/{args.num_shards} done in {(time.time() - t_start) / 60:.1f} min. "
          f"{len(ok)}/{len(df)} ok. Wrote {out_path}", flush=True)
    if len(ok):
        print(
            f"[eval] shard summary: ours@1s={ok['ours_min_ade_1s'].mean():.4f} "
            f"official@1s={ok['official_min_ade_1s'].mean():.4f}  "
            f"ours@3s={ok['ours_min_ade_3s'].mean():.4f} official@3s={ok['official_min_ade_3s'].mean():.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
