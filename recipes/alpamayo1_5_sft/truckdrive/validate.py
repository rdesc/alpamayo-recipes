# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate a TruckDrive sample before training -- shapes, tokenizer, viz.

This is the "does everything look right?" gate for the TruckDrive loader. It
needs **no model weights**: it loads a sample from ``TruckDriveDataset``, prints
tensor shapes, runs the frozen trajectory tokenizer's encode->decode round-trip
on the ground-truth future (the hard constraint everything else depends on), and
renders a PNG showing the camera inputs alongside the GT future trajectory (red)
and its tokenizer reconstruction (blue). A small red/blue gap means the frozen
tokenizer can faithfully represent this dataset's motion; a large gap means the
trajectories fall outside the tokenizer's delta range and training will not learn
them well.

Examples::

    # Point at a local dir (staged data or a mounted mount-s3 filesystem)
    python validate_truckdrive.py --data-root /path/with/scene_dirs --index 0

    # Use the exact tokenizer config from a checkpoint (recommended)
    python validate_truckdrive.py --data-root ... --checkpoint /path/to/ckpt

    # Stream straight from S3 (requires s3torchconnector)
    python validate_truckdrive.py \
        --data-root s3://torc-data/datasets/TruckDrivePublic --backend s3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np
import torch

# Make ``src/`` importable when run directly from the recipe directory.
_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from alpamayo.data.truckdrive import TruckDriveDataset  # noqa: E402
from alpamayo.visualization.viz import visualize_data  # noqa: E402


def build_tokenizer(checkpoint: str | None):
    """Build the trajectory tokenizer -- from a checkpoint config if given.

    Loading from the checkpoint reproduces the *exact* frozen tokenizer used in
    training (ranges, bins, predict_yaw). Without a checkpoint we fall back to
    ``DeltaTrajectoryTokenizer`` defaults and warn.
    """
    from alpamayo_r1.models.delta_tokenizer import DeltaTrajectoryTokenizer

    if checkpoint is not None:
        cfg_path = Path(checkpoint) / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"no config.json in checkpoint: {checkpoint}")
        cfg = json.loads(cfg_path.read_text())
        traj_cfg = cfg.get("traj_tokenizer_cfg")
        if traj_cfg is not None:
            from hydra.utils import instantiate

            print(f"[tokenizer] using traj_tokenizer_cfg from {cfg_path}")
            return instantiate(traj_cfg)
        print(f"[tokenizer] checkpoint has no traj_tokenizer_cfg; using defaults")
    else:
        print("[tokenizer] WARNING: no --checkpoint; using DeltaTrajectoryTokenizer "
              "defaults, which may differ from your training tokenizer")
    return DeltaTrajectoryTokenizer()


def roundtrip(tokenizer, sample: dict) -> tuple[torch.Tensor, dict]:
    """Encode->decode the GT future; return reconstructed xyz + error stats."""
    hist_xyz = sample["ego_history_xyz"]  # (1, Th, 3)
    hist_rot = sample["ego_history_rot"]  # (1, Th, 3, 3)
    fut_xyz = sample["ego_future_xyz"]    # (1, Tf, 3)
    fut_rot = sample["ego_future_rot"]    # (1, Tf, 3, 3)

    tokens = tokenizer.encode(
        hist_xyz=hist_xyz, hist_rot=hist_rot, fut_xyz=fut_xyz, fut_rot=fut_rot
    )
    rec_xyz, _, _ = tokenizer.decode(hist_xyz=hist_xyz, hist_rot=hist_rot, tokens=tokens)

    # The Alpamayo unicycle tokenizer models the ground-plane (x, y) path only;
    # z (elevation) is not encoded, so reconstruction quality is measured in XY.
    err_xy = torch.linalg.norm(rec_xyz[..., :2] - fut_xyz[..., :2], dim=-1)  # (1, Tf)
    z_span = float(fut_xyz[..., 2].max() - fut_xyz[..., 2].min())
    stats = {
        "vocab_size": getattr(tokenizer, "vocab_size", "?"),
        "n_tokens": int(tokens.shape[-1]),
        "recon_err_xy_mean_m": float(err_xy.mean()),
        "recon_err_xy_max_m": float(err_xy.max()),
        "endpoint_err_xy_m": float(err_xy[0, -1]),
        "gt_z_span_m_unmodeled": z_span,
    }
    return rec_xyz, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True,
                    help="Local dir with scene_* subdirs, or s3://bucket/prefix")
    ap.add_argument("--backend", choices=("local", "s3"), default="local")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--index", type=int, default=0, help="Sample index to render")
    ap.add_argument("--views", default="forward_center_medium",
                    help="Comma-separated Leopard camera views")
    ap.add_argument("--num-frames", type=int, default=4)
    ap.add_argument("--num-history-steps", type=int, default=16)
    ap.add_argument("--num-future-steps", type=int, default=64)
    ap.add_argument("--time-step", type=float, default=0.1)
    ap.add_argument("--image-time-step", type=float, default=None)
    ap.add_argument("--checkpoint", default=None,
                    help="Checkpoint dir to source the exact tokenizer config")
    ap.add_argument("--out", default="truckdrive_validation.png")
    args = ap.parse_args()

    dataset = TruckDriveDataset(
        data_root=args.data_root,
        backend=args.backend,
        region=args.region,
        camera_views=[v.strip() for v in args.views.split(",") if v.strip()],
        num_frames=args.num_frames,
        num_history_steps=args.num_history_steps,
        num_future_steps=args.num_future_steps,
        time_step=args.time_step,
        image_time_step=args.image_time_step,
        vla_preprocess_args=None,  # no model needed for validation
    )
    print(f"[dataset] {len(dataset)} samples; rendering index {args.index}")

    sample = dataset[args.index % len(dataset)]

    print("\n[shapes]")
    for k, v in sample.items():
        if torch.is_tensor(v):
            print(f"  {k:22s} {tuple(v.shape)}  {v.dtype}")
        else:
            print(f"  {k:22s} {v!r}")

    tokenizer = build_tokenizer(args.checkpoint)
    rec_xyz, stats = roundtrip(tokenizer, sample)
    print("\n[tokenizer round-trip]  (XY = ground plane; z is unmodeled)")
    for k, v in stats.items():
        print(f"  {k:24s} {v}")
    if stats["recon_err_xy_mean_m"] > 0.25:
        print("  WARNING: XY reconstruction > 0.25 m mean -- check heading_source "
              "and whether this window includes near-stationary (low-speed) motion.")

    # image_frames (N_cam, T, C, H, W) -> visualize_data wants (T, N_cam, C, H, W)
    frames = sample["image_frames"].permute(1, 0, 2, 3, 4)
    info = (
        f"scene: {sample['scene_id']}   t0={sample['t0_us']}us\n"
        f"recon err XY mean/max/endpoint: "
        f"{stats['recon_err_xy_mean_m']:.3f}/{stats['recon_err_xy_max_m']:.3f}/"
        f"{stats['endpoint_err_xy_m']:.3f} m   (z-span unmodeled: "
        f"{stats['gt_z_span_m_unmodeled']:.1f} m)"
    )
    visualize_data(
        image_frames=frames,
        ego_future_xyz_gt=sample["ego_future_xyz"],           # red
        ego_future_xyz_pred=rec_xyz.reshape(1, 1, 1, -1, 3),  # blue: tokenizer recon
        cot_text=None,
        show_waypoint_pai=False,
        save_path=args.out,
        info_text=info,
    )
    print(f"\n[viz] wrote {os.path.abspath(args.out)}")
    print("  red = GT future trajectory, blue = frozen-tokenizer reconstruction")


if __name__ == "__main__":
    main()
