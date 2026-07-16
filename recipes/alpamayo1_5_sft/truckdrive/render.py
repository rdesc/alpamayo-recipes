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

"""Scene inspection renders for the TruckDrive loader.

Three visualisations for eyeballing that inputs + trajectory look right, from a
locally staged scene directory (containing ``camera/``, ``poses/``,
``calibrations/``):

* ``projection`` — future ego trajectory (from the loader) projected onto the
  front camera image via ``projection.py`` (green=near → red=far).
* ``allviews``   — one frame per Leopard camera at a chosen time, spatially
  arranged, to compare coverage and pick views.
* ``filmstrip``  — one front-camera frame per second across the whole scene.

Example::

    python -m truckdrive.render projection --scene /path/scene_10_5 --t0 8
    python -m truckdrive.render allviews  --scene /path/scene_10_5 --t0 8
    python -m truckdrive.render filmstrip --scene /path/scene_10_5
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from alpamayo.data.truckdrive import TruckDriveDataset, _ScenePoses, _sync_key  # noqa: E402
from truckdrive.projection import load_calibration, project_ego_to_image  # noqa: E402

# Spatial layout of the 15 Leopard views for the coverage grid (5 cols x 3 rows).
_VIEW_LAYOUT = [
    ["forward_left_wide", "forward_left_medium", "forward_center_medium",
     "forward_right_medium", "forward_right_wide"],
    ["forward_left_narrow", "sideward_left_front_wide", "sideward_right_front_wide",
     "forward_right_narrow", "sideward_left_back_wide"],
    ["sideward_right_back_wide", "rearward_left_top_medium", "rearward_left_bottom_medium",
     "rearward_right_top_medium", "rearward_right_bottom_medium"],
]


def _label(im: Image.Image, text: str, scale: float = 1.0) -> Image.Image:
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", int(26 * scale)
        )
    except OSError:
        font = ImageFont.load_default()
    d.rectangle([0, 0, 12 + len(text) * int(15 * scale), int(36 * scale)], fill=(0, 0, 0))
    d.text((6, 4), text, fill=(255, 220, 60), font=font)
    return im


def _load_poses(scene_dir: str) -> _ScenePoses:
    with open(os.path.join(scene_dir, "poses", "gt_trajectory.txt")) as f:
        return _ScenePoses(f.read())


def _nearest_frame(scene_dir: str, poses: _ScenePoses, view: str, t_target: float) -> str:
    files = sorted(glob.glob(os.path.join(scene_dir, "camera", "leopard", view, "images", "*.jpg")))
    timed = []
    for p in files:
        t = poses.key_to_t.get(_sync_key(os.path.basename(p)))
        if t is not None:
            timed.append((abs(t - t_target), p))
    if not timed:
        raise FileNotFoundError(f"no camera frames for view {view!r} in {scene_dir}")
    return min(timed)[1]


def render_projection(scene_dir: str, t0: float, out_path: str, view: str = "forward_center_medium") -> str:
    """Overlay the loader's future ego trajectory on the front camera at ~t0."""
    data_root = str(Path(scene_dir).parent)
    ds = TruckDriveDataset(
        data_root=data_root, camera_views=[view], num_frames=1,
        image_time_step=0.2, t0_stride=2, filter_reverse=False,
    )
    idx = int(np.argmin([abs(t - t0) for _, t in ds._samples]))
    sample = ds[idx]
    t0_actual = ds._samples[idx][1]

    hist = sample["ego_history_xyz"][0].numpy()
    fut = sample["ego_future_xyz"][0].numpy()
    pts = np.concatenate([hist, np.zeros((1, 3)), fut], axis=0)

    K, T_vehicle_cam, size = load_calibration(scene_dir, view)
    uv, valid = project_ego_to_image(pts, K, T_vehicle_cam, image_size=size)

    img = sample["image_frames"][0, -1].permute(1, 2, 0).numpy().astype(np.uint8)
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    n_hist = len(hist)
    n_fut = len(fut)
    for i in range(len(pts)):
        if not valid[i]:
            continue
        u, v = uv[i]
        if i < n_hist:
            color, rad = (150, 150, 150), 6
        else:
            frac = (i - n_hist) / max(1, n_fut)
            color, rad = (int(60 + 195 * frac), int(220 - 180 * frac), 60), 10
        draw.ellipse([u - rad, v - rad, u + rad, v + rad], fill=color)
    _label(im, f"{Path(scene_dir).name}  t0={t0_actual:.1f}s  future trajectory "
               f"(green=near -> red=far)", 1.4)
    w, h = im.size
    im.resize((1600, int(1600 * h / w))).save(out_path, quality=85)
    return out_path


def render_all_views(scene_dir: str, t0: float, out_path: str, tile_w: int = 460) -> str:
    """Grid of all 15 Leopard views at ~t0, spatially arranged."""
    poses = _load_poses(scene_dir)
    tiles: dict[str, Image.Image] = {}
    for row in _VIEW_LAYOUT:
        for view in row:
            im = Image.open(_nearest_frame(scene_dir, poses, view, t0)).convert("RGB")
            w, h = im.size
            im = im.resize((tile_w, int(tile_w * h / w)))
            tiles[view] = _label(im, view.replace("_", " "))
    tile_h = next(iter(tiles.values())).size[1]
    cols, rows = 5, 3
    grid = Image.new("RGB", (cols * tile_w, rows * tile_h), (18, 18, 18))
    for r, row in enumerate(_VIEW_LAYOUT):
        for c, view in enumerate(row):
            grid.paste(tiles[view], (c * tile_w, r * tile_h))
    grid.save(out_path, quality=80)
    return out_path


def render_filmstrip(scene_dir: str, out_path: str, view: str = "forward_center_medium",
                     frame_w: int = 520, cols: int = 4) -> str:
    """One front-camera frame per second across the whole scene."""
    poses = _load_poses(scene_dir)
    targets = np.arange(1, int(poses.t_max) + 1)
    thumbs = []
    for tt in targets:
        im = Image.open(_nearest_frame(scene_dir, poses, view, float(tt))).convert("RGB")
        w, h = im.size
        thumbs.append(_label(im.resize((frame_w, int(frame_w * h / w))), f"t={tt}s", 0.8))
    rows = (len(thumbs) + cols - 1) // cols
    cw, ch = thumbs[0].size
    sheet = Image.new("RGB", (cols * cw, rows * ch), (18, 18, 18))
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * cw, (i // cols) * ch))
    sheet.save(out_path, quality=80)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("projection", "allviews", "filmstrip"))
    ap.add_argument("--scene", required=True, help="Path to a staged scene_* directory")
    ap.add_argument("--t0", type=float, default=8.0, help="Time (s) for projection/allviews")
    ap.add_argument("--view", default="forward_center_medium")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or f"truckdrive_{args.command}.jpg"
    if args.command == "projection":
        render_projection(args.scene, args.t0, out, view=args.view)
    elif args.command == "allviews":
        render_all_views(args.scene, args.t0, out)
    else:
        render_filmstrip(args.scene, out, view=args.view)
    print(f"wrote {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
