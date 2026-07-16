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

"""Prebuild the TruckDrive scene-metadata cache (pose text + frame listings).

Dataset init over S3 spends its time on *metadata* (one pose GET + one LIST per
view per scene). Prebuilding that once into a portable pickle makes every later
``TruckDriveDataset(..., metadata_cache=...)`` init S3-free -- only image bytes
are streamed during training. Write the cache to shared storage (EFS) so any
instance can train from it.

Example::

    python -m truckdrive.build_cache \
        --data-root s3://torc-data/datasets/TruckDrivePublic \
        --cache /mnt/efs/users/rod/truckdrive_cache/meta_5views.pkl \
        --views forward_center_medium,sideward_left_front_wide,\
sideward_right_front_wide,rearward_left_bottom_medium,rearward_right_bottom_medium \
        --threads 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from alpamayo.data.truckdrive import build_metadata_cache  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True,
                    help="s3://bucket/prefix (or local dir) with scene_* subdirs")
    ap.add_argument("--cache", required=True,
                    help="Output pickle path (put on EFS to share across instances)")
    ap.add_argument("--views", required=True,
                    help="Comma-separated Leopard camera views to cache listings for")
    ap.add_argument("--backend", choices=("s3", "local"), default="s3")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--pose-file", default="poses/gt_trajectory.txt")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--max-scenes", type=int, default=0,
                    help="Cap scenes (testing only; 0 = all)")
    args = ap.parse_args()

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    cache = build_metadata_cache(
        data_root=args.data_root,
        cache_path=args.cache,
        camera_views=views,
        backend=args.backend,
        region=args.region,
        pose_file=args.pose_file,
        threads=args.threads,
        max_scenes=args.max_scenes,
    )
    n = len(cache["scenes"])
    total_frames = sum(
        len(files) for e in cache["scenes"].values() for files in e["view_files"].values()
    )
    print(f"cache written: {args.cache}")
    print(f"  scenes: {n}   views: {cache['camera_views']}   frame listings: {total_frames}")


if __name__ == "__main__":
    main()
