# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Measure per-clip ego motion over the RL prediction horizon, without loading images.

Why this exists: the shipped motion reward has a degenerate optimum, and it is
**coasting straight ahead**, not standing still. Trajectory tokens are
``(accel, curvature)`` pairs in an ``UnicycleAccelCurvatureActionSpace`` encoded
by a ``DeltaTrajectoryTokenizer``, so one token repeated decodes to constant
accel + constant curvature -- a straight line at the current speed. That output:

* scores a **perfect (0.0) comfort term**, because the six comfort metrics are
  exactly lon/lat accel, jerk, lon jerk, yaw accel and yaw rate, all of which are
  zero for constant-velocity straight-line motion; and
* is **already accurate on most of PAI** -- measured on 891 train clips of the
  19-chunk nav subset, 50.8% have ``cv_ade < 3.0 m``, i.e. coasting alone beats
  the hard ``ADE >= 3.0 -> reward = -1.0`` gate with no learning at all, and
  20.1% are under 1 m.

A 2026-07-30 run collapsed onto exactly that mode: completions became ``<i1498>``
x128 with the CoT dropped entirely, entropy fell 6.5 -> 0.7 and grad_norm
2.9 -> 0.03. Filtering to clips where coasting is *wrong* (turns, braking,
accelerating) removes the clips that teach the policy nothing, which is a
precondition for a meaningful overfit check.

Note this subsumes filtering "stationary" clips: with ``v0 = 0`` the baseline
predicts no motion and GT has none, so stationary clips score ``cv_ade ~ 0`` and
are excluded by the same threshold.

``PAIDataset.__getitem__`` cannot be used for this: it calls
``load_physical_aiavdataset``, which decodes 4 cameras x 4 frames per clip.
Passing ``camera_features=[]`` does not help either -- the function ends in
``torch.stack(image_frames_list)``, which raises on an empty list. So this script
reimplements just the ego-trajectory part of that function (its lines ~102-153:
timestamp construction, then transform into the t0 ego frame) against the same
``PhysicalAIAVDatasetLocalInterface``. Egomotion is a small parquet per chunk, so
a full pass over ~900 clips is minutes rather than hours.

Reported per clip, all over the future horizon only (``num_future_steps *
time_step`` = 6.4 s by default):

* ``cv_ade``    -- ADE of a constant-velocity straight extrapolation vs GT [m].
                   **This is the column that matters**: it measures directly how
                   good the degenerate coast-straight answer is on this clip.
* ``disp``      -- straight-line distance from t0 to the final future pose [m].
* ``path_len``  -- arc length of the future path [m]; > disp when turning.
* ``yaw_deg``   -- absolute net heading change over the horizon [deg].
* ``v0`` / ``vT`` / ``dv`` -- speed at the start/end of the horizon and |change|.

Usage::

    python scripts_fork/profile_clip_motion.py \\
        --dataset-root ~/datasets/pai_nav_subset_19chunks \\
        --clip-index clip_index_mini_891train.parquet.bak \\
        --out /tmp/clip_motion.parquet

    # then build an overfit index of clips where coasting is WRONG
    python scripts_fork/profile_clip_motion.py ... \\
        --min-cv-ade 5.0 --min-disp 5.0 --num-samples 16 \\
        --write-index ~/datasets/pai_nav_subset_19chunks/clip_index_mini.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_T0_US = 5_100_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument(
        "--clip-index",
        default="clip_index.parquet",
        help="Index filename under --dataset-root to profile (default: %(default)s)",
    )
    p.add_argument("--features-metadata", default="features.csv")
    p.add_argument("--num-future-steps", type=int, default=64)
    p.add_argument("--num-history-steps", type=int, default=16)
    p.add_argument("--time-step", type=float, default=0.1)
    p.add_argument(
        "--t0-us",
        type=int,
        default=DEFAULT_T0_US,
        help="Keyframe time; matches use_default_keyframe=True in the RL config.",
    )
    p.add_argument("--limit", type=int, default=None, help="Profile only the first N clips.")
    p.add_argument("--out", type=Path, default=None, help="Write the full per-clip table here.")
    # index-building options
    p.add_argument(
        "--min-disp",
        type=float,
        default=None,
        help="Keep only clips whose future displacement >= this many metres.",
    )
    p.add_argument(
        "--min-cv-ade",
        type=float,
        default=None,
        help=(
            "Keep only clips where the constant-velocity-straight baseline is at least this "
            "wrong [m]. This is the filter that matters: it excludes clips on which the "
            "degenerate coast-straight policy already scores well."
        ),
    )
    p.add_argument("--num-samples", type=int, default=None, help="Sample N of the kept clips.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--write-index",
        type=Path,
        default=None,
        help="Write a curated clip index (same schema as clip_index) to this path.",
    )
    return p.parse_args()


def ego_future_local(avdi, clip_id: str, args) -> np.ndarray | None:
    """Return the future trajectory in the t0 ego frame, shape [T, 3].

    Mirrors ``alpamayo_r1.load_physical_aiavdataset`` lines ~102-153.
    """
    import scipy.spatial.transform as spt

    egomotion = avdi.get_clip_feature(clip_id, avdi.features.LABELS.EGOMOTION)

    ts = args.time_step * 1_000_000
    history_offsets = np.arange(-(args.num_history_steps - 1) * ts, ts / 2, ts).astype(np.int64)
    future_offsets = np.arange(ts, (args.num_future_steps + 0.5) * ts, ts).astype(np.int64)

    ego_hist = egomotion(args.t0_us + history_offsets)
    ego_fut = egomotion(args.t0_us + future_offsets)

    t0_xyz = ego_hist.pose.translation[-1].copy()
    t0_rot = spt.Rotation.from_quat(ego_hist.pose.rotation.as_quat()[-1].copy())
    t0_rot_inv = t0_rot.inv()

    hist_xyz_local = t0_rot_inv.apply(ego_hist.pose.translation - t0_xyz)
    fut_xyz_local = t0_rot_inv.apply(ego_fut.pose.translation - t0_xyz)
    fut_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_fut.pose.rotation.as_quat())).as_matrix()
    return hist_xyz_local, fut_xyz_local, fut_rot_local


def constant_velocity_ade(hist_xy: np.ndarray, fut_xy: np.ndarray, time_step: float) -> float:
    """ADE of a constant-velocity straight-ahead extrapolation against GT.

    This is the metric that actually matters for the degenerate mode. Trajectory
    tokens are (accel, curvature) in a ``UnicycleAccelCurvatureActionSpace``, so a
    constant repeated token decodes to *constant accel + constant curvature* --
    i.e. coasting straight at the current speed, not standing still. Such a
    trajectory also has zero accel / jerk / yaw-rate, so it scores a perfect
    (0.0) comfort term. Clips where this baseline is already accurate teach the
    policy nothing; clips where it is wrong (turns, braking, accelerating) are
    the ones worth overfitting on.
    """
    # Initial velocity from the last history step, in the t0 ego frame.
    v = (hist_xy[-1] - hist_xy[-2]) / time_step if len(hist_xy) >= 2 else np.zeros(2)
    t = np.arange(1, len(fut_xy) + 1) * time_step
    pred = np.outer(t, v)  # straight line at constant velocity
    return float(np.linalg.norm(pred - fut_xy, axis=-1).mean())


def main() -> None:
    args = parse_args()
    from alpamayo.data.pai_utils import PhysicalAIAVDatasetLocalInterface

    root = args.dataset_root.expanduser()
    avdi = PhysicalAIAVDatasetLocalInterface(
        local_dir=str(root),
        chunk_ids=None,
        features_metadata=args.features_metadata,
        clip_index_metadata=args.clip_index,
        reasoning_metadata=None,
    )
    clip_ids = avdi.get_all_clip_ids()
    if args.limit:
        clip_ids = clip_ids[: args.limit]
    print(f"[profile] {len(clip_ids)} clips from {root / args.clip_index}")

    rows = []
    for i, clip_id in enumerate(clip_ids):
        try:
            hist_xyz, fut_xyz, fut_rot = ego_future_local(avdi, clip_id, args)
            xy = fut_xyz[:, :2]
            disp = float(np.linalg.norm(xy[-1]))
            path_len = float(np.linalg.norm(np.diff(xy, axis=0), axis=-1).sum())
            yaw = np.degrees(np.arctan2(fut_rot[:, 1, 0], fut_rot[:, 0, 0]))
            yaw_deg = float(abs(yaw[-1] - yaw[0]))
            # speed at start vs end of the horizon -> how much accel is required
            step = args.time_step
            v0 = float(np.linalg.norm(hist_xyz[-1, :2] - hist_xyz[-2, :2]) / step)
            vT = float(np.linalg.norm(xy[-1] - xy[-2]) / step)
            cv_ade = constant_velocity_ade(hist_xyz[:, :2], xy, step)
            rows.append(
                {
                    "clip_id": clip_id,
                    "disp": disp,
                    "path_len": path_len,
                    "yaw_deg": yaw_deg,
                    "v0": v0,
                    "vT": vT,
                    "dv": abs(vT - v0),
                    "cv_ade": cv_ade,
                }
            )
        except Exception as e:
            print(f"[profile] skip {clip_id}: {type(e).__name__}: {e}")
        if (i + 1) % 100 == 0:
            print(f"[profile]   {i + 1}/{len(clip_ids)}")

    df = pd.DataFrame(rows).set_index("clip_id")
    if df.empty:
        raise SystemExit("[profile] no clips profiled successfully")

    pcts = [0.1, 0.25, 0.5, 0.75, 0.9]
    print(f"\n[profile] {len(df)} clips profiled.")
    print(df[["disp", "yaw_deg", "v0", "dv", "cv_ade"]].describe(percentiles=pcts).to_string())
    # cv_ade is the key column: how good the degenerate coast-straight answer is.
    print("\n[profile] constant-velocity-straight baseline ADE (the degenerate mode):")
    for thr in (0.5, 1.0, 2.0, 3.0, 5.0, 10.0):
        n = int((df["cv_ade"] < thr).sum())
        print(f"[profile]   cv_ade < {thr:5.1f} m : {n:4d} clips ({100 * n / len(df):5.1f}%)"
              + ("   <- coasting already beats the 3m gate" if thr == 3.0 else ""))
    print("\n[profile] turning / speed-change content:")
    for thr in (2.0, 5.0, 10.0, 20.0):
        n = int((df["yaw_deg"] > thr).sum())
        print(f"[profile]   yaw_deg > {thr:5.1f} deg : {n:4d} clips ({100 * n / len(df):5.1f}%)")
    for thr in (1.0, 2.0, 5.0):
        n = int((df["dv"] > thr).sum())
        print(f"[profile]   |dv|    > {thr:5.1f} m/s : {n:4d} clips ({100 * n / len(df):5.1f}%)")

    if args.out:
        df.to_parquet(args.out)
        print(f"[profile] wrote table -> {args.out}")

    if args.write_index:
        if args.min_disp is None and args.min_cv_ade is None:
            raise SystemExit("--write-index requires --min-disp and/or --min-cv-ade")
        keep = df
        crit = []
        if args.min_disp is not None:
            keep = keep[keep["disp"] >= args.min_disp]
            crit.append(f"disp >= {args.min_disp} m")
        if args.min_cv_ade is not None:
            keep = keep[keep["cv_ade"] >= args.min_cv_ade]
            crit.append(f"cv_ade >= {args.min_cv_ade} m")
        print(f"\n[profile] {len(keep)} clips with " + " and ".join(crit))
        if keep.empty:
            raise SystemExit("[profile] nothing left after --min-disp filter")
        src = pd.read_parquet(root / args.clip_index)
        out = src.loc[src.index.intersection(keep.index)]
        if args.num_samples:
            n = min(args.num_samples, len(out))
            if n < args.num_samples:
                print(f"[profile] warning: only {n} clips available, requested {args.num_samples}")
            out = out.sample(n=n, random_state=args.seed)
        out.to_parquet(args.write_index)
        kept = df.loc[out.index]
        print(f"[profile] wrote index ({len(out)} clips) -> {args.write_index}")
        print(f"[profile] disp  min {kept['disp'].min():.1f}  median {kept['disp'].median():.1f}"
              f"  max {kept['disp'].max():.1f} m")
        print(f"[profile] yaw   min {kept['yaw_deg'].min():.1f}  median {kept['yaw_deg'].median():.1f}"
              f"  max {kept['yaw_deg'].max():.1f} deg")
        print(f"[profile] cv_ade min {kept['cv_ade'].min():.2f}  median {kept['cv_ade'].median():.2f}"
              f"  max {kept['cv_ade'].max():.2f} m")


if __name__ == "__main__":
    main()
