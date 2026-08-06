# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Generate navigation-command labels (straight / left / right) for TruckDrive.

TruckDrivePublic ships no navigation, route, or ego-intent annotation (the
devkit has only boxes, tracks and lane lines), so the command has to be derived
from ``poses/gt_trajectory.txt``. This script does that offline and writes a
sideband JSON that ``TruckDriveDataset`` merges into each sample as ``nav_text``
-- the same pattern as ``alpamayo.data.pai_nav``, which feeds the already-plumbed
``route`` chat-template component (``<|route_start|>...<|route_end|>``).

Pose-only: it reuses ``TruckDriveDataset``'s sample index and ``_ego_traj`` and
never loads an image, so a full-dataset pass costs minutes rather than hours.

The JSON keeps the raw geometry per window (heading change, lateral offset, path
length) alongside the label, so thresholds can be re-cut with ``--relabel``
without recomputing anything.

Label rule (three classes), on the ``num_future_steps * time_step`` horizon,
thresholded on **path curvature** expressed as equivalent turn radius
``R = path_length / |net heading|`` -- see :func:`_classify` for why net heading
alone is not enough:

    R <= turn_radius_m and |heading| >= min_turn_deg     -> "left" / "right"
    ...or the same holds for the tightest 1.5 s stretch  -> "left" / "right"
    nothing anywhere in the window bends appreciably     -> "straight"
    in between (the dead band)                           -> unlabeled

The second turn clause matters: whole-window curvature dilutes a turn that only
*begins* near the end of the horizon into "straight", which is precisely the
turn-onset case where the command is most useful.

Unlabeled windows simply carry no ``nav_text``; ``construct_route`` no-ops on
absent nav text, so they train unconditioned rather than on a guessed label.

Lane changes and gentle highway bends both land in "straight" by construction --
a lane change has ~0 net heading with a large lateral offset, and a highway
alignment curve has a radius in the hundreds of metres. That is deliberate: the
command means "which way does the road go", and folding either in with a 90 deg
yard turn would make the turn classes incoherent.

Usage::

    python -m truckdrive.nav_labels \
        --data-root s3://torc-data/datasets/TruckDrivePublic --backend s3 \
        --metadata-cache /mnt/efs/users/rod/truckdrive_cache/meta_5views.pkl \
        --out nav_labels_train.json --split train
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

# Recipe dir on sys.path so `truckdrive.*` and `alpamayo.*` resolve the same way
# the training entrypoints do.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpamayo.data.truckdrive import TruckDriveDataset  # noqa: E402


def _yaw_deg(rot: np.ndarray) -> np.ndarray:
    """Planar yaw (degrees) from a stack of (T, 3, 3) rotation matrices."""
    return np.degrees(np.arctan2(rot[:, 1, 0], rot[:, 0, 0]))


def _window_geometry(fut_xyz: np.ndarray, fut_rot: np.ndarray,
                     sub_window_steps: int = 15,
                     time_step: float = 0.1,
                     min_sub_path_m: float = 1.0) -> dict[str, float]:
    """Per-window scalars. Inputs are already in the ego frame at t0, so yaw at
    t0 is 0 by construction and the last future yaw *is* the net heading change.

    Besides the whole-window quantities this computes the **tightest sustained
    bend** anywhere inside the window, by sliding a short sub-window over the
    future and keeping the minimum equivalent radius. Whole-window curvature
    alone dilutes a turn that only *begins* near the end of the horizon (a long
    straight lead-in averages the bend away), which is exactly the turn-onset
    case where the nav command matters most.

    A sub-window rather than a single step is used because per-step yaw
    differences are noise-dominated at low speed; ``sub_window_steps`` at the
    default 0.1 s time step is 1.5 s of travel.
    """
    yaw = _yaw_deg(fut_rot)
    # Unwrap so a turn through +/-180 deg does not alias to the wrong sign.
    yaw_unwrapped = np.degrees(np.unwrap(np.radians(yaw)))
    step = np.linalg.norm(np.diff(fut_xyz[:, :2], axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(step)])

    # Tightest sustained bend: min radius over all sliding sub-windows.
    min_radius = np.inf
    min_radius_hdg = 0.0
    min_radius_speed = 0.0
    L = min(sub_window_steps, len(yaw_unwrapped) - 1)
    for i in range(0, len(yaw_unwrapped) - L):
        d_hdg = yaw_unwrapped[i + L] - yaw_unwrapped[i]
        d_path = cum[i + L] - cum[i]
        if d_path < min_sub_path_m or abs(d_hdg) < 1e-6:
            continue  # too little travel to define a direction of travel
        r = d_path / np.radians(abs(d_hdg))
        if r < min_radius:
            min_radius = r
            min_radius_hdg = d_hdg  # signed: gives the turn direction
            # Speed over the same stretch, so the label rule can reject
            # physically impossible bends (see _classify's lateral-accel gate).
            min_radius_speed = d_path / (L * time_step)

    return {
        "heading_change_deg": float(yaw_unwrapped[-1]),
        # Total absolute turning, which separates "curved then straightened"
        # (e.g. a lane change) from a sustained turn with the same net heading.
        "abs_yaw_path_deg": float(np.abs(np.diff(yaw_unwrapped)).sum()),
        "lateral_offset_m": float(fut_xyz[-1, 1]),
        "path_length_m": float(step.sum()),
        # inf is not JSON-representable; -1 marks "no valid sub-window".
        "min_radius_m": float(min_radius) if np.isfinite(min_radius) else -1.0,
        "min_radius_hdg_deg": float(min_radius_hdg),
        "min_radius_speed_mps": float(min_radius_speed),
        "mean_speed_mps": float(step.sum() / ((len(step)) * time_step)),
    }


def _classify(g: dict[str, float], turn_radius_m: float, straight_radius_m: float,
              min_turn_deg: float, min_sub_turn_deg: float,
              max_lat_acc_mps2: float, min_path_m: float) -> str | None:
    """Geometry -> class name, or None to leave the window unlabeled.

    Thresholds on **path curvature**, not on net heading change. Net heading
    alone conflates a 90 deg yard turn (20 m of travel) with a gentle highway
    bend (170 m of travel through the same arc is a ~620 m-radius curve, which
    is lane-following, not a turn). Measured on the full dataset, 38% of
    heading-thresholded turn labels were highway windows at a median 15.7 deg --
    i.e. the turn classes meant two unrelated things.

    Curvature is expressed as the equivalent turn radius ``R = path / |heading|``
    (in radians), which is interpretable: an intersection or yard turn is
    R ~ 15-30 m, a freeway ramp R ~ 50-150 m, a highway alignment curve
    R > 500 m.

    Two conditions must BOTH hold for a turn, so that neither a tight-but-tiny
    wiggle nor a long gentle drift can produce one:
      * ``R <= turn_radius_m``  (the path actually bends)
      * ``|net heading| >= min_turn_deg``  (it bends by an amount that matters)

    Windows between ``turn_radius_m`` and ``straight_radius_m`` fall in the dead
    band and are left unlabeled rather than guessed at.
    """
    h = g["heading_change_deg"]
    path = g["path_length_m"]
    # A near-stationary window has no well-defined direction of travel; the
    # position-tangent heading is held rather than measured there.
    if path < min_path_m:
        return None
    # Equivalent turn radius; infinite for a perfectly straight window.
    radius = np.inf if abs(h) < 1e-6 else path / np.radians(abs(h))

    # Tightest sustained bend inside the window (-1 = no valid sub-window).
    sub_r = g.get("min_radius_m", -1.0)
    sub_h = g.get("min_radius_hdg_deg", 0.0)
    sub_v = g.get("min_radius_speed_mps", 0.0)
    has_sub = sub_r > 0

    # PHYSICAL PLAUSIBILITY GATE. A bend of radius R taken at speed v implies
    # lateral acceleration v^2/R. A loaded semi rolls over near 3-4 m/s^2, so a
    # window claiming more than that is measurement noise, not a turn -- short
    # sub-windows at highway speed are especially prone to it (yaw jitter over
    # 1.5 s at 30 m/s manufactures a spurious tight radius). Measured: the
    # ungated sub-window clause recovered 174 "highway turns" at a median
    # 7.3 m/s^2, 94% of them above 3 -- all spurious. Whole-window turns sit at
    # a median 0.95 m/s^2, so this gate barely touches them.
    def _plausible(r: float, v: float) -> bool:
        return r > 0 and (v * v) / r <= max_lat_acc_mps2

    v_mean = g.get("mean_speed_mps", 0.0)

    # Turn if EITHER the whole window bends, OR some 1.5 s stretch of it does.
    # The second clause catches turn onset near the end of the horizon, which the
    # whole-window average dilutes into "straight". Each clause carries its own
    # heading floor because the two are measured over different arc lengths.
    if (radius <= turn_radius_m and abs(h) >= min_turn_deg
            and _plausible(radius, v_mean)):
        return "left" if h > 0 else "right"
    if (has_sub and sub_r <= turn_radius_m and abs(sub_h) >= min_sub_turn_deg
            and _plausible(sub_r, sub_v)):
        return "left" if sub_h > 0 else "right"

    # Straight only if nothing anywhere in the window bends appreciably.
    if radius >= straight_radius_m and (not has_sub or sub_r >= straight_radius_m):
        return "straight"
    if abs(h) < min_turn_deg and (not has_sub or abs(sub_h) < min_sub_turn_deg):
        return "straight"  # bend is tight but too small to matter (slow wiggle)
    return None  # dead band: a coin-flip label is worse than none


NAV_TEXT = {
    # Plain English rather than bare enum tokens: the route content goes through
    # the ordinary text tokenizer, so in-distribution phrasing is likelier to
    # land on useful pretrained embeddings.
    "straight": "Continue straight",
    "left": "Turn left",
    "right": "Turn right",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--backend", default="local", choices=["local", "s3"])
    p.add_argument("--metadata-cache", default=None)
    p.add_argument("--split-metainfo", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--scene-ids", nargs="*", default=None,
                   help="Restrict to these scenes (debugging / sign checks).")
    p.add_argument("--max-scenes", type=int, default=0)
    p.add_argument("--camera-views", nargs="*", default=[
        "forward_center_medium", "sideward_left_front_wide",
        "sideward_right_front_wide", "rearward_left_bottom_medium",
        "rearward_right_bottom_medium",
    ], help="Must match the training config: it decides which scenes are usable.")
    p.add_argument("--num-future-steps", type=int, default=64)
    p.add_argument("--num-history-steps", type=int, default=16)
    p.add_argument("--time-step", type=float, default=0.1)
    p.add_argument("--t0-stride", type=int, default=10)
    p.add_argument("--heading-source", default="position",
                   choices=["position", "quaternion"])
    p.add_argument("--turn-radius-m", type=float, default=200.0,
                   help="Equivalent turn radius at/below which a window is a turn.")
    p.add_argument("--straight-radius-m", type=float, default=300.0,
                   help="Radius at/above which a window is straight. Between this "
                        "and --turn-radius-m is the dead band (unlabeled).")
    p.add_argument("--min-turn-deg", type=float, default=10.0,
                   help="A turn must also change net heading by at least this much.")
    p.add_argument("--min-sub-turn-deg", type=float, default=6.0,
                   help="Heading floor for the tightest-sub-window turn test "
                        "(measured over --sub-window-steps, not the full horizon).")
    p.add_argument("--sub-window-steps", type=int, default=15,
                   help="Sliding sub-window length in steps for the tightest-bend "
                        "scan (15 * time_step = 1.5 s at defaults).")
    p.add_argument("--max-lat-acc-mps2", type=float, default=3.0,
                   help="Reject turns implying more lateral accel than this "
                        "(v^2/R). A loaded semi rolls over near 3-4 m/s^2, so "
                        "anything above is pose noise, not a turn.")
    p.add_argument("--min-path-m", type=float, default=2.0)
    p.add_argument("--out", required=True)
    p.add_argument("--relabel", default=None,
                   help="Re-cut thresholds on an existing JSON; skips all pose I/O.")
    args = p.parse_args()

    if args.relabel:
        with open(args.relabel) as f:
            payload = json.load(f)
        records = payload["records"]
    else:
        ds = TruckDriveDataset(
            data_root=args.data_root,
            backend=args.backend,
            metadata_cache=args.metadata_cache,
            split_metainfo=args.split_metainfo,
            split=args.split,
            scene_ids=args.scene_ids,
            max_scenes=args.max_scenes,
            camera_views=args.camera_views,
            num_history_steps=args.num_history_steps,
            num_future_steps=args.num_future_steps,
            time_step=args.time_step,
            t0_stride=args.t0_stride,
            heading_source=args.heading_source,
            # Labels are keyed by (scene_id, t0_us), so the tokenizer-recon
            # pre-filter is irrelevant here -- label every window the index has
            # and let training-time filtering drop what it drops.
            tok_recon_max_xy_m=None,
            # No images are read; the loader only needs the pose track.
            vla_preprocess_args=None,
        )
        print(f"indexed {len(ds._samples)} windows "  # noqa: SLF001
              f"over {len(set(s for s, _ in ds._samples))} scenes")  # noqa: SLF001

        records = []
        for i, (scene_id, t0) in enumerate(ds._samples):  # noqa: SLF001
            poses = ds._load_scene(scene_id)["poses"]  # noqa: SLF001
            _, _, fut_xyz, fut_rot = ds._ego_traj(poses, t0)  # noqa: SLF001
            rec = {"scene_id": scene_id, "t0_us": int(round(t0 * 1e6))}
            rec.update(_window_geometry(fut_xyz, fut_rot,
                                        sub_window_steps=args.sub_window_steps,
                                        time_step=args.time_step))
            records.append(rec)
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(ds._samples)}", flush=True)  # noqa: SLF001

    counts: Counter[str] = Counter()
    out_records = []
    for rec in records:
        label = _classify(rec, args.turn_radius_m, args.straight_radius_m,
                          args.min_turn_deg, args.min_sub_turn_deg,
                          args.max_lat_acc_mps2, args.min_path_m)
        counts[label or "unlabeled"] += 1
        r = dict(rec)
        r["nav_class"] = label
        if label is not None:
            r["nav_text"] = NAV_TEXT[label]
        out_records.append(r)

    payload = {
        "config": {
            "turn_radius_m": args.turn_radius_m,
            "straight_radius_m": args.straight_radius_m,
            "min_turn_deg": args.min_turn_deg,
            "min_sub_turn_deg": args.min_sub_turn_deg,
            "max_lat_acc_mps2": args.max_lat_acc_mps2,
            "sub_window_steps": args.sub_window_steps,
            "min_path_m": args.min_path_m,
            "heading_source": args.heading_source,
            "horizon_s": args.num_future_steps * args.time_step,
            "nav_text": NAV_TEXT,
        },
        "records": out_records,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f)

    n = len(out_records)
    print(f"\nwrote {n} windows -> {args.out}")
    for k in ("straight", "left", "right", "unlabeled"):
        print(f"  {k:<10} {counts[k]:>7}  {100.0 * counts[k] / max(n, 1):5.1f}%")

    h = np.array([r["heading_change_deg"] for r in out_records])
    print("\n|heading change| percentiles (deg):")
    for q in (50, 75, 90, 95, 99, 99.9):
        print(f"  p{q:<5} {np.percentile(np.abs(h), q):7.2f}")
    print(f"  max    {np.abs(h).max():7.2f}")


if __name__ == "__main__":
    main()
