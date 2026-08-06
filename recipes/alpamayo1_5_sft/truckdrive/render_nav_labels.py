# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Visual spot-check for the derived navigation-command labels.

Renders a stratified sample of windows from a ``nav_labels`` JSON (see
``truckdrive/nav_labels.py``) as one PNG each: the forward camera at t0 with the
GT future trajectory projected onto it, a bird's-eye view of the same future,
and a caption carrying the assigned label plus the geometry that produced it.

Why this exists: the label rule has twice produced physically wrong labels that
the class counts looked perfectly healthy for (net-heading thresholding pulled
in 620 m-radius highway bends; the ungated sub-window clause invented highway
turns from pose jitter). Both were caught by checking a physical quantity, never
by label balance. Pixels are the independent check -- does the label match what
the truck is visibly doing?

Sampling is deliberately adversarial: it over-weights the categories most likely
to be wrong (onset recoveries, near-threshold windows, highway turns, dead-band
rejects) rather than drawing uniformly, so a small render batch actually
exercises the rule's edges.

Reads images straight from S3 -- no staging needed.

Example::

    python -m truckdrive.render_nav_labels \
        --labels /mnt/efs/users/rod/truckdrive_cache/nav_labels_all_v3.json \
        --data-root s3://torc-data/datasets/TruckDrivePublic --backend s3 \
        --metadata-cache /mnt/efs/users/rod/truckdrive_cache/meta_5views.pkl \
        --out-dir /mnt/efs/users/rod/truckdrive_nav_label_renders --per-group 6

``--video SCENE_ID [...]`` renders an MP4 per scene instead: every labelled
window of that scene in time order, with the nav command and t0 burned large
into the camera image and the BEV axes pinned to the scene's extent, so a whole
scene can be scrubbed for label errors rather than eyeballed one PNG at a time::

    python -m truckdrive.render_nav_labels \
        --labels /mnt/efs/users/rod/truckdrive_cache/nav_labels_all_v3.json \
        --data-root s3://torc-data/datasets/TruckDrivePublic --backend s3 \
        --metadata-cache /mnt/efs/users/rod/truckdrive_cache/meta_5views.pkl \
        --out-dir /mnt/efs/users/rod/truckdrive_nav_videos \
        --video scene_25_1083 --fps 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpamayo.data.truckdrive import TruckDriveDataset  # noqa: E402
from truckdrive.projection import load_calibration_via, project_ego_to_image  # noqa: E402

_VIEW = "forward_center_medium"


def _lat_acc(r: dict) -> float:
    """Implied lateral acceleration of the tightest sub-window bend."""
    if r.get("min_radius_m", -1) <= 0:
        return 0.0
    return r.get("min_radius_speed_mps", 0.0) ** 2 / r["min_radius_m"]


def _groups(records: list[dict], turn_radius_m: float) -> dict[str, list[dict]]:
    """Bucket windows into the cases worth eyeballing, hardest first.

    Every bucket targets a specific way the rule could be wrong, so a failure in
    the renders points at which clause to fix rather than just "labels look off".
    """
    g: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        cls = r.get("nav_class")
        speed = r.get("mean_speed_mps", 0.0)
        sub_r = r.get("min_radius_m", -1.0)
        net_r = (r["path_length_m"] / np.radians(abs(r["heading_change_deg"]))
                 if abs(r["heading_change_deg"]) > 1e-6 else np.inf)

        if cls in ("left", "right"):
            # Onset recovery: the whole window does NOT qualify, only a short
            # stretch of it does. These are the labels the sub-window clause
            # added, and the ones most likely to be spurious.
            if net_r > turn_radius_m:
                g["turn_onset_recovery"].append(r)
            elif speed > 18:
                g["turn_highway"].append(r)      # should be ramp geometry only
            elif speed < 3:
                g["turn_lowspeed_yard"].append(r)
            else:
                g["turn_normal"].append(r)
            if _lat_acc(r) > 3.0:
                g["turn_high_lat_acc"].append(r)  # survived the gate; verify
        elif cls == "straight":
            if sub_r > 0 and sub_r < 400:
                g["straight_borderline"].append(r)  # bends, but not enough
            elif speed < 3:
                g["straight_lowspeed"].append(r)
            else:
                g["straight_normal"].append(r)
        else:
            g["unlabeled_deadband"].append(r)
    return g


def _bev_png(rec: dict, hist: np.ndarray, fut: np.ndarray, size_px: int,
             lims: tuple[float, float, float, float] | None = None) -> np.ndarray:
    """Bird's-eye view of the window in the ego@t0 frame (x forward, y left).

    ``lims`` = (y_min, y_max, x_min, x_max) pins the axes; video mode passes the
    whole scene's extent so the view does not jitter frame to frame.
    """
    fig = plt.figure(figsize=(size_px / 100, size_px / 100), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(-hist[:, 1], hist[:, 0], color="0.55", lw=2, label="GT history")
    ax.plot(-fut[:, 1], fut[:, 0], color="crimson", lw=2.5, label="GT future")
    ax.scatter([0], [0], c="k", s=40, zorder=5)
    ax.set_aspect("equal")
    if lims is not None:
        ax.set_xlim(lims[0], lims[1])
        ax.set_ylim(lims[2], lims[3])
    ax.grid(alpha=0.3)
    ax.set_xlabel("← right   y [m]   left →")
    ax.set_ylabel("forward x [m]")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{rec['nav_class'] or 'UNLABELED'}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def _caption(rec: dict) -> str:
    net_r = (rec["path_length_m"] / np.radians(abs(rec["heading_change_deg"]))
             if abs(rec["heading_change_deg"]) > 1e-6 else float("inf"))
    return (
        f"{rec['scene_id']}  t0={rec['t0_us'] / 1e6:.2f}s   ->  "
        f"LABEL = {str(rec['nav_class']).upper()}\n"
        f"net heading {rec['heading_change_deg']:+.1f} deg   net R {net_r:.0f} m   "
        f"sub R {rec.get('min_radius_m', -1):.0f} m (hdg {rec.get('min_radius_hdg_deg', 0):+.1f} deg)\n"
        f"mean speed {rec.get('mean_speed_mps', 0):.1f} m/s   "
        f"lat acc {_lat_acc(rec):.2f} m/s2   lateral offset {rec['lateral_offset_m']:+.1f} m"
    )


def _font(size: int):
    """Monospace truetype at ``size`` px, falling back to PIL's bitmap default."""
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def _compose(ds: TruckDriveDataset, rec: dict,
             lims: tuple[float, float, float, float] | None = None,
             banner: bool = False) -> Image.Image | None:
    """Build the two-panel figure for one window, or None if it has no window.

    ``banner=True`` additionally burns the nav command and t0 into the top-left
    of the camera image at large size -- what makes the video readable while it
    plays, where the small caption at the bottom is not.
    """
    # The label index was built with t0_stride=10; re-find the nearest t0 this
    # dataset actually has (same pose timeline, so it lands on the same window).
    t0_target = rec["t0_us"] / 1e6
    cands = [(abs(t - t0_target), i) for i, (s, t) in enumerate(ds._samples)  # noqa: SLF001
             if s == rec["scene_id"]]
    if not cands:
        return None
    dt, idx = min(cands)
    if dt > 0.15:
        return None
    sample = ds[idx]

    hist = sample["ego_history_xyz"][0].numpy()
    fut = sample["ego_future_xyz"][0].numpy()
    img = sample["image_frames"][0, -1].permute(1, 2, 0).numpy().astype(np.uint8)
    im = Image.fromarray(img)

    try:
        K, T_cam, size = load_calibration_via(
            ds.backend.read_bytes, rec["scene_id"], _VIEW  # noqa: SLF001
        )
        # The loader resizes images; project in native pixels then rescale.
        uv, valid = project_ego_to_image(fut, K, T_cam, image_size=size)
        sx, sy = im.size[0] / size[0], im.size[1] / size[1]
        draw = ImageDraw.Draw(im)
        for i in range(len(fut)):
            if not valid[i]:
                continue
            u, v = uv[i][0] * sx, uv[i][1] * sy
            frac = i / max(1, len(fut) - 1)
            color = (int(60 + 195 * frac), int(220 - 180 * frac), 60)
            r = 7
            draw.ellipse([u - r, v - r, u + r, v + r], fill=color)
    except Exception as exc:  # noqa: BLE001 - a missing calib should not abort the batch
        print(f"    ! projection failed for {rec['scene_id']}: {exc!r}")

    cam_w = 1100
    im = im.resize((cam_w, int(cam_w * im.size[1] / im.size[0])))

    if banner:
        label = str(rec["nav_class"] or "unlabeled").upper()
        line1 = f"NAV: {label}"
        line2 = f"{rec['scene_id']}   t0={rec['t0_us'] / 1e6:6.2f}s"
        f1, f2 = _font(46), _font(30)
        d = ImageDraw.Draw(im, "RGBA")
        d.rectangle([0, 0, 640, 108], fill=(0, 0, 0, 170))
        d.text((14, 6), line1, font=f1,
               fill={"LEFT": (120, 200, 255), "RIGHT": (255, 190, 90),
                     "STRAIGHT": (150, 245, 150)}.get(label, (200, 200, 200)))
        d.text((14, 62), line2, font=f2, fill=(235, 235, 235))

    bev = Image.fromarray(_bev_png(rec, hist, fut, size_px=im.size[1], lims=lims))

    pad, cap_h = 12, 96
    W = im.size[0] + bev.size[0] + 3 * pad
    H = max(im.size[1], bev.size[1]) + cap_h + 2 * pad
    canvas = Image.new("RGB", (W, H), (20, 20, 22))
    canvas.paste(im, (pad, pad))
    canvas.paste(bev, (im.size[0] + 2 * pad, pad))
    ImageDraw.Draw(canvas).multiline_text(
        (pad, max(im.size[1], bev.size[1]) + pad + 8), _caption(rec),
        fill=(235, 235, 235), spacing=6,
    )
    return canvas


def _render_one(ds: TruckDriveDataset, rec: dict, out_path: str) -> bool:
    canvas = _compose(ds, rec)
    if canvas is None:
        return False
    canvas.save(out_path, quality=88)
    return True


def _scene_lims(ds: TruckDriveDataset, recs: list[dict],
                margin_m: float = 5.0) -> tuple[float, float, float, float]:
    """BEV extent covering every window of the scene (pose-only, no image loads).

    Fixed axes across the video are what let the eye read the *path* changing
    rather than the axes rescaling under it.
    """
    ys, xs = [0.0], [0.0]
    for rec in recs:
        poses = ds._load_scene(rec["scene_id"])["poses"]  # noqa: SLF001
        hist_xyz, _, fut_xyz, _ = ds._ego_traj(poses, rec["t0_us"] / 1e6)  # noqa: SLF001
        for arr in (hist_xyz, fut_xyz):
            ys.extend((-arr[:, 1]).tolist())
            xs.extend(arr[:, 0].tolist())
    return (min(ys) - margin_m, max(ys) + margin_m,
            min(xs) - margin_m, max(xs) + margin_m)


def render_scene_video(ds: TruckDriveDataset, recs: list[dict], out_path: str,
                       fps: float = 4.0) -> int:
    """Write an MP4 of one scene's labelled windows in time order.

    One frame per window, each carrying its nav command and t0, so a whole scene
    can be scrubbed for label errors instead of eyeballing per-window PNGs.
    """
    import av  # H.264 via PyAV, as render_scene_video.py does

    recs = sorted(recs, key=lambda r: r["t0_us"])
    lims = _scene_lims(ds, recs)
    container = av.open(out_path, mode="w")
    stream = container.add_stream("libx264", rate=int(round(fps)))
    stream.pix_fmt = "yuv420p"
    size: tuple[int, int] | None = None  # (w, h), both even
    n = 0
    try:
        for rec in recs:
            canvas = _compose(ds, rec, lims=lims, banner=True)
            if canvas is None:
                print(f"    ! no matching window for t0={rec['t0_us'] / 1e6:.2f}s")
                continue
            arr = np.asarray(canvas)
            if size is None:
                # yuv420p needs even dimensions; an odd-sized stream decodes as
                # garbage (this is what made the first cut play back solid green).
                h, w = arr.shape[0] - arr.shape[0] % 2, arr.shape[1] - arr.shape[1] % 2
                size = (w, h)
                stream.width, stream.height = w, h
            w, h = size
            if (arr.shape[1], arr.shape[0]) != (w, h):
                arr = cv2.resize(arr, (w, h))
            frame = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(arr[:h, :w]), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
            n += 1
        if n:
            for packet in stream.encode():
                container.mux(packet)
    finally:
        container.close()
    return n


def _video_main(args: argparse.Namespace, records: list[dict]) -> None:
    """``--video`` path: one MP4 per requested scene."""
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["scene_id"] in set(args.video):
            if args.t_range is not None:
                t = r["t0_us"] / 1e6
                if not (args.t_range[0] <= t <= args.t_range[1]):
                    continue
            by_scene[r["scene_id"]].append(r)

    missing = sorted(set(args.video) - set(by_scene))
    if missing:
        print(f"! no labelled windows for: {', '.join(missing)}")

    os.makedirs(args.out_dir, exist_ok=True)
    for si, scene_id in enumerate(sorted(by_scene), 1):
        recs = by_scene[scene_id]
        print(f"[{si}/{len(by_scene)}] {scene_id} ({len(recs)} windows)", flush=True)
        ds = TruckDriveDataset(
            data_root=args.data_root, backend=args.backend,
            metadata_cache=args.metadata_cache, scene_ids=[scene_id],
            camera_views=[_VIEW], num_frames=1, image_time_step=0.2,
            t0_stride=1, filter_reverse=False, tok_recon_max_xy_m=None,
        )
        out = os.path.join(args.out_dir, f"{scene_id}_nav.mp4")
        n = render_scene_video(ds, recs, out, fps=args.fps)
        print(f"    wrote {n} frames -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--backend", default="s3", choices=["local", "s3"])
    p.add_argument("--metadata-cache", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--per-group", type=int, default=6)
    p.add_argument("--turn-radius-m", type=float, default=200.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--video", nargs="+", metavar="SCENE_ID", default=None,
        help="Instead of the stratified PNG batch, render one MP4 per named "
             "scene: every labelled window in time order, nav command and t0 "
             "burned in. Pass 'worst' style scene ids as printed by the PNG "
             "filenames (e.g. scene_25_1083).",
    )
    p.add_argument("--fps", type=float, default=4.0, help="Video frame rate")
    p.add_argument(
        "--t-range", nargs=2, type=float, default=None, metavar=("T0", "T1"),
        help="Video only: restrict to windows with t0 in [T0, T1] seconds",
    )
    args = p.parse_args()

    with open(args.labels) as f:
        payload = json.load(f)

    if args.video:
        _video_main(args, payload["records"])
        return

    groups = _groups(payload["records"], args.turn_radius_m)

    rng = random.Random(args.seed)
    picks: list[tuple[str, dict]] = []
    for name in sorted(groups):
        recs = groups[name]
        for r in rng.sample(recs, min(args.per_group, len(recs))):
            picks.append((name, r))
    print(f"groups: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(groups.items())))
    print(f"rendering {len(picks)} windows\n")

    # Group picks by scene so each scene's dataset/index is built once.
    by_scene: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for name, r in picks:
        by_scene[r["scene_id"]].append((name, r))

    os.makedirs(args.out_dir, exist_ok=True)
    n_ok = 0
    for si, (scene_id, items) in enumerate(sorted(by_scene.items()), 1):
        print(f"[{si}/{len(by_scene)}] {scene_id} ({len(items)} windows)", flush=True)
        try:
            ds = TruckDriveDataset(
                data_root=args.data_root, backend=args.backend,
                metadata_cache=args.metadata_cache, scene_ids=[scene_id],
                camera_views=[_VIEW], num_frames=1, image_time_step=0.2,
                t0_stride=1, filter_reverse=False, tok_recon_max_xy_m=None,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    ! skipping scene: {exc!r}")
            continue
        for name, rec in items:
            sub = os.path.join(args.out_dir, name)
            os.makedirs(sub, exist_ok=True)
            out = os.path.join(sub, f"{scene_id}_t{rec['t0_us'] / 1e6:.2f}s.jpg")
            try:
                if _render_one(ds, rec, out):
                    n_ok += 1
                else:
                    print(f"    ! no matching window for t0={rec['t0_us'] / 1e6:.2f}s")
            except Exception as exc:  # noqa: BLE001
                print(f"    ! render failed: {exc!r}")

    print(f"\nwrote {n_ok} renders -> {args.out_dir}")
    print("one subfolder per group; filename carries scene + t0")


if __name__ == "__main__":
    main()
