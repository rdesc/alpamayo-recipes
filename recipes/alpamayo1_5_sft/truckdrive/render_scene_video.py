# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Render an MP4 walkthrough of a whole val scene: camera inputs + BEV overlay.

Turns the *saved* eval predictions (``<ckpt>/eval*/predictions.pt``, one record
per val window with ``pred_xyz`` / ``ego_future_xyz`` / ``ego_history_xyz``)
into a video. **No model and no GPU are needed** -- the rollout already
happened; this only re-loads the camera images for each window and draws.

Each video frame is one val window (they are ~1 s apart, so the default 2 fps
plays a scene back at ~2x real time):

* left  -- the 5 camera views that were fed to the model at that window's t0,
  laid out spatially (forward row on top, rearward row below), with the same
  GT history, GT future and one predicted sample (sample 0;
  ``+video.overlay_sample=``) projected onto each image via the scene's pinhole
  calibration (``projection.py``). Segments behind or off the image are dropped,
  so a view shows only the part of the path it actually sees. Disable with
  ``+video.overlay=false``.
* right -- bird's-eye view in the ego@t0 frame: grey = GT history, solid red =
  GT future, one colour per sampled prediction (best-of-K drawn heavier). Axis
  limits are fixed across the whole scene so the view does not jitter.
* bottom (only if the eval run saved ``gen_text/cot``) -- each sample's generated
  reasoning, coloured to match its BEV curve and sorted best-ADE first, so a
  wrong path can be read against the reasoning that produced it. Disable with
  ``+video.show_cot=false``.

Two models can be compared in one video by passing a second predictions file,
``+video.predictions_b=`` (e.g. the Stage-2 diffusion action head against the
Stage-1 discrete trajectory tokens). Hue encodes the model and shade only the
sample within it -- set A solid in blues, set B dashed in one orange -- so model
identity, not sample index, is the salient channel; the legend names each set
and its best-of-K draw rather than all six samples. Name them with
``+video.label=`` / ``+video.label_b=``; both minADEs appear in the BEV title.

Predictions are joined to dataset windows on ``(scene_id, t0_us)``, which is
exactly how ``__getitem__`` stamps them, so the join is exact rather than
positional -- the eval file's order does not have to match the dataset's.

List the scenes in a predictions file, worst mean-minADE first::

    python -m alpamayo1_5_sft.truckdrive.render_scene_video \
      --config-path pkg://alpamayo1_5_sft/configs --config-name sft_truckdrive \
      +video.predictions=/path/to/checkpoint-4119/eval/predictions.pt \
      +video.list_scenes=true

Render one (or several) scenes::

    python -m alpamayo1_5_sft.truckdrive.render_scene_video \
      --config-path pkg://alpamayo1_5_sft/configs --config-name sft_truckdrive \
      data.val_dataset.data_root=s3://torc-data/datasets/TruckDrivePublic \
      data.val_dataset.backend=s3 \
      data.val_dataset.metadata_cache=/mnt/efs/users/rod/truckdrive_cache/meta_5views.pkl \
      data.val_dataset.split_metainfo=/mnt/efs/users/rod/repos/alpamayo-recipes/recipes/alpamayo1_5_sft/truckdrive/truckdrive_metainfo.json \
      +video.predictions=/path/to/checkpoint-4119/eval/predictions.pt \
      +video.scenes=[scene_28_24,scene_42_12] \
      +video.out_dir=/mnt/efs/users/rod/truckdrive_val_videos
"""
from __future__ import annotations

import os
from collections import defaultdict

import hydra
import hydra.utils as hyu
import matplotlib

matplotlib.use("Agg")  # headless

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import alpamayo.common.constants as constants
try:  # normal: run as `python -m alpamayo1_5_sft.truckdrive.render_scene_video`
    from .projection import load_calibration_via, project_ego_to_image
except ImportError:  # run as a plain script from the recipe directory
    from truckdrive.projection import load_calibration_via, project_ego_to_image

# Spatial layout of the 5 training views: forward-facing row, then rearward row.
# Views absent from the run's camera_views are dropped, and any view not named
# here is appended as an extra row, so this never hard-fails on a different rig.
_VIEW_ROWS = [
    ["sideward_left_front_wide", "forward_center_medium", "sideward_right_front_wide"],
    ["rearward_left_bottom_medium", "rearward_right_bottom_medium"],
]

_PANEL_W = 1280  # width of the camera panel; BEV is squared off to its height

# Per-trajectory-sample colours for set A, used for BOTH the BEV curve and its
# CoT line so the text can be tied to the path it explains. All one hue (blue):
# the set a curve belongs to must read at a glance, so hue encodes the model and
# only shade separates samples within it. Shades stay dark enough to be legible
# as CoT text on white. Deliberately clear of red (GT future) and grey (history).
_SAMPLE_COLORS = ["#08306B", "#08519C", "#2171B5", "#3182BD", "#4292C6", "#5AA3D0"]

# Second predictions file (``+video.predictions_b=``), e.g. the Stage-2 diffusion
# action head against the Stage-1 discrete trajectory tokens. One orange for all
# its samples -- it has no CoT rows to tie back to, so it needs no shade ramp.
_COLOR_B = "#E69F00"

_COT_PANEL_H = 210  # fixed: every video frame must have identical dimensions
_COT_TEXT_X = 320   # x where the sentence starts, clear of the swatch+ADE column


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]


def _get(cfg: DictConfig, path: str, default):
    v = OmegaConf.select(cfg, path)
    return default if v is None else v


def _load_predictions(path: str) -> dict[str, list[dict]]:
    """Group saved eval records by scene, each list sorted by t0."""
    records = torch.load(path, map_location="cpu", weights_only=False)
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_scene[r["scene_id"]].append(r)
    for v in by_scene.values():
        v.sort(key=lambda r: r["t0_us"])
    return dict(by_scene)


def _scene_table(by_scene: dict[str, list[dict]]) -> list[tuple[str, int, float, float]]:
    """(scene, n_windows, mean minADE, worst minADE), worst mean first."""
    rows = []
    for scene, recs in by_scene.items():
        ades = [float(r["metric/min_ade"]) for r in recs if "metric/min_ade" in r]
        if not ades:
            continue
        rows.append((scene, len(recs), sum(ades) / len(ades), max(ades)))
    rows.sort(key=lambda t: -t[2])
    return rows


def _draw_projected(
    tile: np.ndarray, xyz: torch.Tensor, calib, color: tuple[int, int, int], thickness: int
) -> None:
    """Draw one ego-frame trajectory onto an already-resized camera tile.

    ``calib`` is ``(K, T_vehicle_cam, (orig_w, orig_h))`` at full sensor
    resolution; the tile has been resized twice (loader + layout), so pixel
    coordinates are rescaled by the width/height ratios independently -- the
    loader does not preserve aspect ratio.
    """
    K, T, (ow, oh) = calib
    uv, valid = project_ego_to_image(xyz.reshape(-1, 3).numpy(), K, T, image_size=(ow, oh))
    h, w = tile.shape[:2]
    uv = uv * np.array([w / ow, h / oh])
    pts = uv.astype(np.int32)
    # Only connect consecutive points that are both in front of the camera and
    # on-image; a segment spanning the horizon would sweep across the frame.
    for i in range(len(pts) - 1):
        if valid[i] and valid[i + 1]:
            cv2.line(tile, tuple(pts[i]), tuple(pts[i + 1]), color, thickness, cv2.LINE_AA)


def _overlay_trajectories(
    tile: np.ndarray, rec: dict, calib, sample_idx: int = 0, rec_b: dict | None = None
) -> None:
    """Draw GT history, one predicted sample per model and GT future onto a tile.

    Only a single sample per model is projected -- all six overlap heavily in
    image space and turn the view into a smear; the full fan stays available in
    the BEV panel. Sample 0 by default (``+video.overlay_sample=``): an
    unselected draw from the policy, unlike best-of-K, which is chosen using the
    ground truth and so flatters what the model would actually do. Colours match
    the BEV panel.
    """
    _draw_projected(tile, rec["ego_history_xyz"], calib, (119, 119, 119), 2)
    pred = rec["pred_xyz"].reshape(-1, rec["pred_xyz"].shape[-2], 3)
    k = min(sample_idx, pred.shape[0] - 1)
    _draw_projected(tile, pred[k], calib,
                    _hex_to_rgb(_SAMPLE_COLORS[k % len(_SAMPLE_COLORS)]), 2)
    if rec_b is not None:
        pb = rec_b["pred_xyz"].reshape(-1, rec_b["pred_xyz"].shape[-2], 3)
        _draw_projected(tile, pb[min(sample_idx, pb.shape[0] - 1)], calib,
                        _hex_to_rgb(_COLOR_B), 2)
    _draw_projected(tile, rec["ego_future_xyz"], calib, (214, 39, 40), 2)


def _camera_panel(
    frames: torch.Tensor,
    view_order: list[str],
    width: int,
    rec: dict | None = None,
    calib_by_view: dict | None = None,
    sample_idx: int = 0,
    rec_b: dict | None = None,
) -> np.ndarray:
    """Tile the t0 camera images into the spatial layout. frames: (N_cam,C,H,W) uint8.

    When ``rec`` and ``calib_by_view`` are given, the GT and predicted paths are
    also projected onto each view they are visible in.
    """
    imgs = {}
    for view, f in zip(view_order, frames):
        imgs[view] = f.permute(1, 2, 0).numpy()  # (H, W, 3) uint8

    rows = [[v for v in row if v in imgs] for row in _VIEW_ROWS]
    laid_out = {v for row in rows for v in row}
    extra = [v for v in view_order if v not in laid_out]
    if extra:
        rows.append(extra)
    rows = [r for r in rows if r]

    row_imgs = []
    for row in rows:
        tile_w = width // len(row)
        tiles = []
        for view in row:
            img = imgs[view]
            h = max(1, int(round(img.shape[0] * tile_w / img.shape[1])))
            tile = cv2.resize(img, (tile_w, h), interpolation=cv2.INTER_AREA)
            if rec is not None and calib_by_view and view in calib_by_view:
                _overlay_trajectories(tile, rec, calib_by_view[view], sample_idx, rec_b)
            cv2.putText(tile, view, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        # Rows with fewer views get shorter tiles; pad each row to a common height.
        h_max = max(t.shape[0] for t in tiles)
        tiles = [
            np.pad(t, ((0, h_max - t.shape[0]), (0, 0), (0, 0))) if t.shape[0] < h_max else t
            for t in tiles
        ]
        row_img = np.hstack(tiles)
        if row_img.shape[1] < width:  # integer division leftover
            row_img = np.pad(row_img, ((0, 0), (0, width - row_img.shape[1]), (0, 0)))
        row_imgs.append(row_img[:, :width])
    return np.vstack(row_imgs)


def _bev_limits(recs: list[dict], pad: float = 5.0) -> tuple[float, float, float, float]:
    """Fixed, equal-aspect xy limits covering every trajectory in the scene.

    Pass both models' records when comparing, so neither set is ever clipped.
    """
    xs, ys = [], []
    for r in recs:
        for key in ("ego_history_xyz", "ego_future_xyz", "pred_xyz"):
            t = r[key].reshape(-1, 3)
            xs.append(t[:, 0].numpy())
            ys.append(t[:, 1].numpy())
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    cx, cy = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
    half = 0.5 * max(x.max() - x.min(), y.max() - y.min()) + pad
    return cx - half, cx + half, cy - half, cy + half


def _bev_panel(
    rec: dict,
    limits,
    size_px: int,
    dpi: int = 100,
    rec_b: dict | None = None,
    label_a: str = "pred",
    label_b: str = "pred B",
) -> np.ndarray:
    """Render the BEV plot for one window to an RGB array of size_px x size_px."""
    fig = plt.figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    ax = fig.add_subplot(111)

    # Vehicle frame: x forward, y left. Rotate so forward points up, matching
    # the camera panel above it (same convention as viz.rotate_90cc).
    def _xy(t):
        a = t.reshape(-1, 3).numpy()
        return -a[:, 1], a[:, 0]

    hx, hy = _xy(rec["ego_history_xyz"])
    ax.plot(hx, hy, "-", color="#777777", lw=2.0, label="GT history")

    pred = rec["pred_xyz"].reshape(-1, rec["pred_xyz"].shape[-2], 3)
    best = int(rec["sample_ade"].reshape(-1).argmin()) if "sample_ade" in rec else -1
    for k in range(pred.shape[0]):
        px, py = _xy(pred[k])
        # The best-of-K sample (the one minADE reports) is drawn heavier -- it is
        # the number in the title, and its CoT is the one worth reading first.
        ax.plot(px, py, "-", color=_SAMPLE_COLORS[k % len(_SAMPLE_COLORS)],
                lw=2.2 if k == best else 1.2, alpha=0.95 if k == best else 0.6,
                # Now that the samples share a hue, per-sample legend rows are six
                # near-identical swatches: label only the set and its best draw,
                # matching set B. Sample index is still readable off the CoT rows.
                label=(f"{label_a} (best)" if k == best else
                       (label_a if k == 0 and best != 0 else None)))

    if rec_b is not None:
        pb = rec_b["pred_xyz"].reshape(-1, rec_b["pred_xyz"].shape[-2], 3)
        best_b = int(rec_b["sample_ade"].reshape(-1).argmin()) if "sample_ade" in rec_b else -1
        for k in range(pb.shape[0]):
            px, py = _xy(pb[k])
            ax.plot(px, py, "--", color=_COLOR_B,
                    lw=2.2 if k == best_b else 1.2, alpha=0.95 if k == best_b else 0.5,
                    label=(f"{label_b} (best)" if k == best_b else
                           (label_b if k == 0 and best_b != 0 else None)))

    gx, gy = _xy(rec["ego_future_xyz"])
    ax.plot(gx, gy, "-", color="tab:red", lw=2.2, label="GT future")
    ax.scatter([0], [0], c="k", marker="s", s=28, zorder=5)  # ego at t0

    x_lo, x_hi, y_lo, y_hi = limits
    ax.set_xlim(-y_hi, -y_lo)  # limits are in raw xy; the plot is rotated
    ax.set_ylim(x_lo, x_hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.set_xlabel("lateral (m)")
    ax.set_ylabel("forward (m)")
    # Window identity on the first line, metrics on the second: with two model
    # labels in it the one-line form overruns the panel and gets clipped.
    ade = float(rec["metric/min_ade"]) if "metric/min_ade" in rec else float("nan")
    metrics = f"minADE {label_a}={ade:.2f}m"
    if rec_b is not None and "metric/min_ade" in rec_b:
        metrics += f"  {label_b}={float(rec_b['metric/min_ade']):.2f}m"
    ax.set_title(
        f"{rec['scene_id']}  t0={rec['t0_us'] / 1e6:.1f}s\n{metrics}", fontsize=10
    )
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8, ncol=2)
    fig.tight_layout()

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def _cot_panel(rec: dict, width: int, height: int = _COT_PANEL_H) -> np.ndarray:
    """Strip listing each sample's generated CoT, coloured to match its BEV curve.

    One row per trajectory sample: colour swatch, sample index, that sample's ADE,
    then the reasoning text it generated. Rows are ordered best-ADE first, so the
    explanation behind the reported minADE reads at the top. Height is fixed (a
    video needs constant frame dimensions); overflow is dropped rather than
    resizing the frame.
    """
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    cot = rec.get("gen_text/cot")
    if not cot:
        return panel
    texts = list(cot[0]) if isinstance(cot[0], (list, tuple)) else list(cot)
    ades = rec["sample_ade"].reshape(-1).tolist() if "sample_ade" in rec else [
        float("nan")] * len(texts)

    order = sorted(range(len(texts)), key=lambda k: ades[k])
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    y = 22
    cv2.putText(panel, "generated CoT per trajectory sample (best ADE first)",
                (10, y), font, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    y += 22
    for rank, k in enumerate(order):
        if y > height - 10:
            break
        color = _hex_to_rgb(_SAMPLE_COLORS[k % len(_SAMPLE_COLORS)])
        cv2.rectangle(panel, (10, y - 10), (24, y + 2), color, -1)
        head = f"s{k}  ADE {ades[k]:5.2f} m" + ("  <- best" if rank == 0 else "")
        cv2.putText(panel, head, (32, y), font, scale, color, thick, cv2.LINE_AA)
        # Wrap the sentence into the remaining width (~9.5 px/char at scale 0.5).
        avail = max(20, int((width - _COT_TEXT_X) / 9.5))
        words, line, lines = texts[k].split(), "", []
        for w in words:
            if len(line) + len(w) + 1 > avail:
                lines.append(line)
                line = w
            else:
                line = f"{line} {w}".strip()
        if line:
            lines.append(line)
        for i, ln in enumerate(lines[:2]):  # 2 lines max keeps rows aligned
            cv2.putText(panel, ln, (_COT_TEXT_X, y + i * 16), font, scale, (30, 30, 30),
                        thick, cv2.LINE_AA)
        y += 16 * max(1, min(len(lines), 2)) + 10
    return panel


def _write_mp4(frames: list[np.ndarray], path: str, fps: int) -> None:
    """Encode RGB frames to H.264. Uses PyAV (present in the recipe venv)."""
    import av

    h, w = frames[0].shape[:2]
    # yuv420p needs even dimensions.
    w -= w % 2
    h -= h % 2
    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height = w, h
    stream.pix_fmt = "yuv420p"
    for f in frames:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(f[:h, :w]), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    pred_path = _get(cfg, "video.predictions", None)
    if pred_path is None:
        raise ValueError("set +video.predictions=<ckpt>/eval/predictions.pt")
    by_scene = _load_predictions(pred_path)
    print(f"[video] {sum(len(v) for v in by_scene.values())} windows "
          f"over {len(by_scene)} scenes from {pred_path}", flush=True)

    # Optional second predictions file to compare against, joined on the same
    # (scene_id, t0_us) key. Windows missing from it simply get no B curves.
    pred_path_b = _get(cfg, "video.predictions_b", None)
    by_scene_b: dict[str, list[dict]] = {}
    if pred_path_b is not None:
        by_scene_b = _load_predictions(pred_path_b)
        print(f"[video] compare: {sum(len(v) for v in by_scene_b.values())} windows "
              f"over {len(by_scene_b)} scenes from {pred_path_b}", flush=True)

    if bool(_get(cfg, "video.list_scenes", False)):
        print(f"\n{'scene':<20}{'n':>4}{'mean minADE':>13}{'worst':>9}")
        for scene, n, mean_ade, worst in _scene_table(by_scene):
            print(f"{scene:<20}{n:>4}{mean_ade:>13.3f}{worst:>9.3f}")
        return

    scenes = _get(cfg, "video.scenes", None)
    if scenes is None:
        one = _get(cfg, "video.scene", None)
        if one is None:
            # Default to the scene the checkpoint does worst on -- the one worth
            # watching. Everything else is available via +video.scenes=[...].
            one = _scene_table(by_scene)[0][0]
            print(f"[video] no scene given; defaulting to worst mean minADE: {one}")
        scenes = [one]
    scenes = [s for s in list(scenes)]

    out_dir = _get(cfg, "video.out_dir", "truckdrive_scene_videos")
    fps = int(_get(cfg, "video.fps", 2))
    show_cot = bool(_get(cfg, "video.show_cot", True))
    overlay = bool(_get(cfg, "video.overlay", True))
    overlay_sample = int(_get(cfg, "video.overlay_sample", 0))
    label_a = str(_get(cfg, "video.label", "pred"))
    label_b = str(_get(cfg, "video.label_b", "pred B"))
    os.makedirs(out_dir, exist_ok=True)

    # Images only -- no model, so no tokenization. num_frames=1 loads just the
    # t0 image per camera instead of the 4-frame training stack (4x fewer reads).
    OmegaConf.update(cfg, "data.val_dataset.vla_preprocess_args", None)
    OmegaConf.update(cfg, "data.val_dataset.num_frames",
                     int(_get(cfg, "video.num_frames", 1)))
    # The val split has no tok-recon pre-filter by default; clear it only if the
    # config set one, so the window indexing matches the eval run that produced
    # these predictions.
    if OmegaConf.select(cfg, "data.val_dataset.tok_recon_max_xy_m") is not None:
        OmegaConf.update(cfg, "data.val_dataset.tok_recon_max_xy_m", None)
    ds = hyu.instantiate(cfg.data.val_dataset, _convert_="partial", model_config=None)

    # (scene_id, t0_us) -> dataset index, matching __getitem__'s stamping.
    index_of = {
        (scene_id, int(round(t0 * 1e6))): i
        for i, (scene_id, t0) in enumerate(ds._samples)
    }
    # image_frames comes back sorted by alpamayo camera index; mirror that here
    # so each tile gets its true Leopard view name.
    view_order = sorted(ds.camera_views,
                        key=lambda v: constants.CAMERA_NAMES_TO_INDICES[ds.view_to_alpamayo[v]])

    for scene in scenes:
        recs = by_scene.get(scene)
        if not recs:
            print(f"[video] no predictions for {scene}; skipping", flush=True)
            continue
        recs_b = {int(r["t0_us"]): r for r in by_scene_b.get(scene, [])}
        limits = _bev_limits(recs + list(recs_b.values()))
        # Calibration is per scene and constant across its windows, so load it
        # once here. A view without usable calibration simply gets no overlay.
        calib_by_view: dict[str, tuple] = {}
        if overlay:
            for view in view_order:
                try:
                    calib_by_view[view] = load_calibration_via(ds.backend.read_bytes, scene, view)
                except Exception as exc:  # noqa: BLE001 - overlay is best-effort
                    print(f"[video]   no calibration for {scene}/{view}: {exc!r}", flush=True)
        frames = []
        rendered: list[dict] = []
        for rec in recs:
            key = (scene, int(rec["t0_us"]))
            if key not in index_of:
                print(f"[video]   no dataset window for {key}; skipping frame", flush=True)
                continue
            rec_b = recs_b.get(int(rec["t0_us"]))
            sample = ds[index_of[key]]
            imgs = sample["image_frames"][:, -1]  # (N_cam, C, H, W) at t0
            cam = _camera_panel(imgs, view_order, _PANEL_W, rec=rec if overlay else None,
                                calib_by_view=calib_by_view, sample_idx=overlay_sample,
                                rec_b=rec_b)
            bev = _bev_panel(rec, limits, size_px=cam.shape[0], rec_b=rec_b,
                             label_a=label_a, label_b=label_b)
            top = np.hstack([cam, bev])
            # CoT strip only when the eval run saved generated text (older eval
            # dirs have gen_text/cot; newer trajectory-only ones do not).
            if show_cot and rec.get("gen_text/cot"):
                top = np.vstack([top, _cot_panel(rec, top.shape[1])])
            frames.append(top)
            rendered.append(rec)
            print(f"[video]   {scene} t0={rec['t0_us'] / 1e6:6.1f}s "
                  f"({len(frames)}/{len(recs)})", flush=True)
        if not frames:
            print(f"[video] {scene}: nothing rendered", flush=True)
            continue
        # Mean minADE over the windows actually rendered goes in the filename, so
        # a directory listing sorts/reads as a difficulty ranking.
        ades = [float(r["metric/min_ade"]) for r in rendered if "metric/min_ade" in r]
        mean_ade = sum(ades) / len(ades) if ades else float("nan")
        out = os.path.join(out_dir, f"{scene}_minADE{mean_ade:.2f}m.mp4")
        _write_mp4(frames, out, fps)
        print(f"[video] wrote {out}  ({len(frames)} frames @ {fps} fps, "
              f"mean minADE {mean_ade:.3f} m)", flush=True)


if __name__ == "__main__":
    main()
