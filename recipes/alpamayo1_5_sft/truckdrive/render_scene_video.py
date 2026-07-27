# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Render an MP4 walkthrough of a whole val scene: camera inputs + BEV overlay.

Turns the *saved* eval predictions (``<ckpt>/eval*/predictions.pt``, one record
per val window with ``pred_xyz`` / ``ego_future_xyz`` / ``ego_history_xyz``)
into a video. **No model and no GPU are needed** -- the rollout already
happened; this only re-loads the camera images for each window and draws.

Each video frame is one val window (they are ~1 s apart, so the default 2 fps
plays a scene back at ~2x real time):

* left  -- the 5 camera views that were fed to the model at that window's t0,
  laid out spatially (forward row on top, rearward row below).
* right -- bird's-eye view in the ego@t0 frame: grey = GT history, solid red =
  GT future, one colour per sampled prediction (best-of-K drawn heavier). Axis
  limits are fixed across the whole scene so the view does not jitter.
* bottom (only if the eval run saved ``gen_text/cot``) -- each sample's generated
  reasoning, coloured to match its BEV curve and sorted best-ADE first, so a
  wrong path can be read against the reasoning that produced it. Disable with
  ``+video.show_cot=false``.

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

# Spatial layout of the 5 training views: forward-facing row, then rearward row.
# Views absent from the run's camera_views are dropped, and any view not named
# here is appended as an extra row, so this never hard-fails on a different rig.
_VIEW_ROWS = [
    ["sideward_left_front_wide", "forward_center_medium", "sideward_right_front_wide"],
    ["rearward_left_bottom_medium", "rearward_right_bottom_medium"],
]

_PANEL_W = 1280  # width of the camera panel; BEV is squared off to its height

# Per-trajectory-sample colours, used for BOTH the BEV curve and its CoT line so
# the text can be tied to the path it explains. Deliberately excludes red (GT
# future) and grey (GT history). CVD-safe set, assigned in fixed sample order.
_SAMPLE_COLORS = ["#0072B2", "#009E73", "#56B4E9", "#CC79A7", "#785EF0", "#8C6D31"]

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


def _camera_panel(frames: torch.Tensor, view_order: list[str], width: int) -> np.ndarray:
    """Tile the t0 camera images into the spatial layout. frames: (N_cam,C,H,W) uint8."""
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
    """Fixed, equal-aspect xy limits covering every trajectory in the scene."""
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


def _bev_panel(rec: dict, limits, size_px: int, dpi: int = 100) -> np.ndarray:
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
                label=f"pred {k}" + (" (best)" if k == best else ""))

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
    ade = float(rec["metric/min_ade"]) if "metric/min_ade" in rec else float("nan")
    ax.set_title(f"{rec['scene_id']}  t0={rec['t0_us'] / 1e6:.1f}s  minADE={ade:.2f}m",
                 fontsize=10)
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
        limits = _bev_limits(recs)
        frames = []
        rendered: list[dict] = []
        for rec in recs:
            key = (scene, int(rec["t0_us"]))
            if key not in index_of:
                print(f"[video]   no dataset window for {key}; skipping frame", flush=True)
                continue
            sample = ds[index_of[key]]
            imgs = sample["image_frames"][:, -1]  # (N_cam, C, H, W) at t0
            cam = _camera_panel(imgs, view_order, _PANEL_W)
            bev = _bev_panel(rec, limits, size_px=cam.shape[0])
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
