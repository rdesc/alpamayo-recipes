# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Asset extractor for the CORRECTED Phase-0 LLR artifact: 4 examples per min/mean/max bucket.

Differs from ``extract_llr_viz_v2.py`` in what it selects on. v2 bucketed events by the
RETRACTED metric (``phase0_llr_action_direction.py``'s mask-mean, which was dominated by a
constant delimiter offset), so its "min"/"max" examples were extremes of an artifact. This reads
the corrected per-token output (``phase0_llr_per_token.py --blank-mode splice``) and aggregates
to a per-event mean over the 128 future trajectory tokens before bucketing.

No model forward passes -- purely data loading (camera frames + GT trajectory) and rendering.
Plots are GT-only: history + future ground truth, no self-generated overlays (those mix
conditioning regimes with the scoring, which conditions on gold CoC and GT trajectory).
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path[:0] = ["../../../src", "../.."]
import physical_ai_av  # noqa: E402
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402

MERGED_PARQUET = "/opt/dlami/nvme/rod/results/llr_splice/merged.parquet"  # corrected, per-token
OUT_ROOT = (
    "/tmp/claude-1001/-mnt-efs-users-rod-repos-alpamayo-recipes/"
    "2313ddf8-517c-4f41-8959-3dd623c291a8/scratchpad/llr_viz/alpamayo1_5_corrected"
)
CAM_SIZE = (960, 540)  # up from 640x360
MIN_LATERAL_SPAN_M = 16.0  # +/-8m minimum around center
N_PER_BUCKET = 4
SOURCE_PARQUET = "/opt/dlami/nvme/rod/datasets/pai_av_dataset/reasoning/ood_reasoning.parquet"
MIN_T0_US = 1_700_000


def select_events() -> list[dict]:
    """Aggregate the per-token parquet to one row per event, then take 4 per bucket.

    The corrected output has one row per (event, trajectory token), so the per-event LLR is the
    mean over its 128 future tokens -- the same quantity the headline number averages.
    """
    df = pd.read_parquet(MERGED_PARQUET)
    keys = ["clip_id", "event_idx", "event_cluster", "split", "gold_coc"]
    ev = df.groupby(keys, as_index=False).agg(llr_act=("llr_act", "mean"))
    # t0_us isn't in the per-token output; recover it from the source parquet the run used.
    src_df = pd.read_parquet(SOURCE_PARQUET)
    src_df["events"] = src_df["events"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    t0 = {}
    for clip_id, row in src_df.dropna(subset=["events"]).iterrows():
        for ei, e in enumerate(row["events"]):
            t0[(clip_id, ei)] = max(int(e["event_start_timestamp"]), MIN_T0_US)
    ev["t0_us"] = [t0.get((c, int(i))) for c, i in zip(ev["clip_id"], ev["event_idx"])]
    ev = ev.dropna(subset=["t0_us"])

    mean_val = ev["llr_act"].mean()
    ev = ev.assign(absdiff=(ev["llr_act"] - mean_val).abs())
    picks = []
    for bucket, sub in [
        ("min", ev.nsmallest(N_PER_BUCKET, "llr_act")),
        ("mean", ev.nsmallest(N_PER_BUCKET, "absdiff")),
        ("max", ev.nlargest(N_PER_BUCKET, "llr_act")),
    ]:
        for rank, (_, row) in enumerate(sub.iterrows(), start=1):
            picks.append(
                {
                    "label": bucket,
                    "rank": rank,
                    "clip_id": row["clip_id"],
                    "event_idx": int(row["event_idx"]),
                    "t0_us": float(row["t0_us"]),
                    "llr_act": float(row["llr_act"]),
                    "event_cluster": row["event_cluster"],
                    "split": row["split"],
                    "gold_coc": row["gold_coc"],
                }
            )
    print(f"[extract-v3] dataset mean llr_act = {mean_val:+.4f} over {len(ev)} events", flush=True)
    return picks


def render_bev(history_xy: np.ndarray, future_xy: np.ndarray, out_path: str) -> None:
    """Top-down plot: x = forward (m), y = lateral (m). Min lateral span so straight
    trajectories don't collapse to a flat line under autoscale."""
    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=150)
    fig.patch.set_facecolor("#0a0d10")
    ax.set_facecolor("#0a0d10")

    ax.plot(history_xy[:, 0], history_xy[:, 1], color="#4f8fd6", lw=2.2, marker="o", ms=3, label="history")
    ax.plot(future_xy[:, 0], future_xy[:, 1], color="#4fc7c7", lw=2.2, marker="o", ms=3, label="future (GT)")
    ax.scatter([0], [0], color="#e7ecf1", marker="*", s=140, zorder=5, label="ego @ t0")

    all_xy = np.concatenate([history_xy, future_xy], axis=0)
    x_min, x_max = all_xy[:, 0].min(), all_xy[:, 0].max()
    y_min, y_max = all_xy[:, 1].min(), all_xy[:, 1].max()

    y_center = (y_min + y_max) / 2.0
    y_span = max(y_max - y_min, MIN_LATERAL_SPAN_M)
    ax.set_ylim(y_center - y_span / 2.0, y_center + y_span / 2.0)

    x_pad = max((x_max - x_min) * 0.08, 2.0)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("forward (m)", color="#9aa7b4", fontsize=9)
    ax.set_ylabel("lateral (m)", color="#9aa7b4", fontsize=9)
    ax.tick_params(colors="#64707c", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#262f3a")
    ax.grid(True, color="#1a222b", lw=0.6)
    ax.legend(loc="upper left", fontsize=7, facecolor="#131920", edgecolor="#262f3a", labelcolor="#e7ecf1")

    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def save_camera_frames(image_frames: np.ndarray, camera_indices: list[int], out_dir: str) -> list[str]:
    """image_frames: [n_cams, n_frames, 3, H, W] uint8 -- use most recent frame per cam."""
    files = []
    for ci, cam_idx in enumerate(camera_indices):
        frame = image_frames[ci, -1]  # [3, H, W]
        if hasattr(frame, "numpy"):
            frame = frame.numpy()
        arr = np.transpose(frame, (1, 2, 0))  # HWC
        arr = np.ascontiguousarray(arr)
        img = Image.fromarray(arr)
        img = img.resize(CAM_SIZE, Image.LANCZOS)
        fname = f"cam_{cam_idx}.jpg"
        img.save(os.path.join(out_dir, fname), format="JPEG", quality=85)
        files.append(fname)
    return files


def main() -> None:
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    picks = select_events()
    manifest = []

    for p in picks:
        subdir_name = f"{p['label']}{p['rank']}_{p['clip_id'][:8]}"
        out_dir = os.path.join(OUT_ROOT, subdir_name)
        os.makedirs(out_dir, exist_ok=True)

        data = load_physical_aiavdataset(p["clip_id"], t0_us=p["t0_us"], avdi=avdi)

        camera_indices = list(data["camera_indices"])
        camera_files = save_camera_frames(data["image_frames"], camera_indices, out_dir)

        hist_xyz = data["ego_history_xyz"].squeeze(0).squeeze(0).numpy()  # [16,3]
        fut_xyz = data["ego_future_xyz"].squeeze(0).squeeze(0).numpy()  # [64,3]
        render_bev(hist_xyz[:, :2], fut_xyz[:, :2], os.path.join(out_dir, "bev.png"))

        entry = dict(p)
        entry["out_dir"] = out_dir
        entry["camera_files"] = camera_files
        entry["camera_indices"] = camera_indices
        entry["bev_file"] = "bev.png"
        manifest.append(entry)
        print(f"[extract-v3] done {subdir_name} llr_act={p['llr_act']:.3f}", flush=True)

    with open(os.path.join(OUT_ROOT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[extract-v3] wrote manifest with {len(manifest)} events -> {OUT_ROOT}/manifest.json")


if __name__ == "__main__":
    main()
