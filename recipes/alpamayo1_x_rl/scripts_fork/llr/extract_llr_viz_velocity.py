# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Velocity-vs-time layer for the Phase-0 LLR visualization artifact's 9 example events
(alpamayo1_5_v2/manifest.json). Companion to extract_llr_viz_predictions.py's bev.png --
same color mapping, same events, no model forward pass needed (speed is derived from
already-available position arrays: GT via load_physical_aiavdataset, predicted trajectories
already stored inline in the manifest from the prediction step).

Saves velocity.png per event's out_dir and adds a velocity_file field to manifest.json.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path[:0] = ["../../../src", "../.."]
import physical_ai_av  # noqa: E402
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402

OUT_ROOT = (
    "/tmp/claude-1001/-mnt-efs-users-rod-repos-alpamayo-recipes/"
    "2313ddf8-517c-4f41-8959-3dd623c291a8/scratchpad/llr_viz/alpamayo1_5_v2"
)
MANIFEST_PATH = os.path.join(OUT_ROOT, "manifest.json")
DT = 0.1  # seconds, 10 Hz -- matches history_xyz/future_xyz spacing used throughout this repo


def speed_from_xy(xy: np.ndarray, dt: float = DT) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference speed (m/s) at each point except the endpoints (forward/backward diff there)."""
    n = len(xy)
    v = np.zeros(n)
    if n >= 2:
        v[0] = np.linalg.norm(xy[1] - xy[0]) / dt
        v[-1] = np.linalg.norm(xy[-1] - xy[-2]) / dt
        for i in range(1, n - 1):
            v[i] = np.linalg.norm(xy[i + 1] - xy[i - 1]) / (2 * dt)
    return v


def render_velocity(
    t_hist: np.ndarray, v_hist: np.ndarray,
    t_fut: np.ndarray, v_fut: np.ndarray,
    t_pred_w: np.ndarray, v_pred_w: np.ndarray,
    t_pred_wo: np.ndarray, v_pred_wo: np.ndarray,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 2.6), dpi=150)
    fig.patch.set_facecolor("#0a0d10")
    ax.set_facecolor("#0a0d10")

    ax.plot(t_hist, v_hist, color="#4f8fd6", lw=2.0, marker="o", ms=2.5, label="history")
    ax.plot(t_fut, v_fut, color="#4fc7c7", lw=2.0, marker="o", ms=2.5, label="future (GT)")
    ax.plot(t_pred_w, v_pred_w, color="#e8a23a", lw=1.8, ls="--", marker="^", ms=2.5,
            label="pred (with reasoning)", alpha=0.95)
    ax.plot(t_pred_wo, v_pred_wo, color="#c86ee2", lw=1.8, ls="--", marker="v", ms=2.5,
            label="pred (no reasoning)", alpha=0.95)
    ax.axvline(0, color="#e7ecf1", lw=1.0, alpha=0.5, zorder=1)

    ax.set_xlabel("time (s)  [0 = ego @ t0]", color="#9aa7b4", fontsize=9)
    ax.set_ylabel("speed (m/s)", color="#9aa7b4", fontsize=9)
    ax.tick_params(colors="#64707c", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#262f3a")
    ax.grid(True, color="#1a222b", lw=0.6)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=6, facecolor="#131920", edgecolor="#262f3a", labelcolor="#e7ecf1", ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    for entry in manifest:
        clip_id, t0_us = entry["clip_id"], entry["t0_us"]
        out_dir = entry["out_dir"]

        data = load_physical_aiavdataset(clip_id, t0_us=t0_us, avdi=avdi)
        hist_xyz = data["ego_history_xyz"].squeeze(0).squeeze(0).numpy()
        fut_xyz = data["ego_future_xyz"].squeeze(0).squeeze(0).numpy()

        hist_xy = hist_xyz[:, :2]
        fut_xy = fut_xyz[:, :2]
        pred_w_xy = np.array(entry["pred_traj_with_reasoning"])
        pred_wo_xy = np.array(entry["pred_traj_without_reasoning"])

        v_hist = speed_from_xy(hist_xy)
        v_fut = speed_from_xy(fut_xy)
        v_pred_w = speed_from_xy(pred_w_xy)
        v_pred_wo = speed_from_xy(pred_wo_xy)

        n_hist = len(hist_xy)
        t_hist = (np.arange(n_hist) - (n_hist - 1)) * DT  # ends at 0
        t_fut = (np.arange(len(fut_xy)) + 1) * DT
        t_pred_w = (np.arange(len(pred_w_xy)) + 1) * DT
        t_pred_wo = (np.arange(len(pred_wo_xy)) + 1) * DT

        out_path = os.path.join(out_dir, "velocity.png")
        render_velocity(
            t_hist, v_hist, t_fut, v_fut, t_pred_w, v_pred_w, t_pred_wo, v_pred_wo, out_path,
        )
        entry["velocity_file"] = "velocity.png"

        print(
            f"[velocity] {entry['label']}{entry['rank']} "
            f"v_hist(end)={v_hist[-1]:.2f} v_fut(start/end)={v_fut[0]:.2f}/{v_fut[-1]:.2f} "
            f"v_pred_w(start/end)={v_pred_w[0]:.2f}/{v_pred_w[-1]:.2f} "
            f"v_pred_wo(start/end)={v_pred_wo[0]:.2f}/{v_pred_wo[-1]:.2f}",
            flush=True,
        )

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[velocity] wrote manifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
