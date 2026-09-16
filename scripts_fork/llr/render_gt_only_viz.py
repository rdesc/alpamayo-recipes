# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Re-render the Phase-0 LLR example plots (alpamayo1_5_v2) with GROUND TRUTH ONLY.

Why this exists: the earlier renderers overlaid the model's own *self-generated* trajectory
(with vs. without reasoning) on top of the GT paths. That came from the Sec 4.3 behavioral
ablation, which conditions on the model's *self-generated* CoC -- a different regime from what
`llr_act` actually scores, which is **gold CoC + GT trajectory**, teacher-forced
(see `phase0_llr_action_direction.py` and `results/phase0_llr_action_direction.md`).

Putting both in one figure invites reading them as the same measurement, so the published
artifact's panels show only what the measurement conditions on: gold reasoning (rendered as
caption text from the manifest's `gold_coc`) and the GT trajectory (these plots).

The manifest's `pred_*` fields are deliberately left in place -- they are a real prior result
and stay on disk -- they are just no longer plotted.

Overwrites `bev.png` and `velocity.png` in each event's existing `out_dir`. No model forward
pass, no GPU: purely data loading + matplotlib.

Usage (CWD matters -- the sys.path bootstrap below is CWD-relative; see this dir's README)::

    cd recipes/alpamayo1_x_rl
    a1x_rl_b300/bin/python ../../scripts_fork/llr/render_gt_only_viz.py
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

MIN_LATERAL_SPAN_M = 16.0  # +/-8m minimum around center, so straight paths stay readable
DT = 0.1  # 10 Hz

C_HIST = "#4f8fd6"
C_FUT = "#4fc7c7"
C_BG = "#0a0d10"
C_PANEL = "#131920"
C_EDGE = "#262f3a"
C_GRID = "#1a222b"
C_TEXT = "#e7ecf1"
C_LABEL = "#9aa7b4"
C_TICK = "#64707c"


def speed_from_xy(xy: np.ndarray, dt: float = DT) -> np.ndarray:
    """Central-difference speed (m/s); forward/backward diff at the endpoints."""
    n = len(xy)
    v = np.zeros(n)
    if n >= 2:
        v[0] = np.linalg.norm(xy[1] - xy[0]) / dt
        v[-1] = np.linalg.norm(xy[-1] - xy[-2]) / dt
        for i in range(1, n - 1):
            v[i] = np.linalg.norm(xy[i + 1] - xy[i - 1]) / (2 * dt)
    return v


def render_bev(history_xy: np.ndarray, future_xy: np.ndarray, out_path: str) -> None:
    """Top-down: x = forward (m), y = lateral (m), GT history + GT future only."""
    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=150)
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    ax.plot(history_xy[:, 0], history_xy[:, 1], color=C_HIST, lw=2.2, marker="o", ms=3,
            label="GT history")
    ax.plot(future_xy[:, 0], future_xy[:, 1], color=C_FUT, lw=2.2, marker="o", ms=3,
            label="GT future")
    ax.scatter([0], [0], color=C_TEXT, marker="*", s=140, zorder=5, label="ego @ t0")

    all_xy = np.concatenate([history_xy, future_xy], axis=0)
    x_min, x_max = all_xy[:, 0].min(), all_xy[:, 0].max()
    y_min, y_max = all_xy[:, 1].min(), all_xy[:, 1].max()

    y_center = (y_min + y_max) / 2.0
    y_span = max(y_max - y_min, MIN_LATERAL_SPAN_M)
    ax.set_ylim(y_center - y_span / 2.0, y_center + y_span / 2.0)

    x_pad = max((x_max - x_min) * 0.08, 2.0)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("forward (m)", color=C_LABEL, fontsize=9)
    ax.set_ylabel("lateral (m)", color=C_LABEL, fontsize=9)
    ax.tick_params(colors=C_TICK, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(C_EDGE)
    ax.grid(True, color=C_GRID, lw=0.6)
    ax.legend(loc="upper left", fontsize=7, facecolor=C_PANEL, edgecolor=C_EDGE, labelcolor=C_TEXT)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_velocity(
    t_hist: np.ndarray, v_hist: np.ndarray,
    t_fut: np.ndarray, v_fut: np.ndarray,
    out_path: str,
) -> None:
    """Speed vs time, GT history + GT future only. t=0 is the ego pose at t0."""
    fig, ax = plt.subplots(figsize=(4.6, 2.6), dpi=150)
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)

    ax.plot(t_hist, v_hist, color=C_HIST, lw=2.0, marker="o", ms=2.5, label="GT history")
    ax.plot(t_fut, v_fut, color=C_FUT, lw=2.0, marker="o", ms=2.5, label="GT future")
    ax.axvline(0, color=C_TEXT, lw=1.0, alpha=0.5, zorder=1)

    ax.set_xlabel("time (s)  [0 = ego @ t0]", color=C_LABEL, fontsize=9)
    ax.set_ylabel("speed (m/s)", color=C_LABEL, fontsize=9)
    ax.tick_params(colors=C_TICK, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(C_EDGE)
    ax.grid(True, color=C_GRID, lw=0.6)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=7, facecolor=C_PANEL, edgecolor=C_EDGE, labelcolor=C_TEXT)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(cache_dir="/mnt/efs/users/rod/hf_cache")

    for entry in manifest:
        out_dir = entry["out_dir"]
        data = load_physical_aiavdataset(entry["clip_id"], t0_us=entry["t0_us"], avdi=avdi)

        hist_xy = data["ego_history_xyz"].squeeze(0).squeeze(0).numpy()[:, :2]
        fut_xy = data["ego_future_xyz"].squeeze(0).squeeze(0).numpy()[:, :2]

        render_bev(hist_xy, fut_xy, os.path.join(out_dir, "bev.png"))

        v_hist = speed_from_xy(hist_xy)
        v_fut = speed_from_xy(fut_xy)
        n_hist = len(hist_xy)
        t_hist = (np.arange(n_hist) - (n_hist - 1)) * DT  # ends at 0
        t_fut = (np.arange(len(fut_xy)) + 1) * DT

        render_velocity(t_hist, v_hist, t_fut, v_fut, os.path.join(out_dir, "velocity.png"))

        # pred_* fields intentionally preserved but no longer plotted.
        entry["bev_file"] = "bev.png"
        entry["velocity_file"] = "velocity.png"
        entry["plots_show"] = "gt_only"

        print(
            f"[gt-only] {entry['label']}{entry['rank']} llr={entry['llr_act']:.3f} "
            f"v_hist(end)={v_hist[-1]:.2f} v_fut(start/end)={v_fut[0]:.2f}/{v_fut[-1]:.2f} "
            f"lat_span={max(np.ptp(np.concatenate([hist_xy, fut_xy])[:, 1]), MIN_LATERAL_SPAN_M):.1f}m",
            flush=True,
        )

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[gt-only] re-rendered {len(manifest)} events, manifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
