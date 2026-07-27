# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Plot per-scene minADE from a saved eval ``predictions.pt``.

Answers "which scenes does this checkpoint fail on, and is the failure per-scene
or per-window?" -- the question the aggregate ``val/metric/min_ade`` hides. Needs
only the predictions file (no model, no dataset, no S3).

Two panels:

* left  -- every val scene as a bar, sorted worst-first, coloured by TruckDrive
  chunk (the ``scene_<chunk>_<n>`` prefix, which tracks collection site/session).
  Dashed line = the val-wide mean over windows.
* right -- per-window minADE for each chunk, so a chunk that is uniformly hard is
  distinguishable from one carrying a few catastrophic windows.

Usage::

    python -m alpamayo1_5_sft.truckdrive.plot_scene_ade \
      --predictions /path/to/checkpoint-4119/eval/predictions.pt \
      --out /mnt/efs/users/rod/truckdrive_scene_ade.png
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import torch

# Okabe-Ito: a published colour-vision-deficiency-safe categorical set. Assigned
# in fixed order by sorted chunk id, so a chunk keeps its colour across runs.
_CHUNK_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
                 "#D55E00", "#F0E442", "#000000"]


def _chunk_of(scene_id: str) -> str:
    """'scene_28_24' -> '28' (the TruckDrive chunk / collection grouping)."""
    m = re.match(r"scene_(\d+)_", scene_id)
    return m.group(1) if m else "?"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--predictions", required=True, help="eval predictions.pt")
    ap.add_argument("--out", default="scene_ade.png")
    ap.add_argument("--title", default=None, help="Extra title text (e.g. checkpoint name)")
    args = ap.parse_args()

    records = torch.load(args.predictions, map_location="cpu", weights_only=False)
    per_scene: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if "metric/min_ade" in r:
            per_scene[r["scene_id"]].append(float(r["metric/min_ade"]))

    scenes = sorted(per_scene, key=lambda s: -float(np.mean(per_scene[s])))
    means = np.array([np.mean(per_scene[s]) for s in scenes])
    chunks = [_chunk_of(s) for s in scenes]
    chunk_order = sorted(set(chunks), key=lambda c: (len(c), c))
    color_of = {c: _CHUNK_COLORS[i % len(_CHUNK_COLORS)] for i, c in enumerate(chunk_order)}

    all_windows = np.array([v for vals in per_scene.values() for v in vals])
    overall = float(all_windows.mean())

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [2.4, 1]}
    )

    x = np.arange(len(scenes))
    ax.bar(x, means, color=[color_of[c] for c in chunks], width=0.85, linewidth=0)
    ax.axhline(overall, ls="--", lw=1.2, color="#555555")
    ax.text(len(scenes) * 0.99, overall, f"  val mean {overall:.2f} m",
            ha="right", va="bottom", fontsize=9, color="#555555")

    # Direct-label only the worst few, stepping the labels down and to the right
    # so they never overplot each other (the bars are ~1 unit apart).
    for i in range(min(5, len(scenes))):
        ax.annotate(
            f"{scenes[i]}  {means[i]:.2f} m",
            xy=(x[i], means[i]),
            xytext=(6 + 14 * i, -4 - 15 * i),
            textcoords="offset points",
            fontsize=8, color="#333333", va="center", ha="left",
            arrowprops=dict(arrowstyle="-", lw=0.6, color="#999999",
                            shrinkA=0, shrinkB=2),
        )

    ax.set_xlim(-1, len(scenes))
    ax.set_ylim(0, means.max() * 1.35)
    ax.set_xlabel(f"val scene (n={len(scenes)}), sorted worst first")
    ax.set_ylabel("mean minADE (m)")
    ax.set_xticks([])
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_of[c]) for c in chunk_order]
    ax.legend(handles, [f"chunk {c}" for c in chunk_order], frameon=False,
              fontsize=9, ncol=len(chunk_order), loc="upper right")

    # Right: per-window spread within each chunk (jittered strip + median bar).
    rng = np.random.default_rng(0)
    for i, c in enumerate(chunk_order):
        vals = np.array([v for s, vs in per_scene.items() if _chunk_of(s) == c for v in vs])
        ax2.scatter(i + rng.uniform(-0.28, 0.28, vals.size), vals, s=6, alpha=0.35,
                    color=color_of[c], linewidths=0)
        ax2.plot([i - 0.34, i + 0.34], [np.median(vals)] * 2, color="#222222", lw=2)
    ax2.axhline(overall, ls="--", lw=1.2, color="#555555")
    ax2.set_xticks(range(len(chunk_order)))
    ax2.set_xticklabels(chunk_order)
    ax2.set_ylabel("per-window minADE (m)")
    ax2.set_yscale("log")
    ax2.set_xlabel("TruckDrive chunk  (black bar = chunk median)")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.25)

    title = f"Per-scene minADE over {len(all_windows)} val windows"
    if args.title:
        title += f" -- {args.title}"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"[plot] wrote {args.out}")
    print(f"[plot] {len(scenes)} scenes, val mean {overall:.3f} m, "
          f"median {float(np.median(all_windows)):.3f} m")
    for c in chunk_order:
        vals = np.array([v for s, vs in per_scene.items() if _chunk_of(s) == c for v in vs])
        n_scenes = sum(1 for s in per_scene if _chunk_of(s) == c)
        print(f"[plot]   chunk {c}: {n_scenes:>3} scenes, {vals.size:>4} windows, "
              f"mean {vals.mean():.3f} m, median {np.median(vals):.3f} m")


if __name__ == "__main__":
    main()
