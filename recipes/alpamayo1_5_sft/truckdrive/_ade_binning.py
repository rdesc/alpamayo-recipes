# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Pure-python (numpy-only) difficulty-stratified selection of val windows by ADE.

Shared by score_val_ade.py (scores then selects) and rebin_val_ade.py (re-selects
from an existing manifest without re-scoring). No torch/hydra imports so re-binning
is instant and dependency-light.
"""
from typing import Any

import numpy as np


def select_stratified(records: list[dict[str, Any]], num_bins: int,
                      per_bin: int) -> tuple[list[int], list[dict]]:
    """Bin ``records`` by ``min_ade`` quantiles, pick ``per_bin`` per bin.

    Within each bin, picks are spread evenly along the (sorted) ADE axis and
    prefer distinct ``scene_id`` so the fixed viz set spans scenes, not just
    consecutive windows. Returns (sorted sample_indices, per-bin summary).
    """
    ades = np.array([r["min_ade"] for r in records], dtype=float)
    order = list(np.argsort(ades))                 # ascending difficulty
    edges = np.quantile(ades, np.linspace(0, 1, num_bins + 1))

    selected: list[int] = []
    summary: list[dict] = []
    for b in range(num_bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = [p for p in order
                  if ades[p] >= lo and (ades[p] < hi or b == num_bins - 1)]
        picks: list[int] = []
        seen: set = set()
        if in_bin:
            cand_pos = np.linspace(0, len(in_bin) - 1,
                                   min(per_bin * 3, len(in_bin)))
            cand = [in_bin[int(round(t))] for t in cand_pos]
            for p in cand:                          # distinct scenes first
                sid = records[p].get("scene_id")
                if sid in seen:
                    continue
                seen.add(sid)
                picks.append(p)
                if len(picks) == per_bin:
                    break
            for p in cand:                          # backfill if too few scenes
                if len(picks) == per_bin:
                    break
                if p not in picks:
                    picks.append(p)
        selected.extend(picks)
        summary.append({
            "bin": b, "ade_lo": float(lo), "ade_hi": float(hi),
            "n_in_bin": len(in_bin),
            "picked": [{"idx": int(records[p]["idx"]),
                        "scene_id": records[p].get("scene_id"),
                        "min_ade": round(float(records[p]["min_ade"]), 3)}
                       for p in picks],
        })
    sample_indices = sorted(int(records[p]["idx"]) for p in selected)
    return sample_indices, summary
