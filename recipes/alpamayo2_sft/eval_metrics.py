"""Shared BEV displacement metrics at fixed horizons for the A2 offline eval.

Mirrors ``alpamayo1_5_sft/truckdrive/horizon_metrics.py`` so the two recipes'
eval outputs are directly comparable. ``evaluate_hf.py``'s per-window records
here are ``{"pred_xyz": [K, T, 3], "gt_xyz": [T, 3]}`` (single group, no ``N``
trajectory-set axis -- A2 rollout only ever produces one set of K samples),
unlike A1.5's ``[B, N, K, T, 3]`` / ``[B, T, 3]`` batched tensors.

Metrics are BEV (XY-plane) L2, matching ``alpamayo.metrics.distance_metrics``
(``only_xy=True``):

  min_ade/by_t=T  best-of-K ADE, **convention B**: the single trajectory with
                  the lowest error over the FULL predicted horizon is selected
                  once, then its mean error over 0..T is reported (matches
                  ``alpamayo_r1``'s ``DistanceMetrics``, not a per-horizon
                  re-selection).
  ade/by_t=T      ADE of sample 0 over 0..T (sample 0, not a confidence pick --
                  A1.5's dummy zero logprob argmax also always lands on index 0).
  min_fde/by_t=T  best-of-K final displacement: min over K of the L2 at step T.
  fde/by_t=T      final displacement of sample 0 at step T.

T is in seconds; the step index is ``round(T / time_step)``. A2's full horizon
is 64 steps @ 0.1 s = 6.4 s. Bare ``min_ade`` / ``ade`` (no ``by_t`` suffix)
are the full-horizon values, matching A1.5's naming.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

# Union of the horizons A1.5 reports across DistanceMetrics (0.5, 1.0, 3.0,
# 5.0) and the fork's horizon_metrics.py addition (3.0, 6.0, 6.4).
DEFAULT_HORIZONS_S: tuple[float, ...] = (0.5, 1.0, 3.0, 5.0, 6.0, 6.4)


def displacement_metrics_from_records(
    records: Sequence[dict],
    horizons_s: Iterable[float] = DEFAULT_HORIZONS_S,
    time_step: float = 0.1,
) -> dict[str, float]:
    """Compute mean min_ade/ade/min_fde/fde at each horizon, averaged over records.

    Args:
        records: per-window dicts with ``pred_xyz`` [K, T, 3] and ``gt_xyz`` [T, 3]
            (as dumped by ``evaluate_hf.py`` into ``eval_predictions.json``).
        horizons_s: horizons in seconds to report.
        time_step: seconds per trajectory step.

    Returns:
        Flat ``{name: float}`` dict, e.g. ``{"min_ade/by_t=3.0": ..., "min_ade": ...}``.
    """
    pred = torch.stack([torch.as_tensor(r["pred_xyz"]) for r in records]).float()  # [B, K, T, 3]
    gt = torch.stack([torch.as_tensor(r["gt_xyz"]) for r in records]).float()  # [B, T, 3]

    # BEV (XY) per-step L2: [B, K, T]
    l2 = torch.linalg.norm((pred - gt[:, None])[..., :2], dim=-1)
    _, _, total_steps = l2.shape

    # Convention-B best mode: the trajectory minimising mean error over the
    # FULL horizon, selected once per window. [B]
    k_best = l2.mean(dim=-1).argmin(dim=1)
    sel = torch.take_along_dim(l2, k_best[:, None, None], dim=1).squeeze(1)  # [B, T]

    out: dict[str, float] = {}
    horizons_s = sorted(set(horizons_s))
    full_sec = None
    for sec in horizons_s:
        t = int(round(sec / time_step))
        if t < 1 or t > total_steps:
            continue
        idx = t - 1
        out[f"min_ade/by_t={sec:.1f}"] = sel[..., :t].mean(dim=-1).mean().item()
        out[f"ade/by_t={sec:.1f}"] = l2[:, 0, :t].mean(dim=-1).mean().item()
        out[f"min_fde/by_t={sec:.1f}"] = l2[..., idx].min(dim=1).values.mean().item()
        out[f"fde/by_t={sec:.1f}"] = l2[:, 0, idx].mean().item()
        if t == total_steps:
            full_sec = sec

    # Bare aliases for the full-horizon values, matching A1.5's naming
    # ("min_ade" / "ade" with no "by_t" suffix means the full trajectory).
    if full_sec is not None:
        out["min_ade"] = out[f"min_ade/by_t={full_sec:.1f}"]
        out["ade"] = out[f"ade/by_t={full_sec:.1f}"]
    else:
        out["min_ade"] = sel.mean(dim=-1).mean().item()
        out["ade"] = l2[:, 0, :].mean(dim=-1).mean().item()

    return out
