# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Shared BEV displacement metrics at fixed horizons (single source of truth).

Both the offline scorer (``score_val_horizons.py``) and the online eval
(``evaluate_hf.py`` -> ``metrics.json``) call this so their numbers agree.

Metrics are BEV (XY-plane) L2, matching ``alpamayo.metrics.distance_metrics``
(``only_xy=True``). For a set of K sampled trajectories per window:

  min_ade/by_t=T  best-of-K ADE, **convention B**: the single trajectory with the
                  lowest error over the FULL predicted horizon is selected once,
                  then its mean error over 0..T is reported. This matches
                  ``alpamayo.metrics.metric_api.DistanceMetrics`` (and hence the
                  streamed ``metrics.json`` ``min_ade`` / ``min_ade/by_t=*``),
                  NOT a per-horizon re-selection.
  ade/by_t=T      ADE of sample 0 over 0..T. Sample 0 is what the streamed ``ade``
                  reports too, because its logprob tensor is a dummy zeros tensor
                  so argmax picks index 0 -- i.e. this is the FIRST sample, not a
                  confidence-selected "most likely" one.
  min_fde/by_t=T  best-of-K final displacement: min over K of the L2 at step T.
  fde/by_t=T      final displacement of sample 0 at step T.

T is in seconds; the step index is ``round(T / time_step)``. The model's full
horizon is 64 steps @ 0.1 s = 6.4 s, so e.g. ``by_t=6.0`` truncates to 60 steps
and ``by_t=6.4`` is the full trajectory.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

# Horizons (seconds) reported by default. 6.4 s is the full trajectory; 6.0 s is
# the literal "6 s" truncation; 3.0 s mirrors the existing metrics.json by_t.
DEFAULT_HORIZONS_S: tuple[float, ...] = (3.0, 6.0, 6.4)


def displacement_metrics_from_tensors(
    pred_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
    horizons_s: Iterable[float] = DEFAULT_HORIZONS_S,
    time_step: float = 0.1,
) -> dict[str, float]:
    """Compute horizon metrics from batched tensors.

    Args:
        pred_xyz: [B, N, K, T, 3] predicted trajectories (N traj-sets, K samples).
        gt_xyz:   [B, T, 3] ground-truth future (single traj group).
        horizons_s: horizons in seconds to report.
        time_step: seconds per trajectory step.

    Returns:
        Flat ``{name: float}`` dict, e.g. ``{"min_ade/by_t=3.0": ..., ...}``.
    """
    pred_xyz = pred_xyz.float()
    gt_xyz = gt_xyz.float()
    # BEV (XY) per-step L2: [B, N, K, T]
    l2 = torch.linalg.norm((pred_xyz - gt_xyz[:, None, None])[..., :2], dim=-1)
    _, _, _, total_steps = l2.shape

    # Convention-B best mode: the trajectory minimising mean error over the FULL
    # horizon, selected once per (B, N). [B, N]
    k_best = l2.mean(dim=-1).argmin(dim=2)
    sel = torch.take_along_dim(l2, k_best[:, :, None, None], dim=2).squeeze(2)  # [B, N, T]

    out: dict[str, float] = {}
    for sec in horizons_s:
        t = int(round(sec / time_step))
        if t < 1 or t > total_steps:
            continue
        idx = t - 1  # final step of this horizon
        out[f"min_ade/by_t={sec:.1f}"] = sel[..., :t].mean(dim=-1).mean().item()
        out[f"ade/by_t={sec:.1f}"] = l2[:, :, 0, :t].mean(dim=-1).mean().item()
        out[f"min_fde/by_t={sec:.1f}"] = l2[..., idx].min(dim=2).values.mean().item()
        out[f"fde/by_t={sec:.1f}"] = l2[:, :, 0, idx].mean().item()
    return out


def displacement_metrics_from_records(
    records: Sequence[dict],
    horizons_s: Iterable[float] = DEFAULT_HORIZONS_S,
    time_step: float = 0.1,
) -> dict[str, float]:
    """Same as :func:`displacement_metrics_from_tensors` but from dumped records.

    ``records`` are the per-sample dicts saved by ``evaluate_hf.py``: each has
    ``pred_xyz`` [N, K, T, 3] and ``ego_future_xyz`` [n_traj_group, T, 3] (the
    last group is used, matching ``DistanceMetrics``).
    """
    pred = torch.stack([torch.as_tensor(r["pred_xyz"]) for r in records])  # [B, N, K, T, 3]
    gt = torch.stack([torch.as_tensor(r["ego_future_xyz"])[-1] for r in records])  # [B, T, 3]
    return displacement_metrics_from_tensors(pred, gt, horizons_s, time_step)
