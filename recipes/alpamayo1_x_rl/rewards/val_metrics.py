# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""SFT-parity geometric metrics over a *group* of rollouts sharing one prompt.

Motivation: the shipped reward logs only per-completion scalars (``reward``,
``traj_L2``, ``comfort_reward``). ``traj_L2`` is the mean ADE over the sampled
completions, which is not the number Alpamayo is evaluated on -- the SFT recipe
reports **minADE over K samples** (``alpamayo.metrics.metric_api.DistanceMetrics``,
wired into ``configs/sft_base.yaml`` and ``callbacks/val_metric_callback.py``).
A policy can improve mean ADE while its best-of-K gets worse, and vice versa, so
the two curves are not interchangeable.

This module computes exactly the SFT quantities, by calling the *same*
``alpamayo.metrics.distance_metrics`` functions the SFT metric runner calls
rather than reimplementing them, so an RL ``val/min_ade`` is directly comparable
to an SFT ``val/minADE_mean``:

* ``min_ade``               -- min over K of per-sample ADE (XY plane).
* ``min_ade/by_t={0.5,1.0,3.0,5.0}`` -- the same, truncated to each horizon.
  Horizons past the available future length are silently skipped.
* ``ade``                   -- mean ADE over the K samples. Kept because it is
  the group-level counterpart of ``traj_L2`` and shows the mean-vs-best gap.
* ``corner_distance``       -- min-over-K mean distance between the 8 corners of
  the predicted and GT ego boxes. Unlike ADE this is sensitive to *heading*,
  which pure-XY ADE ignores entirely.

**Why a group function and not a per-completion one.** minADE needs every
completion for a prompt at once. Cosmos-RL exposes this via
``[train.train_policy].group_reward_calculation = true``, which hands the whole
group to the reward fn in one call (``cosmos_rl/dispatcher/algo/reward.py:396``).
That flag is required for these metrics to appear; without it only the
per-completion keys are logged.

**Why the values are attached to every completion in the group.** Cosmos-RL's
``aggregate_report_data`` (``cosmos_rl/utils/util.py:1299``) averages each metric
key over the flat list of *rollouts*, substituting 0 for rollouts missing the
key. So a group-level value written onto only the first rollout of each group --
the idiom upstream uses for ``most_likely_mode_reward_mean`` -- comes out
diluted by exactly ``1/n_generation``, not averaged over groups. Writing the same
value onto all K rollouts makes the mean come out right.

**Deviation from SFT, deliberate.** SFT's ``DistanceMetrics`` also reports an
``ade`` that selects the single highest-logprob sample. Cosmos-RL does surface
``cumulative_logprob`` per completion, but only inside ``RolloutGroup`` -- it is
not passed to the reward fn -- so mode-selection is not reproducible here. The
``ade`` reported below is the plain mean over K.
"""

from __future__ import annotations

from typing import Any

import torch

# Horizons in seconds, and the trajectory timestep. Both mirror
# `alpamayo.metrics.metric_api.DistanceMetrics` (which hardcodes the same list
# with a `TODO: move this to a config`) so RL and SFT bucket identically.
TIMESTEP_HORIZONS_S: tuple[float, ...] = (0.5, 1.0, 3.0, 5.0)
TIME_STEP: float = 0.1


def _as_bnkt(traj_list: list[torch.Tensor]) -> torch.Tensor:
    """Stack K per-completion trajectories into the [B=1, N=1, K, T, ...] layout.

    ``decode_rollout_trajectory`` returns [1, T, 3] (xyz) or [1, T, 3, 3] (rot);
    ``alpamayo.metrics.distance_metrics`` wants a leading batch dim and an
    explicit "group of candidates" dim.
    """
    stacked = torch.stack([t[0] for t in traj_list], dim=0)  # [K, T, ...]
    return stacked[None, None]  # [1, 1, K, T, ...]


def group_distance_metrics(
    pred_xyz_list: list[torch.Tensor],
    pred_rot_list: list[torch.Tensor],
    gt_xyz: torch.Tensor,
    gt_rot: torch.Tensor | None = None,
    ego_lwh: Any = None,
) -> dict[str, float]:
    """Compute SFT-parity group metrics for one prompt's K completions.

    Args:
        pred_xyz_list: K decoded trajectories, each [1, T, 3].
        pred_rot_list: K decoded rotations, each [1, T, 3, 3].
        gt_xyz: ground-truth future, [G, T, 3]; group 0 is used, matching
            ``calculate_ade``'s ``gt_fut_xyz[0]`` so ``min_ade`` and ``traj_L2``
            are computed against the same target.
        gt_rot: ground-truth rotations, [G, T, 3, 3]. Corner distance is skipped
            when this is None.
        ego_lwh: ego box dims. Falls back to SFT's ``EGO_VEHICLE_LWH`` constant,
            which is itself a placeholder -- so treat ``corner_distance`` as
            comparable to SFT, not as true-to-vehicle.

    Returns:
        Flat ``{metric_name: float}``. Never raises: a failure returns {} so a
        metric can't kill a training run.
    """
    from alpamayo.metrics import distance_metrics

    out: dict[str, float] = {}
    if not pred_xyz_list:
        return out

    pred = _as_bnkt(pred_xyz_list).float()  # [1, 1, K, T, 3]
    gt = gt_xyz[0][None].float().to(pred.device)  # [1, T, 3]

    # Guard against a decode that produced a different future length than the GT
    # (e.g. a truncated completion): the metric functions broadcast silently.
    if pred.shape[-2] != gt.shape[-2]:
        return out

    horizons = [int(t / TIME_STEP) for t in TIMESTEP_HORIZONS_S]
    minade = distance_metrics.compute_minade(
        pred,
        gt,
        disable_summary=True,  # N == 1: no cross-group mean/std to summarize
        timestep_horizons=horizons,
        time_step=TIME_STEP,
    )
    for k, v in minade.items():
        out[k] = float(v.mean())

    # Mean-over-K ADE: the group counterpart of the per-completion traj_L2, and
    # the gap against min_ade is how much of the improvement is best-of-K luck.
    out["ade"] = float(distance_metrics.compute_ade(pred, gt).mean())

    if gt_rot is not None and pred_rot_list:
        try:
            from alpamayo.metrics.metric_api import EGO_VEHICLE_LWH

            dims = torch.tensor(
                tuple(ego_lwh) if ego_lwh is not None else EGO_VEHICLE_LWH,
                dtype=torch.float32,
                device=pred.device,
            ).reshape(3)
            corner = distance_metrics.compute_grouped_corner_distance(
                pred,
                _as_bnkt(pred_rot_list).float(),
                gt,
                gt_rot[0][None].float().to(pred.device),
                dims,
                disable_summary=True,
            )
            for k, v in corner.items():
                out[k] = float(v.mean())
        except Exception:  # noqa: BLE001 -- a metric must never kill training
            pass

    return out


def safe_group_distance_metrics(*args, **kwargs) -> dict[str, float]:
    """:func:`group_distance_metrics`, but swallowing every failure.

    Rewards run inside the rollout replica; an exception here would fail the
    whole group's scoring and stall the step. Metrics are strictly observational,
    so they degrade to "absent" instead.
    """
    from cosmos_rl.utils.logging import logger  # pyright: ignore[reportMissingImports]

    try:
        return group_distance_metrics(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[val_metrics] group metrics skipped ({type(e).__name__}: {e})")
        return {}
