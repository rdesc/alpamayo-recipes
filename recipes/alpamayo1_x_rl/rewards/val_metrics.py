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
* ``min_ade/by_t={0.5,1.0,3.0,6.0}`` -- the same, truncated to each horizon.
  Horizons past the available future length are silently skipped.
* ``ade``                   -- mean ADE over the K samples. Kept because it is
  the group-level counterpart of ``traj_L2`` and shows the mean-vs-best gap.

``corner_distance`` was here until 2026-08-18 -- see the NOTE where it was
removed for why it went.

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

# Horizons in seconds, and the trajectory timestep, mirroring
# `alpamayo.metrics.metric_api.DistanceMetrics` (which hardcodes the list with a
# `TODO: move this to a config`).
#
# NOTE: fork change, 2026-08-18. Final horizon 5.0 -> 6.0. The model predicts
# `num_future_steps: 64` (hydra_configs/alpamayo1_5_rvla_rl_pai.yaml:114) =
# 6.4s, so a 5.0s last bucket left the final 1.4s of every predicted trajectory
# unmeasured by any per-horizon metric -- and the late horizon is where the
# trajectories actually diverge. 6.0s = 60 steps, comfortably inside 64.
#
# The remaining three (0.5, 1.0, 3.0) are unchanged and still bucket identically
# to SFT, so those are directly comparable; `min_ade/by_t=6.0` has no SFT
# counterpart and should not be compared against one. Aggregate `min_ade` and
# `ade` are unaffected -- they cover the whole trajectory either way.
TIMESTEP_HORIZONS_S: tuple[float, ...] = (0.5, 1.0, 3.0, 6.0)
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
        gt_rot: ground-truth rotations, [G, T, 3, 3]. Currently unused.
        ego_lwh: ego box dims. Currently unused.

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

    # NOTE: fork change, 2026-08-18. `corner_distance` removed. It tracked
    # min_ade closely enough to add no decision the ADE curves did not already
    # support, and its absolute scale was never trustworthy anyway: ego_lwh
    # falls back to SFT's EGO_VEHICLE_LWH, itself a placeholder. The gt_rot and
    # ego_lwh parameters are kept in the signature -- both callers pass them
    # positionally, and a rotation-aware metric is the obvious thing to add back
    # here if one is ever wanted.

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
