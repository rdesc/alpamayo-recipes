# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Aggregated reward computation for Alpamayo Cosmos-RL training."""

from __future__ import annotations

from typing import Any

_REQUIRED_REWARD_KEYS: list[str] = [
    "traj_l2_weight",
    "comfort_weight",
]

# NOTE: fork addition. Optional keys, with the upstream values as defaults so an
# unmodified TOML keeps the original behaviour.
#   ade_threshold -- ADE [m] that maps to a full `traj_l2_weight` penalty.
#   ade_cap       -- ADE [m] beyond which the penalty stops growing. Upstream
#                    hard-gated instead: `ADE >= ade_threshold -> reward = -1.0`,
#                    a flat region with NO gradient. Since GRPO advantages are
#                    (r - mean)[/std], a group whose completions all land past
#                    the gate gets identical rewards -> zero advantage -> zero
#                    gradient, so a policy that drifts past it cannot learn its
#                    way back. Clipping the ADE instead keeps the reward monotone
#                    from 0 to ade_cap while still bounding it. Set
#                    ade_cap == ade_threshold to recover the upstream cliff.
_OPTIONAL_REWARD_DEFAULTS: dict[str, float] = {
    "ade_threshold": 3.0,
    "ade_cap": 9.0,
}


def _get_reward_cfg(config: object | None) -> dict[str, float]:
    """Extract reward parameters from Cosmos TOML [custom.alpamayo.reward]."""
    try:
        reward_cfg = getattr(config, "custom")["alpamayo"]["reward"]
    except (TypeError, KeyError, AttributeError) as e:
        raise ValueError(
            "Reward config not found in TOML. "
            f"Required keys under [custom.alpamayo.reward]: {_REQUIRED_REWARD_KEYS}"
        ) from e

    missing = [k for k in _REQUIRED_REWARD_KEYS if k not in reward_cfg]
    if missing:
        raise ValueError(f"Missing key(s) in [custom.alpamayo.reward]: {missing}")

    out = {k: float(reward_cfg[k]) for k in _REQUIRED_REWARD_KEYS}
    for k, default in _OPTIONAL_REWARD_DEFAULTS.items():
        out[k] = float(reward_cfg.get(k, default))
    if out["ade_cap"] < out["ade_threshold"]:
        raise ValueError(
            f"ade_cap ({out['ade_cap']}) must be >= ade_threshold ({out['ade_threshold']})"
        )
    return out


def compute_reward(
    to_be_evaluated: str,
    reference: dict[str, Any],
    *,
    tokenizer: Any,
    traj_tokenizer: Any,
    config: object | None = None,
    model_config: Any,
) -> tuple[float, dict[str, float]]:
    """Compute the aggregated reward for a single rollout against reference data."""
    from alpamayo1_x_rl.rewards.comfort_reward import compute_comfort
    from cosmos_rl.utils.logging import logger  # pyright: ignore[reportMissingImports]

    from alpamayo1_x_rl.rewards.traj_reward import calculate_ade
    from alpamayo1_x_rl.utils.trajectory_decode import decode_rollout_trajectory

    w = _get_reward_cfg(config)

    gt_fut_xyz = reference["ego_future_xyz"]
    predicted_fut_xyz, predicted_fut_rot = decode_rollout_trajectory(
        to_be_evaluated,
        reference["ego_history_xyz"],
        reference["ego_history_rot"],
        tokenizer=tokenizer,
        traj_tokenizer=traj_tokenizer,
        model_config=model_config,
    )

    l2_dist = calculate_ade(predicted_fut_xyz[0], gt_fut_xyz[0])

    comfort_dict_t = compute_comfort(
        predicted_fut_xyz[:, None, None, ...],
        predicted_fut_rot[:, None, None, ...],
    )
    comfort_score = float(sum(comfort_dict_t.values()) / len(comfort_dict_t))
    comfort_score = comfort_score - 1.0

    # NOTE: fork change -- clip the ADE rather than hard-gating the reward, so the
    # penalty stays monotone out to `ade_cap` instead of collapsing to a flat
    # -1.0 past `ade_threshold`. Below `ade_threshold` this is numerically
    # identical to upstream. See _OPTIONAL_REWARD_DEFAULTS for the rationale.
    ade_threshold = w["ade_threshold"]
    ade_cap = w["ade_cap"]
    ade_eff = min(l2_dist, ade_cap)
    final_reward = (
        -w["traj_l2_weight"] * (ade_eff / ade_threshold) + w["comfort_weight"] * comfort_score
    )
    logger.debug(f"[compute_reward] Final reward: {final_reward}")
    reward_dict: dict[str, float] = {
        # traj_L2 stays RAW (unclipped) so the logged ADE remains interpretable;
        # only the reward uses the clipped value.
        "traj_L2": float(l2_dist),
        "comfort_reward": float(comfort_score),
        "reward": float(final_reward),
        # Fraction of rollouts pinned at the cap. Averaged over a batch this is
        # the saturation rate -- the replacement for "reward == -1.0" as the
        # signal that rollouts are far outside the useful range.
        "ade_capped": float(l2_dist >= ade_cap),
    }

    # NOTE: fork addition -- BEV plot of rollout vs reference policy vs GT.
    # No-op unless [custom.alpamayo.traj_viz].enable is true; never raises.
    from alpamayo1_x_rl.rewards.traj_viz import record_and_maybe_log

    record_and_maybe_log(
        idx=reference.get("idx"),
        gt_xyz=gt_fut_xyz,
        pred_xyz=predicted_fut_xyz,
        hist_xyz=reference.get("ego_history_xyz"),
        cot_text=reference.get("cot"),
        reward_dict=reward_dict,
        config=config,
    )

    return reward_dict["reward"], reward_dict
