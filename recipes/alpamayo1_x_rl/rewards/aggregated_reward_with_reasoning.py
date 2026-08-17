# NOTE: modified in this fork -- see upstream NVlabs/alpamayo-recipes for the original.
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


"""Aggregated reward with chain-of-thought (CoT) scoring for Alpamayo Cosmos-RL."""

from __future__ import annotations

from typing import Any

_REQUIRED_REWARD_KEYS: list[str] = [
    "traj_l2_weight",
    "comfort_weight",
    "reasoning_weight",
]

# NOTE: fork addition -- CoC<->action consistency reward term (see
# rewards/coc_action_consistency_reward.py). Optional, defaulting to 0.0 so an
# unmodified TOML is byte-for-byte unaffected: the extra extraction/classification
# work below only runs when a caller explicitly sets `coc_consistency_weight > 0`.
# Status: the matching logic is a validated port of an offline-selected design
# (alpamayo-coc-autolabeler, E3xM5-abstain); the trajectory-side classifier feeding it
# here is a segmentation+merge design validated at 0.932-0.933 gold binary against
# ground-truth future motion -- an oracle ceiling (never run against real rollout
# output), not yet a forecast of in-loop reward values. See
# coc_action_consistency_trajectory.py's docstring before trusting this term's scale
# relative to the other three.
_OPTIONAL_REWARD_DEFAULTS: dict[str, float] = {
    "coc_consistency_weight": 0.0,
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
    """Aggregate traj, comfort, and CoT reasoning into one scalar reward.

    Trajectory and comfort match :func:`alpamayo1_x_rl.rewards.aggregated_reward.compute_reward`.
    Additionally parses CoT from ``to_be_evaluated``, grades it against ground-truth
    CoT when both are present, and adds the weighted reasoning component (or a
    penalty when CoT is missing after decode).
    """
    from alpamayo_r1.models.token_utils import extract_between_special_tokens
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

    pred_cot = extract_between_special_tokens([to_be_evaluated], token="cot")[0]
    gt_cot = reference.get("cot", "")

    logger.debug(f"[compute_reward] Pred_cot: {pred_cot}")
    logger.debug(f"[compute_reward] GT_cot: {gt_cot}")

    # FIX 2026-08-16: only load/run the Lingo-Judge grader when reasoning_weight != 0.
    # Previously this ran unconditionally whenever both pred_cot and gt_cot were present,
    # regardless of the weight -- which (a) crashes with FileNotFoundError if
    # reasoning_grading_model_path isn't configured/available, even for a run that never
    # wanted reasoning scored, and (b) more importantly, the gate below
    # (`reasoning_score > reasoning_threshold`) used the resulting score to hard-clamp
    # the ENTIRE reward to -1.0 -- so a run without a working grader silently produced a
    # constant -1.0 reward for every rollout regardless of coc_consistency_score, which
    # would have looked exactly like "the new reward term doesn't work."
    reasoning_score = -1.0
    if w["reasoning_weight"] != 0.0 and pred_cot and gt_cot:
        from alpamayo1_x_rl.utils.light_weight_reasoning_grading_model import (
            get_reasoning_grader_from_config,
        )

        grader = get_reasoning_grader_from_config(config)
        raw_score = float(grader.score(pred_cot, gt_cot).item())
        reasoning_score = raw_score - 1.0

    # NOTE: fork addition -- CoC<->action consistency (see
    # rewards/coc_action_consistency_reward.py for the full design). Computed when the
    # weight is nonzero (trained on) OR traj_viz is enabled (monitored but not trained
    # on -- the consistency card below needs `claims`/`traj_state` to draw anything
    # useful), so an unmodified TOML with both off pays no extraction cost. Uses the
    # REGEX fallback extractor unless a GPU extractor is explicitly configured under
    # [custom.alpamayo.reward].coc_extractor_model_path -- the Qwen in-loop extractor
    # works (validated 2026-08-16) but is unbatched and ~600x slower than regex per
    # call, see coc_action_consistency_extract.py's docstring before enabling it here.
    from alpamayo1_x_rl.rewards import traj_viz as _traj_viz

    coc_consistency_score = 0.0
    coc_consistency_abstained = 0.0
    coc_comp: dict[str, Any] | None = None
    if w["coc_consistency_weight"] != 0.0 or _traj_viz.is_enabled(config):
        from alpamayo1_x_rl.rewards.coc_action_consistency_extract import (
            get_claim_extractor_from_config,
        )
        from alpamayo1_x_rl.rewards.coc_action_consistency_reward import compute_component

        coc_extractor = get_claim_extractor_from_config(config)
        coc_comp = compute_component(
            pred_cot, predicted_fut_xyz[0], predicted_fut_rot[0], extractor=coc_extractor
        )
        coc_consistency_score = coc_comp["reward_contribution"]
        coc_consistency_abstained = float(coc_comp["abstained"])

    # Continuous reward: each component contributes independently, no hard gates.
    ade_threshold = 3.0
    reasoning_threshold = -0.4
    pred_cot_decoded = bool(pred_cot and len(pred_cot.strip()) > 0)
    # Only require a passing reasoning score when reasoning is actually weighted -- see
    # the FIX note above. Unweighted, reasoning_score stays at its -1.0 default and would
    # otherwise clamp every reward to -1.0 regardless of the other (weighted) terms.
    reasoning_ok = w["reasoning_weight"] == 0.0 or reasoning_score > reasoning_threshold

    if pred_cot_decoded and reasoning_ok and l2_dist < ade_threshold:
        final_reward = (
            -w["traj_l2_weight"] * (l2_dist / ade_threshold)
            + w["comfort_weight"] * comfort_score
            + w["reasoning_weight"] * (reasoning_score / reasoning_threshold)
            + w["coc_consistency_weight"] * coc_consistency_score
        )
    else:
        final_reward = -1.0

    logger.debug(
        f"[compute_reward] l2={l2_dist:.3f} reasoning={reasoning_score:.3f} "
        f"coc_consistency={coc_consistency_score:.3f} "
        f"cot_decoded={pred_cot_decoded} final={final_reward:.4f}"
    )
    reward_dict: dict[str, float] = {
        "traj_L2": float(l2_dist),
        "comfort_reward": float(comfort_score),
        "coc_consistency": float(coc_consistency_score),
        "coc_consistency_abstained": coc_consistency_abstained,
        "reasoning_score": float(reasoning_score),
        "reward": float(final_reward),
    }

    # NOTE: fork addition -- BEV plot of rollout vs reference policy vs GT, captioned
    # with predicted vs ground-truth CoC side by side, PLUS (when pred_rot/coc_comp are
    # given) a separate consistency card: predicted CoC, extracted claims, extracted
    # trajectory meta-action, camera, BEV, and the long/lat segmentation strip -- see
    # traj_viz.py's docstring. No-op unless [custom.alpamayo.traj_viz].enable is true;
    # never raises.
    _traj_viz.record_and_maybe_log(
        idx=reference.get("idx"),
        gt_xyz=gt_fut_xyz,
        pred_xyz=predicted_fut_xyz,
        hist_xyz=reference.get("ego_history_xyz"),
        cot_text=f"pred: {pred_cot or '(none)'}\ngt:   {gt_cot or '(none)'}",
        reward_dict=reward_dict,
        config=config,
        pred_rot=predicted_fut_rot[0],
        coc_comp=coc_comp,
    )

    return reward_dict["reward"], reward_dict
