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

"""CoC <-> action consistency reward: ties extraction + trajectory classification +
matching into a scalar component, and (optionally) a standalone reward entry point.

See the sibling modules for what is solid vs. draft:
  - ``coc_action_consistency_matcher.py``    -- validated, verbatim-ported matching logic.
  - ``coc_action_consistency_trajectory.py`` -- segmentation+merge trajectory-side
    classifier, validated at 0.932/0.902 gold-binary/hard-sep against ground-truth
    future motion (an oracle ceiling -- never run against real rollout output; see its
    docstring). This module adds one thing on top: a longitudinal `reverse` claim reads
    the trajectory state at t0+1s instead of t0 (see ``REVERSE_OFFSET_S`` below).
  - ``coc_action_consistency_extract.py``    -- regex fallback (solid port) + DRAFT
    in-loop Qwen extractor (untested in-loop).

Usage as a component inside another reward (the intended integration -- see the
opt-in ``coc_consistency_weight`` term added to ``aggregated_reward_with_reasoning.py``):

    from alpamayo1_x_rl.rewards.coc_action_consistency_reward import compute_component
    comp = compute_component(pred_cot, predicted_fut_xyz[0], predicted_fut_rot[0])
    final_reward += weight * comp["reward_contribution"]

Usage as a standalone reward_fn (useful for the "overfit on 1->16 samples" smoke-test
workflow the recipes-side scoping doc, ``docs/coc_action_consistency_rl.md``,
recommends before scaling up any new reward): ``compute_reward`` below matches the same
``(to_be_evaluated, reference, *, tokenizer, traj_tokenizer, config, model_config) ->
(float, dict)`` signature as ``aggregated_reward.compute_reward`` /
``aggregated_reward_with_reasoning.compute_reward``, so it can be registered as its own
``reward_fn`` on a ``ModelSpec`` (see either of those two entry-point files under
``models/reasoning_vla/`` for the registration pattern) without needing a new entry
script beyond swapping the import. This standalone path has NOT been run against
Cosmos-RL; it is provided for smoke-testing, not as a verified alternative to the
integrated component above.
"""

from __future__ import annotations

from typing import Any, Optional

from alpamayo1_x_rl.rewards.coc_action_consistency_extract import (
    extract_claims_regex,
    get_claim_extractor_from_config,
)
from alpamayo1_x_rl.rewards.coc_action_consistency_matcher import Claim, score_claims
from alpamayo1_x_rl.rewards.coc_action_consistency_trajectory import trajectory_state_from_pred

# ---------------------------------------------------------------------------------
# DESIGN DECISION -- flagged per the task, needs a human sign-off before this ships:
#
# What should an ABSTAINED event (every extracted claim was a hedged lateral claim,
# `event_abstained=True`) contribute to a single rollout's reward?
#
# The offline corpus-accuracy convention (cac_scorer_design_results.md /
# cac_abstain_variant.py) is to EXCLUDE these events from the accuracy denominator
# entirely when reporting one aggregate number over a fixed 2077-event corpus. That
# convention does not directly translate to a single scalar reward for ONE rollout --
# "exclude from the population" isn't a value you can add into a weighted sum.
#
# Three options, mirroring the *offline* doc's own "Abstention handling" section
# (which covers a related but distinct case -- an axis with NO claim at all, not an
# event where every claim was withdrawn by the hedge rule):
#   1. MONITOR, DON'T PENALIZE (chosen here): drop the term from the weighted sum for
#      this rollout (contribute 0 * weight, i.e. same as if this reward term didn't
#      exist for this one rollout) and emit `coc_consistency_abstained=1.0` as a
#      separate metric key so abstention rate can be tracked across training. This is
#      the option the offline doc explicitly recommends for the analogous axis-silence
#      question ("recommended... Track abstention rate as a separate diagnostic during
#      RL... a rising trend is direct evidence of gaming"), generalized here to
#      full-event abstention. It cannot reward a policy for learning to phrase its way
#      into abstention (abstention isn't scored at all, positively or negatively) and
#      needs no cross-worker state.
#   2. Fixed value (e.g. score as 0, or as 1): rejected -- scoring 0 directly
#      contradicts the reason M5-abstain exists (penalizing exactly the claims it
#      judged unreliable to judge); scoring 1 is a free pass exploitable by a policy
#      that learns to hedge every lateral claim.
#   3. Impute at a running average (reward-neutral relative to typical behavior):
#      considered and rejected FOR NOW on engineering grounds, not principle -- it
#      requires synchronized cross-process state across GRPO's distributed rollout
#      workers, which is real plumbing this repo doesn't have a precedent for (the
#      existing group-reward path in aggregated_reward.py only aggregates within one
#      prompt's completions, not across the whole training run). Worth revisiting if
#      option 1 turns out to make the reward too sparse in practice (e.g. if
#      model-generated CoC turns out to hedge far more than the terse gold corpus did
#      -- the offline doc explicitly warns the 2.1% abstention rate measured on gold
#      CoC may not transfer to longer, more discursive model output).
#
# NEEDS A DECISION: whether option 1 is actually the right call once real rollout
# abstention rates are measured (per the offline doc's own caveat, re-measure before
# trusting any of these rates on model-generated CoC), and whether "drop the term"
# vs. "exclude the whole rollout from the reward computation" (a bigger structural
# change touching L2/comfort too) is preferred once this is observed in practice.
# ---------------------------------------------------------------------------------

# Longitudinal `reverse` claims read the trajectory a beat later than every other claim.
# Validated on the gold corpus: 7/7 correct at 1.0s vs 5/7 at 0.0s, with zero effect on
# every other bucket (offset only ever hurts them -- see
# coc_action_consistency_trajectory.py's docstring). Reverse maneuvers in this corpus
# often begin from a brief stop ("back out of the driveway after stopping"), so the
# backward-motion signal is genuinely absent at t0, not just noisy -- this is NOT a
# general "try a later offset" knob, it is specific to this one bucket.
REVERSE_OFFSET_S = 1.0


def compute_component(
    pred_cot: Optional[str],
    pred_xyz,
    pred_rot,
    *,
    claims: Optional[list[Claim]] = None,
    extractor: Any = None,
) -> dict[str, Any]:
    """Score one rollout's (CoC text, predicted future trajectory) pair.

    Args:
        pred_cot: decoded CoC text for this rollout (e.g. via
            ``extract_between_special_tokens([to_be_evaluated], token="cot")[0]``,
            the same call ``aggregated_reward_with_reasoning.py`` already makes).
            May be empty/None (e.g. the model didn't emit a CoT before running out of
            budget) -- scores as "no claim extracted", not a crash.
        pred_xyz: ``[T, 3]`` predicted future ego positions (already batch-indexed,
            i.e. ``predicted_fut_xyz[0]`` from ``decode_rollout_trajectory``).
        pred_rot: ``[T, 3, 3]`` predicted future ego rotations.
        claims: pre-extracted claims, bypassing extraction entirely (used by the gold-
            label parity test, and available to callers that already have claims from
            elsewhere).
        extractor: an object with an ``.extract(text) -> list[Claim]`` method (e.g. a
            ``QwenClaimExtractor``). If ``None``, falls back to
            ``extract_claims_regex``. Ignored if ``claims`` is passed.

    Returns:
        dict with:
          reward_contribution: float in [0, 1], the value to multiply by a weight and
            add into an aggregated reward. Always 0.0 when abstained or unparseable
            (see the DESIGN DECISION note above for why abstained is 0 contribution
            rather than a penalty -- it is dropped, not failed).
          graded: the raw graded score in [0, 1] (NaN if abstained), i.e. the mean
            per-axis match score. Chosen over the binary all-axes-pass score for the
            RL reward specifically: GRPO needs a gradient, and this repo's own
            aggregated rewards already avoid hard pass/fail gating for exactly that
            reason (see `aggregated_reward.py`'s `ade_cap` rationale: a flat/gated
            region yields identical rewards within a group -> zero advantage -> zero
            gradient). The *offline validation numbers* (0.941 gold agreement) are for
            the BINARY score; `binary` is also returned below for monitoring/parity
            with those numbers, but `graded` is what feeds the reward.
          binary: 0/1, all-axes-pass score (offline-comparable metric, not the reward).
          abstained: bool, whether every claim was withdrawn (event_abstained).
          parsed: bool, whether any claim was extracted at all.
          n_scored: number of (axis) claims actually scored.
          claims: the claims used (for logging/debugging).
          traj_state: the trajectory-side classification used (for logging/debugging).
    """
    if claims is None:
        text = pred_cot or ""
        if extractor is not None:
            claims = extractor.extract(text)
        else:
            claims = extract_claims_regex(text)

    ts = trajectory_state_from_pred(pred_xyz, pred_rot)
    if any(c.get("axis") == "longitudinal" and c.get("bucket") == "reverse" for c in claims):
        ts = dict(ts)
        ts["longitudinal"] = trajectory_state_from_pred(
            pred_xyz, pred_rot, offset_s=REVERSE_OFFSET_S
        )["longitudinal"]
    result = score_claims(claims, ts)

    graded = result["graded"]
    if result["event_abstained"]:
        reward_contribution = 0.0  # dropped, not failed -- see DESIGN DECISION above
    elif graded != graded:  # NaN guard (shouldn't happen when not abstained, but safe)
        reward_contribution = 0.0
    else:
        reward_contribution = float(graded)

    return {
        "reward_contribution": reward_contribution,
        "graded": graded,
        "binary": result["binary"],
        "abstained": result["event_abstained"],
        "parsed": result["parsed"],
        "n_scored": result["n_scored"],
        "claims": claims,
        "traj_state": ts,
    }


# ---------------------------------------------------------------------------------
# Standalone reward_fn (draft, untested against Cosmos-RL -- see module docstring).
# ---------------------------------------------------------------------------------

_REQUIRED_REWARD_KEYS = ["traj_l2_weight", "coc_consistency_weight"]


def _get_reward_cfg(config: object | None) -> dict[str, float]:
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
    return {k: float(reward_cfg[k]) for k in _REQUIRED_REWARD_KEYS}


def compute_reward(
    to_be_evaluated: str,
    reference: dict[str, Any],
    *,
    tokenizer: Any,
    traj_tokenizer: Any,
    config: object | None = None,
    model_config: Any,
) -> tuple[float, dict[str, float]]:
    """Standalone entry point: traj_l2 + coc_consistency only (no comfort/reasoning-
    quality terms -- add those back in by copying the pattern from
    ``aggregated_reward_with_reasoning.compute_reward`` if this is promoted beyond a
    smoke test).
    """
    from alpamayo_r1.models.token_utils import extract_between_special_tokens

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

    pred_cot = extract_between_special_tokens([to_be_evaluated], token="cot")[0]
    extractor = get_claim_extractor_from_config(config)
    comp = compute_component(pred_cot, predicted_fut_xyz[0], predicted_fut_rot[0], extractor=extractor)

    ade_threshold = 3.0
    final_reward = -w["traj_l2_weight"] * (l2_dist / ade_threshold) + w["coc_consistency_weight"] * comp[
        "reward_contribution"
    ]

    reward_dict = {
        "traj_L2": float(l2_dist),
        "coc_consistency": comp["reward_contribution"],
        "coc_consistency_binary": float(comp["binary"]),
        "coc_consistency_abstained": float(comp["abstained"]),
        "reward": float(final_reward),
    }
    return reward_dict["reward"], reward_dict
