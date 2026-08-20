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
# NOTE: fork addition -- ADE clipping, ported verbatim in spirit from
# aggregated_reward.py's _OPTIONAL_REWARD_DEFAULTS (this file never got the fix).
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
#
# NOTE: fork addition -- reasoning gate escape hatch.
#   reasoning_gate_enable -- 1.0 (default, = existing behaviour) hard-clamps the
#                    WHOLE reward to -1.0 when the graded CoT falls below
#                    `reasoning_threshold`. 0.0 removes that clamp and lets the
#                    (now monotone, see below) reasoning term carry the signal on
#                    its own. Kept ON by default deliberately -- see the block
#                    comment at the gate in _score_one for the full argument.
_OPTIONAL_REWARD_DEFAULTS: dict[str, float] = {
    "coc_consistency_weight": 0.0,
    "ade_threshold": 3.0,
    "ade_cap": 9.0,
    "reasoning_gate_enable": 1.0,
    # NOTE: fork addition, 2026-08-18. Cost subtracted from the group mean when
    # imputing an ABSTAINED rollout -- see the imputation NOTE in
    # compute_group_reward. 0.0 makes abstention exactly advantage-neutral,
    # which is a FREE OPTION: for any input where the policy expects to score
    # below its group mean, hedging beats committing, so uncosted abstention
    # gets over-used (the "learns to phrase its way into abstention" failure the
    # DESIGN DECISION block in coc_action_consistency_reward.py names). A small
    # positive value makes abstention worth taking only when the policy expects
    # to do worse than `mean - cost`, i.e. the honest "I don't know" case.
    # Scale is that of coc_consistency itself, [0, 1]. Setting this to the group
    # mean would recover the old, broken "abstained scores 0.0" behaviour.
    # Defaults to 0.0 so no existing TOML changes behaviour silently; the
    # 50chunks config sets it explicitly.
    "coc_abstain_cost": 0.0,
}


def _apply_cli_overrides(out: dict[str, float]) -> dict[str, float]:
    """NOTE: fork addition. Let ``--<key>`` on the launch command override a TOML weight.

    Sweeping a reward weight otherwise means editing the TOML per run, which
    turns the committed config into a mutable thing several runs share -- it
    stops describing what any particular run actually used.

    Cosmos-RL has no override flag of its own, but it does forward unrecognised
    trailing arguments all the way to every worker process, so a real flag is
    available without patching upstream:

        cosmos-rl --config run.toml entry.py --reasoning-weight 0.3

    The chain, verified end to end:
        launch_all.py:343      `script_args` captured with nargs=REMAINDER
        utility.py:660,720     appended to the per-node/per-replica commands
        launch_replica.sh:87   unknown args collected into SCRIPT_ARGS
        launch_replica.sh:217  expanded into the python invocation

    Flags are the dashed form of the config key (``reasoning_weight`` ->
    ``--reasoning-weight``); every key in the resolved config is overridable, so
    traj_l2/comfort/coc sweeps need no further changes. Both ``--k v`` and
    ``--k=v`` are accepted. Preferred over an environment variable because the
    value lands in the worker's command line, where `ps` and the launcher log
    both show it, and cannot leak into unrelated processes.

    Two deliberate choices:

    * A bad value raises instead of falling back to the TOML. A typo'd weight
      that silently trained at the config default would produce a run whose
      recorded settings are wrong, which is worse than a launch failure.
    * Every applied override is logged at INFO. An override invisible in the
      logs would make W&B's config snapshot (which reflects the TOML) a lie
      about the run.
    """
    import sys

    argv = sys.argv[1:]
    for key in list(out):
        flag = "--" + key.replace("_", "-")
        raw = None
        for i, arg in enumerate(argv):
            if arg == flag and i + 1 < len(argv):
                raw = argv[i + 1]
            elif arg.startswith(flag + "="):
                raw = arg.split("=", 1)[1]
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"{flag} {raw!r} is not a float. Fix or drop the flag; refusing "
                "to silently fall back to the TOML weight."
            ) from e
        if value != out[key]:
            try:
                from cosmos_rl.utils.logging import (  # pyright: ignore[reportMissingImports]
                    logger,
                )

                logger.info(
                    f"[alpamayo1_x_rl.reward] {key}: {out[key]} -> {value} "
                    f"(overridden by {flag}; TOML value ignored)"
                )
            except Exception:  # pragma: no cover - logging must never block a run
                pass
        out[key] = value
    return out


def _get_reward_cfg(config: object | None) -> dict[str, float]:
    """Extract reward parameters from Cosmos TOML [custom.alpamayo.reward].

    ``--<key>`` flags on the launch command take precedence over the TOML --
    see :func:`_apply_cli_overrides`.
    """
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
    out = _apply_cli_overrides(out)
    # Validated after CLI overrides so a bad --ade-cap fails the same way a bad
    # TOML value does (matches aggregated_reward.py:63-66).
    if out["ade_cap"] < out["ade_threshold"]:
        raise ValueError(
            f"ade_cap ({out['ade_cap']}) must be >= ade_threshold ({out['ade_threshold']})"
        )
    return out


def _score_one(
    to_be_evaluated: str,
    reference: dict[str, Any],
    w: dict[str, float],
    *,
    tokenizer: Any,
    traj_tokenizer: Any,
    config: object | None,
    model_config: Any,
    claims: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, float], Any, Any]:
    """Score a single rollout; shared by :func:`compute_reward` (extracts its own CoC
    claims) and :func:`compute_group_reward` (passes in claims already batch-extracted
    for the whole group in one Qwen call). Mirrors the ``_score_one`` /
    ``compute_reward`` / ``compute_group_reward`` split already used in
    ``aggregated_reward.py``, generalized here so the caller can hand in pre-extracted
    ``claims`` -- the actual point of grouping for THIS reward is batching the Qwen
    extraction call, not a minADE@K-style post-hoc metric (this reward has no group-
    level metric of its own, unlike the motion-only reward).

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
    # [custom.alpamayo.reward].coc_extractor_model_path. When ``claims`` is already
    # supplied (group mode: batch-extracted by the caller), extraction is skipped
    # entirely regardless of which extractor would otherwise have been used -- see
    # compute_group_reward below, and coc_action_consistency_extract.py's docstring for
    # the Qwen extractor's own batched-throughput numbers.
    from alpamayo1_x_rl.rewards import traj_viz as _traj_viz

    coc_consistency_score = 0.0
    coc_consistency_abstained = 0.0
    coc_comp: dict[str, Any] | None = None
    if w["coc_consistency_weight"] != 0.0 or _traj_viz.is_enabled(config):
        from alpamayo1_x_rl.rewards.coc_action_consistency_reward import compute_component

        if claims is not None:
            coc_comp = compute_component(
                pred_cot, predicted_fut_xyz[0], predicted_fut_rot[0], claims=claims
            )
        else:
            from alpamayo1_x_rl.rewards.coc_action_consistency_extract import (
                get_claim_extractor_from_config,
            )

            coc_extractor = get_claim_extractor_from_config(config)
            coc_comp = compute_component(
                pred_cot, predicted_fut_xyz[0], predicted_fut_rot[0], extractor=coc_extractor
            )
        coc_consistency_score = coc_comp["reward_contribution"]
        coc_consistency_abstained = float(coc_comp["abstained"])

    # Continuous reward: each component contributes independently, no hard gates.
    ade_threshold = w["ade_threshold"]
    ade_cap = w["ade_cap"]
    reasoning_threshold = -0.4
    pred_cot_decoded = bool(pred_cot and len(pred_cot.strip()) > 0)
    # Only require a passing reasoning score when reasoning is actually weighted -- see
    # the FIX note above. Unweighted, reasoning_score stays at its -1.0 default and would
    # otherwise clamp every reward to -1.0 regardless of the other (weighted) terms.
    reasoning_passed = reasoning_score > reasoning_threshold
    reasoning_ok = (
        w["reasoning_weight"] == 0.0
        or w["reasoning_gate_enable"] == 0.0
        or reasoning_passed
    )

    # NOTE: fork change -- clip the ADE rather than hard-gating the reward, so the
    # penalty stays monotone out to `ade_cap` instead of collapsing to a flat
    # -1.0 past `ade_threshold`. Below `ade_threshold` this is numerically
    # identical to upstream. See _OPTIONAL_REWARD_DEFAULTS for the rationale.
    # This is the same fix aggregated_reward.py:111-120 already carries; this file
    # was simply never updated alongside it.
    ade_eff = min(l2_dist, ade_cap)

    # NOTE: fork change -- FIX SIGN INVERSION in the reasoning term.
    #
    # `reasoning_score` is `sigmoid(logits) - 1.0` (:215, grader returns sigmoid at
    # light_weight_reasoning_grading_model.py:170), so it lives in (-1, 0] with 0 =
    # PERFECT CoT. The previous term was
    #     w * (reasoning_score / reasoning_threshold)
    # which divides a negative score by a negative threshold. The sign comes out
    # positive -- which is presumably why it looked correct -- but the MAGNITUDE is
    # proportional to how far the CoT is from perfect:
    #     raw 1.00 (perfect)      -> reasoning_score  0.00 -> term 0.000  (NO credit)
    #     raw 0.61 (barely passes)-> reasoning_score -0.39 -> term 0.975w (MAX credit)
    # i.e. the reward INCREASED as CoT quality degraded. Training against that would
    # have driven the policy toward CoT sitting just above the acceptance threshold
    # and penalised it for improving -- strictly worse than leaving reasoning off.
    #
    # Option A (chosen): `1 - reasoning_score / reasoning_threshold`
    #   -> 0 at the gate, 1 at perfect, monotone increasing, bounded [0, 1] across
    #      the passing band, so the term is bounded [0, w]. Written against
    #      `reasoning_threshold` rather than a literal 0.4 so the two stay coupled
    #      if the threshold is ever retuned.
    # Option B (rejected): `w * raw_score` (== `w * (reasoning_score + 1)`)
    #   -> also monotone, and has the appeal that the term equals the metric plotted
    #      in W&B. Rejected because it pays 0.6w at the gate -- a standing bonus for
    #      CoT the gate has just declared marginal -- so the floor of the reward
    #      shifts with `reasoning_weight` and the term stops being comparable across
    #      weight sweeps. Option A keeps "barely acceptable" worth exactly nothing.
    reasoning_term = 1.0 - (reasoning_score / reasoning_threshold)

    if pred_cot_decoded and reasoning_ok:
        final_reward = (
            -w["traj_l2_weight"] * (ade_eff / ade_threshold)
            + w["comfort_weight"] * comfort_score
            + w["reasoning_weight"] * reasoning_term
            + w["coc_consistency_weight"] * coc_consistency_score
        )
    else:
        # DECISION 2026-08-17 -- the reasoning gate is KEPT ON by default, unlike the
        # ADE gate removed just above. The two look symmetric but are not:
        #
        #  * The ADE gate was removed because ADE is a continuum -- "3.0m" is an
        #    arbitrary cut through a smooth quantity, and a group that all lands past
        #    it gets identical rewards, zero advantage, and no gradient home. Clipping
        #    preserves the ordering that already existed in the data.
        #  * The reasoning gate encodes a genuine constraint: CoT graded below
        #    raw 0.6 is not "slightly worse reasoning", it is reasoning we do not want
        #    the policy to explore toward at all. -1.0 is also exactly the value used
        #    when CoT fails to decode (`pred_cot_decoded` false), so one floor covers
        #    both "no CoT" and "unacceptable CoT" rather than two different penalties.
        #
        # The honest caveat: this flat region has the SAME zero-advantage pathology
        # the ADE fix addresses. Whether that matters depends on how often rollouts
        # actually fail the gate, which is unknown -- reasoning has never run with a
        # nonzero weight, so there is no measured pass-rate. Rather than guess, the
        # pass rate is now logged (`reasoning_gate_passed` below): if the first run
        # shows a large fraction failing, the flat region is real and
        # `reasoning_gate_enable = 0.0` turns it off without a code change, leaving
        # the monotone reasoning term to carry the signal by itself.
        #
        # Note the fix above already defused the worst of it: the bonus at the gate
        # is now 0 rather than 0.975w, so the cliff is a drop from ~0 to -1.0 instead
        # of from a maximum. The policy is no longer rewarded for loitering there.
        final_reward = -1.0

    logger.debug(
        f"[compute_reward] l2={l2_dist:.3f} (capped {ade_eff:.3f}) "
        f"reasoning={reasoning_score:.3f} term={reasoning_term:.3f} "
        f"gate_passed={reasoning_passed} coc_consistency={coc_consistency_score:.3f} "
        f"cot_decoded={pred_cot_decoded} final={final_reward:.4f}"
    )
    reward_dict: dict[str, float] = {
        # traj_L2 stays RAW (unclipped) so the logged ADE remains interpretable;
        # only the reward uses the clipped value. Matches aggregated_reward.py:123-125.
        "traj_L2": float(l2_dist),
        "comfort_reward": float(comfort_score),
        "coc_consistency": float(coc_consistency_score),
        "coc_consistency_abstained": coc_consistency_abstained,
        "reasoning_score": float(reasoning_score),
        # NOTE: fork addition. Fraction of rollouts clearing the reasoning gate,
        # averaged into train/reasoning_gate_passed by the controller's report
        # aggregation. This is the measurement the "keep the gate on" decision above
        # is waiting on: a low pass rate means the flat -1.0 region is being hit
        # often enough to matter, and `reasoning_gate_enable = 0.0` is the answer.
        # Only meaningful when reasoning_weight != 0 (otherwise the grader never runs
        # and reasoning_score stays at its -1.0 default, reading as 0.0 here).
        "reasoning_gate_passed": float(reasoning_passed),
        # Clipped ADE actually used by the reward, so the effect of ade_cap is
        # visible next to the raw value rather than having to be inferred.
        "ade_capped": float(ade_eff),
        # NOTE: fork addition, 2026-08-18. 1.0 when this rollout took the flat
        # -1.0 reasoning-gate floor instead of the continuous branch. Needed by
        # compute_group_reward's abstention imputation, which must not adjust a
        # reward the coc term never entered; useful on its own as the gate-floor
        # rate, which is the measurement `reasoning_gate_enable` is waiting on.
        "reward_floored": float(not (pred_cot_decoded and reasoning_ok)),
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

    return reward_dict, predicted_fut_xyz, predicted_fut_rot


def compute_reward(
    to_be_evaluated: str,
    reference: dict[str, Any],
    *,
    tokenizer: Any,
    traj_tokenizer: Any,
    config: object | None = None,
    model_config: Any,
) -> tuple[float, dict[str, float]]:
    """Aggregate traj, comfort, and CoT reasoning into one scalar reward for a single
    rollout. See :func:`_score_one` for the actual scoring logic and
    :func:`compute_group_reward` for the group-batched counterpart.
    """
    w = _get_reward_cfg(config)
    reward_dict, _, _ = _score_one(
        to_be_evaluated,
        reference,
        w,
        tokenizer=tokenizer,
        traj_tokenizer=traj_tokenizer,
        config=config,
        model_config=model_config,
    )
    return reward_dict["reward"], reward_dict


def compute_group_reward(
    to_be_evaluated_list: list[str],
    reference: dict[str, Any],
    *,
    tokenizer: Any,
    traj_tokenizer: Any,
    config: object | None = None,
    model_config: Any,
) -> tuple[list[float], list[dict[str, float]]]:
    """Score every completion sharing one prompt/group, batching the CoC claim
    extraction across the whole group in a single call -- the actual point of setting
    ``[train.train_policy].group_reward_calculation = true`` for this reward (see
    ``cosmos_rl/dispatcher/algo/reward.py::compute_per_reward_func`` for the exact
    calling contract this satisfies). Everything else (trajectory decode, comfort,
    reasoning grading) is no more expensive per-item than :func:`compute_reward`
    already pays, so it stays a per-item loop via :func:`_score_one` -- same shape as
    ``aggregated_reward.py``'s ``compute_group_reward``, which only groups for a
    post-hoc minADE@K metric; this reward groups specifically to amortize the Qwen
    forward pass (measured 2.2-2.7x faster at batch 8-32, see
    coc_action_consistency_extract.py's docstring).

    Only actually batches a model call when a GPU extractor is configured
    (``coc_extractor_model_path``) and exposes ``extract_batch`` (``QwenClaimExtractor``
    does). The regex fallback has no model call to batch -- it's cheap enough per-item
    that grouping it would add complexity for no measured benefit, so it still runs one
    string at a time, same as the single-item path.
    """
    from alpamayo_r1.models.token_utils import extract_between_special_tokens
    from alpamayo1_x_rl.rewards import traj_viz as _traj_viz
    from alpamayo1_x_rl.rewards.val_metrics import safe_group_distance_metrics

    w = _get_reward_cfg(config)

    claims_list: list[list[dict[str, Any]]] | None = None
    if w["coc_consistency_weight"] != 0.0 or _traj_viz.is_enabled(config):
        from alpamayo1_x_rl.rewards.coc_action_consistency_extract import (
            extract_claims_regex,
            get_claim_extractor_from_config,
        )

        pred_cots = [pc or "" for pc in extract_between_special_tokens(to_be_evaluated_list, token="cot")]
        coc_extractor = get_claim_extractor_from_config(config)
        if coc_extractor is not None and hasattr(coc_extractor, "extract_batch"):
            claims_list = coc_extractor.extract_batch(pred_cots)
        else:
            claims_list = [
                coc_extractor.extract(text) if coc_extractor is not None else extract_claims_regex(text)
                for text in pred_cots
            ]

    reward_dicts: list[dict[str, float]] = []
    pred_xyz_list = []
    pred_rot_list = []
    for i, to_be_evaluated in enumerate(to_be_evaluated_list):
        reward_dict, pred_xyz, pred_rot = _score_one(
            to_be_evaluated,
            reference,
            w,
            tokenizer=tokenizer,
            traj_tokenizer=traj_tokenizer,
            config=config,
            model_config=model_config,
            claims=claims_list[i] if claims_list is not None else None,
        )
        reward_dicts.append(reward_dict)
        pred_xyz_list.append(pred_xyz)
        pred_rot_list.append(pred_rot)

    # NOTE: fork addition, 2026-08-18. Abstention imputation.
    #
    # An ABSTAINED rollout (every extracted claim was a hedged lateral claim and
    # so withdrawn by the M5-abstain rule) is meant to be MONITORED, NOT
    # PENALIZED -- see the DESIGN DECISION block in
    # coc_action_consistency_reward.py. That module implements "not penalized"
    # as `reward_contribution = 0.0`, reasoning that this is "the same as if
    # this reward term didn't exist for this one rollout".
    #
    # That equivalence is false under GRPO, and this is the bug it caused.
    # Only DIFFERENCES WITHIN A GROUP reach the gradient (advantage = r - mean,
    # and `unbiased_advantage = true` does not even divide by std), so 0.0 is
    # not a neutral value -- it is the MINIMUM of the term's [0, 1] range. An
    # abstained rollout therefore received the most negative advantage in its
    # group, identical to a rollout that emitted unparseable garbage. The reward
    # actively pushed the policy AWAY from hedging, which is the opposite of the
    # documented intent and creates pressure toward confident unhedged lateral
    # claims whether or not the confidence is warranted.
    #
    # "The term does not exist for this rollout" has to mean ZERO ADVANTAGE
    # CONTRIBUTION, not zero reward. So an abstained rollout is imputed at the
    # mean coc score of the non-abstained rollouts in its own group. That value
    # is exactly the whole group's coc mean, for any number of abstainers:
    #     mean_all = (S + a*m) / (n + a),  m = S/n
    #              = (n*m + a*m) / (n + a) = m
    # so at `coc_abstain_cost = 0` the term contributes precisely nothing to the
    # abstainer's advantage. Note this neutralises only the COC TERM: the
    # rollout keeps its traj_l2/comfort advantage, which is the reason to impute
    # rather than drop it from the group. The matcher declined to grade its CoC
    # TEXT, which says nothing about whether it drove well -- dropping it would
    # discard a trajectory signal that is still perfectly good, and would shrink
    # the effective group size besides.
    #
    # ABSTENTION COST. Exact neutrality is not actually what we want, because it
    # makes abstention a free option. A scored rollout gets advantage
    # `coc_i - m`; an abstainer gets 0. So whenever the policy expects to score
    # below its group mean -- i.e. on exactly the inputs it finds hard --
    # hedging strictly beats committing, and nothing else in the reward charges
    # for it (traj_l2 grades the trajectory, and hedging the CoC text does not
    # change the trajectory). That is the classic uncosted-abstention failure,
    # and it is the "learns to phrase its way into abstention" outcome the
    # DESIGN DECISION block hoped option 1 would avoid.
    #
    # `coc_abstain_cost` (delta) shifts the imputed value to `mean - delta`, so
    # abstention is only worth taking when the policy expects to do worse than
    # that -- the honest "I don't know" case. delta = 0 restores exact
    # neutrality. The exploit is narrower than "hedge everything": abstention
    # requires EVERY extracted claim to be lateral AND gentle (a hedged
    # longitudinal claim is still scored -- there was no accuracy gap on that
    # axis, 0.940 vs 0.942), and emitting no claims at all scores 0.0 as a
    # genuine failure. So the target shape is specifically "one gentle lateral
    # claim, no speed claim at all" -- hard to stumble into, but a badly
    # degenerate CoC if it is ever found.
    #
    # The DESIGN DECISION block rejected imputation (its option 3) on the
    # grounds that a running average "requires synchronized cross-process state
    # across GRPO's distributed rollout workers". That objection does not apply
    # here: `group_reward_calculation = true` hands this function all K
    # completions of one prompt in a single call, so the mean is purely local
    # and needs no plumbing. This is why the fix lives here and NOT in
    # compute_reward -- the per-completion path has no group to average over and
    # keeps the old 0.0 behaviour, which is one more reason to run this reward
    # with group_reward_calculation = true.
    #
    # Deliberately NOT imputed:
    #   * "no claims extracted at all" -- `abstained` is False there, and
    #     score_claims separates the two cases precisely because that one IS a
    #     real failure ("unparseable reasoning"). It stays at 0.0 and stays in
    #     the mean below, so the imputed value reflects the group's honest
    #     typical score rather than only its successes.
    #   * rollouts that took the flat -1.0 reasoning-gate floor -- the coc term
    #     never entered their reward, so adjusting it would corrupt the floor.
    # `coc_consistency_imputed` is the rate at which this actually fired. It is
    # distinct from `coc_consistency_abstained`: that counts abstentions, this
    # counts the ones that had a group mean to be neutralised against. Written
    # onto every rollout for the same aggregation reason as the group metrics
    # below (Cosmos means each key over the flat rollout list).
    for d in reward_dicts:
        d["coc_consistency_imputed"] = 0.0
    if w["coc_consistency_weight"] != 0.0:
        eligible = [d for d in reward_dicts if not d["reward_floored"]]
        scored = [d for d in eligible if not d["coc_consistency_abstained"]]
        abstainers = [d for d in eligible if d["coc_consistency_abstained"]]
        # With no scored rollout to average against, every eligible rollout
        # abstained: the term is already constant across the group, so the
        # advantage from it is zero anyway and there is nothing to correct.
        if scored and abstainers:
            group_mean = sum(float(d["coc_consistency"]) for d in scored) / len(scored)
            # Clamped at 0.0: a cost larger than the group mean would push
            # abstention below the worst achievable score, which is the old
            # over-penalizing behaviour this whole block exists to remove.
            imputed = max(0.0, group_mean - w["coc_abstain_cost"])
            for d in abstainers:
                # Replace the 0.0 the component contributed with the imputed value.
                d["reward"] = float(d["reward"]) + w["coc_consistency_weight"] * imputed
                d["coc_consistency"] = float(imputed)
                d["coc_consistency_imputed"] = 1.0

    # NOTE: fork addition, 2026-08-18. Ported from aggregated_reward.py:217 --
    # this file's group path had been discarding the pred_xyz/pred_rot that
    # `_score_one` already returns, so a run on this reward logged no `min_ade`
    # at all. That is a much bigger hole than a missing
    # nice-to-have: `traj_L2` is a mean over all K completions, whereas the SFT
    # eval this has to be comparable against reports min-over-K, and on the
    # isolated coc_consistency config (traj_l2_weight = 0) these are the ONLY
    # numbers not part of the objective -- i.e. the sole evidence that the
    # policy is improving rather than gaming the consistency term.
    group_metrics = safe_group_distance_metrics(
        pred_xyz_list,
        pred_rot_list,
        reference["ego_future_xyz"],
        reference.get("ego_future_rot"),
        reference.get("ego_lwh"),
    )
    # Written onto EVERY completion, not just the first: Cosmos-RL averages each
    # key over the flat list of rollouts and treats a missing key as 0, so a
    # value present on one rollout per group comes out diluted by 1/K rather
    # than averaged over groups. See rewards/val_metrics.py for the full note.
    for reward_dict in reward_dicts:
        reward_dict.update(group_metrics)

    return [d["reward"] for d in reward_dicts], reward_dicts
