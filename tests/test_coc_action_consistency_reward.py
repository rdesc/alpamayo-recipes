# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Tests for the CoC<->action consistency reward (rewards/coc_action_consistency_*.py).

Two tiers, matching how this repo's CPU CI lane works (`.github/workflows/ci.yml` runs
`pytest tests` with only `pytest`+`pyyaml` installed -- no torch, no `alpamayo1_x_rl`):

  - Matcher-only tests load ``coc_action_consistency_matcher.py`` directly by file path
    (``importlib.util``, same pattern as ``test_pai_reasoning_compatibility.py``), since
    that module has zero external dependencies. These run unconditionally, including in
    the lightweight CPU CI job, and are the ones that matter most: they pin the ported
    matching logic against a frozen sample of real output from
    ``alpamayo-coc-autolabeler``'s own validated scorer
    (E3(P3) x M5-abstain, `tests/fixtures/cac_gold_sample.json`), which is the offline
    validation section 8 of the original design proposal calls for, applied to this port.

  - Trajectory-classifier and full end-to-end tests need torch + the installed
    `alpamayo1_x_rl` package (they exercise `coc_action_consistency_trajectory.py` and
    `coc_action_consistency_reward.py`, both of which import real project code, not just
    stdlib). These `pytest.importorskip`, so they skip cleanly in the CPU CI lane and run
    fully wherever a real training env (e.g. the `a1x_rl`/`a1x_rl_b300` venvs under
    `recipes/alpamayo1_x_rl/`) is used to invoke pytest.

`tests/fixtures/cac_gold_sample.json` provenance: 150 events sampled (seed=7) from the
2077-event PAI-AV OOD gold corpus used throughout `alpamayo-coc-autolabeler`'s scorer
design docs, scored there with the real E3 (Qwen3-VL-8B, P3 prompt) extractor and the
real 10 Hz meta-action timelines, reduced to the ``(claims, traj_state)`` pair that is
this port's actual interface boundary -- no timeline files or trajdata needed here.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REWARDS_DIR = ROOT / "recipes" / "alpamayo1_x_rl" / "rewards"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "cac_gold_sample.json"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def matcher():
    return _load_module(
        "coc_action_consistency_matcher", REWARDS_DIR / "coc_action_consistency_matcher.py"
    )


@pytest.fixture(scope="module")
def gold_sample():
    with FIXTURE.open() as fh:
        return json.load(fh)


def test_matcher_selftest_passes(matcher):
    assert matcher._selftest() is True


def test_matcher_parity_with_gold_corpus(matcher, gold_sample):
    """The ported `score_claims` must reproduce alpamayo-coc-autolabeler's own
    `score_event_abstain` output exactly, event by event, on a real sample of its gold
    corpus. Any mismatch here means the port has drifted from the source repo's matcher.
    """
    assert len(gold_sample) >= 100, "fixture should carry a decent-sized sample"

    mismatches = []
    for row in gold_sample:
        result = matcher.score_claims(row["claims"], row["traj_state"])
        if (
            result["binary"] != row["expected_binary"]
            or result["event_abstained"] != row["expected_event_abstained"]
            or result["n_scored"] != row["expected_n_scored"]
        ):
            mismatches.append(
                {
                    "clip_id": row["clip_id"],
                    "coc": row["coc"],
                    "got": {
                        "binary": result["binary"],
                        "event_abstained": result["event_abstained"],
                        "n_scored": result["n_scored"],
                    },
                    "want": {
                        "binary": row["expected_binary"],
                        "event_abstained": row["expected_event_abstained"],
                        "n_scored": row["expected_n_scored"],
                    },
                }
            )

    assert not mismatches, f"{len(mismatches)}/{len(gold_sample)} events diverged: {mismatches[:5]}"


def test_matcher_sample_accuracy_matches_offline_headline(matcher, gold_sample):
    """Sanity check on the fixture itself: accuracy over the *scored* (non-abstained)
    sample should land near the offline corpus's 0.941 headline (E3(P3) x M5-abstain,
    all 2077 events) -- not exact, since this is a 150-event sample, but well within
    sampling noise. Guards against a fixture-generation bug producing something
    unrepresentative.
    """
    scored = [r for r in gold_sample if not r["expected_event_abstained"]]
    acc = sum(r["expected_binary"] for r in scored) / len(scored)
    assert acc > 0.85, f"sample accuracy {acc:.3f} is suspiciously far from the 0.941 offline headline"


def test_matcher_abstain_only_on_hedged_lateral(matcher):
    """M5-abstain must never abstain a claim that isn't a hedged lateral one -- this is
    the property the offline docs build the whole mechanism around (longitudinal hedge
    claims show no accuracy gap and must not be abstained).
    """
    ts = {"longitudinal": {"bucket": "slow_down", "token": "GentleDeceleration", "magnitude": "gentle"}}
    claim = {"axis": "longitudinal", "bucket": "slow_down", "magnitude": "gentle"}
    score, scored = matcher.match_axis_m5_abstain(claim, ts["longitudinal"])
    assert scored is True

    ts_lat = {"bucket": "left", "token": "SteerLeft", "magnitude": "gentle"}
    claim_lat = {"axis": "lateral", "bucket": "left", "magnitude": "gentle"}
    score, scored = matcher.match_axis_m5_abstain(claim_lat, ts_lat)
    assert scored is False


# --------------------------------------------------------------------- torch-gated tests
# NOTE: deliberately NOT `pytest.importorskip` at module level -- that raises during
# collection and skips the *whole file*, including the dependency-free matcher tests
# above. Import defensively instead and skip only the tests that actually need torch /
# the installed `alpamayo1_x_rl` package (absent in this repo's CPU CI lane).
try:
    import sys

    import torch  # noqa: F401

    sys.path.insert(0, str(ROOT / "recipes"))
    from alpamayo1_x_rl.rewards import coc_action_consistency_trajectory as traj_mod
    from alpamayo1_x_rl.rewards.coc_action_consistency_extract import extract_claims_regex
    from alpamayo1_x_rl.rewards.coc_action_consistency_reward import compute_component

    _TORCH_AND_ALPAMAYO_AVAILABLE = True
except ImportError:
    _TORCH_AND_ALPAMAYO_AVAILABLE = False

skip_without_torch = pytest.mark.skipif(
    not _TORCH_AND_ALPAMAYO_AVAILABLE,
    reason="needs torch + the installed alpamayo1_x_rl package (absent in the CPU CI lane)",
)


@skip_without_torch
def test_trajectory_classifier_selftest():
    assert traj_mod._selftest() is True


@skip_without_torch
def test_regex_extracts_slow_down_from_lead_vehicle_justification():
    """Found 2026-08-17 from real RL rollout text, not the gold corpus: "Keep distance to
    the lead vehicle since it is slowing ahead" must extract as slow_down, not steady --
    keeping distance behind a decelerating lead vehicle mechanically requires decelerating
    too, even though the deceleration is only stated in the (normally-discarded)
    justification clause. Measured at 6.1% of 6146 real training-run CoC lines -- still a
    clear majority over the phrasing already handled correctly ("Slow down to keep
    distance...", covered by the second case below). See _LEAD_DECEL_JUSTIFICATION's
    comment in coc_action_consistency_extract.py for the full reasoning and false-positive
    check.
    """
    upgraded = extract_claims_regex("Keep distance to the lead vehicle since it is slowing ahead.")
    assert upgraded == [{"axis": "longitudinal", "bucket": "slow_down", "magnitude": None}]

    already_correct = extract_claims_regex(
        "Slow down to keep distance to the lead vehicle since it is decelerating ahead."
    )
    assert any(c["axis"] == "longitudinal" and c["bucket"] == "slow_down" for c in already_correct)

    # Must NOT fire when the justification is unrelated to the lead vehicle's own motion --
    # this is the false-positive check the rule is deliberately scoped to avoid.
    unrelated_justification = extract_claims_regex(
        "Keep distance to the lead vehicle while navigating through the construction zone "
        "marked by traffic cones."
    )
    assert unrelated_justification == [{"axis": "longitudinal", "bucket": "steady", "magnitude": None}]

    # Must NOT trigger on the bare adverb "slowly" (manner/pace, not a transition) --
    # deliberately narrower than a literal "slow_down"'s conditions.
    # "moving slowly" can mean a constant slow speed just as well as a gradual
    # deceleration; cross-checked live against Qwen3-VL-8B-Instruct, which reads this
    # exact phrasing as steady on its own (no equivalent rule at all), so this regex was
    # narrowed to match that same distinction rather than over-trigger past it.
    ambiguous_pace = extract_claims_regex(
        "Keep distance to the lead vehicle since it is moving slowly ahead."
    )
    assert ambiguous_pace == [{"axis": "longitudinal", "bucket": "steady", "magnitude": None}]


@skip_without_torch
def test_compute_component_end_to_end():
    """Small integration test: synthetic strong-braking trajectory scored against
    matching / mismatching / hedged / empty CoC text, through the full
    extract -> classify -> match pipeline (regex fallback extractor, no GPU).
    """
    n, freq = 40, 10.0
    dt = 1.0 / freq
    t = torch.arange(n, dtype=torch.float32) * dt
    v = torch.clamp(15.0 - 3.0 * t, min=0.0)  # 15 -> 0 m/s, strong braking
    x = torch.cumsum(v * dt, dim=0)
    xyz = torch.stack([x, torch.zeros(n), torch.zeros(n)], dim=-1)
    rot = torch.eye(3).unsqueeze(0).repeat(n, 1, 1)

    matching = compute_component("Strong deceleration to yield to the pedestrian.", xyz, rot)
    assert matching["binary"] == 1
    assert matching["reward_contribution"] == pytest.approx(1.0)

    mismatching = compute_component("Accelerate to overtake the truck ahead.", xyz, rot)
    assert mismatching["binary"] == 0
    assert mismatching["reward_contribution"] == pytest.approx(0.0)

    hedged_lateral = compute_component("Slightly shift left following the cones.", xyz, rot)
    assert hedged_lateral["abstained"] is True
    assert hedged_lateral["reward_contribution"] == pytest.approx(0.0)

    empty = compute_component("", xyz, rot)
    assert empty["parsed"] is False
    assert empty["abstained"] is False
    assert empty["reward_contribution"] == pytest.approx(0.0)


@skip_without_torch
def test_reverse_claim_reads_trajectory_at_offset(monkeypatch):
    """compute_component must read the trajectory at `REVERSE_OFFSET_S` specifically for
    a longitudinal `reverse` claim, and at the default offset (0.0) for every other claim
    -- validated on the gold corpus (7/7 vs 5/7 correct at 1.0s vs 0.0s) because reverse
    maneuvers there often start from a brief stop, so the backward-motion signal is
    genuinely absent at t0, not just noisy.

    Verified by spying on `trajectory_state_from_pred`'s `offset_s` argument rather than
    by engineering a synthetic trajectory to straddle the exact validated timing: a clean
    step-function pause-then-reverse trajectory turns out to already read `Reverse` at
    offset=0 too (a short-enough pause gets merged into the following segment by the
    same min-length-by-caption merge that makes the classifier accurate in the first
    place -- see coc_action_consistency_trajectory.py). The offset exception matters on
    the real corpus's longer/noisier pauses; this test pins the WIRING, not that specific
    numeric edge case.
    """
    import alpamayo1_x_rl.rewards.coc_action_consistency_reward as reward_mod

    calls: list[float] = []
    real_fn = reward_mod.trajectory_state_from_pred

    def spy(pred_xyz, pred_rot, *, offset_s=0.0):
        calls.append(offset_s)
        return real_fn(pred_xyz, pred_rot, offset_s=offset_s)

    monkeypatch.setattr(reward_mod, "trajectory_state_from_pred", spy)

    n, freq = 61, 10.0  # 6s @ 10Hz, matching a realistic predicted-future length
    dt = 1.0 / freq
    xyz = torch.zeros(n, 3)
    xyz[:, 0] = torch.arange(n, dtype=torch.float32) * dt * 10.0  # steady 10 m/s forward
    rot = torch.eye(3).unsqueeze(0).repeat(n, 1, 1)

    calls.clear()
    compute_component("Maintain speed in the current lane.", xyz, rot)
    assert calls == [0.0], f"non-reverse claim should read the trajectory once, at offset=0.0 (got {calls})"

    calls.clear()
    compute_component("Reverse to back out of the parking space.", xyz, rot)
    assert calls == [0.0, reward_mod.REVERSE_OFFSET_S], (
        f"reverse claim should ALSO read the trajectory at REVERSE_OFFSET_S (got {calls})"
    )


@skip_without_torch
def test_group_reward_batches_extraction_and_matches_single_item(monkeypatch):
    """`aggregated_reward_with_reasoning.compute_group_reward` (added 2026-08-17, wiring
    `[train.train_policy].group_reward_calculation = true` support -- see that module's
    docstring) must (a) call a GPU extractor's `extract_batch` exactly ONCE for a whole
    group instead of once per completion, and (b) produce byte-identical per-item
    rewards to calling `compute_reward` on each completion separately -- grouping must
    only change how extraction is batched, never the scored result.
    """
    import alpamayo1_x_rl.rewards.aggregated_reward_with_reasoning as arwr

    n, freq = 40, 10.0
    dt = 1.0 / freq
    t = torch.arange(n, dtype=torch.float32) * dt
    v = torch.clamp(15.0 - 3.0 * t, min=0.0)
    x = torch.cumsum(v * dt, dim=0)
    xyz = torch.stack([x, torch.zeros(n), torch.zeros(n)], dim=-1).unsqueeze(0)
    rot = torch.eye(3).unsqueeze(0).repeat(n, 1, 1).unsqueeze(0)

    texts = [
        "Strong deceleration to yield to the pedestrian.",
        "Accelerate to overtake the truck ahead.",
        "Maintain speed in the current lane.",
    ]

    class FakeConfig:
        custom = {
            "alpamayo": {
                "reward": {
                    "traj_l2_weight": 0.0,
                    "comfort_weight": 0.0,
                    "reasoning_weight": 0.0,
                    "coc_consistency_weight": 1.0,
                }
            }
        }

    reference = {
        "ego_future_xyz": xyz,
        "ego_history_xyz": None,
        "ego_history_rot": None,
        "cot": "",
        "idx": None,
    }

    calls = {"extract_batch": 0}

    class FakeBatchExtractor:
        def extract_batch(self, coc_texts):
            calls["extract_batch"] += 1
            return [extract_claims_regex(text) for text in coc_texts]

        def extract(self, coc_text):
            return self.extract_batch([coc_text])[0]

    monkeypatch.setattr(
        "alpamayo1_x_rl.utils.trajectory_decode.decode_rollout_trajectory",
        lambda to_be_evaluated, *a, **kw: (xyz, rot),
    )
    monkeypatch.setattr(
        "alpamayo_r1.models.token_utils.extract_between_special_tokens",
        lambda lst, token: list(lst),
    )
    monkeypatch.setattr(
        "alpamayo1_x_rl.rewards.coc_action_consistency_extract.get_claim_extractor_from_config",
        lambda config: FakeBatchExtractor(),
    )

    single_results = [
        arwr.compute_reward(
            text, reference, tokenizer=None, traj_tokenizer=None, config=FakeConfig(), model_config=None
        )
        for text in texts
    ]
    calls["extract_batch"] = 0
    group_rewards, group_dicts = arwr.compute_group_reward(
        texts, reference, tokenizer=None, traj_tokenizer=None, config=FakeConfig(), model_config=None
    )

    assert calls["extract_batch"] == 1, (
        f"expected exactly one batched extract_batch call for the whole group, got {calls['extract_batch']}"
    )
    for i, (single_reward, single_dict) in enumerate(single_results):
        assert group_rewards[i] == single_reward, f"reward mismatch at index {i}"
        assert group_dicts[i]["coc_consistency"] == single_dict["coc_consistency"], (
            f"coc_consistency mismatch at index {i}"
        )
