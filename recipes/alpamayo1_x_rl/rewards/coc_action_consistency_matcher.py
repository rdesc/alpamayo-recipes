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

"""CoC <-> action consistency: deterministic matching stage ("M5-abstain").

This is a **verbatim port** of the matcher selected and empirically validated in
``alpamayo-coc-autolabeler`` (a sibling repo, not a runtime dependency of this one --
ported rather than imported because this repo does not cross-import sibling repos at
runtime). Source of truth for the design and the numbers below:

  - ``alpamayo-coc-autolabeler/docs/cac_scorer_design_results.md`` (design comparison,
    "Recommendation (decided)" section + the "Update -- M5-abstain" note at the top)
  - ``alpamayo-coc-autolabeler/docs/cac_failure_modes.md`` (failure taxonomy)
  - ``alpamayo-coc-autolabeler/scripts/coc_action_consistency_v2.py`` (packaged scorer,
    taxonomy + boundary-tolerant matching)
  - ``alpamayo-coc-autolabeler/scripts/cac_grid.py`` (``match_axis`` M5 branch,
    ``score_event``)
  - ``alpamayo-coc-autolabeler/scripts/cac_abstain_variant.py`` (``match_axis_abstain``,
    ``score_event_abstain`` -- the M5-abstain refinement, now the recommended matcher)

If the source repo's matcher changes, re-sync this file by hand; there is no shared
package. ``tests/test_coc_action_consistency_reward.py`` in this repo pins this port
against a frozen sample of the source repo's own gold-label outputs
(``tests/fixtures/cac_gold_sample.json``) so drift is caught by CI rather than assumed
away.

Selected offline numbers (PAI-AV OOD gold corpus, 2077 events; a perfect scorer would
score ~1.0 since the gold CoC is auto-labeled to agree with the trajectory):

    E3(P3, Qwen3-VL-8B, schema-only prompt) x M5-abstain: 0.941 gold agreement,
    0.931 separation from a direction-flipped hard negative, on 2034/2077 events
    (43 fully abstained -- every claim on those events was a hedged lateral claim).

**These numbers are for the matcher scored against `alpamayo-coc-autolabeler`'s own
10 Hz meta-action *timelines*, which are built by full clip segmentation over minutes
of driving.** In this repo, the trajectory side is instead a single short predicted
future window (see ``coc_action_consistency_trajectory.py``), which is a materially
different and *not yet separately validated* signal source. Only the matching logic
below -- claims + a reduced per-axis trajectory state -> score -- is validated as a
faithful port; see that module's docstring for the trajectory-side caveat.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

# --------------------------------------------------------------------------- taxonomy
# Coarse comparison grain (verbatim from coc_action_consistency_v2.py). Both the claim
# extractor and the trajectory-side classifier must emit buckets from these same sets.
LON_SPEED_UP, LON_SLOW_DOWN, LON_STEADY, LON_REVERSE = (
    "speed_up",
    "slow_down",
    "steady",
    "reverse",
)
LAT_LEFT, LAT_RIGHT, LAT_STRAIGHT = "left", "right", "straight"

# Fine autolabeler token -> coarse bucket. The trajectory-side classifier
# (coc_action_consistency_trajectory.py) must emit one of these token names.
LON_BUCKET: dict[str, str] = {
    "GentleAcceleration": LON_SPEED_UP,
    "StrongAcceleration": LON_SPEED_UP,
    "GentleDeceleration": LON_SLOW_DOWN,
    "StrongDeceleration": LON_SLOW_DOWN,
    "Stop": LON_SLOW_DOWN,
    "MaintainSpeed": LON_STEADY,
    "Reverse": LON_REVERSE,
}
LAT_BUCKET: dict[str, str] = {
    "GoStraight": LAT_STRAIGHT,
    "SteerLeft": LAT_LEFT,
    "SharpSteerLeft": LAT_LEFT,
    "ReverseLeft": LAT_LEFT,
    "SteerRight": LAT_RIGHT,
    "SharpSteerRight": LAT_RIGHT,
    "ReverseRight": LAT_RIGHT,
}
# Fine token -> declared magnitude, used only by the trajectory-side classifier to
# populate `ts["magnitude"]` (the matcher's boundary-tolerance gate reads this).
TOKEN_MAGNITUDE: dict[str, str] = {
    "GentleAcceleration": "gentle",
    "StrongAcceleration": "strong",
    "GentleDeceleration": "gentle",
    "StrongDeceleration": "strong",
    "SteerLeft": "gentle",
    "SteerRight": "gentle",
    "SharpSteerLeft": "strong",
    "SharpSteerRight": "strong",
}
GENTLE_TOKENS = {"GentleAcceleration", "GentleDeceleration", "SteerLeft", "SteerRight"}

_ADJACENT = ({LON_STEADY, LON_SLOW_DOWN}, {LON_STEADY, LON_SPEED_UP})

# Claims corrupted into every alternative bucket must never all pass -- see
# ``_selftest`` below. Kept here (not just in the offline repo) because a matcher
# that goes vacuous under refactor is exactly the failure mode this scorer exists to
# avoid, and it is a five-line check.
LON_ALTS = [LON_SPEED_UP, LON_SLOW_DOWN, LON_STEADY, LON_REVERSE]
LAT_ALTS = [LAT_LEFT, LAT_RIGHT, LAT_STRAIGHT]


class Claim(TypedDict, total=False):
    axis: str  # "longitudinal" | "lateral"
    bucket: str
    magnitude: Optional[str]  # "gentle" | "strong" | None


class AxisState(TypedDict, total=False):
    bucket: Optional[str]
    token: Optional[str]
    magnitude: Optional[str]


def match_axis_m5(claim: Claim, ts: AxisState) -> tuple[float, bool]:
    """Plain M5: boundary-tolerant from the gentle side only.

    Verbatim port of ``cac_grid.match_axis``'s "M5" branch. Returns ``(score, scored)``;
    ``scored=False`` means the trajectory side has no decision to check against (should
    not happen once ``coc_action_consistency_trajectory.py`` always emits a bucket, but
    kept for parity with the source and as a defensive guard).
    """
    if ts.get("bucket") is None:
        return 0.0, False
    if claim["bucket"] == ts["bucket"]:
        return 1.0, True
    gentle = ts.get("magnitude") == "gentle"
    if claim["axis"] == "longitudinal":
        adjacent = {claim["bucket"], ts["bucket"]} in _ADJACENT
    else:
        adjacent = LAT_STRAIGHT in (claim["bucket"], ts["bucket"])
    return (1.0 if (gentle and adjacent) else 0.0), True


def match_axis_m5_abstain(claim: Claim, ts: AxisState) -> tuple[float, bool]:
    """M5, plus: a LATERAL claim whose own extracted magnitude is "gentle" is not scored.

    Verbatim port of ``cac_abstain_variant.match_axis_abstain``. Motivated by a direct
    empirical finding on the gold corpus: hedge-qualified lateral claims ("steer
    *slightly* left") score 0.512 vs 0.952 for unhedged ones, while hedged longitudinal
    claims show no such gap (0.940 vs 0.942) -- so only the lateral axis abstains.
    This is the shipped/recommended matcher (supersedes plain M5 above, which is kept
    for reference and for the parity test).
    """
    if claim["axis"] == "lateral" and claim.get("magnitude") == "gentle":
        return 0.0, False
    return match_axis_m5(claim, ts)


class ScoreResult(TypedDict):
    binary: int
    graded: float
    parsed: bool
    n_scored: int
    event_abstained: bool


def score_claims(
    claims: list[Claim], traj_state: dict[str, AxisState]
) -> ScoreResult:
    """Score every claim against the trajectory state on its own axis.

    Verbatim port of ``cac_abstain_variant.score_event_abstain``, generalized to take a
    pre-computed per-axis trajectory state directly (this repo has no meta-action
    *timeline* to reduce -- see ``coc_action_consistency_trajectory.py``).

    Three distinct "nothing to score" cases, per the source repo's convention:
      - No claims extracted at all (`claims` is empty): real failure, Sec. 5.3.2's
        "unparseable reasoning" rule. `binary=0`, `event_abstained=False`.
      - Claims were extracted but every one was withdrawn by the abstain rule (hedged
        lateral claims only): NOT a failure -- `event_abstained=True`. Caller decides
        how to fold this into a reward (see ``coc_action_consistency_reward.py`` for
        the RL-specific policy and why it differs from the offline corpus-accuracy
        convention of simply excluding these from the denominator).
      - Claims were extracted and scored, multi-claim best-match-per-axis (a compound
        sentence is not penalized for the clause the trajectory isn't showing).
    """
    per_axis: dict[str, float] = {}
    for c in claims:
        s, scored = match_axis_m5_abstain(c, traj_state.get(c["axis"], {}))
        if not scored:
            continue
        per_axis[c["axis"]] = max(per_axis.get(c["axis"], 0.0), s)

    if not claims:
        return {
            "binary": 0,
            "graded": 0.0,
            "parsed": False,
            "n_scored": 0,
            "event_abstained": False,
        }
    if not per_axis:
        return {
            "binary": 0,
            "graded": float("nan"),
            "parsed": True,
            "n_scored": 0,
            "event_abstained": True,
        }
    graded = sum(per_axis.values()) / len(per_axis)
    return {
        "binary": int(all(v >= 1.0 for v in per_axis.values())),
        "graded": float(graded),
        "parsed": True,
        "n_scored": len(per_axis),
        "event_abstained": False,
    }


# --------------------------------------------------------------------------- self-test
def _selftest() -> bool:
    """Sanity checks independent of any fixture: run with `python -m ...matcher`."""
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        ok &= cond
        print(f"  [{'OK ' if cond else 'BAD'}] {name}")

    steady_vs_gentle_decel: dict[str, AxisState] = {
        "longitudinal": {"bucket": LON_SLOW_DOWN, "token": "GentleDeceleration", "magnitude": "gentle"},
        "lateral": {"bucket": LAT_STRAIGHT, "token": "GoStraight", "magnitude": None},
    }
    steady_vs_strong_decel: dict[str, AxisState] = {
        "longitudinal": {"bucket": LON_SLOW_DOWN, "token": "StrongDeceleration", "magnitude": "strong"},
        "lateral": {"bucket": LAT_STRAIGHT, "token": "GoStraight", "magnitude": None},
    }

    r = score_claims(
        [{"axis": "longitudinal", "bucket": LON_STEADY, "magnitude": None}],
        steady_vs_gentle_decel,
    )
    check("steady vs GentleDeceleration -- boundary-tolerant, gentle side", r["binary"] == 1)

    r = score_claims(
        [{"axis": "longitudinal", "bucket": LON_STEADY, "magnitude": None}],
        steady_vs_strong_decel,
    )
    check("steady vs StrongDeceleration -- never tolerated", r["binary"] == 0)

    r = score_claims([], steady_vs_gentle_decel)
    check("no claims extracted -> binary 0, not abstained", r["binary"] == 0 and not r["event_abstained"])

    hedge_lat: dict[str, AxisState] = {
        "longitudinal": {"bucket": None, "token": None, "magnitude": None},
        "lateral": {"bucket": LAT_STRAIGHT, "token": "GoStraight", "magnitude": None},
    }
    r = score_claims(
        [{"axis": "lateral", "bucket": LAT_LEFT, "magnitude": "gentle"}],
        hedge_lat,
    )
    check(
        "hedged lateral claim ('slightly left') abstains rather than failing",
        r["event_abstained"] and r["n_scored"] == 0,
    )

    r = score_claims(
        [{"axis": "longitudinal", "bucket": LON_SLOW_DOWN, "magnitude": "gentle"}],
        {"longitudinal": {"bucket": LON_SLOW_DOWN, "token": "GentleDeceleration", "magnitude": "gentle"}},
    )
    check(
        "hedged LONGITUDINAL claim is NOT abstained (no accuracy gap on that axis)",
        not r["event_abstained"] and r["binary"] == 1,
    )

    # Vacuity guard: no claim should pass against every possible trajectory state.
    for ts in (steady_vs_gentle_decel, steady_vs_strong_decel):
        for axis, alts in (("lateral", LAT_ALTS), ("longitudinal", LON_ALTS)):
            scores = [
                match_axis_m5({"axis": axis, "bucket": a, "magnitude": None}, ts[axis])[0]
                for a in alts
            ]
            check(f"not vacuous: not every {axis} claim passes", not all(s >= 1.0 for s in scores))

    print(f"\nSelf-test {'PASSED' if ok else 'FAILED'}.")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if _selftest() else 1)
