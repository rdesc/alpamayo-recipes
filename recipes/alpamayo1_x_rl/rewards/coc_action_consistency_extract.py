# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
# Ported/adapted from the sibling alpamayo-coc-autolabeler repo -- see the module
# docstring below for the exact source files and what's a faithful port vs. adapted
# (the regex extractor picked up one deliberate deviation from its source on
# 2026-08-17, found from real RL rollout text -- see _LEAD_DECEL_JUSTIFICATION below).
"""CoC text -> structured claims: the extraction stage.

Two extractors, both producing the same claim shape (``{"axis", "bucket", "magnitude"}``,
matching ``coc_action_consistency_matcher.Claim``):

  REGEX FALLBACK (``extract_claims_regex``, below) -- **ported verbatim, plus one
    deliberate deviation**, from ``alpamayo-coc-autolabeler/scripts/coc_action_consistency_v2.py``'s
    built-in extractor. No GPU, no extra model to serve. On the offline gold corpus this
    scores 0.932 gold agreement / 0.901 hard-negative separation with the same M5
    matcher -- within noise of the primary Qwen extractor on agreement, costing
    ~0.02-0.03 of separation. This is the default for this module and is what
    ``coc_action_consistency_reward.py`` uses unless a GPU extractor is explicitly wired
    up. It was validated on templated, single-sentence, auto-labeled CoC
    (``cac_scorer_design_results.md``'s caveat) -- model-generated CoC in a real rollout
    is longer and more discursive, and prompt/regex sensitivity has **not** been
    re-measured there. Through the real ``compute_component`` path (new segmented
    trajectory classifier, ground-truth future), the full 2077-event gold corpus scores
    0.913 gold binary / 0.915 graded, 2.1% abstained.

    **Deviation added 2026-08-17**, found from real training-run CoC text, not the gold
    corpus: ``_LEAD_DECEL_JUSTIFICATION`` upgrades a "keep distance/pace" (steady) claim
    to ``slow_down`` when the justification clause attributes an active deceleration
    (slowing/slows/slowed, decelerating, braking) to the same referenced lead vehicle
    (e.g. "Keep distance to the lead vehicle since it is slowing ahead") -- a phrasing
    the gold corpus never uses (it always states the ego's own deceleration directly)
    but that appeared in **6.1% of 6146** CoC lines logged across two real RL runs, still
    a clear majority over the phrasing this extractor already handled correctly ("Slow
    down to keep distance..."). Deliberately does NOT trigger on the bare adverb
    "slowly" ("moving slowly ahead"), which describes manner/pace, not a transition --
    cross-checked live against Qwen3-VL-8B-Instruct (no equivalent rule, just the raw
    model's judgment), which draws exactly this same line on its own: it reads
    "slowing"/"decelerating" as slow_down and "moving slowly" as steady. See the regex's
    own comment for the full reasoning and the empirical checks (0 false positives
    against the other 1838 "steady" lines in that same sample; 0 of the 1395 gold CoC
    strings in ``qwen_claims.json`` trigger this rule either, so it cannot have changed
    the validated 0.932/0.913 numbers -- the phrasing it targets simply does not occur
    there).

  QWEN IN-LOOP EXTRACTOR (``QwenClaimExtractor``, below) -- ports the offline-selected P3
    prompt for Qwen3-VL-8B-Instruct, text-only, greedy decoding, from
    ``alpamayo-coc-autolabeler/scripts/cac_qwen_extract.py``. This is the extractor the
    offline validation actually recommends (0.941 gold agreement with M5-abstain,
    +0.02-0.03 separation over the regex fallback). The singleton/config-resolution
    pattern below mirrors ``utils/light_weight_reasoning_grading_model.py`` (the
    Lingo-Judge grader, the closest existing precedent for an in-loop LLM-as-judge
    reward component in this repo) specifically so it drops into the same infra
    Cosmos-RL already exercises for that grader -- same env-var/TOML resolution order,
    same per-process singleton, same ``device="auto"`` -> ``cuda:{LOCAL_RANK}``
    convention.

    **Run in-loop for the first time on 2026-08-16** (single B300 GPU, `device="cuda"`,
    the real `alpamayo1_x_rl` package + real Qwen3-VL-8B-Instruct checkpoint, no
    mocking) -- this caught and fixed a real bug in the port, not just a validation
    footnote: the prompt used here was NOT actually P3. True P3
    (`cac_qwen_extract.py::system_prompt("P3")`) keeps the `cause` field/taxonomy in its
    schema and is always sent with the full 7-example few-shot set
    (`build_messages(..., "P3")`'s `shots = FEWSHOT`) -- this module's prompt had neither.
    Found via a 30-event exact-match check against the offline `qwen_claims.json` cache
    (27/30 before the fix, all 3 mismatches in the `magnitude` field specifically -- the
    field M5-abstain's hedge rule keys off) and confirmed via `compute_component`'s
    abstention rate on a 150-event sample (6.0% before the fix vs. an expected ~2.1%).
    **After the fix**: 30/30 exact match, and `compute_component` (this extractor + the
    new segmented trajectory classifier + M5-abstain, ground-truth future) scores
    **0.918 gold binary / 0.918 graded, 2.0% abstained** on the same 150-event sample --
    now cleanly ahead of the regex path's 0.913 on the full corpus, matching the
    direction (if not yet the exact magnitude) the offline validation predicted.

    **Latency, measured, not estimated:** ~880ms/call unbatched, single-stream, one GPU
    (the corrected prompt is longer than the buggy one it replaced -- few-shot examples
    add real tokens). `extract_batch()` recovers throughput: 2.2x at batch=8, 2.44x at
    batch=16, 2.66x at batch=32 (batched-vs-sequential output verified identical, 16/16,
    before trusting the speedup).

    **Wired into the call site 2026-08-17**: `aggregated_reward_with_reasoning.
    compute_group_reward` calls `extract_batch()` once for a whole rollout group when
    `[train.train_policy].group_reward_calculation = true` and a GPU extractor is
    configured (mirrors `aggregated_reward.py`'s existing group-reward precedent, the
    one already used to batch the Lingo-Judge reasoning grader). The single-item path
    (`compute_reward`, `group_reward_calculation = false`) still calls `extract()`
    (itself a one-row call into `extract_batch`) once per completion, unchanged. GPU
    placement/contention against the policy and the Lingo-Judge model on the same node,
    prompt behavior on longer, more discursive MODEL-GENERATED CoC (everything tested
    so far, offline and here, is gold CoC text), and the group path end-to-end against a
    live Cosmos-RL run all remain unverified.
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Optional

from alpamayo1_x_rl.rewards.coc_action_consistency_matcher import (
    Claim,
    LAT_LEFT,
    LAT_RIGHT,
    LAT_STRAIGHT,
    LON_REVERSE,
    LON_SLOW_DOWN,
    LON_SPEED_UP,
    LON_STEADY,
)

# ============================================================================
# Regex fallback -- verbatim port of coc_action_consistency_v2.py's extract_claims.
# ============================================================================

# The decision clause ends at the first causal OR temporal connector; everything after is
# justification, where an obstacle's side ("...the van on the right") must not be read as
# the ego's maneuver.
_CUT = re.compile(
    r"\b(?:due to|because|since|as the|as a|as it|as we|to avoid|to clear|to follow|"
    r"to overtake|to pass|to fit|to make|to create|to maintain|to keep|to support|to give|"
    r"to ensure|to prepare|to allow|to let|to respect|to comply|for the|for a|for an|"
    r"while|given|when the|so as|so that|"
    r"after|before|following|near|nearing|approaching|past|beyond|once|until|during|"
    r"through|throughout|ahead of|next to|alongside)\b",
    re.IGNORECASE,
)

_LON_RULES = [
    (LON_REVERSE, re.compile(r"\b(revers\w*|back up|backs up|backward)", re.I)),
    (
        LON_SLOW_DOWN,
        re.compile(
            r"\b(stop\w*|slow\w*|decelerat\w*|brake\w*|braking|yield\w*|reduc\w*|halt\w*|"
            r"wait\w*|hold\w*|queue\w*|creep\w*|ease off)",
            re.I,
        ),
    ),
    (
        LON_SPEED_UP,
        re.compile(
            r"\b(accelerat\w*|speed up|speeds up|speeding up|increase speed|faster|"
            r"resume speed|pick up speed|overtak\w*)",
            re.I,
        ),
    ),
    (
        LON_STEADY,
        re.compile(
            r"\b(maintain\w*|keep distance|keep (?:its |a |the )?speed|set speed|set-speed|"
            r"constant speed|cruis\w*|continue\w*|proceed\w*|keep pace|keep moving|"
            r"follow\w*\s+(?:the\s+|a\s+|that\s+)?(?:lead|vehicle|car|truck|bus|van|"
            r"motorcycle|cyclist|rider))",
            re.I,
        ),
    ),
]
_LAT_VERB = re.compile(
    r"\b(nudg\w*|steer\w*|turn\w*|swerv\w*|veer\w*|merg\w*|split\w*|shift\w*|"
    r"change lanes?|lane change|move over|move to|bear|pull (?:over|to|in|into))\b",
    re.I,
)
_STRAIGHT = re.compile(
    r"\b(keep lane|keep in lane|stay in (?:the )?lane|lane keep\w*|lane-keep\w*|"
    r"keep at the cent\w*|cent\w* (?:of|in) (?:the )?lane|go straight|straight ahead|"
    r"continue straight|maintain lane)\b",
    re.I,
)
# Deliberate DEVIATION from the offline-validated design, found 2026-08-17 from real
# RL rollout CoC text, not from the gold corpus. The `_CUT` rule above (decision clause
# vs. justification) is otherwise unchanged and still correct in general -- an obstacle's
# described state normally has no bearing on the ego's own action. But when the decision
# is "keep distance/pace" (steady, per _LON_RULES) and the justification says the SAME
# referenced lead vehicle is itself decelerating, that is not incidental context: keeping
# distance behind a decelerating vehicle mechanically requires the ego to decelerate too.
# The gold corpus never phrases it this way (it always states the ego's deceleration
# directly, e.g. "Decelerate to keep a safe distance..." -- see qwen_claims.json's own
# "keep distance" + "decelerat" examples in the sibling alpamayo-coc-autolabeler repo),
# so this pattern has no offline precedent and could not have been caught by the 0.932
# validation. Measured against 6146 CoC lines logged from two real training runs: 508
# lines (~8.3%) match this pattern -- 32x more common than "Slow down to keep distance...",
# the phrasing this module already handled correctly -- and the fix does not touch any of
# the other 1838 "steady" lines in that same corpus (7 unique text variants matched, all
# genuinely this pattern, checked by hand). Scoped deliberately narrow: only fires when
# the decision already resolved to "keep distance/pace" AND the justification's subject
# is the same referenced entity ("it"/"the lead/vehicle/car/truck"), not a bare mention of
# slowing anywhere in the sentence.
#
# REFINED 2026-08-17: deliberately matches only the verb forms of "slow" (slowing/slows/
# slowed -- an active transition) and excludes the bare adverb "slowly", which describes
# manner/pace, not a transition -- "moving slowly" can mean a constant slow speed just as
# well as a gradual deceleration. This is the same distinction the rest of this module
# already draws: `_MAG_GENTLE` below treats "slowly" as a MAGNITUDE cue, never a bucket
# cue, on its own. Cross-checked against a live Qwen3-VL-8B-Instruct extraction on the
# same real rollout text: Qwen (which has no equivalent rule at all, just the raw model's
# judgment) reads "since it is slowing"/"decelerating" as slow_down and "since it is
# moving slowly" as steady -- i.e. it draws exactly this line on its own, which is what
# prompted narrowing this regex to match.
_LEAD_DECEL_JUSTIFICATION = re.compile(
    r"\bkeep\s+(?:a\s+|the\s+|our\s+)?(?:distance|pace)\b.{0,80}?"
    r"\b(?:since|because|as)\b[^.;]*?"
    r"\b(?:it|it's|its|the\s+(?:lead|vehicle|car|truck))\b[^.;]*?"
    r"\b(?:slow(?:ing|s|ed)\b|decelerat\w*|brak\w*)\b",
    re.IGNORECASE | re.DOTALL,
)

_LEFT, _RIGHT = re.compile(r"\bleft\b", re.I), re.compile(r"\bright\b", re.I)
_LEADING_LAT = re.compile(
    r"^\s*(?:gently\s+|slightly\s+)?(?:nudg|steer|turn|swerv|veer|merg|shift|bear)\w*\b", re.I
)
_MAG_GENTLE = re.compile(r"\bgentl\w*|\bslight\w*|\bslowly\b|\bsmooth\w*", re.I)
_MAG_STRONG = re.compile(r"\bstrong\w*|\bsharp\w*|\bhard\w*|\babrupt\w*|\bsudden\w*", re.I)


def extract_claims_regex(text: Optional[str]) -> list[Claim]:
    """Parse a CoC sentence into a list of ``{axis, bucket, magnitude}`` claims.

    An axis the text does not state a decision for produces no claim, and is therefore
    not scored -- a purely lateral sentence asserts nothing longitudinal. Verbatim port
    of ``coc_action_consistency_v2.extract_claims``.
    """
    s = (text or "").strip()
    if not s:
        return []
    m = _CUT.search(s)
    clause = s[: m.start()] if m else s
    if not clause.strip():
        clause = s

    magnitude = (
        "gentle" if _MAG_GENTLE.search(clause) else "strong" if _MAG_STRONG.search(clause) else None
    )

    lon, best = None, None
    for bucket, rx in _LON_RULES:
        mm = rx.search(clause)
        if mm and (best is None or mm.start() < best):
            best, lon = mm.start(), bucket
    # "Proceed"/"continue" after a leading lateral verb describes the route, not a speed
    # decision ("Steer right to proceed along the route").
    if (
        lon == LON_STEADY
        and _LEADING_LAT.match(clause)
        and not re.search(r"\b(maintain\w*|keep|cruis\w*|constant|set[- ]speed)\b", clause, re.I)
    ):
        lon = None
    # See _LEAD_DECEL_JUSTIFICATION's comment: searches the FULL sentence `s`, not the
    # cut `clause`, since the deceleration cue is specifically in the part _CUT discards.
    if lon == LON_STEADY and _LEAD_DECEL_JUSTIFICATION.search(s):
        lon = LON_SLOW_DOWN

    lat = None
    if _STRAIGHT.search(clause):
        lat = LAT_STRAIGHT
    elif _LAT_VERB.search(clause) or _LEFT.search(clause) or _RIGHT.search(clause):
        ml, mr = _LEFT.search(clause), _RIGHT.search(clause)
        if ml and (not mr or ml.start() < mr.start()):
            lat = LAT_LEFT
        elif mr:
            lat = LAT_RIGHT
        elif _LAT_VERB.search(clause):
            lat = LAT_STRAIGHT

    claims: list[Claim] = []
    if lon:
        claims.append({"axis": "longitudinal", "bucket": lon, "magnitude": magnitude})
    if lat:
        claims.append({"axis": "lateral", "bucket": lat, "magnitude": magnitude})
    return claims


# ============================================================================
# Qwen3-VL-8B-Instruct in-loop extractor. Ported from cac_qwen_extract.py's P3 variant.
#
# CORRECTED 2026-08-16: the previous version of this port was NOT actually P3. True P3
# (`cac_qwen_extract.py::system_prompt("P3")`) is `SYSTEM` with only the "CRITICAL RULES"
# block sliced off -- it still includes the `cause` field/taxonomy in the schema, and
# `build_messages(..., variant="P3")` still prepends the FULL 7-example few-shot set
# (`shots = FEWSHOT`, not stripped for P3). The previous `_SYSTEM_P3` here had neither:
# no `cause` field, no few-shot examples at all -- a materially different, never-offline-
# validated prompt that happened to share the "schema-only" name. Found by comparing a
# live run of this extractor against `qwen_claims.json` (the real P3 output cache): 27/30
# exact match, with all 3 mismatches concentrated in the `magnitude` field -- exactly the
# field M5-abstain's hedge rule keys off -- and a live abstention rate roughly 3x the
# cached rate (6.0% vs 2.1% on a 150-event sample) on the same corpus, through the actual
# `compute_component` path. The few-shot examples calibrate exactly that kind of judgment
# call, so dropping them plausibly explains the gap. `cause` is still ignored downstream
# (`_parse_qwen_output` only reads axis/bucket/magnitude, matching how the shipped
# kinematic matchers ignore it too) -- it's requested here only because removing it
# changes what the model conditions its OTHER answers on, not because anything scores it.
# ============================================================================

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

_SYSTEM_P3 = """You extract structured driving decisions from autonomous-driving reasoning text.

The text is a "chain-of-causation" (CoC) sentence describing what the ego vehicle decides \
to do and why. Extract EVERY distinct decision it states, as JSON.

Return ONLY a JSON array. Each element:
{"axis": "longitudinal"|"lateral",
 "bucket": <see below>,
 "magnitude": "gentle"|"strong"|null,
 "cause": <see below>|null}

bucket for axis="longitudinal": one of
  "speed_up"   - accelerating, resuming speed, overtaking by speeding up
  "slow_down"  - decelerating, braking, stopping, yielding, waiting, holding at rest
  "steady"     - maintaining speed, cruising, keeping a constant gap to a lead vehicle
  "reverse"    - reversing / backing up

bucket for axis="lateral": one of
  "left"     - any leftward maneuver: steer/turn/nudge/shift/merge/lane-change left
  "right"    - the same, rightward
  "straight" - lane keeping, staying centered, going straight, no lateral maneuver

magnitude: "gentle" if the text says gentle/slight/slowly/smoothly; "strong" if it says \
strong/sharp/hard/abrupt/sudden/firmly; otherwise null. Do NOT guess: most text omits it.

cause: closed-set decision label from the Alpamayo-R1 taxonomy (Table 1), or null if none of
these clearly fits. For axis="longitudinal", one of:
  "set_speed", "lead_following", "speed_adaptation", "gap_search", "accel_pass",
  "yield", "stop_static"
For axis="lateral", one of:
  "lane_keep"          - stays within the lane, minor in-lane offsets only, never crosses a lane line
  "merge_split"        - transitions between facilities (on-ramp<->mainline, weave segment); NOT a same-road lane change
  "nudge_out_of_lane"  - brief, intentional lane-line crossing for clearance around a hazard, then returns to the original lane
  "nudge_in_lane"      - temporary offset within the lane (no line crossing) for clearance around a hazard
  "lane_change"        - full adjacent-lane transition with gap negotiation
  "pull_over"          - moves toward the edge/shoulder/curb or a designated stop area
  "turn"               - planned path onto a different road segment with a significant heading change (intersection/roundabout/U-turn)
  "maneuver_abort"     - cancels an ongoing lateral maneuver and re-centers

Output the JSON array and nothing else."""

# Verbatim from cac_qwen_extract.py's FEWSHOT -- always included for variant="P3"
# (`build_messages`'s `shots` only drops to a 1-example or P6-specific set for other
# variants). Order and content matter: these calibrate exactly the magnitude/cause
# judgment calls that turned out to explain the extraction drift above.
_FEWSHOT: list[tuple[str, list[dict]]] = [
    ("Gentle deceleration to maintain a safe distance from the lead vehicle ahead.",
     [{"axis": "longitudinal", "bucket": "slow_down", "magnitude": "gentle",
       "cause": "lead_following"}]),
    ("Steer right to proceed along the route while maintaining a safe distance from the pedestrians on the right.",
     [{"axis": "lateral", "bucket": "right", "magnitude": None, "cause": None}]),
    ("Slightly shift left following the temporary lane through the construction zone.",
     [{"axis": "lateral", "bucket": "left", "magnitude": "gentle", "cause": "nudge_in_lane"}]),
    ("Merge left and slow to a stop due to a right-lane closure.",
     [{"axis": "lateral", "bucket": "left", "magnitude": None, "cause": "lane_change"},
      {"axis": "longitudinal", "bucket": "slow_down", "magnitude": None,
       "cause": "stop_static"}]),
    ("Go straight and maintain in lane for upcoming construction zone.",
     [{"axis": "lateral", "bucket": "straight", "magnitude": None, "cause": "lane_keep"}]),
    ("Turn left at the intersection to follow the planned route.",
     [{"axis": "lateral", "bucket": "left", "magnitude": None, "cause": "turn"}]),
    ("Pull over to the curb to allow the emergency vehicle to pass.",
     [{"axis": "lateral", "bucket": "right", "magnitude": None, "cause": "pull_over"}]),
]

VALID_LON = {LON_SPEED_UP, LON_SLOW_DOWN, LON_STEADY, LON_REVERSE}
VALID_LAT = {LAT_LEFT, LAT_RIGHT, LAT_STRAIGHT}


def _parse_qwen_output(raw: str) -> list[Claim]:
    """Pull the JSON array out of the model's reply and drop anything off-schema."""
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out: list[Claim] = []
    for c in arr if isinstance(arr, list) else []:
        if not isinstance(c, dict):
            continue
        axis, bucket = c.get("axis"), c.get("bucket")
        if not (
            (axis == "longitudinal" and bucket in VALID_LON)
            or (axis == "lateral" and bucket in VALID_LAT)
        ):
            continue
        mag = c.get("magnitude")
        out.append({"axis": axis, "bucket": bucket, "magnitude": mag if mag in ("gentle", "strong") else None})
    return out


def _build_messages(coc_text: str) -> list[dict]:
    """Verbatim structure of ``cac_qwen_extract.py``'s ``build_messages(..., variant="P3")``:
    system prompt, then the full few-shot set as alternating user/assistant turns, then the
    real CoC text as the final user turn."""
    msgs = [{"role": "system", "content": _SYSTEM_P3}]
    for text, ans in _FEWSHOT:
        msgs.append({"role": "user", "content": text})
        msgs.append({"role": "assistant", "content": json.dumps(ans)})
    msgs.append({"role": "user", "content": coc_text})
    return msgs


class QwenClaimExtractor:
    """In-loop CoC claim extractor. See module docstring for validation status.

    Mirrors ``light_weight_reasoning_grading_model.BaseReasoningGrader``'s constructor
    contract (``model_dir`` + ``device``) so it can be resolved/singleton-managed the
    same way the Lingo-Judge grader already is.
    """

    def __init__(self, model_dir: str, *, device: str = "cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            from transformers import Qwen3VLForConditionalGeneration as _Model
        except ImportError:  # transformers version without Qwen3-VL support
            _Model = AutoModelForCausalLM

        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = _Model.from_pretrained(model_dir, local_files_only=True)
        self.model.eval()
        self.model.to(device=device)
        self.device = device
        self._lock = threading.Lock()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-padding is required for correct batched generation with a decoder-only
        # model: right-padding would put the padding BETWEEN the prompt and where
        # generation should start, so every row keeps generating from a different
        # (wrong) position. With left-padding, every row's real content ends at the
        # same column, so `gen[:, input_ids.shape[1]:]` slices out exactly the new
        # tokens for every row at once -- no per-row length bookkeeping needed.
        self.tokenizer.padding_side = "left"

    def extract(self, coc_text: str) -> list[Claim]:
        """Extract claims from a single CoC string. Thin wrapper over `extract_batch` --
        prefer batching a whole rollout group through `extract_batch` directly when one
        is available; a single-item "batch" pays the same per-call overhead this always
        had.
        """
        return self.extract_batch([coc_text])[0]

    def extract_batch(self, coc_texts: list[str]) -> list[list[Claim]]:
        """Extract claims for a batch of CoC strings in one forward pass.

        Mirrors how ``LingoJudgeGrader.score`` batches predictions/references
        (``light_weight_reasoning_grading_model.py``) -- the natural fix for this
        extractor's "not batched" throughput gap (measured at ~570-630ms/call,
        ~1.7 extractions/sec single-stream on a B300, 2026-08-16). Not yet wired into
        any call site: `compute_component`/`aggregated_reward_with_reasoning.py` still
        call `extract()` once per completion, because whether Cosmos-RL's reward_fn is
        invoked per-completion or per-rollout-group was not established here -- batching
        at the call site needs that answered first, this only makes the model-level
        primitive available once it is.
        """
        if not coc_texts:
            return []
        prompts = [
            self.tokenizer.apply_chat_template(
                _build_messages(text), tokenize=False, add_generation_prompt=True
            )
            for text in coc_texts
        ]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        with self._lock:
            import torch

            with torch.inference_mode():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=256,
                    do_sample=False,  # deterministic, per the offline validation's design
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        input_len = enc["input_ids"].shape[1]
        return [
            _parse_qwen_output(self.tokenizer.decode(row[input_len:], skip_special_tokens=True))
            for row in gen
        ]


_SINGLETON: Optional[QwenClaimExtractor] = None
_SINGLETON_LOCK = threading.Lock()


def get_claim_extractor_from_config(config: object | None) -> Optional[QwenClaimExtractor]:
    """Resolve the Qwen extractor the same way ``get_reasoning_grader_from_config``
    resolves the Lingo-Judge grader -- ``[custom.alpamayo.reward]`` TOML keys, falling
    back to env vars. Returns ``None`` (triggering the regex fallback) when no model
    path is configured, so an unmodified TOML keeps working with zero GPU cost from
    this reward term -- same "fallback runs today, primary drops in" story as the
    offline scorer's own design (``coc_action_consistency_v2.py``'s docstring).
    """
    global _SINGLETON
    custom = getattr(config, "custom", {}) if config is not None else {}
    alp_custom = custom.get("alpamayo", {}) if isinstance(custom, dict) else {}
    reward_cfg = alp_custom.get("reward", {}) if isinstance(alp_custom, dict) else {}
    model_dir = (
        reward_cfg.get("coc_extractor_model_path")
        if isinstance(reward_cfg, dict)
        else None
    ) or os.getenv("ALPAMAYO_COC_EXTRACTOR_MODEL_PATH")
    if not model_dir:
        return None
    if _SINGLETON is None:
        with _SINGLETON_LOCK:
            if _SINGLETON is None:
                device = (
                    reward_cfg.get("coc_extractor_device", "cpu")
                    if isinstance(reward_cfg, dict)
                    else "cpu"
                )
                if device == "auto":
                    import torch

                    local_rank = int(os.getenv("LOCAL_RANK", "0"))
                    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
                _SINGLETON = QwenClaimExtractor(model_dir, device=device)
    return _SINGLETON
