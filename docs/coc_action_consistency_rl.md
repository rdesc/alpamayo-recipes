# CoC–action consistency via RL — scoping notes

Observation that kicked this off: the released HuggingFace Alpamayo
checkpoints show **poor chain-of-causation (CoC) ↔ action consistency** —
noticeably lower than the paper reports. The idea on the table: additional
fine-tuning on PAI-AV using this repo's RL pipeline
([`recipes/alpamayo1_x_rl/`](../recipes/alpamayo1_x_rl/)) to tighten it.

This doc records the reasoning about whether/how that works, so we don't
re-derive it. Nothing built yet — this is scoping.

> **Status update (2026-08-16).** The consistency reward this doc scopes below is now
> implemented as an opt-in component:
> `recipes/alpamayo1_x_rl/rewards/coc_action_consistency_*.py` (matcher, trajectory
> classifier, extractor, and the top-level reward), wired into
> `rewards/aggregated_reward_with_reasoning.py` behind a `coc_consistency_weight` TOML
> key that defaults to 0.0 (no behavior change unless set). The matching logic is a
> verbatim, tested port of a design selected and validated offline against 2077 PAI-AV
> OOD gold events in the sibling `alpamayo-coc-autolabeler` repo (E3(P3)xM5-abstain,
> 0.941 gold agreement, 0.931 hard-negative separation, 0.945/0.933 read at t0+1s — see
> that repo's `docs/cac_scorer_design_results.md`).
>
> The trajectory-side classifier was originally a fixed-window kinematic average
> (structurally close to the *rejected* B0 baseline, 0.79–0.83) but has since been
> replaced with a segmentation+min-length-merge design ported from the same production
> pipeline the 0.941 figure was measured against, adapted to run directly on
> `pred_xyz`/`pred_rot` instead of needing a full trajdata clip. Validated at **0.932
> gold binary / 0.902 hard-negative separation** (0.933 with a documented offset
> exception for `reverse` claims specifically — see `coc_action_consistency_reward.py`'s
> `REVERSE_OFFSET_S`), closing nearly the entire gap to the 0.941/0.945 production
> number. **Still not validated:** this number is measured against each gold event's
> real recorded future motion, not a rollout's own decoded prediction — it is an oracle
> ceiling, not a forecast of in-loop reward values; the trajectory classifier has never
> been run against real rollout output. A further, unimplemented improvement is stubbed
> in `coc_action_consistency_trajectory.py` (`trajectory_state_from_action`): Alpamayo's
> action space is literally `(accel, kappa)` per waypoint, which map exactly onto this
> module's own features — reading them directly would skip differentiating a rolled-out
> trajectory back into the quantities the model already predicted, removing a real noise
> source, but needs new plumbing to surface the decoded action tensor and can't be
> validated against the gold corpus (no model-predicted control tokens there).
>
> The in-loop Qwen3-VL-8B extractor is ported but has never been run inside a rollout
> loop — the reward currently defaults to a regex fallback extractor (also ported,
> ~0.93 offline gold agreement on its own) so it runs without a GPU dependency. See
> `coc_action_consistency_reward.py`'s module docstring for the full status and the
> still-open abstained-event reward-shaping decision.

## The two structural facts that constrain the plan

### 1. The RL pipeline trains the VLM backbone only — not the action expert

Per the `alpamayo1_x_rl` SKILL/FAQ, GRPO here updates **only** the
autoregressive pathway that emits the CoC text *and* the discrete trajectory
tokens (both live in one rollout completion: reasoning before `<|cot_end|>`,
trajectory tokens after `<|traj_future_start|>`). The flow-matching **action
expert** (continuous-trajectory head) is **not** trained and passes through
unused in the export.

So the deciding question is **which "action" is inconsistent with the CoC**:

- **Discrete trajectory tokens** (same autoregressive stream as the CoC) →
  RL here is the right lever. A reward can see both halves of the completion
  and score their agreement.
- **Action-expert continuous trajectory** → RL in this repo **cannot** reach
  it. Aligning the CoC to a pathway the deployed model doesn't use for its
  final action would be optimizing the wrong thing — and that pathway
  mismatch could itself be the observed "poor consistency."

### 2. None of the shipped rewards reward *consistency*

GRPO optimizes exactly what you measure. The shipped rewards don't measure
CoC↔action agreement:

- **Motion reward** (`rewards/aggregated_reward.py`) = ADE + comfort on the
  trajectory only. Blind to the reasoning.
- **Joint reward** (`rewards/aggregated_reward_with_reasoning.py`) = motion +
  Lingo-Judge grading the CoC against a *reference* answer. Rewards reasoning
  *quality/plausibility*, not whether the reasoning agrees with the action the
  model then takes.

⇒ Out of the box, neither improves consistency. This needs a **custom reward**
that extracts both halves of the completion and scores their agreement (e.g.
decode the reasoning's implied maneuver, decode the trajectory's actual
maneuver, penalize divergence). The repo is built for custom rewards
(subclass the reward interface, register in the entry script), so it's
feasible — but the reward *is* the real work here, not the plumbing.

## Diagnose before spending RL compute

"Way lower than the paper" has boring explanations that RL will **not** fix.
Rule these out first:

- **Eval-methodology mismatch** — how we measure consistency vs. how the paper
  did. Most common cause of a large reported-vs-observed gap.
- **Released checkpoint ≠ paper model**, or a conversion/format step
  (`to-a1`, RL config remap) altering behavior.
- **Decoding / prompt-format** — this model is sensitive to prompt/token
  conventions (see the camera-prompt-format finding in the TruckDrive work);
  a subtle mismatch degrades the CoC stream specifically.

## Recommended sequencing

1. **Reproduce the paper's consistency metric** on the raw HF checkpoint and
   confirm the gap survives a correct eval. If it doesn't, stop here.
2. **Pin the target pathway** (discrete tokens vs action expert). Only the
   former is RL-addressable in this repo.
3. If discrete-token: **design a CoC-action-consistency reward**, overfit it
   on 1 → ~16 samples (the repo's recommended smoke workflow), watch it climb,
   *then* scale.
4. **Consider SFT first.** If the model *never* learned the alignment,
   supervised fine-tuning on consistent CoC+trajectory pairs is more direct.
   RL shines when the capability exists but is degraded — which fits the
   "lower than paper" framing, so RL is plausible, but weigh it against SFT.

## Open questions to resolve

- Which action pathway is the inconsistent one (discrete tokens / action
  expert / not yet pinned)?
- Do we have a concrete consistency metric, or is it qualitative so far? In
  RL, that metric becomes the reward — it has to be defined either way.

## Survey: paper vs. this repo vs. NVIDIA's other repos vs. external forks

Paper: *Alpamayo-R1: Bridging Reasoning and Action Prediction for
Generalizable Autonomous Driving in the Long Tail* (arXiv:2511.00088),
RL post-training in Sec 5.3/6.3.

### `recipes/alpamayo1_x_rl/` vs. the paper

| Paper element | Status here |
|---|---|
| GRPO algorithm | Implemented (Cosmos-RL). |
| KL penalty vs reference policy | Implemented (`kl_beta=0.01` in the TOML). |
| Reasoning-quality reward | Implemented but simplified — uses **Lingo-Judge**
  (`rewards/aggregated_reward_with_reasoning.py`), a smaller/older judge than
  the paper's large-reasoning-model critic. |
| Trajectory-quality / comfort reward | Implemented (`rewards/traj_reward.py`,
  `rewards/comfort_reward.py`). |
| **Reasoning↔action consistency reward** | **Missing.** Motion and reasoning
  rewards are scored independently; nothing measures whether the CoC text and
  the executed trajectory agree with each other. This is the gap this doc is
  about. |
| **Safety / collision / off-road reward** | **Missing.** `base_dataset.py`
  already loads obstacle-bbox and lane-boundary fields but no reward file
  consumes them. |
| Closed-loop simulator rollout | **Missing** — this recipe is explicitly
  open-loop (single-step, scored against a fixed logged future), per its
  SKILL.md. |

### NVIDIA's other repo, `NVlabs/alpagym`

Public, separate reward stack (does not import anything from
`recipes/alpamayo1_x_rl/rewards/`), confirmed genuinely **closed-loop**:
AlpaSim drives per-tick, calling back into the policy each tick, and only the
accumulated episode result resolves at the end.

Its `progress_safety` reward config:
- `progress` (+1.0)
- `collision_any` (−10.0)
- `offroad` (−5.0)
- `distance_to_gt` (−0.01, plain trajectory RMSE)

So AlpaGym **covers the safety/collision/off-road gap** — but the underlying
collision/offroad geometry is computed inside the separate `NVlabs/alpasim`
simulator repo, not inspected here. It does **not** cover the
reasoning-consistency gap: zero language/CoT/judge terms anywhere in the
repo (grepped for reason/consist/judge/lingo/cot, no hits outside model I/O).

Net: between NVIDIA's two repos, you get closed-loop safety rewards
(AlpaGym) XOR reasoning-quality rewards (`alpamayo1_x_rl`) — never both, and
**neither implements the paper's actual consistency term**.

### External reimplementations — none found

Swept GitHub search, the HF paper page's discussion tab, and
PapersWithCode (which has effectively retired into HF's paper index — no
independent code-links listing). Found only:
- NVIDIA's own repos (`alpamayo`, `alpamayo1.5`, `alpamayo-recipes`,
  `alpagym`, `alpasim`).
- Personal forks/mirrors and inference/eval/pruning projects that reuse the
  released checkpoint without touching RL training.
- One repo, `IP127000/Alpamayo1.5-finetune`, that *advertises* "RL
  fine-tuning, consistency training" — but its README says the code isn't
  released yet ("Stay tuned").

**Conclusion: nobody has published a reasoning-action-consistency reward
implementation.** If this gets built, it'd be original work, not a
port of something that already exists — reinforces "the reward *is* the
real work here" above.
