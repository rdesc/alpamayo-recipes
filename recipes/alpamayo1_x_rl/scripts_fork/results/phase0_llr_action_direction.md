# Phase 0 — action-direction LLR (langforce_readme.md §4.2), full OOD reasoning split

Run of `scripts_fork/phase0_llr_action_direction.py` against the full PAI-AV OOD
reasoning subset (`reasoning/ood_reasoning.parquet`, train+val splits), on
`/mnt/efs/users/rod/ckpts/Alpamayo-1.5-10B-rl-training` via `TrainableReasoningVLA`
(Stage-1 discrete-AR-trajectory-token pathway).

## What this measures

`langforce_readme.md` §4 asks, before any training: is the model's Chain-of-Causation
reasoning already load-bearing for its trajectory prediction, or decorative? The
language-direction form (§4.1) was tried first and found broken — see
`phase0_llr_ordering_control.py` — its `[v, a*, cot]` reordering put the trajectory in
a sequence position the model was never trained on, and a wrong-trajectory control
showed the resulting "signal" was ~95% positional-novelty artifact (real vs. wrong
trajectory differed by 0.19 nats out of a 3.3–3.5 nat drop vs. vision-only).

This run instead uses §4.2's **action-direction** form, which needs no reordering
(both passes keep the model's natural trained order `[v, cot, traj_future]`):

```
llr_act = mean_a[log p(a* | v, ℓ)] − mean_a[log p(a* | v, ℓ-blanked)]
```

- Pass C (`logp_num`): real gold CoC (`ℓ`) teacher-forced, score the ground-truth
  trajectory tokens (`a*`).
- Pass D (`logp_den`): CoC span present but token-content-blanked (pad tokens, same
  length/position — not removed, not attention-masked, to avoid reintroducing a
  positional shift), score the same `a*` tokens.

Validated on an 8-event smoke test before this run (llr_act +0.31 nats, an order of
magnitude smaller than the broken §4.1 number and the right sign) — see prior smoke
test in `phase0_llr_action_direction.py`'s own history.

### Why blanking, not `coc_text=""` — two different ablations, do not conflate

A separate, later piece of work (`extract_llr_viz_predictions_v2.py`, illustrative
BEV/velocity examples for a handful of events) generates the model's actual predicted
trajectory with reasoning suppressed, using this repo's validated `coc_text=""`
mechanism (`~/repos/alpamayo1.5/eval_pai_av_val.py --no_coc`): the CoC span is spliced
down to essentially zero tokens (`[cot_end_id, traj_future_start_id]` appended directly)
before generation continues. That is the *correct* choice for a **generation-time**
ablation — the model just continues from whatever prefix it's given, and it already
sees variable-length real CoC during training, so a short/empty-but-structurally-correct
span isn't out-of-distribution the way a *structurally dropped* CoT component is (an
earlier, broken attempt at this that produced garbage — see that script's own commit
history / the results doc for the RL recipe's viz work).

**Do not port `coc_text=""` back into this script's `llr_act` scoring.** Scoring needs
the numerator and denominator passes to differ *only* in content, never in sequence
length or position — that's the entire reason Pass D blanks the CoC span with pad
tokens at the *same length* as the real reasoning, rather than shortening it. Splicing
in a near-zero-length CoC for Pass D would shift `a*`'s position between the two passes,
reintroducing the exact positional-novelty confound that broke §4.1 in the first place
(see the ordering-control note above — real vs. wrong content differed by only 0.19 nats
out of a 3.3–3.5 nat drop that was pure position). The two ablations look similar
(“remove the reasoning”) but serve different purposes and are not interchangeable:
blanking-at-fixed-length for measurement, `coc_text=""` for generation.

## Run

- Model: `/mnt/efs/users/rod/ckpts/Alpamayo-1.5-10B-rl-training`
- `--split both`, all annotated events (no `--first-event-only`)
- Sharded 4-way across GPUs 0-3, ~44 min/shard (parallel, so ~44 min wall clock)
- Output: `/opt/dlami/nvme/rod/results/phase0_llr_action_direction/llr.shard{0..3}-of-4.parquet`,
  merged to `llr_merged.parquet` (same directory)

1,731 unique clips / 2,077 annotated events — same event set as the sibling
`eval_baseline_ood_reasoning*.py` scripts.

## Overall (n=2,071 ok; 6 failed)

| metric | value |
|---|---|
| llr_act mean | **+0.227 nats** |
| llr_act median | +0.248 nats |
| llr_act std | 0.124 |
| logp_num (real CoC) mean | -7.334 |
| logp_den (blanked CoC) mean | -7.561 |
| mean CoC span length | ~17 tokens |

## By split

| split | n | llr_act mean | llr_act median | std |
|---|---|---|---|---|
| train | 1,722 | 0.227 | 0.249 | 0.128 |
| val | 349 | 0.230 | 0.245 | 0.097 |

Train and val are essentially identical — no sign of the train/val leakage gap seen
in the trajectory-accuracy metrics (`baseline_ood_reasoning*.md` docs show train ADE
noticeably better than val). Consistent with this being a property of the
reasoning/action relationship itself rather than of trajectory-prediction skill,
which is where the leakage effect shows up.

## By event cluster

| cluster | n | llr_act mean | llr_act median |
|---|---|---|---|
| PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY | 433 | 0.161 | 0.204 |
| ANIMALS_BIRDS_ROADKILL | 29 | 0.164 | 0.232 |
| ROAD_DEBRIS_OR_SAFETY_TRACES | 22 | 0.210 | 0.256 |
| CYCLISTS_AND_MICROMOBILITY_COMPLEX | 99 | 0.220 | 0.230 |
| WORK_ZONES_TEMP_TRAFFIC_CONTROL | 1,054 | 0.245 | 0.250 |
| SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR | 303 | 0.247 | 0.256 |
| EMERGENCY_INCIDENT_SCENE | 32 | 0.280 | 0.283 |
| COMPLEX_INTERSECTION_INTERACTION | 57 | 0.281 | 0.275 |
| OTHER_LONGTAIL | 42 | 0.295 | 0.278 |

Every cluster is positive, none near zero, none negative — the signal is uniform in
direction across scene types, just varying in magnitude (roughly 2x between the
weakest and strongest clusters).

## Failures

6/2077 events failed (`ok=False`), all `train` split. Same known data-loading edge
case documented in the sibling eval docs (`baseline_ood_reasoning*.md` "Failures"
sections): `Interpolation times must be within the range [...]`, i.e. `t0` outside
the clip's ego-pose interpolation range for those specific clip/event pairs —
independent of this script or measurement, a pre-existing dataset quirk.

## Reading against the §4.4 decision gate

The doc's gate frames two clean outcomes: `llr` clearly `> 0` with reasoning-masking
hurting the trajectory ("reasoning already used, little to fix"), or `llr ≈ 0` with
masking barely changing the trajectory ("shortcut confirmed, proceed to Phase 1").

+0.227 nats mean, positive in every cluster, is a real, small, consistently
positive signal — closer to the first bucket than the second, but far too modest to
call "reasoning already fully used." In plain terms: conditioning on the gold
reasoning text does make the ground-truth trajectory tokens measurably more likely
than conditioning on vision alone with the reasoning content blanked out — the
model's discrete-trajectory-token pathway is not purely ignoring its own reasoning.
But the effect size (~0.23 nats, against ~7.3–7.6 nats of baseline per-token
negative log-likelihood) is small — reasoning is doing *something*, not nothing, but
it's a modest contribution, not the dominant driver of the trajectory prediction.

Caveats carried over from the confirm-first investigation, still load-bearing here:

- This measures the **Stage-1 discrete-AR-token pathway only** (`TrainableReasoningVLA`).
  It says nothing about whether the same holds for the actually **deployed** Stage-2
  diffusion/flow-matching head (`TrainableAlpamayoR1`), which has no trajectory-token
  logits to measure this way — that needs the §8 diffusion-consistency variant on a
  separate run.
- This checkpoint (`Alpamayo-1.5-10B-rl-training`) is the one from the RL recipe,
  where the discrete-token pathway is real and actively trained — not a fresh SFT
  checkpoint. The result should not be assumed to transfer unchanged to other
  checkpoints without RL post-training on this pathway.
- Behavioral confirmation (§4.3 — does masking reasoning at *inference* actually
  change the *decoded* trajectory, not just its likelihood under teacher forcing) has
  not been run yet. A small positive LLR is consistent with either a small real
  effect or a systematic-but-small bias in how blanking interacts with the loss;
  §4.3 would corroborate this from a different angle.

**Recommendation:** the shortcut is not fully confirmed (llr is not ≈0), so a blind
"proceed straight to Phase 1 training" is not obviously right, but the effect is
small enough that Phase 1's regularizer plausibly still has real room to grow it.
Worth running §4.3's behavioral ablation next (mask reasoning at generation time,
measure decoded-trajectory displacement) before committing training compute, since
that's a cheaper/faster sanity check than a full Phase-1 SFT run and would confirm
whether this modest likelihood-level signal corresponds to any real inference-time
behavior change.
