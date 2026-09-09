# Baseline eval: OOD reasoning subset -- ACTION-HEAD (continuous/diffusion) variant

> ## ⚠️ STALE: these numbers predate the prompt fix
>
> **Every number in this doc comes from the pre-prompt-fix eval script.** The script itself
> has since been corrected (it now builds prompts via
> `alpamayo.processor.qwen_processor.get_preprocess_data_fn_from_model_config`, matching the
> real Stage-1/Stage-2 SFT nav config including camera-identity/frame-index prompt text), but
> **it has not been re-run**, so nothing below reflects that fix.
>
> **Do not compare this doc's numbers against the current
> `alpamayo1_x_rl/scripts_fork/results/baseline_ood_reasoning.md`.** That doc *was* re-run and
> now reports post-fix numbers (min-ADE 0.96 m). Read side by side, the action-head path looks
> ~52% worse than discrete-token — that is almost entirely the prompt difference, not the
> trajectory-generation pathway.
>
> The **"Comparison vs. discrete-token baseline" section at the bottom of this doc is still
> valid on its own terms**: it compares pre-fix against pre-fix (discrete 1.31 m vs. action-head
> 1.46 m), which is apples-to-apples and shows the real pathway gap of ~11%. Expect that
> relative gap to roughly hold post-fix, but it is unverified until the re-run happens.
>
> See `alpamayo1_x_rl/scripts_fork/results/eval_prompt_format_discrepancy.md` for the
> investigation.

Run of `scripts_fork/eval_baseline_ood_reasoning_actionhead.py` against the same
PAI-AV OOD reasoning subset (`reasoning/ood_reasoning.parquet`, train+val splits)
as `alpamayo1_x_rl/scripts_fork/results/baseline_ood_reasoning.md` -- same events,
same K, same metrics. The only difference is how the trajectory is generated.

- Model: `/mnt/efs/users/rod/ckpts/Alpamayo-1.5-10B-rl-training` (same checkpoint
  as the discrete-token baseline -- it carries real action-expert weights that
  `RLWrapperReasoningVLA` simply never loads; loaded here via Stage-2's
  `TrainableAlpamayoR1` instead)
- Trajectory pathway: VLM autoregressively generates reasoning/CoC text up to
  `<traj_future_start>`, then a flow-matching diffusion expert (cross-attending
  into the VLM's KV-cache) denoises a *continuous* trajectory in action space
  (`UnicycleAccelCurvatureActionSpace`) -- no discrete trajectory tokens involved
- CoC extractor: `Qwen/Qwen3-VL-8B-Instruct` (same as discrete-token baseline)
- `--num-traj-samples 6` (min-ADE₆), all annotated events per clip (no `--first-event-only`)
- Sharded 4-way across GPUs 4-7, ~142 min/shard (~2.4h wall clock -- faster than
  the discrete-token run's ~4h, consistent with skipping ~128 sequential
  autoregressive trajectory-token decode steps per sample)
- Output: `/opt/dlami/nvme/rod/results/baseline_ood_reasoning_actionhead/baseline.shard{0..3}-of-4.parquet`

Same 1,731 unique clips / 2,077 annotated events / x6 samples = 12,432 rows as the
discrete-token run (same `load_events` loader, same dataset).

## Overall (n=12,426 ok rows; 6 failed)

| metric | mean | median |
|---|---|---|
| ADE | 2.98 m | 2.37 m |
| min-ADE | 1.46 m | 0.97 m |
| min-ADE @ t=0.5s | 0.019 m | 0.014 m |
| min-ADE @ t=1.0s | 0.062 m | 0.046 m |
| min-ADE @ t=3.0s | 0.427 m | 0.304 m |
| min-ADE @ t=6.0s | 1.301 m | 0.853 m |

> Horizon note: `rewards/val_metrics.py` dropped `corner_distance` and its
> longest horizon changed from 5.0s to 6.0s in a same-day fork edit (see
> `git log -- recipes/alpamayo1_x_rl/rewards/val_metrics.py`), so this table's
> shape differs slightly from the discrete-token doc's (no corner distance,
> 6.0s instead of 5.0s). Both scripts import the same `group_distance_metrics`,
> so re-running the discrete-token eval today would show the same shape.

CoC-action-consistency (model's own predicted CoC vs. its own predicted trajectory,
not graded against `gold_coc`):

- coc_binary accuracy: **0.595**
- coc_graded (soft score): **0.603**
- coc_reward_contribution: **0.603**
- abstain rate: ~0.008%
- parsed rate: ~99.9%

## By split

| split | n | ADE mean | coc_binary mean |
|---|---|---|---|
| train | 10,332 | 2.94 m | 0.600 |
| val | 2,094 | 3.16 m | 0.573 |

## By event cluster

| cluster | n | ADE mean | coc_binary mean |
|---|---|---|---|
| WORK_ZONES_TEMP_TRAFFIC_CONTROL | 6,324 | 2.76 m | 0.578 |
| PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY | 2,598 | 3.04 m | 0.638 |
| SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR | 1,818 | 3.20 m | 0.648 |
| CYCLISTS_AND_MICROMOBILITY_COMPLEX | 594 | 3.20 m | 0.453 |
| COMPLEX_INTERSECTION_INTERACTION | 342 | 4.52 m | 0.643 |
| OTHER_LONGTAIL | 252 | 3.13 m | 0.663 |
| EMERGENCY_INCIDENT_SCENE | 192 | 3.26 m | 0.531 |
| ANIMALS_BIRDS_ROADKILL | 174 | 2.96 m | 0.575 |
| ROAD_DEBRIS_OR_SAFETY_TRACES | 132 | 3.41 m | 0.386 |

Weakest CoC accuracy: `ROAD_DEBRIS_OR_SAFETY_TRACES` (0.39), `CYCLISTS_AND_MICROMOBILITY_COMPLEX` (0.45) --
same two weakest clusters as the discrete-token baseline, same rank order.
Strongest: `OTHER_LONGTAIL` (0.66), `SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR` (0.65).
Worst ADE: `COMPLEX_INTERSECTION_INTERACTION` (4.52 m) -- also the worst cluster in
the discrete-token baseline, and the gap versus the rest of the set is larger here.

## Failures

Same 6 clips/events fail as in the discrete-token baseline, with the identical
`Interpolation times must be within the range [...]` error -- confirms this is a
data-loading edge case (t0 outside the ego-pose interpolation range for those
specific clips), independent of which trajectory-generation pathway is used.

## Comparison vs. discrete-token baseline

(`alpamayo1_x_rl/scripts_fork/results/baseline_ood_reasoning.md`, same checkpoint,
same events, same K=6)

| metric | discrete-token | action-head (this doc) | delta |
|---|---|---|---|
| ADE mean | 2.67 m | 2.98 m | +0.31 m (+12%) |
| ADE median | 2.01 m | 2.37 m | +0.36 m (+18%) |
| min-ADE mean | 1.31 m | 1.46 m | +0.15 m (+11%) |
| min-ADE median | 0.80 m | 0.97 m | +0.17 m (+21%) |
| min-ADE @ 0.5s | 0.018 m | 0.019 m | ~flat |
| min-ADE @ 1.0s | 0.057 m | 0.062 m | +9% |
| min-ADE @ 3.0s | 0.387 m | 0.427 m | +10% |
| coc_binary | 0.599 | 0.595 | ~flat (-0.004) |
| coc_graded | 0.607 | 0.603 | ~flat (-0.004) |
| coc_reward_contribution | 0.607 | 0.603 | ~flat (-0.004) |
| abstain rate | ~0.03% | ~0.008% | ~flat |

Takeaways:

- **Discrete-token trajectories are consistently more accurate** on this checkpoint
  and this subset -- roughly 10-20% lower ADE/min-ADE across every horizon and
  aggregation (mean and median). The gap widens at the median more than the mean,
  suggesting the action-head path has a heavier tail of larger errors rather than
  being uniformly worse.
- **CoC-action-consistency is essentially identical** between the two pathways
  (within noise, ~0.004 on every variant of the metric). This makes sense: the
  consistency score grades the model's *reasoning text* against its *own*
  trajectory, and the reasoning-generation step (autoregressive decode up to
  `<traj_future_start>`) is the same VLM, same weights, same sampling params in
  both scripts -- only what happens *after* that token differs.
- **Cluster ranking is stable across both pathways** -- `ROAD_DEBRIS_OR_SAFETY_TRACES`
  and `CYCLISTS_AND_MICROMOBILITY_COMPLEX` are the two weakest CoC-accuracy clusters
  in both evals, and `COMPLEX_INTERSECTION_INTERACTION` is the worst-ADE cluster in
  both -- but the action-head path's gap on `COMPLEX_INTERSECTION_INTERACTION`
  (4.52 m vs. the baseline's 3.99 m) is proportionally larger than its overall ADE
  gap, suggesting the diffusion path degrades more than average on the hardest
  geometric cases.
