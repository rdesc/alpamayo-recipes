# Baseline eval: OOD reasoning subset

Run of `scripts_fork/eval_baseline_ood_reasoning.py` (no-RL baseline) against the
PAI-AV OOD reasoning subset (`reasoning/ood_reasoning.parquet`, train+val splits).

> **2026-08-19: superseded prompt fix.** The numbers below (as of this update) use the
> corrected prompt construction -- `alpamayo.processor.qwen_processor.get_preprocess_data_fn_from_model_config`
> with the same `components_order`/`chat_template_version` as actual RL training
> (`alpamayo1_x_rl/hydra_configs/alpamayo1_5_rvla_rl_pai.yaml`), including camera-identity/
> frame-index prompt text. The original run (numbers preserved at the bottom of this doc) used
> `alpamayo_r1.helper.create_message`, which omits that camera annotation entirely -- an
> eval-script-only bug, not present in actual training. See
> `scripts_fork/results/eval_prompt_format_discrepancy.md` for the full investigation and the
> experiments that isolated and confirmed this. Metric definition is unchanged (still this
> repo's `compute_minade` full-trajectory-committed-selection convention, NOT the per-horizon
> convention `~/repos/alpamayo1.5`'s published numbers use -- see the discrepancy doc for why
> these two numbers still aren't apples-to-apples even after this fix).

- Model: `/mnt/efs/users/rod/ckpts/Alpamayo-1.5-10B-rl-training`
- CoC extractor: `Qwen/Qwen3-VL-8B-Instruct`
- `--num-traj-samples 6` (min-ADE₆), all annotated events per clip (no `--first-event-only`)
- Sharded 4-way across GPUs 4-7, ~200 min wall clock (via `a1x_rl_b300` venv -- see note below)
- Output: `/opt/dlami/nvme/rod/results/baseline_ood_reasoning_v2/baseline.shard{0..3}-of-4.parquet`

> **Venv note.** This recipe's default `a1x_rl` venv currently fails on every event on this
> B300 box with `nvrtc: invalid value for --gpu-architecture` (vendored nvrtc 12.8 doesn't
> recognize this GPU's architecture string when JIT-compiling a reduction kernel -- confirmed
> unrelated to this script, the original unmodified script fails identically). Use `a1x_rl_b300`
> (nvrtc 13.0) instead until that's fixed.

Every scored row comes from a clip with a human-annotated decision-moment event
(`gold_coc` reasoning trace) — clips without annotated `events` are dropped before
inference (`load_events` in the eval script). 1,731 unique clips, 2,077 annotated
events, x6 trajectory samples = 12,432 rows.

## Overall (n=12,426 ok rows; 6 failed)

| metric | mean | median |
|---|---|---|
| ADE | 1.96 m | 1.47 m |
| min-ADE | 0.96 m | 0.58 m |
| min-ADE @ t=0.5s | 0.012 m | 0.008 m |
| min-ADE @ t=1.0s | 0.040 m | 0.026 m |
| min-ADE @ t=3.0s | 0.283 m | 0.184 m |
| min-ADE @ t=6.0s | 0.854 m | 0.511 m |

CoC-action-consistency (model's own predicted CoC vs. its own predicted trajectory,
not graded against `gold_coc`):

- coc_binary accuracy: **0.593**
- coc_graded (soft score): **0.603**
- coc_reward_contribution: **0.603**
- abstain rate: ~0.08%

## By split

| split | n | ADE mean | min-ADE mean | coc_binary mean |
|---|---|---|---|---|
| train | 10,332 | 1.93 m | 0.94 m | 0.600 |
| val | 2,094 | 2.10 m | 1.06 m | 0.562 |

Val-split-only minADE₆ (n=349 events, directly comparable in scope, not metric convention,
to `~/repos/alpamayo1.5/docs/first_round_eval_results.md`'s published numbers):

| horizon | this eval (post prompt-fix) | officially published |
|---|---|---|
| @1.0s | 0.042 m | 0.0247 m |
| @3.0s | 0.307 m | 0.2241 m |
| @6.0s | 0.946 m | 0.8528 m |

The remaining gap here is the metric-convention difference, not a prompt issue -- see the
discrepancy doc's `eval_metric_convention_compare.py` experiment, which showed switching only
the metric convention (same predictions) closes this down to ~5-11%.

## By event cluster

| cluster | n | ADE mean | coc_binary mean |
|---|---|---|---|
| WORK_ZONES_TEMP_TRAFFIC_CONTROL | 6,324 | 1.80 m | 0.555 |
| PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY | 2,598 | 1.99 m | 0.643 |
| SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR | 1,818 | 2.02 m | 0.673 |
| CYCLISTS_AND_MICROMOBILITY_COMPLEX | 594 | 2.53 m | 0.503 |
| COMPLEX_INTERSECTION_INTERACTION | 342 | 2.74 m | 0.623 |
| OTHER_LONGTAIL | 252 | 2.09 m | 0.694 |
| EMERGENCY_INCIDENT_SCENE | 192 | 2.41 m | 0.620 |
| ANIMALS_BIRDS_ROADKILL | 174 | 2.49 m | 0.580 |
| ROAD_DEBRIS_OR_SAFETY_TRACES | 132 | 2.19 m | 0.492 |

Same rank pattern as before the fix: `ROAD_DEBRIS_OR_SAFETY_TRACES` and
`CYCLISTS_AND_MICROMOBILITY_COMPLEX` remain the weakest CoC-accuracy clusters;
`COMPLEX_INTERSECTION_INTERACTION` remains the worst-ADE cluster.

## Failures

6 rows failed with `Interpolation times must be within the range [0.0, X]...` —
trajectory query time fell outside the ego-pose interpolation range for those
specific clips (edge-of-clip timing issue, not a model failure; same 6 clips fail
identically regardless of prompt construction, since it's a data-loading issue).

---

## Superseded: original run (pre prompt-fix, kept for reference)

Numbers below used `alpamayo_r1.helper.create_message`, which omits camera-identity/frame-index
prompt annotation. See the note at the top of this doc.

- Sharded 4-way across GPUs 4-7, ~4h wall clock
- Output: `/opt/dlami/nvme/rod/results/baseline_ood_reasoning/baseline.shard{0..3}-of-4.parquet`

| metric | mean | median |
|---|---|---|
| ADE | 2.67 m | 2.01 m |
| min-ADE | 1.31 m | 0.80 m |
| min-ADE @ t=0.5s | 0.018 m | 0.013 m |
| min-ADE @ t=1.0s | 0.057 m | 0.040 m |
| min-ADE @ t=3.0s | 0.387 m | 0.257 m |
| min-ADE @ t=5.0s | 0.860 m | 0.533 m |
| corner distance | 1.25 m | 0.73 m |

- coc_binary accuracy: 0.599
- coc_graded: 0.607
- coc_reward_contribution: 0.607
- abstain rate: ~0.03%

| split | n | ADE mean | coc_binary mean |
|---|---|---|---|
| train | 10,332 | 2.65 m | 0.604 |
| val | 2,094 | 2.78 m | 0.570 |

| cluster | n | ADE mean | coc_binary mean |
|---|---|---|---|
| WORK_ZONES_TEMP_TRAFFIC_CONTROL | 6,324 | 2.41 m | 0.591 |
| PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY | 2,598 | 2.88 m | 0.609 |
| SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR | 1,818 | 2.68 m | 0.678 |
| CYCLISTS_AND_MICROMOBILITY_COMPLEX | 594 | 3.16 m | 0.406 |
| COMPLEX_INTERSECTION_INTERACTION | 342 | 3.99 m | 0.632 |
| OTHER_LONGTAIL | 252 | 2.70 m | 0.667 |
| EMERGENCY_INCIDENT_SCENE | 192 | 3.55 m | 0.547 |
| ANIMALS_BIRDS_ROADKILL | 174 | 3.22 m | 0.580 |
| ROAD_DEBRIS_OR_SAFETY_TRACES | 132 | 2.98 m | 0.402 |
