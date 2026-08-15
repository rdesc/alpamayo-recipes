# Throughput & training observations (run `20260809182549`)

Notes from watching the first working B300 RL run end-to-end (wandb run
`ReasoningVLA_Post_Training/20260809182549`,
`https://wandb.ai/rdesc1-milaquebec/alpamayo/runs/20260809182549`,
log dir `/mnt/efs/users/rod/logs/logs_20260809-182427`). This is a
performance/tuning investigation, not a bug list — see
`B300_MIGRATION_NOTES.md` for the crash history that got this run running
in the first place.

## Dataset actually in use

- Local dataset: `pai_nav_subset_19chunks` (19 downloaded PAI chunks,
  306,152 raw clip-index rows before curation).
- Curated **training pool: 891 windows = 891 distinct scenes** (unique
  `clip_id`, no repeats), spanning **9 of the 19** downloaded chunks
  (~99-100 windows/chunk). Confirmed directly from
  `clip_index_mini_891train.parquet.bak` (`split=train` for all 891 rows).
- Curated **validation pool: 891 windows**, of which 90 are sampled each
  validation pass (`stride=10`).
- Reward mode: **motion only** (trajectory ADE + comfort). This run does
  *not* touch the OOD-reasoning subset or Lingo-Judge CoT grading — those
  are only used by the separate "joint reward" entry point
  (`alpamayo_cosmos_rl_post_training_reasoning_entry.py`), which this run
  doesn't use.
- Caveat (from `SETUP_NOTES.md`): `curate_pai_samples.py` filters only on
  `--chunk`, not `split` — train/val chunk disjointness was not
  independently re-verified here, just assumed from the file naming.

## Step/epoch math

- `train_batch_per_replica=48` ÷ `rollout.n_generation=8` = **6 unique
  prompts consumed per training step**.
- 891 windows ÷ 6 prompts/step ≈ **149 steps/epoch** (matches a toml
  comment: "bs 24->48 halves steps/epoch (297 -> 149)").
- `epoch=30` planned → ~4470 steps, matches the `total_steps=4455` logged
  by the run (small rounding since 891/6 isn't a whole number).

## Reward curve: flat through step ~250

Bucketed 40-step averages of `train/reward_mean` (all in same range, no
directional trend):

| Steps | Mean reward | Best | Worst |
|---|---|---|---|
| 1-40 | -0.2802 | -0.1242 | -0.5361 |
| 41-80 | -0.2620 | -0.1504 | -0.4173 |
| 81-120 | -0.2726 | -0.1201 | -0.4840 |
| 121-160 | -0.2537 | -0.1165 | -0.4071 |
| 161-200 | -0.2564 | -0.0907 | -0.5253 |
| 201-240 | -0.2721 | -0.1635 | -0.4898 |

Validation (the cleaner, less noisy signal) confirms it — step 0 vs step
150:

| Metric | Step 0 | Step 150 | Δ |
|---|---|---|---|
| val/reward_avg | -0.2433 | -0.2349 | ~flat |
| val/min_ade | 0.8702 | 0.8736 | ~flat |
| val/ade | 1.7702 | 1.7165 | ~flat |
| val/corner_distance | 0.9049 | 0.9079 | ~flat |

Not disqualifying at ~5-6% of planned training, but genuinely flat, not
"slow but trending." Worth re-checking at the step-300 and step-450
validation passes before concluding anything about whether the LR/KL-beta
combination is workable.

## Config vs. NVIDIA's own documented recommendations

`SKILL.md` gives two reference configs (Local smoke-test, Cluster
production). This run's actual config matches neither:

| Parameter | Doc: Local | Doc: Cluster | **This run** |
|---|---|---|---|
| `rollout.parallelism.n_init_replicas` | 1 | 128 | **6** |
| `train.optm_lr` | 2e-6 | 2e-6 | **5e-6** |
| `rollout.n_generation` | 12 | 12 | **8** |
| `custom.alpamayo.prefetch.capacity` | 16 | 128 | **48** |

Notable: this run's LR (5e-6) is **2.5x higher** than NVIDIA's documented
value at *both* local and cluster scale — so if the flat reward curve
above turns out to be a real problem, "LR too small" is not the obvious
first suspect; if anything the LR is already on the aggressive side
relative to the only documented reference points we have.

The `n_init_replicas=6` rollout / `dp_shard_size=2` policy split is a
custom mid-point between the two documented configs, inherited from an
older, pre-B300-migration profiling run (`20260807045227`) referenced in
the toml's own comments — never re-validated against current (B300,
post-bugfix) hardware/software behavior. See below — there's direct
evidence this ratio is now stale.

## GPU utilization: policy-bound, not rollout-bound

Precisely timed one full step cycle (step 264→265, iteration_time=106.97s)
by correlating `controller.log` timestamps with `nvidia-smi` sampled at
1 Hz across all 8 GPUs:

| Phase | Duration | % of step |
|---|---|---|
| Post-step housekeeping/dispatch | 10.0s | 9% |
| Rollout completions arriving (6 replicas reporting) | 4.6s | 4% |
| **Silent gap** (no controller.log activity) | **94.8s** | **87%** |

GPU utilization measured *during that silent gap specifically*:

| GPU | avg util |
|---|---|
| 0-1 (policy, `dp_shard_size=2`) | 36.5% / 39.8%, peaks to 100% |
| 2-7 (rollout ×6) | **0%** flat |

So: rollout generation is fast (<5s for all 6 replicas combined) and then
those 6 GPUs sit **completely idle** for the remaining ~87% of every step
while the 2 policy GPUs do the actual training compute (confirmed via
utilization, not just log silence — redis timeout/retry noise in
`policy_0.log` during this window is a separate background coordination
channel, not evidence of the GPU compute itself stalling).

**This means the current 6-rollout / 2-policy GPU split is backwards for
this run's actual bottleneck.** The historical toml comment justifying
6:2 ("rollout ~3x costlier than policy, ~1.63s/rollout vs ~0.57s/rollout")
is from a pre-B300-migration run and evidently no longer holds — on
current hardware, policy compute dominates by roughly 20x
(94.8s policy-bound vs ~4.6s rollout-bound), not the other way around.

### Options worth testing on a future launch (not mid-run)

1. **Rebalance GPU allocation toward policy** — e.g. fewer rollout
   replicas (`--rollout 2-4`), more policy FSDP shards
   (`policy.parallelism.dp_shard_size=4` or `6`). Directly motivated by
   the utilization data above, not just a hunch.
2. **`rollout.mode = "async"`** — real, first-class config option
   (`choices=["sync","async"]`, with its own `async_config` block already
   present in the toml but unused since `mode="sync"`). Would let rollout
   generation for the next batch overlap with policy training on the
   current batch instead of strictly alternating. Untested here — the
   usual tradeoff is training against slightly stale weights, and
   `allowed_outdated_steps=50` in this toml suggests the recipe already
   has some staleness tolerance built in, but the actual overlap behavior
   wasn't verified.
3. Neither of these should be applied to a currently-running job — both
   require a fresh launch.
