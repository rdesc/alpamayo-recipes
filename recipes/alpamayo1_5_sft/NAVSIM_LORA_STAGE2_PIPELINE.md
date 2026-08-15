# NAVSIM: LoRA Stage-1 -> merge -> navhard eval -> Stage-2

How this fork's NAVSIM navtrain experiment is wired end-to-end: a LoRA
Stage-1 VLM finetune, merging it back into a plain checkpoint, evaluating that
checkpoint on NAVSIM's navhard split via a *new* eval path built for this
(Stage-1 checkpoints can't use the existing `eval_navsim.py`), and launching
Stage-2 (flow-matching action-head finetune) on top of it.

Status as of 2026-08-12: pipeline complete end-to-end. Stage-1 LoRA run
(`checkpoint-1660`, epoch 1.0, train_loss 1.656), merged, Stage-1 navhard eval
done; Stage-2 seed0 and seed1 both trained and scored on navhard. See
"Results" below for the numbers.

## Pipeline

```
Stage-1 LoRA SFT  --------->  merge_lora.py  --------->  eval_navsim_stage1.py  --------->  NAVSIM PDM scorer
(sft_stage1_navsim_lora)      (fold adapter        (Stage-1 checkpoint,             (navsim repo, py3.9 venv,
                                into base weights)   navhard split -> parquet)        unchanged)
                                     |
                                     v
                          Stage-2 SFT (sft_stage2_navsim)
                          model.stage1_vlm_checkpoint_path = merged dir
                                     |
                                     v
                    (existing) eval_navsim.py + convert_checkpoint.py to-a15
                    -- works for Stage-2 checkpoints, see "Why two eval paths" below
```

## 1. Stage-1 LoRA -> merge

Stage-1 LoRA training (`configs/sft_stage1_navsim_lora.yaml`) saves adapter
weights alongside each checkpoint (`checkpoint-N/adapter/`). Trainer
checkpoints store adapted projections as `...q_proj.base_layer.weight` +
separate `lora_A`/`lora_B` tensors -- anything that loads weights by name
(notably `load_alpamayo1_vlm`, used by Stage-2's
`model.stage1_vlm_checkpoint_path`) won't match those keys and will either
error (there's a guard for this) or, if unguarded, silently leave every
adapted projection randomly initialized.

Merge before using a LoRA Stage-1 checkpoint for anything downstream:

```bash
a1_5_sft_b300/bin/python -m alpamayo1_5_sft.truckdrive.merge_lora \
    --checkpoint /path/to/output_stage1_navsim_lora/checkpoint-1660 \
    --output /path/to/output_stage1_navsim_lora/checkpoint-1660-merged
```

Pure CPU op (folds `delta_W = B @ A * (alpha/r)` into the base weights,
re-saves under the original key names), no GPU needed, runs in a couple
minutes. Output is "indistinguishable from a full fine-tune's output" and
safe to point `model.stage1_vlm_checkpoint_path` at directly.

## 2. Stage-1 navhard eval: why a new script was needed

The existing navhard eval pipeline (`navsim/docs/alpamayo_navsim.md`,
`alpamayo1.5/eval_navsim.py`) loads checkpoints via the released
`Alpamayo1_5` class. `Alpamayo1_5.__init__`
(`alpamayo1.5/src/alpamayo1_5/models/alpamayo1_5.py`) unconditionally does:

```python
self.action_space = hyu.instantiate(config.action_space_cfg)
self.diffusion = hyu.instantiate(config.diffusion_cfg, ...)
self.action_in_proj = hyu.instantiate(config.action_in_proj_cfg, ...)
```

A Stage-1 checkpoint's `config.json` (`TrainableReasoningVLA`,
`model_type: alpamayo_reasoning_vla`) has **none** of these keys -- Stage 1
never trains a diffusion/flow-matching action expert, only the VLM + discrete
trajectory-token head. Loading it through `Alpamayo1_5` crashes at
construction time, before any weights load. `convert_checkpoint.py to-a15`
(the existing Stage-2 checkpoint -> A15-format converter) doesn't fix this --
it's a pure metadata/`_target_`-prefix rewrite, and the missing config keys
would still be missing after conversion.

**`eval_navsim_stage1.py`** (this recipe) sidesteps the problem entirely by
loading the checkpoint with its own native class
(`TrainableReasoningVLA.from_pretrained`, zero config surgery) and calling its
`sample_trajectories_from_data` (`models/sft_base_model.py`) -- "generate
discrete trajectory tokens autoregressively -> decode via the frozen
`traj_tokenizer`", no diffusion involved. Not new/experimental:
`callbacks/viz_callback.py` calls the exact same method for Stage-1's own
per-eval-step trajectory viz during training.

It writes the **same parquet schema** `eval_navsim.py` writes, so the
downstream scoring pipeline needs zero changes:

```bash
python scripts/navsim_submission_from_parquet.py --parquet ... --out submission.pkl
python navsim/planning/script/run_pdm_score_from_submission.py \
    train_test_split=navhard_two_stage submission_file_path=submission.pkl ...
```

### Running it

Single command, sharded across GPUs (spawns one subprocess per GPU, merges
parquets automatically -- no manual for-loop needed):

```bash
cd recipes/alpamayo1_5_sft
a1_5_sft_b300/bin/python -m alpamayo1_5_sft.eval_navsim_stage1 \
    --checkpoint /path/to/checkpoint-1660-merged \
    --npz_dir /opt/dlami/nvme/rod/datasets/navsim_npz/navhard_two_stage \
    --gpus 0,1,2,3,4,5,6,7
```

Smoke-test first with `--limit 8` (drop `--gpus`/`--num_shards` for a single
quick run) -- same rationale as `docs/alpamayo_navsim.md`'s own advice:
spot-check `poses` for NaN by hand, don't trust the failure counter alone.

Throughput estimate going in was ~8.4s/token steady-state on one GPU, i.e.
5912 tokens / 8 GPUs -> ~1.75h -- the real run came in slower, ~16.7s/token
(739 tokens/shard in 12358s on 8 GPUs, ~3.4h wall clock), plus ~4 min for
`run_pdm_score_from_submission.py` to score all 5912. Budget ~3.5h end-to-end
for a full Stage-1 navhard eval, not the original ~1.75h estimate.

### Gotchas hit building this

- **`attn_implementation`**: `TrainableReasoningVLA.from_pretrained(...,
  attn_implementation="flash_attention_2")` (or `"sdpa"`) raises `ValueError`
  at construction. The *wrapper* class declares neither `_supports_sdpa` nor
  `_supports_flash_attn` (only its `self.vlm` submodule does, and that's
  governed separately by the checkpoint's own saved `config.attn_implementation`,
  unaffected by this flag). Training never hits this because it never goes
  through the generic `from_pretrained()` entrypoint. Only `"eager"` loads --
  that's the script's default now.
- **Waypoint count != token count**: `model.config.tokens_per_future_traj`
  (128) is the discrete-token count, not the decoded waypoint count (64, at a
  fixed 10 Hz cadence spanning 6.4s -- 2 tokens per waypoint, e.g. accel+
  curvature bins). Building the pose-interpolation time grid from
  `tokens_per_future_traj` instead of the actual decoded array length breaks
  `to_navsim_poses`'s `np.interp` with a length mismatch. Derive it from
  `pred_xy.shape[1]` after generation, not before.
- **`load_navsim_npz`'s extra batch dim**: `alpamayo1_5.load_navsim.load_navsim_npz`
  (borrowed cross-repo -- pure numpy/torch/PIL, no `alpamayo1_5` model
  dependency, safe to import) bakes in an extra leading "batch of 1" dim
  (`ego_history_xyz` is `(1, 1, 16, 3)`), sized for `eval_navsim.py`'s own
  manual `torch.cat`-based batching. `NavsimDataset`'s collate_fn instead
  *stacks* raw per-item tensors shaped `(1, 16, 3)` with no pre-existing batch
  dim -- squeeze it off (`_build_sample`) or collate produces a 5D tensor and
  `sample_trajectories_from_data`'s `B, n_traj_group, _, _ = shape` unpack
  throws.
- **Preprocessing must match training exactly**: `components_order`,
  `components_prompt`, `label_components`, `chat_template_version`,
  `include_camera_ids`, `include_frame_nums` are hardcoded defaults in
  `eval_navsim_stage1.py`, copied from `data.val_dataset.vla_preprocess_args`
  in `configs/sft_stage1_navsim*.yaml` as of 2026-08-11, with
  `generation_mode=True` (verified by hand that the resulting prompt cuts off
  exactly at `<|traj_future_start|>`, ready for `model.vlm.generate()` to
  continue). If Stage-1's preprocessing config ever changes, update these
  defaults too -- a mismatch produces a syntactically valid but
  off-distribution prompt, not an error.
- **Stdout buffering**: redirecting stdout to a file/pipe (any background
  `... > log 2>&1 &` launch) switches Python from line- to block-buffered, so
  `print()` output sits invisible in the buffer for the process's lifetime
  instead of showing progress as it happens -- looks exactly like a hang. Not
  a training-loop-specific gotcha, but bit us here; the script now calls
  `sys.stdout.reconfigure(line_buffering=True)` itself so it doesn't depend on
  the caller remembering `python -u`.
- **Running on a different box**: `--checkpoint` paths under `/home/rod/ckpts`
  do **not** survive a machine change -- despite living under `/home/rod`,
  `ckpts` itself is a symlink to `/opt/dlami/nvme/rod/ckpts`, local NVMe, same
  as `--npz_dir`'s `/opt/dlami/nvme/...` (confirmed via `readlink -f`, after
  wrongly assuming otherwise here first). Copy checkpoints to a genuinely
  EFS-backed path (e.g. the `efs_ckpts` convention used for the Stage-1
  navhard eval on a second box) before switching boxes. The `a1_5_sft_b300`-style venv is per-box (compiled
  CUDA extensions) -- if the other box has a different GPU architecture,
  check it works before trusting output; `navsim/docs/alpamayo_navsim.md`
  documents this exact failure mode for the *other* eval venv (silent
  `poses=NaN` with `0 failed` reported, on a GPU-arch mismatch).

## 3. Stage-2 launch

`configs/sft_stage2_navsim.yaml` as committed has
`stage1_vlm_checkpoint_path: null` (built as a "no Stage-1 overlay" ablation
baseline -- see the file's own comments). Override it to the **merged**
checkpoint:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 a1_5_sft_b300/bin/torchrun --nproc_per_node 4 --master_port 29500 \
    -m alpamayo1_5_sft.train_hf \
    --config-path pkg://alpamayo1_5_sft/configs \
    --config-name sft_stage2_navsim \
    +trainer.seed=0 \
    model.stage1_vlm_checkpoint_path=/path/to/checkpoint-1660-merged \
    paths.output_dir=/path/to/output_stage2_navsim_seed0 \
    task_name=alpamayo_1_5_sft_stage2_navsim_seed0
```

Run seed1 the same way (`+trainer.seed=1`, distinct `output_dir`/`task_name`)
**after** seed0 finishes if both need to share the same reserved GPUs --
sequential, not parallel, when GPU 0-3 belong to someone else on a shared box.

### Batch size: Stage 1 vs Stage 2

Effective global batch is 4x larger in Stage 2 despite the opposite
per-GPU-batch direction, because Stage 1 does a full VLM forward+backward
(memory-heavy, never profiled on B300s -- started conservative) while
Stage 2 freezes the VLM and only trains the tiny diffusion action-expert head
(measured 65% memory headroom at batch 16, no grad accumulation needed):

| | Stage 1 (LoRA) | Stage 2 |
|---|---|---|
| `per_device_train_batch_size` | 1 | 16 |
| `gradient_accumulation_steps` | 4 | 1 |
| Effective global batch (4 GPUs) | 16 | 64 |

### The `[tok-recon]` diagnostic and the rank-0 straggler

`trainer.py`'s `ReasoningVLA_Trainer.compute_loss` prints a per-step
`[tok-recon] step N xy_mean=... xy_max=...` diagnostic (frozen-tokenizer XY
reconstruction error on the batch's GT future) -- meaningful for Stage 1
(discrete tokens ARE the prediction), meaningless for Stage 2 (prediction
comes from the continuous diffusion head, never touching that tokenizer). It
still ran unconditionally for both stages because the guard was only
`traj_tokenizer is not None` (still true for Stage 2 -- inherited from the
base checkpoint's architecture) and `"ego_future_xyz" in inputs` (always
true), not "which stage is this."

It's not free: it's rank-0-only synchronous CPU work (a Python for-loop
encode+decode over the whole batch, only on `is_world_process_zero()`), and
in a live Stage-2 run this showed up as a literal DDP straggler -- 5x
`nvidia-smi` samples on the running seed1 job showed rank 0's GPU idling at
0% while ranks 1-3 sat at 100%, consistent with the other 3 ranks waiting on
rank 0's extra per-step CPU work at the next all-reduce.

Fixed with a new toggle: `trainer.TrainingArguments.log_tokenizer_recon`
(default `True`, unchanged everywhere except where explicitly overridden).
Set `log_tokenizer_recon: false` in `sft_stage2_navsim.yaml`. Not yet set in
`sft_stage2_truckdrive.yaml`/`sft_stage2_nav.yaml` -- same argument applies
there if anyone hits it.

### `dataloader_num_workers`

Raised from 2 in `sft_base.yaml` (so it's the default everywhere, no
per-config overrides existed). These boxes have 192 cores and sit at ~15%
load even with other users' jobs running; each sample decodes several
full-res camera JPEGs per camera, exactly the per-sample CPU work more
workers parallelizes.

Tried 32 first -- a live 4-GPU Stage-2 seed0 run (128 total workers) stalled
for 20+ minutes: log frozen, one rank's 32 workers all `R`/CPU-busy but not
producing a batch, the other 3 ranks `S`/idle waiting at the collective. No
OOM in `dmesg`, `/dev/shm` had 1.8TB free -- looked like worker-count
oversubscription on a shared box (128 workers + another user's concurrent job
competing for the same cores/IO), not a fundamental deadlock. Backed off to
**16**. Re-check if ever run on a smaller/quieter box, and don't re-raise past
16 without confirming it holds up under a real multi-hour run first.

## 4. Stage-2 checkpoints on navhard: the existing path still applies

Once a Stage-2 checkpoint exists, it's `TrainableAlpamayoR1`
(`alpamayo_r1.*` format) -- state-dict-layout-identical to `Alpamayo1_5`, so
the *existing* `convert_checkpoint.py to-a15` + `eval_navsim.py` path (no new
code) works as documented in `navsim/docs/alpamayo_navsim.md`:

```bash
PYTHONPATH=src python3 scripts/convert_checkpoint.py to-a15 \
    --input  /path/to/output_stage2_navsim_seed0/checkpoint-417 \
    --output /path/to/output_stage2_navsim_seed0/checkpoint-417-A15-format
```

then run inference exactly as that doc's "Full run" section describes. Use
`eval_navsim_stage1.py` only for Stage-1-only checkpoints; use the
`to-a15`/`eval_navsim.py` path for Stage-2 checkpoints.

`run_eval_multigpu.py --eval_script eval_navsim.py --model_path <ckpt-dir>`
already defaulted its merged prediction parquet (`--out`) into
`<ckpt-dir>/eval_navsim/`, but its per-shard logs (`--logdir`) used to default
to a shared `/tmp/pai_av_val_shards` that every run overwrites regardless of
which checkpoint it was for. Fixed (2026-08-12) so `--logdir` follows the same
checkpoint-relative rule as `--out`, defaulting to
`<ckpt-dir>/eval_navsim/shard_logs` -- no need to pass `--logdir` by hand
anymore. `navsim_submission_from_parquet.py` / `run_pdm_score_from_submission.py`
still have no such default (generic NAVSIM infra, not Alpamayo-specific) --
keep pointing `--out`/`output_dir` at `<ckpt-dir>/eval_navsim/` explicitly for
those, as shown in `navsim/docs/alpamayo_navsim.md`'s own example.

### Gotcha: metric cache metadata bakes in absolute paths at build time

`MetricCacheLoader` (`navsim/common/dataloader.py`) doesn't rescan its cache
directory -- it reads a pre-built `metadata/*.csv` listing each token's
**absolute** cache file path, recorded once when the cache was built (via
`run_metric_caching.py`). Renaming or moving the directory afterward (e.g.
`$NAVSIM_EXP_ROOT` pointing somewhere new) breaks every entry with a silent
`FileNotFoundError` per token at scoring time, even though `metric_cache_path=`
on the CLI looks like it should control this -- it only controls where the
metadata CSV itself is read from, not the paths recorded inside it. Hit this
after `$NAVSIM_EXP_ROOT`'s target directory was renamed from `drivor_exp` to
`navsim_exp`: 100% of `metric_cache_navhard_two_stage`'s 5912 entries still
pointed at the old `drivor_exp` path. Fixed with a symlink
(`ln -s navsim_exp drivor_exp`) rather than regenerating the cache -- cheap,
additive, and doesn't touch the actual cached data. Re-run
`run_metric_caching.py` instead if a real rebuild is ever needed; don't rename
`$NAVSIM_EXP_ROOT`'s target directory again without expecting this.

### Eval invocation details (what actually ran, for the numbers below)

Neither `run_stage2_navhard_sequential.sh` (Stage-2, via `eval_navsim.py`) nor
the Stage-1 navhard run (`eval_navsim_stage1.py`) passed any of the flags
below -- so all three navhard runs used the tool defaults, not overrides:

- **CoC (chain-of-causation)**: generated. `eval_navsim.py --no_coc` is
  `action="store_true"` (default off), and it wasn't passed -- `coc_text` was
  populated from the model's own chain-of-causation trace for every row, not
  blanked to `""`.
- **Trajectory samples per token**: 6, not 1. `eval_navsim.py
  --num_traj_samples` defaults to `6` and wasn't overridden -- each navhard
  token got 6 sampled trajectories (medoid used for scoring), not a single
  greedy rollout.
- **`human_penalty_filter`**: on (`True`, the default in
  `navsim/planning/script/config/pdm_scoring/scorer/pdm_scorer.yaml`,
  unset anywhere in `score_stage2_navhard.sh`). Per `navsim/evaluate/pdm_score.py`
  (~line 171), this only fires when `scorer._config.human_penalty_filter and
  metric_cache.scene_type == SceneFrameType.ORIGINAL` -- i.e. only on stage-1
  "real" scenes (not stage-2 synthetic ones, which have no human-driven
  trajectory to compare against), and only affects the
  `driving_direction_compliance` sub-metric: the human's actual trajectory is
  simulated and scored too, so the ego isn't penalized for a direction-
  compliance mistake the human driver made on that same route.

Measured wall-clock, generation-only, full 5912-token navhard split (`run_eval_multigpu.py`'s "All shards done in..." line): seed0 70.0 min, seed1 70.9 min, on the reserved 4-GPU set (GPUs 4-7). Scoring (`run_pdm_score_from_submission.py`, 192 workers) adds ~2 min each -- call it **~72 min end-to-end per Stage-2 checkpoint**. For comparison, the Stage-1 navhard eval (`eval_navsim_stage1.py`, 8 GPUs) took ~3.4h generation + ~4 min scoring -- see its "Running it" section above for why (discrete-token autoregressive decoding is much slower per-sample than the diffusion head's parallel denoising steps).

## 5. Results (navhard_two_stage, 5912 tokens, 0 failed in all runs)

| Checkpoint | Eval path | Extended PDM Score |
|---|---|---|
| Constant-velocity baseline | -- | 0.1148 |
| Zero-shot Alpamayo 1.5 (`Alpamayo-1.5-10B`), greedy, no CoC | `eval_navsim.py`, diffusion | 0.1342 |
| Zero-shot Alpamayo 1.5, sampled (6/tok), CoC off | `eval_navsim.py`, diffusion | 0.1469 |
| Zero-shot Alpamayo 1.5, sampled (6/tok), CoC on | `eval_navsim.py`, diffusion | 0.1675 |
| Stage-2 seed0 (`checkpoint-417-A15-format`) | `eval_navsim.py`, diffusion | 0.2025 |
| Stage-2 seed1 (`checkpoint-417-A15-format`), sampled (6/tok) | `eval_navsim.py`, diffusion | 0.2188 |
| Stage-2 seed1 (`checkpoint-417-A15-format`), sampled (16/tok) | `eval_navsim.py`, diffusion | 0.2217 |
| Stage-2 seed0, 4 epochs (`checkpoint-1668-A15-format`), sampled (16/tok) | `eval_navsim.py`, diffusion | 0.2121 |
| Stage-2 seed0, 10 epochs (`checkpoint-4170-A15-format`) | `eval_navsim.py`, diffusion | 0.2068 |
| Stage-1 LoRA merged (`checkpoint-1660-merged`) | `eval_navsim_stage1.py`, discrete tokens | **0.2522** |

Notable: the Stage-1-only checkpoint (discrete trajectory tokens, no
diffusion action expert) scored *higher* than either Stage-2 seed on this
split -- both Stage-2 runs still clear the constant-velocity baseline by a
wide margin, but neither surpasses Stage-1 here. Worth revisiting if Stage-2
hyperparameters (LR, batch size, steps -- `checkpoint-417` is the only
checkpoint evaluated per seed, not a sweep) get tuned further; not
investigated further in this session.

**Medoid vs. sample-0 selection (2026-08-12).** Rescored the two 6-sample
zero-shot runs (CoC on/off) with `poses` taken from sample 0 instead of the
medoid -- no re-inference needed, just recomputed `poses` from the existing
`pred_xy_10hz` column via `eval_navsim.py`'s `to_navsim_poses` and rescored:

| Config | Selection | Extended PDM Score |
|---|---|---|
| CoC on | medoid | 0.1675 |
| CoC on | sample 0 | 0.1510 |
| CoC off | medoid | 0.1469 |
| CoC off | sample 0 | 0.1439 |

Medoid selection is worth ~0.017 EPDMS with CoC on but only ~0.003 with CoC
off. Consistent with CoC text itself varying sample-to-sample when CoC is
on (different reasoning -> more divergent trajectories -> more for medoid
to filter), while CoC-off samples only vary via the diffusion head's own
noise, a smaller source of spread.

The zero-shot release checkpoint (no SFT on this fork's NAVSIM data at all)
sits well below every fine-tuned checkpoint here, though still above the
constant-velocity floor -- confirms the SFT stages are doing real work on
this domain, not just matching a strong prior. Within the zero-shot rows the
ordering (CoC on > CoC off > greedy) matches the earlier warmup-split finding
in `navsim/docs/alpamayo_navsim.md` that CoC reasoning measurably helps, and
adds evidence that sampling (6 trajectories + medoid) also beats one greedy
rollout -- consistent with medoid selection filtering out worse samples
rather than just adding noise. Zero-shot checkpoint used:
`/home/rod/efs_ckpts/Alpamayo-1.5-10B-A15-format` (metadata-converted from
`Alpamayo-1.5-10B-A1-format` via `convert_checkpoint.py to-a15`, despite the
source directory's name -- its `config.json` had `alpamayo_r1.*` `_target_`
paths, i.e. real "A1" format, not already A15 as the name suggests). The
`--greedy` flag on `eval_navsim.py` (do_sample=False, forces
`--num_traj_samples 1`) and the underlying `do_sample` passthrough on
`Alpamayo1_5.sample_trajectories_from_data_with_vlm_rollout` did not exist
before this run -- added `alpamayo1.5/eval_navsim.py` and
`alpamayo1.5/src/alpamayo1_5/models/alpamayo1_5.py`. Note "greedy" only
makes the VLM's CoC/text rollout deterministic; the diffusion action head
still draws fresh noise per call, so the trajectory itself isn't fully
deterministic run-to-run even with `--greedy`.

Score CSVs: `<ckpt-dir>/eval_navsim/pdm_score/<timestamp>.csv` under each
checkpoint's own directory (Stage-1: on the second box's EFS copy,
`.../output_stage1_navsim_lora_checkpoint-1660-merged/eval_navsim/pdm_score/`;
Stage-2: `/home/rod/ckpts/alpamayo_navsim_finetuning/output_from_lora_sft_stage2_navsim_seed{0,1}/checkpoint-417-A15-format/eval_navsim/pdm_score/`;
zero-shot: `/home/rod/efs_ckpts/Alpamayo-1.5-10B-A15-format/eval_navsim/pdm_score_{coc,nococ,greedy}/`).

## 6. `num_traj_samples=16` re-eval, CoC on (2026-08-13)

Two Stage-2 checkpoints re-evaluated on navhard_two_stage with 16 trajectory
samples per scene instead of the default 6 (CoC left on -- that's already
`eval_navsim.py`'s default, no flag needed):

- `~/efs_ckpts/alpamayo_navsim_finetuning/output_from_lora_sft_stage2_navsim_seed1_checkpoint-417-A15-format`
  (already A15-format; had an existing default k=6 eval in the same
  `eval_navsim/` dir, scored 0.2188 -- the k=16 outputs use a `_k16` filename
  suffix, `navsim_navhard_two_stage_k16.parquet` /
  `submission_k16.pkl` / `pdm_score_k16/`, so as not to overwrite it).
- `~/efs_ckpts/alpamayo_navsim_finetuning/output_from_lora_sft_stage2_navsim_seed0_4_epochs_checkpoint-1668`
  -- still in raw `alpamayo_r1.*` format at the time of this run (unlike its
  `checkpoint-417` sibling, no pre-existing `-A15-format` copy), converted via
  `convert_checkpoint.py to-a15` first into a sibling
  `..._checkpoint-1668-A15-format` dir, smoke-tested (`--limit 8`, 1 GPU,
  NaN-pose check) before committing to the full run.

Launched via a single script,
`run_navhard_k16_two_ckpts.sh` (scratch, not checked into the repo), on 8
free GPUs, sequentially (seed1 k16 fully done -- inference + score -- before
starting the seed0_4epochs conversion/smoke-test/eval). Ran unattended via a
background subagent that monitored the log roughly hourly and reported step
transitions/results; completed cleanly (`ALL DONE`, both smoke test and full
runs 0 NaN / 0 failed across all 5912 tokens each), **~5h 20min** wall-clock
total (02:43 -> 08:03), longer than the ~3-4h estimate -- `num_traj_samples=16`
(vs. the usual 6) is the likely driver, on top of CoC still being on.

| Checkpoint | stage_one | stage_two | combined |
|---|---|---|---|
| Stage-2 seed1, k=16 | 0.5315 | 0.3686 | 0.2217 |
| Stage-2 seed0 4-epoch, k=16 | 0.5260 | 0.3793 | 0.2121 |

Both now recorded in the results table above.

## 7. Seed0, 10-epoch re-run (2026-08-13)

Trained `sft_stage2_navsim` seed0 for 10 epochs (`num_train_epochs=10`,
`+trainer.resume_from_checkpoint=true` after an interruption around step
2019 -- resumed cleanly from `checkpoint-2000`, see the resume-support work
in `train_hf.py`/`sft_base.yaml`). Final checkpoint `checkpoint-4170`
(global_step 4170, epoch 10.0). Converted to A15, smoke-tested (`--limit 8`,
0 NaN), then full navhard_two_stage run on GPUs 4-7 (default k=6 samples,
CoC on): 70.3 min inference + ~2 min scoring, 5912/5912 succeeded.

| | stage_one | stage_two | combined |
|---|---|---|---|
| Stage-2 seed0, 10 epochs (`checkpoint-4170-A15-format`) | 0.5142 | 0.3697 | **0.2068** |

Not monotonic with epoch count on seed0: 1 epoch (`checkpoint-417`) 0.2025 ->
4 epochs (`checkpoint-1668`) 0.2121 -> 10 epochs (`checkpoint-4170`) 0.2068.
4 epochs is the best of the three; 10 epochs gives back some of that gain --
consistent with overfitting past ~4 epochs on this dataset size, though this
is one checkpoint per epoch-count on one seed, not a sweep, so not confirmed.
Still clears the constant-velocity baseline (0.1148) comfortably and is in
the same range as seed1 (0.2188) and Stage-1 (0.2522), just not better than
either.

Takeaways:
- Going from k=6 to k=16 moved seed1's combined score only marginally
  (0.2188 -> 0.2217, +0.0029) -- most of medoid selection's benefit is
  apparently already captured at k=6; diminishing returns past that on this
  split.
- The seed0 4-epoch checkpoint's first-ever navhard number (0.2121) sits
  between seed0's original single-epoch result (0.2025) and seed1's k=16
  result (0.2217) -- more epochs helped seed0, but seed1 (fewer epochs) still
  edges it out here, so epoch count alone isn't the deciding factor between
  these two runs.
- stage_one and stage_two individually (~0.53 / ~0.38) are both well above
  their respective combined contributions -- consistent with the zero-shot
  numbers above, `combined` is not a simple average of the two stage scores.

**Gotcha: `--npz_dir` moved.** The path baked into earlier shard logs for
this checkpoint family, `/opt/dlami/nvme/rod/datasets/navsim_npz/navhard_two_stage`,
no longer exists on this box. The current canonical location (5912 `.npz`
files, verified 2026-08-13) is
`/opt/dlami/nvme/rod/datasets/navsim/npz_exports/navhard_two_stage` -- matches
the S3-bundled `npz_exports/` layout described in
`navsim/docs/alpamayo_navsim.md`. Unclear when/why the old path stopped
existing; if a future run can't find `--npz_dir`, check for another rename
before assuming data loss.
