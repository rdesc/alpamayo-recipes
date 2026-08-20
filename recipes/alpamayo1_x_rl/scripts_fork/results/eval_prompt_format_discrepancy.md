# Discrepancy: our fork's OOD-reasoning eval numbers vs. `~/repos/alpamayo1.5`'s published numbers

## Symptom

Our fork's baseline evals over the PAI-AV OOD reasoning subset (`baseline_ood_reasoning.md`
[discrete-token, `alpamayo1_x_rl`] and `alpamayo1_5_sft/scripts_fork/results/baseline_ood_reasoning_actionhead.md`
[action-head]) report minADE₆ roughly **1.5-2.6x worse**, at every horizon, than
`~/repos/alpamayo1.5/docs/first_round_eval_results.md`'s "Alpamayo first-round eval" table --
despite both covering the *same* 349-event val split (same `reasoning/ood_reasoning.parquet`,
same `split == 'val'`), same K=6, same nominal checkpoint family, same base processor
(`Qwen/Qwen3-VL-2B-Instruct`).

| horizon | ours (discrete-token, val) | ours (action-head, val) | `alpamayo1.5` published (val, CoC) |
|---|---|---|---|
| minADE₆ @1s | 0.058 | 0.064 | 0.0247 |
| minADE₆ @3s | 0.392 | 0.452 | 0.2241 |
| minADE₆ @5s | 0.891 | -- | 0.6059 |
| minADE₆ @6s | -- | 1.421 | 0.8528 |

(`--` = that horizon bucket wasn't computed by that run; see `val_metrics.py`'s 2026-08-18
horizon change, 5.0s -> 6.0s, noted in `baseline_ood_reasoning_actionhead.md`.)

n=349 val events matches exactly on both sides, confirming it's the same underlying event set,
not a different sample.

## Leading hypothesis: prompt-format mismatch (camera-identity text)

Our fork's `scripts_fork/eval_baseline_*.py` scripts build the VLM prompt via
`alpamayo_r1.helper.create_message(frames)` -- imported from the *shared, older* `alpamayo_r1`
package used across this repo's recipes. The official `~/repos/alpamayo1.5/eval_pai_av_val.py`
(the script that produced the published numbers above) instead uses
`alpamayo1_5.helper.create_message(frames, camera_indices=data["camera_indices"], ...)` -- a
different, 1.5-specific helper.

The two are structurally identical (same system/assistant messages, same
`<|traj_history_start|>...`, same final prompt text, same `<|cot_start|>` sentinel) **except**
for how the image content is built:

- `alpamayo_r1.helper.create_message`: `[{"type": "image", "image": frame} for frame in frames]`
  -- raw frames, no camera identity, no frame index.
- `alpamayo1_5.helper.create_message` -> `_build_image_content(frames, camera_indices, ...)`:
  when `camera_indices` is passed, prepends text like `"Front left camera: "` before each
  camera's first frame and `"frame {i} "` before every frame -- **and its own docstring says
  this is "matching the format used during model training."**

The data needed to fix this is already being loaded and silently discarded: `alpamayo_r1.load_physical_aiavdataset`
(what our eval scripts already call) returns `data["camera_indices"]` -- our scripts just never
pass it anywhere, because `alpamayo_r1.helper.create_message` has no parameter to accept it.

Corroborating evidence that Alpamayo 1.5 training itself expects camera annotations:
`recipes/alpamayo1_5_sft/eval_navsim_stage1.py` builds its prompts via
`alpamayo.processor.qwen_processor.get_preprocess_data_fn_from_model_config(..., include_camera_ids=True, include_frame_nums=True, ...)`
-- the actual training-time preprocessing path -- with those two flags explicitly turned on. So
the *training* pipeline includes camera-id/frame-number conditioning; our `scripts_fork` eval
scripts, via `alpamayo_r1.helper`, do not. That's a real train/inference prompt mismatch, not
just a different metric definition.

## What this does and doesn't explain

- Both of our fork's OOD-reasoning eval scripts (discrete-token and action-head) import the same
  `alpamayo_r1.helper.create_message`, so this affects both equally -- the **relative**
  discrete-token-vs-action-head comparison in `baseline_ood_reasoning_actionhead.md` is likely
  still valid.
- It does **not** explain everything by itself until verified experimentally -- logged here as
  the leading hypothesis, not a confirmed root cause. Other candidate contributors not yet ruled
  out: exact checkpoint weights (`Alpamayo-1.5-10B-rl-training`, the RL recipe's converted
  training-ready checkpoint, vs. `nvidia/Alpamayo-1.5-10B` pulled fresh from HF hub by the
  official script) -- though `vlm_name_or_path`/`vlm_backend`/architectures match across all
  three local checkpoint format copies we have (A1-format, A15-format, rl-training), so a gross
  weight mismatch looks unlikely.

## Status: CONFIRMED as a major contributor (not the sole cause)

Ran `scripts_fork/eval_baseline_ood_reasoning_camprompt.py` -- a variant of
`eval_baseline_ood_reasoning.py` (same checkpoint, same `RLWrapperReasoningVLA`, same K=6, same
val split, same everything) with ONLY the prompt-construction changed: images are annotated with
camera-identity/frame-index text (mirroring `alpamayo1_5.helper._build_image_content`) instead of
being passed as raw unlabeled frames. Full 349-event val split, 4-way sharded, 2,094/2,094 rows ok.

### Aside: unrelated infra bug hit along the way

The `alpamayo1_x_rl` recipe's default venv (`a1x_rl`) failed on **every single event** with
`nvrtc: error: invalid value for --gpu-architecture (-arch)` on this B300 box -- confirmed
unrelated to this prompt change (the *original*, unmodified `eval_baseline_ood_reasoning.py`
fails identically through `a1x_rl` right now, on two different GPUs). Root cause: `a1x_rl`
vendors `nvidia-cuda-nvrtc-cu12` 12.8, which doesn't recognize this GPU's architecture string when
JIT-compiling a reduction kernel (AOT-compiled kernels are unaffected, so model load and most of
the forward pass work fine -- it's specifically the JIT path that breaks). `alpamayo1_5_sft`'s
`a1_5_sft` venv (nvrtc-builtins 12.9) and `alpamayo1_x_rl`'s own `a1x_rl_b300` venv (nvrtc 13.0,
purpose-built for this hardware) are both unaffected. All camera-prompt numbers below were run
through `a1x_rl_b300`. Anyone else hitting `nvrtc: invalid value for --gpu-architecture` on
`alpamayo1_x_rl`'s default venv on a B300 box should just switch to `a1x_rl_b300`.

### Results: val split (349 events, K=6)

| metric | original (no camera ID) | camera-annotated prompt | official `alpamayo1.5` (CoC, val) | gap closed |
|---|---|---|---|---|
| minADE₆ @1.0s | 0.0580 | 0.0392 | 0.0247 | ~57% |
| minADE₆ @3.0s | 0.3921 | 0.3037 | 0.2241 | ~53% |
| min_ade (full, 6.4s) | 1.3683 | 1.0549 | 0.8528 @6.0s (not directly comparable, shorter horizon) | ~39% |
| ADE mean (full, 6.4s) | 2.7775 | 2.0950 | n/a (not reported at this horizon) | -- |
| coc_binary | 0.5697 | 0.5654 | 0.587 (CAC v2 val, different scorer version) | ~flat |
| coc_graded | 0.5759 | 0.5776 | -- | ~flat |

**The camera-identity/frame-index prompt annotation is a real, substantial contributor** --
closing roughly half the gap to the official published numbers at every horizon checked (minADE
dropped 23-32% depending on horizon, ADE mean dropped 25%). This confirms the
train/inference prompt-format mismatch hypothesis above.

**It is not the complete explanation.** A ~1.3-1.6x gap remains even with the camera-annotated
prompt (e.g. @1.0s: 0.039 ours vs. 0.025 official; @3.0s: 0.304 vs. 0.224). As expected,
CoC-action-consistency barely moved (0.570 -> 0.565 binary) -- it grades the model's reasoning
text against its own trajectory, and that self-consistency isn't sensitive to how accurate the
trajectory is in an absolute sense.

### Second contributor found: minADE selection-convention mismatch (explains the residual gap)

Checkpoint-weight identity was checked and ruled out: spot-compared overlapping `vlm.*` tensors
(embeddings, attention layers, vision tower, lm_head) between `Alpamayo-1.5-10B-rl-training` and
the local `Alpamayo-1.5-10B-A1-format` checkpoint (release-derived, `scripts/convert_checkpoint.py
to-a1`) -- every sampled tensor is `torch.equal` bit-identical. `alpamayo_r1.load_physical_aiavdataset`
vs. `alpamayo1_5.load_physical_aiavdataset` was also checked -- the two files are identical modulo
a copyright year and one docstring word. Neither data loading nor checkpoint weights explain the
residual gap.

The actual second cause: **`alpamayo.metrics.distance_metrics.compute_minade`** (what our
`rewards/val_metrics.py` calls, and what `alpamayo.metrics.metric_api.DistanceMetrics` -- the
actual SFT/RL training-time val metric, i.e. every `val/min_ade` curve logged during training in
this repo -- also calls) selects its best-of-K trajectory **once, using the full 6.4s
trajectory's mean error**, then reports that SAME trajectory's error truncated to each shorter
horizon:

```python
# src/alpamayo/metrics/distance_metrics.py, compute_minade
idx = l2.mean(dim=-1).argmin(dim=2)                       # best-of-K by FULL-length error
min_diff = torch.take_along_dim(l2, idx[...], dim=2)       # then truncate THAT pick per horizon
```

`~/repos/alpamayo1.5/eval_common.py`'s `clip_metrics` (what produced the official published
numbers) instead **re-selects the best-of-K independently at every horizon**:

```python
# eval_common.py, clip_metrics
for h in ADE_HORIZONS:
    out[f"min_ade_{h}s"] = float(err[:, :idx].mean(1).min())   # best-of-K re-picked AT h
```

Per-horizon best-of-K selection can only match or beat a single fixed full-trajectory pick, at
every horizon shorter than the full trajectory -- so this is a systematic, one-directional bias,
consistent with what we observed (our numbers worse at *every* horizon, by a roughly similar
proportion at each). This is not a bug isolated to our eval scripts; `compute_minade` is a shared,
repo-wide library function -- this convention is baked into every `val/min_ade` metric this repo
logs during SFT and RL training, not something introduced by `scripts_fork`.

**Combined picture:** the original gap had (at least) two real, independent causes -- a
train/inference prompt-format mismatch (fixed, see above, closes roughly half the gap) and a
genuine metric-definition difference between this repo's shared `compute_minade` and the
standalone `~/repos/alpamayo1.5` benchmark script's per-horizon selection (not fixed here --
would require either patching `compute_minade` or accepting the two numbers are answering
subtly different questions and are not directly comparable at face value). Neither checkpoint
weights nor the data-loading path contribute to the gap -- both were checked and are identical.

### Confirmed experimentally (not just argued): `scripts_fork/eval_metric_convention_compare.py`

Ran the camera-prompt-fixed model over the full 349-event val split a second time, this time
computing BOTH conventions on the exact same raw predicted trajectories per event: our
`compute_minade` vs. `~/repos/alpamayo1.5/eval_common.clip_metrics` imported directly (pure
numpy/pandas, no reimplementation risk). 349/349 events ok.

| stage | @1.0s | @3.0s | @6.0s |
|---|---|---|---|
| original (no camera prompt, our metric) | 0.058 | 0.392 | -- |
| + camera-prompt fix (still our metric) | 0.039 | 0.304 | -- |
| + same predictions, **official metric convention** | **0.026** | **0.249** | **0.937** |
| officially published (`first_round_eval_results.md`) | 0.0247 | 0.2241 | 0.8528 |

Switching *only* the metric convention (same checkpoint, same camera-annotated predictions, same
raw trajectories) takes us to within ~5-11% of the published numbers at every horizon checked --
down from the original 1.5-2.6x gap. This confirms the two identified causes (prompt format +
metric convention) together explain nearly all of the original discrepancy.

**Residual ~5-11% gap, same direction at every horizon:** small enough to plausibly be
sampling/seed variance (different RNG realizations of a stochastic K=6 draw) rather than a third
systematic cause, but not proven -- not chased further here.

Raw data: `/opt/dlami/nvme/rod/results/metric_convention_compare/results.shard{0..3}-of-4.parquet`.
Script: `scripts_fork/eval_metric_convention_compare.py`.

### Raw data

- Camera-annotated run: `/opt/dlami/nvme/rod/results/baseline_ood_reasoning_camprompt/baseline.shard{0..3}-of-4.parquet`
- Script: `scripts_fork/eval_baseline_ood_reasoning_camprompt.py`
