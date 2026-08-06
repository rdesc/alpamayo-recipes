# Alpamayo 2 Super -- LoRA SFT (trajectory-only)

LoRA fine-tuning of the released **Alpamayo 2 Super** (32B Qwen3-VL backbone +
2B diffusion expert) on PhysicalAI-AV clips. Supervises the VLM's discrete
future-trajectory tokens only -- no ground-truth Chain-of-Causation text is
required, matching this fork's existing Alpamayo-1.5 TruckDrive trajectory-only
recipes.

Verified end to end (2026-08-05) against the real `nvidia/Alpamayo2-Super`
checkpoint on a single GPU: real PhysicalAI-AV clips, LoRA injection, a 3-step
HF `Trainer` run with finite/decreasing loss, split-loss logging, a W&B
trajectory-viz callback with a true autoregressive rollout (offline-mode
tested), and a full adapter save -> reload -> merge round trip.

## What this recipe does

- Loads the release checkpoint directly via `Alpamayo2Super.from_pretrained`
  (no "A1-format" conversion step -- unlike Alpamayo-1/1.5, Alpamayo 2 Super
  ships as a native HF `PreTrainedModel`).
- Injects LoRA adapters into `model.vlm` only (`peft.inject_adapter_in_model`,
  targeting the Qwen3-VL attention/MLP projections), leaving the ~2B diffusion
  expert exactly as trainable as the checkpoint's own config makes it
  (`cotrain_expert_vlm: false` in the release config already freezes the VLM
  and leaves the expert fully trainable -- LoRA composes with that for free).
- Builds a teacher-forced (`generation_mode=False`) conversation via
  `alpamayo2_super.chat_template.conversation.build_conversation`, supervising
  only the requested components (default: `["traj_future"]`) with a
  `labels_mask` built the same way `alpamayo-recipes`' own
  `get_label_mask.py` does for Alpamayo-1.5 (locate the paired
  `<label>_start`/`<label>_end` special tokens in `input_ids`).
- Trains with `Alpamayo2SFTTrainer` (`trainer.py`), a thin `transformers.Trainer`
  subclass -- the model's `forward(tokenized_data, traj_data, labels_mask)`
  signature matches the batch dict shape directly, so the only reason for a
  subclass is to surface the `loss_future_traj`/`loss_others` split (see next
  point) to the console/W&B each logging step, mirroring
  `alpamayo1_5_sft.trainer.ReasoningVLA_Trainer`.
- `Alpamayo2SuperModelOutput` (in the `alpamayo2` repo) now carries
  `loss_future_traj`/`loss_others` alongside `loss` -- a small, additive,
  non-breaking field addition mirroring Alpamayo-1.5's `ReasoningVLAOutput`.
  This is *not* optional plumbing: `fuse_traj_tokens` mutates `input_ids` on a
  local copy inside `forward()`, so the split cannot be reconstructed correctly
  from outside the model.
- Optional W&B trajectory visualization (`viz_callback.py`, off by default --
  `viz.enabled=true`): renders GT-vs-predicted BEV plot + the 6-camera/4-frame
  image grid in one figure, logged as a `wandb.Image`. The prediction is a
  **true autoregressive rollout** (`rollout.py`): `model.vlm.generate()` over
  the discrete future-trajectory-token span with the trajectory-token vocab
  left *unmasked* (the opposite of the release model's
  `sample_trajectories_from_data`, which masks it out and always routes the
  real trajectory through the frozen diffusion expert instead -- not
  something this recipe trains). Each token is conditioned on the model's own
  previously generated tokens, not ground truth, so it reflects what the
  model actually predicts on its own -- verified to differ substantially from
  a teacher-forced decode of the same sample (~0.01m vs. 0.8-1.9m ADE; see
  git history for the teacher-forced version this replaced). Also plots the
  frozen tokenizer's encode->decode reconstruction of the GT future (the
  "ceiling" a discrete-token prediction can approach but not beat), same
  concept as `alpamayo1_5_sft`'s viz. Supports multiple sampled rollouts per
  input (`viz.num_traj_samples`, default 2), each plotted as its own curve.

## Hardware assumptions

Validated on 1x and 8x 275GB-class GPUs (NVIDIA B300) with gradient
checkpointing on `model.vlm` (not optional at this model scale --
`per_device_train_batch_size=2` without it OOM'd even on a single 275GB GPU).

Multi-GPU uses plain PyTorch DDP via `torchrun`, not DeepSpeed. This
deliberately does NOT match `alpamayo1_5_sft`'s Stage-1-LoRA precedent (the
more directly analogous case, since it also LoRA-trains the VLM backbone),
which uses DeepSpeed ZeRO-2 successfully -- but Stage 1's model class has no
expert/action-head submodule at all, so it never hits the issue this recipe
has: the diffusion expert's ~2.4B trainable params are *structurally* never
in the loss graph (see below), and there's no verified evidence DeepSpeed
handles a permanently-unused trainable parameter group as cleanly as plain
DDP's `find_unused_parameters` does. Rather than assume it would, this
recipe sticks to the empirically-verified plain-DDP path. A
`configs/deepspeed/zero2.json` is still included (copied from
`alpamayo1_5_sft`, model-agnostic) if someone wants to actually test that.

```bash
torchrun --nproc_per_node=8 train_hf.py \
  data.train_dataset.manifest_path=/path/to/train_manifest.json \
  data.val_dataset.manifest_path=/path/to/val_manifest.json \
  paths.output_dir=./output_sft
```

`trainer.ddp_find_unused_parameters=true` is already the default in
`sft_base.yaml` -- required because the diffusion expert's ~2.4B trainable
params never receive a gradient from this VLM-only loss, which would
otherwise hang DDP waiting to sync them. Verified end to end on 8x GPU
(2026-08-05): LoRA injection, gradient checkpointing, split-loss logging, and
adapter checkpointing all behave identically to the single-GPU path, with
only rank 0 logging/saving.

## Install

```bash
cd recipes/alpamayo2_sft
source /mnt/efs/users/rod/repos/alpamayo2/env.sh   # sets CUDA_HOME -- required, flash-attn/deepspeed build against it
export UV_PROJECT_ENVIRONMENT=a2_sft               # named after the package, matching alpamayo1_5_sft's a1_5_sft convention
uv sync --locked
source a2_sft/bin/activate
```

Verified (2026-08-05): builds cleanly end to end (109 packages, including
`flash-attn` and `deepspeed` compiled from source) and every recipe module
(`alpamayo2_sft.*`) plus the shared `alpamayo.*` (from `alpamayo-recipes`'
`src/`) import with no `PYTHONPATH` hacks needed -- this venv is fully
self-contained, unlike earlier testing in this recipe's development which
borrowed the `alpamayo2` repo's own `.venv` with manually `pip install`'d
extras.

`[tool.uv.sources]` points `alpamayo2_super` at `../../../alpamayo2` (a sibling
checkout of this `alpamayo2` repo) as an editable path dependency, not a git
dependency -- both repos are under active local iteration. Switch to a
`git = "..."` source once the model-side changes (if any) are pushed and
stable, matching every other recipe's convention
(`alpamayo_r1 = { git = "https://github.com/NVlabs/alpamayo.git" }`).

You also need Hugging Face access to `nvidia/Alpamayo2-Super` and
`nvidia/PhysicalAI-Autonomous-Vehicles` (`hf auth login`), same as the
`alpamayo2` repo's own inference README.

## Required inputs

| Input | Where it comes from |
|---|---|
| Checkpoint | `nvidia/Alpamayo2-Super` (HF hub id, or a local release-native checkpoint dir) |
| Training clips | A JSON manifest with `{"samples": [{"clip_id": ..., "t0_us": ...}, ...]}` -- the `alpamayo2` repo ships `examples/validation_samples.json` (5 clips) and `examples/public_golden_validation_samples.json`, both usable as-is for a smoke test. **Not a real training set** -- see Known limitations. |

## Commands

Smoke test (a few steps, proves the pipeline works before a real run):

```bash
python train_hf.py \
  data.train_dataset.manifest_path=/path/to/alpamayo2/examples/validation_samples.json \
  data.val_dataset.manifest_path=/path/to/alpamayo2/examples/validation_samples.json \
  +trainer.max_steps=3 \
  paths.output_dir=./output_smoke
```

Real run (once you have a real clip manifest):

```bash
python train_hf.py \
  data.train_dataset.manifest_path=/path/to/train_manifest.json \
  data.val_dataset.manifest_path=/path/to/val_manifest.json \
  paths.output_dir=./output_sft
```

Enable W&B + trajectory viz:

```bash
python train_hf.py \
  data.train_dataset.manifest_path=/path/to/train_manifest.json \
  data.val_dataset.manifest_path=/path/to/val_manifest.json \
  +wandb=default wandb.key=$WANDB_API_KEY trainer.report_to=wandb \
  viz.enabled=true viz.every_n_steps=100 \
  paths.output_dir=./output_sft
```

`+wandb=default` loads `configs/wandb/default.yaml` (edit `team`/`project`
there, or override on the CLI). Logged scalars: `loss`, `loss_future_traj`,
`loss_others` every `logging_steps`; `viz/val_sample_<idx>` (image) and
`val/ade_viz_mean` every `viz.every_n_steps`. Tested with `WANDB_MODE=offline`
(no real credentials needed) -- `wandb.key=null` skips `wandb.login`.

## Expected outputs

- `<output_dir>/checkpoint-<step>/` -- full HF `Trainer` checkpoint (base model
  + LoRA-wrapped state, i.e. `...base_layer.weight` + `lora_A`/`lora_B` keys).
- `<output_dir>/checkpoint-<step>/adapter/` -- adapter-only artifacts written by
  the `LoraSaveCallback` (`adapter_model.safetensors`, `adapter_config.json`,
  `alpamayo_lora.json`). Use `lora.py`'s `load_lora_settings` +
  `load_lora_adapter` (inject LoRA into a fresh model first) or
  `merge_lora_weights` to fold the adapter into a plain checkpoint for
  inference -- both round-trip verified.

## Known limitations

- **Toy dataset only.** `Alpamayo2TrajectoryDataset` reads a flat JSON manifest
  of a handful of clips, not a full local PAI chunk directory. Scaling to a
  real training set means either adding chunk-directory enumeration (mirroring
  `alpamayo.data.pai.PAIDataset` / `PhysicalAIAVDatasetLocalInterface` from
  `alpamayo-recipes`' shared `src/alpamayo/`) or a TruckDrive-style loader
  (mirroring `alpamayo.data.truckdrive`), matching this fork's existing
  Alpamayo-1.5 TruckDrive work.
- **VLM-side loss only.** `Alpamayo2Super.forward()` only computes the
  discrete-trajectory-token + text CE loss through the VLM. The diffusion
  expert's own loss (`model.expert(traj_data, vlm_outputs, mask)`, verified
  separately to have a real, working training-mode forward pass) is not wired
  into this Trainer -- the expert stays at whatever trainability the
  checkpoint config gives it (fully trainable, but never actually trained by
  this recipe yet).
- **No Chain-of-Causation supervision.** `label_components` defaults to
  `["traj_future"]` only. Add `"cot"` and populate `data["cot"]` with real GT
  reasoning text (e.g. from a PAI `ood_reasoning.parquet`'s `coc` field) to
  supervise text generation too.
- **No minADE eval loop.** `alpamayo1_5_sft`'s `ValMinADECallback` equivalent
  (a representative-subset metric distinct from the fixed viz set) is not
  wired in yet -- only the viz callback's per-sample best-of-K ADE scalar
  (`val/ade_viz_mean`).
- **Rollout viz is real generation, so it's slow.** ~15-25s per sample per
  rollout on a single GPU in local testing (128 autoregressive steps) --
  budget `every_n_steps`/`num_samples`/`num_traj_samples` accordingly. The
  expert's continuous rollout (`sample_trajectories_from_data`) is still not
  visualized here, since the expert isn't trained by this recipe (see above).
- **DeepSpeed config included but untested.** `configs/deepspeed/zero2.json`
  is copied over (generic, model-agnostic) for if ZeRO sharding is ever
  needed, but this recipe uses plain DDP for multi-GPU (see Hardware
  assumptions) and has never actually run with DeepSpeed wired in.
