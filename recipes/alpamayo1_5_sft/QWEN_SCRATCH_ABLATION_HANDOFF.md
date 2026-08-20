# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
# Handoff: Qwen-from-scratch pretraining ablation

**Goal:** measure how much of Alpamayo-1.5's TruckDrive performance comes from
NVIDIA's physical-AI VLM pretraining vs. the TruckDrive fine-tune itself, by
running the *same* Stage-1 TruckDrive SFT recipe but starting from a vanilla
HF `Qwen/Qwen3-VL-*-Instruct` checkpoint instead of the Alpamayo-1.5-pretrained
one. Stage 1 (discrete-trajectory-token VLM SFT) only -- no Stage-2 action
head, to keep the ablation simple.

**STATUS (2026-08-20): running and training on 8xA100.** The B300 hardware
wall is gone, the venv is repaired, and the run is past first loss and
learning. See "Current run" below. The rest of this doc is now a record of
what was fixed plus what the *next* person needs to watch.

---

## Current run

- **Node:** `ip-10-225-141-207` (8x A100-SXM4-80GB, compute cap 8.0, system
  CUDA at `/usr/local/cuda`, 1121 GB host RAM).
- **Backbone:** `Qwen/Qwen3-VL-8B-Instruct`, deepspeed **on** (recipe default).
- **wandb:** https://wandb.ai/rdesc1-milaquebec/alpamayo/runs/fbutalhs
- **Output dir:** `output_truckdrive_qwen_scratch/`
- **Log:** `logs/qwen_scratch_20260819_234502.log`
- **Launch script (exact command used):**
  `/tmp/claude-1001/.../scratchpad/launch_qwen_scratch.sh` -- reproduced under
  "How to launch" below; copy it somewhere durable, the scratchpad is
  session-scoped.

Health at ~step 80:

| metric | value |
|---|---|
| step-0 `ValMinADE` (K=6, n=248) | mean **76.135 m**, median 87.416, p95 96.246 |
| step-0 `eval_loss` | 23.338 |
| first logged `loss` (step 5) | 22.638, `grad_norm` 732 |
| `loss` at step ~80 | **~4.9-5.6**, `grad_norm` ~18 |
| throughput | ~6-8 s/it |
| host RAM | ~291 GB / 1121 GB, stable |

The step-0 numbers are the *untrained-backbone baseline* and are exactly what
you want for this ablation: a from-scratch Qwen emits essentially random
trajectory tokens (76 m minADE). Step 0 reproduced bit-identically across two
separate launches, so it's deterministic and safe to quote. The fast
22.6 -> 5.0 loss drop in 80 steps is the model learning the trajectory-token
format, not yet driving skill -- judge the ablation on `ValMinADE`, not loss.

**Full run is 12357 steps (3 epochs) ≈ 24-27 h wall clock.** `save_steps=500`,
`eval_steps=500`.

---

## What's built

1. **`models/sft_base_model.py`** -- added
   `TrainableReasoningVLA.from_scratch_vlm()`. Unlike
   `from_alpamayo_checkpoint` (builds an empty VLM shell and overwrites it
   with Alpamayo's fine-tuned weights), this calls the upstream
   `ReasoningVLA.from_pretrained_submodules()` path, which loads **real
   vanilla HF Qwen weights** via `Qwen3VLForConditionalGeneration.from_pretrained(vlm_name_or_path, ...)`
   -- no Alpamayo pretraining anywhere in the loop. It reads the
   trajectory-tokenizer *codec* (a fixed kinematic binning scheme --
   `DiscreteTrajectoryTokenizer` / `DeltaTrajectoryTokenizer`, not a learned
   network) from an existing Alpamayo checkpoint's `config.json` so the label
   format matches the rest of the recipe; that checkpoint's actual weights
   are never loaded.

2. **`configs/models/ar1_5_base_qwen_scratch.yaml`** -- Hydra model config
   wiring `from_scratch_vlm`. `vlm_name_or_path` defaults to
   `Qwen/Qwen3-VL-8B-Instruct` and is freely overridable on the CLI for
   smaller/larger Qwen3-VL siblings, e.g.
   `model.vlm_name_or_path=Qwen/Qwen3-VL-2B-Instruct`.

3. **`configs/sft_truckdrive_qwen_scratch.yaml`** -- top-level training
   config. Identical to `sft_truckdrive.yaml` (same data/viz/val_metric/
   trainer wiring), just overrides `/models@model: ar1_5_base_qwen_scratch`.

**Verified correctness** (CPU, with `Qwen/Qwen3-VL-2B-Instruct`): instantiated
`from_scratch_vlm` and confirmed vocab size, every entry in
`special_token_ids`, and the trajectory tokenizer class all match byte-for-byte
what `from_alpamayo_checkpoint` produces from the real Alpamayo checkpoint. So
the TruckDrive dataset/collate/viz/val_metric code needs zero further changes
-- the scratch-Qwen model is a drop-in swap.

**Verified end-to-end on A100:** checkpoint load, token-embedding resize,
S3 dataset load, deepspeed init, step-0 eval + viz, and sustained
forward/backward with a decreasing loss.

---

## How to launch

```bash
cd recipes/alpamayo1_5_sft
source a1_5_sft/bin/activate
export CUDA_HOME=/usr/local/cuda
export WANDB_ENTITY=models
R=$PWD

torchrun --nproc_per_node 8 \
  -m alpamayo1_5_sft.train_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_truckdrive_qwen_scratch \
  trainer.dataloader_num_workers=2 \
  data.train_dataset.data_root=s3://torc-data/datasets/TruckDrivePublic \
  data.train_dataset.backend=s3 \
  data.train_dataset.metadata_cache=/mnt/efs/users/rod/truckdrive_cache/meta_5views.pkl \
  data.train_dataset.split_metainfo=$R/truckdrive/truckdrive_metainfo.json \
  data.val_dataset.data_root=s3://torc-data/datasets/TruckDrivePublic \
  data.val_dataset.backend=s3 \
  data.val_dataset.metadata_cache=/mnt/efs/users/rod/truckdrive_cache/meta_5views.pkl \
  data.val_dataset.split_metainfo=$R/truckdrive/truckdrive_metainfo.json
```

`trainer.dataloader_num_workers=2` is **required**, not optional -- see
"Host RAM OOM" below. Keep deepspeed **on** (the recipe default) on A100.

To compare against a smaller backbone, override
`model.vlm_name_or_path=Qwen/Qwen3-VL-2B-Instruct` (or `-4B-Instruct`).

---

## Traps hit on the A100 node, and their fixes

### 1. Host RAM OOM -- the big one

The first A100 launch died with all 8 ranks `SIGKILL`ed (exitcode -9) about
45 min in, during the step-0 eval. Not GPU memory -- the **kernel OOM killer**
(confirmed in `journalctl -k`: `Out of memory: Killed process ... (pt_data_worker)`,
26.9 GB anon-rss each).

Cause: the recipe default `dataloader_num_workers: 16`, times 8 ranks, is
**128 worker processes**, each forked from a ~21 GB-RSS parent. That exceeds
1121 GB of host RAM. `dataloader_num_workers=2` (16 workers total) holds
steady at ~291 GB and costs nothing in throughput here, because the loop is
S3-latency-bound rather than worker-bound.

**This is a landmine for any 8-rank run of this recipe on this box, not just
the ablation.** If a TruckDrive run ever dies with exitcode -9 and no Python
traceback, this is why.

### 2. `wandb.util.generate_id` is gone in wandb 0.28.x

`src/alpamayo/common/wandb_utils.py` calls `wandb.util.generate_id()` when
starting a *new* run; that attribute was removed in the wandb 0.28 line, so a
fresh run dies with
`AttributeError: module 'wandb.util' has no attribute 'generate_id'`.
Existing runs are unaffected because they take the resume branch.

**Workaround used (no upstream edit):** `init_wandb` reads the run id from
`<output_dir>/.wandb_id` if present and calls `wandb.init(..., resume="allow")`,
so pre-seeding that file with any unused 8-char id skips `generate_id`
entirely and wandb creates the run on first contact:

```bash
mkdir -p output_truckdrive_qwen_scratch
python -c "import random,string;print(''.join(random.choices(string.ascii_lowercase+string.digits,k=8)))" \
  > output_truckdrive_qwen_scratch/.wandb_id
```

**Do this for every fresh output dir.** A permanent fix would be patching
`wandb_utils.py` to fall back when `generate_id` is missing (needs the fork
NOTE header per repo convention) -- not done, to keep the upstream file clean.

### 3. venv corruption -- fixed, and the fix is on EFS so it follows you

The corruption described in the previous handoff was real and did follow to
this node (`a1_5_sft/` is on EFS): `deepspeed` was missing `__init__.py` and
`comm/__init__.py`, and `wandb` was missing `wandb_promise` *and had no
dist-info at all*, so no package manager considered it installed.

`uv sync --active` could not repair this (the broken deepspeed dist-info has
no `RECORD`, so uv can't uninstall it). What worked:

```bash
# quarantine the broken trees rather than deleting them
mv a1_5_sft/lib/python3.12/site-packages/{deepspeed,deepspeed-0.18.2.dist-info,wandb} /some/scratch/
CUDA_HOME=/usr/local/cuda uv pip install --python a1_5_sft/bin/python "deepspeed==0.19.1" "wandb>=0.25.0"
```

Result: `deepspeed 0.19.1`, `wandb 0.28.2`, both importing cleanly.
`s3torchconnector`/`s3torchconnectorclient` 1.5.0 were **already installed**
in this venv -- no action needed. Verify with:

```bash
CUDA_HOME=/usr/local/cuda a1_5_sft/bin/python -c "import torch,transformers,deepspeed,wandb,s3torchconnector; print('ok')"
```

### 4. Things the previous handoff warned about that turned out fine

- **wandb 0.26.0 login bug** (`json.loads(None)` in `_get_username`): gone on
  0.28.2. `wandb.Api()` authenticates from `~/.netrc` as `rdesc1`. No
  `WANDB_MODE=disabled` workaround needed.
- **HF weights needing a re-pull:** they don't. `HF_HOME` is
  `/mnt/efs/users/rod/hf_cache` (**EFS, not `~/.cache`**), and both
  Qwen3-VL-8B and -2B are already there -- 4 shards load in under a second.
  The previous handoff's "local to this box, re-pull on A100" note was wrong;
  the 11 MB under `~/.cache/huggingface` is just stray config files.
- **`CUDA_HOME` on EFS:** unnecessary here, the node has a real system toolkit
  at `/usr/local/cuda` (plus `/usr/local/cuda-13.0`).

### 5. It looks stalled for ~40 minutes at startup. It isn't.

With `eval_on_start`, HF runs a full 155-iteration eval pass at ~11.5 s/it
(~36 min) *before* step 1, and tqdm's `0/12357` training bar is printed up
front the whole time. GPUs sit near 0% because the loop is S3-fetch-bound on
5 camera views. Don't kill it; wait for `{'eval_loss': ...}` and then the
first `{'loss': ...}`. Also ignore the flood of
`Invalid token ids found` / `Number of tokens is not equal to the expected
number` warnings at step 0 -- an untrained backbone emitting non-trajectory
tokens is the expected baseline behavior, and they stop as it learns the
format.

---

## Resolved: the B300 blocker

Previously the run died in Qwen3-VL's vision tower with
`RuntimeError: nvrtc: error: invalid value for --gpu-architecture (-arch)`,
because `torch==2.8.0+cu128`'s bundled NVRTC can't JIT for B300's `sm_103`.

**Not reproducible on A100** (compute cap 8.0, fully supported by this torch
build), exactly as predicted. Confirmed by the step-0 eval running full
generation through the vision tower and by sustained 100% GPU utilization.
Nothing further needed. The `a1_5_sft_b300` venv is irrelevant on this node.

---

## Next steps

- [ ] Let the run reach step 500 for the first mid-training `ValMinADE` +
      checkpoint, and confirm minADE is falling well below the 76.1 m
      step-0 baseline.
- [ ] Run to completion (~24-27 h) or until `ValMinADE` plateaus.
- [ ] **Answer the research question:** compare this run's `ValMinADE` curve
      against the existing Alpamayo-pretrained TruckDrive Stage-1 run
      (`output_stage1_nav`-style checkpoints / wandb). The gap between the two
      curves *is* the value of NVIDIA's physical-AI pretraining. Also worth
      putting both against the ego-only MLP baseline (ADE 1.425 m), which
      already beats the finetuned Alpamayo on K=1 ADE -- if scratch-Qwen lands
      near the MLP, that says the VLM contributes little on this dataset.
- [ ] Optional: repeat with `model.vlm_name_or_path=Qwen/Qwen3-VL-2B-Instruct`
      for a backbone-size sweep (weights already cached on EFS).
