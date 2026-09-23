# alpamayo-recipes — project instructions

## ⚠️ VENVS: CHECK WHICH BOX YOU ARE ON FIRST

**Do this before anything else — do not assume the hardware from this file:**

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1
```

This repo is used on more than one machine, and the right venv depends on the
answer. Each recipe has **two** venvs, and neither is on `PATH` — always invoke
`<venv>/bin/python` explicitly.

| recipe | default | alternate |
|---|---|---|
| `recipes/alpamayo1_5_sft/` | `a1_5_sft/` | `a1_5_sft_b300/` |
| `recipes/alpamayo1_x_rl/` | `a1x_rl/` | `a1x_rl_b300/` |

What is actually installed in each (verified 2026-09-22):

| venv | torch | bundled nvrtc |
|---|---|---|
| `a1_5_sft` | 2.8.0+cu128 | 12.8 |
| `a1_5_sft_b300` | 2.8.0+cu128 | **12.9** |
| `a1x_rl` | 2.8.0+cu128 | 12.8 |
| `a1x_rl_b300` | **2.11.0+cu130** | (cu130 wheel layout) |

**Note the `_b300` suffix is misleading.** It is a name, not a guarantee of
sameness: `a1_5_sft_b300` differs from its default only in nvrtc, whereas
`a1x_rl_b300` is a whole different torch. That difference has measurable
numerical consequences — see the LLR note below.

### On an sm_103 box (B300 / GH200-class), use the `_b300` venvs

**Symptom if you pick the wrong one:** every step dies with

```
nvrtc: error: invalid value for --gpu-architecture (-arch)
```

**Why:** the default venvs bundle **nvrtc 12.8**, and `sm_103` did not exist
until CUDA 12.9 — so any *JIT-compiled* kernel (e.g. a `torch.prod` reduction)
fails to compile, while *precompiled* kernels work either way. That is why a
script can load a model, run for a while, and only then blow up once it hits the
JIT path. Do not "fix" this by patching nvrtc, setting `LD_LIBRARY_PATH`, or
editing the default venv; just use `_b300`.

One-line check before a long run:

```bash
./a1_5_sft_b300/bin/python -c "import torch; torch.prod(torch.randn(4,8,device='cuda'),dim=1); print('NVRTC OK')"
```

### On an sm_80 box (A100), the nvrtc error cannot occur

nvrtc 12.8 knows `sm_80` perfectly well, so **either venv runs**. Pick on
*consistency with whatever produced the numbers you are comparing against*, not
on this error.

### For LLR / reasoning-consistency work: always `a1x_rl_b300`

`scripts_fork/llr/` has its own rule that overrides the table above, for a
reason that is about reproducibility rather than correctness: the two venvs
give answers that differ by **~0.01 nats** on identical inputs (torch 2.8 vs
2.11 use different CUDA reduction kernels in `Qwen2RMSNorm`). Neither is more
correct — against an fp64 CPU reference they are statistically
indistinguishable — but 0.01 nats is 25× the content term that study measures.
*Within* one venv the measurement is bit-exact across processes (re-verified
2026-09-22: 12/12 events reproduced to max |Δ| = 0). See
`scripts_fork/llr/README.md`, trap 6.

## READ THE RECIPE SKILL DOC FIRST

Each recipe under `recipes/` ships a NVIDIA-authored `SKILL.md` — an
end-to-end playbook (install/venv, checkpoint conversion, datasets, training
stages, eval, Hydra overrides, and a large "common failure modes" table). They
are **authoritative** for how each pipeline is meant to be run.

Before doing anything on a recipe (training, configs, checkpoints, datasets,
eval, or debugging), **read that recipe's `SKILL.md` first** rather than
reconstructing the pipeline from code:

- `recipes/alpamayo1_5_sft/SKILL.md` — Alpamayo-1.5 SFT (Stage-1 nav/vqa,
  Stage-2 action expert). **The one this fork's TruckDrive work builds on.**
- `recipes/alpamayo1_sft/SKILL.md` — Alpamayo-1 SFT (older gen; holds the eval
  reference numbers 1.5 is compared against).
- `recipes/alpamayo1_x_rl/SKILL.md` — Alpamayo RL post-training.

These `SKILL.md` files are documentation, not auto-loaded skills — open the one
matching whatever you're working on at the start of the task.

## This fork's TruckDrive work

This working tree (`feature/lora-and-traj-viz`) adds a TruckDrivePublic
trajectory fine-tune on top of the recipe. The stage-2 contract from
`SKILL.md` still holds (`model.pretrained_model_name_or_path` = A1-format base,
`model.stage1_vlm_checkpoint_path` = Stage-1 trainer output — never swap them).
Fork-specific configs: `configs/sft_truckdrive.yaml`,
`configs/sft_stage2_truckdrive.yaml`. Venv: `recipes/alpamayo1_5_sft/a1_5_sft/`
(python at `a1_5_sft/bin/python`; not on PATH).
