# alpamayo-recipes — project instructions

## ⚠️ ON THIS BOX (B300 / sm_103): ALWAYS USE THE `_b300` VENV

This machine's GPUs are **NVIDIA B300 SXM6, compute capability 10.3 (`sm_103`)**.
Each recipe has **two** venvs — use the `_b300` one, always:

| recipe | ❌ do not use | ✅ use on this box |
|---|---|---|
| `recipes/alpamayo1_5_sft/` | `a1_5_sft/` | **`a1_5_sft_b300/`** |
| `recipes/alpamayo1_x_rl/` | `a1x_rl/` | **`a1x_rl_b300/`** |

**Symptom if you pick the wrong one:** every step dies with

```
nvrtc: error: invalid value for --gpu-architecture (-arch)
```

**Why:** the default venvs bundle **nvrtc 12.8**, and `sm_103` didn't exist until
CUDA 12.9 — so any *JIT-compiled* kernel (e.g. a `torch.prod` reduction) fails to
compile. The `_b300` venvs bundle **nvrtc 12.9**, which knows `sm_103`. Torch is
the same version (`2.8.0+cu128`) in both, and its *precompiled* kernels work
either way — which is why a script can load a model, run for a while, and only
then blow up once it hits the JIT path. Do not "fix" this by patching nvrtc,
setting `LD_LIBRARY_PATH`, or editing the default venv; just use `_b300`.

One-line check before a long run:

```bash
./a1_5_sft_b300/bin/python -c "import torch; torch.prod(torch.randn(4,8,device='cuda'),dim=1); print('NVRTC OK')"
```

(Neither venv is on `PATH` — always invoke `<venv>/bin/python` explicitly.)

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
