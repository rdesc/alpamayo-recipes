# alpamayo-recipes — project instructions

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
