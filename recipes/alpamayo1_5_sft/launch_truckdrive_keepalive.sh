#!/bin/bash
# Keep-the-box-busy TruckDrive Stage-1 SFT run, 8x B300.
#
# Purpose: this box must not sit idle. GPUs 0-3 are already running other
# in-flight jobs (motion_confidence_smoke.py, eval_baseline_ood_reasoning_actionhead.py)
# owned by this user, so this run uses ALL 8 GPUs (per explicit instruction)
# but keeps per-GPU footprint small (batch size 1, no checkpoint writes) to
# avoid starving those jobs of memory.
#
# 10 epochs, checkpointing disabled entirely (+trainer.save_strategy=no) --
# this run is disposable, nothing here needs to survive a restart.
#
# W&B disabled (~wandb): the a1_5_sft_b300 venv's wandb==0.28.2 dropped
# wandb.util.generate_id (alpamayo.common.wandb_utils.init_wandb calls it
# unconditionally whenever cfg.wandb is set), so init_wandb crashes with
# AttributeError before training starts. viz/val_metric callbacks already
# guard on `wandb.run is None`, so dropping the group is safe -- no code change.
set -euo pipefail
cd "$(dirname "$0")"
source a1_5_sft_b300/bin/activate
export CUDA_HOME=/mnt/efs/users/rod/cuda-13.0/cuda-13.0
R=$PWD

torchrun --nproc_per_node 8 \
  -m alpamayo1_5_sft.train_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_truckdrive \
  ~wandb \
  trainer.report_to=none \
  model.attn_implementation=sdpa \
  trainer.num_train_epochs=10 \
  trainer.dataloader_num_workers=8 \
  trainer.per_device_train_batch_size=1 \
  trainer.per_device_eval_batch_size=1 \
  +trainer.save_strategy=no \
  data.train_dataset.data_root=s3://torc-data/datasets/TruckDrivePublic \
  data.train_dataset.backend=s3 \
  data.train_dataset.metadata_cache=/mnt/efs/users/rod/truckdrive_cache/meta_5views.pkl \
  data.train_dataset.split_metainfo=$R/truckdrive/truckdrive_metainfo.json \
  data.val_dataset.data_root=s3://torc-data/datasets/TruckDrivePublic \
  data.val_dataset.backend=s3 \
  data.val_dataset.metadata_cache=/mnt/efs/users/rod/truckdrive_cache/meta_5views.pkl \
  data.val_dataset.split_metainfo=$R/truckdrive/truckdrive_metainfo.json \
  paths.output_dir=output_truckdrive_keepalive \
  task_name=alpamayo_1_5_sft_truckdrive_keepalive
