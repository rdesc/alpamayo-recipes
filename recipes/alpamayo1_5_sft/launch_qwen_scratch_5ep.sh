#!/bin/bash
# Qwen-scratch TruckDrive ablation, 5 epochs, bigger batch -- for the 8x B300 box.
# See QWEN_SCRATCH_ABLATION.md for background on the ablation itself.
#
# DO NOT RAISE `trainer.dataloader_num_workers` back to the recipe default of 16.
# 16 x 8 ranks = 128 worker processes; a 2026-08-26 run at that default died at
# step 500 -- workers were OOM-killed, the parent read EOF off a dead worker's
# pipe -> EOFError('Ran out of input') -> that rank left the collective -> NCCL
# watchdog timeout -> SIGABRT on all 8 ranks. ~2h of training lost, no checkpoint.
#
# The limit is RAM-per-fork, NOT CPU (this box has 192 cores and sits near idle).
# Each worker is fork()ed from a ~22 GB-RSS parent and Python's refcounting breaks
# copy-on-write, so a worker costs ~22-27 GB of anon-RSS -- measured at 26.9 GB on
# the A100 box. Budget on THIS box (4011 GB):
#     8 main ranks x 22 GB                    ~= 176 GB
#     8 workers/rank x 8 ranks x ~25 GB       ~= 1.6 TB
#     + ValMinADE image cache + page cache    ~= 2 TB headroom
#   => ~55% of RAM. The 16/rank default came to >80% and died.
# NOTE: the ValMinADE callback does NOT spawn its own workers -- it indexes the
# dataset inline (callbacks/val_metric_callback.py), inflating each MAIN process
# at step 500. That spike is what tips an already-overcommitted box over.
# 8/rank keeps GPUs fed (they ran 81-100% at the old default); 2/rank risks
# starving them and stretching a 20 h run.
set -euo pipefail
cd "$(dirname "$0")"
source a1_5_sft_b300/bin/activate
export CUDA_HOME=/opt/pytorch/cuda
export WANDB_ENTITY=models
R=$PWD

torchrun --nproc_per_node 8 \
  -m alpamayo1_5_sft.train_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_truckdrive_qwen_scratch \
  model.attn_implementation=sdpa \
  trainer.num_train_epochs=5 \
  trainer.dataloader_num_workers=8 \
  trainer.per_device_train_batch_size=4 \
  trainer.per_device_eval_batch_size=4 \
  +trainer.save_strategy=epoch \
  trainer.save_total_limit=5 \
  data.train_dataset.data_root=s3://torc-data/datasets/TruckDrivePublic \
  data.train_dataset.backend=s3 \
  data.train_dataset.metadata_cache=/mnt/efs/users/rod/truckdrive_cache/meta_5views.pkl \
  data.train_dataset.split_metainfo=$R/truckdrive/truckdrive_metainfo.json \
  data.val_dataset.data_root=s3://torc-data/datasets/TruckDrivePublic \
  data.val_dataset.backend=s3 \
  data.val_dataset.metadata_cache=/mnt/efs/users/rod/truckdrive_cache/meta_5views.pkl \
  data.val_dataset.split_metainfo=$R/truckdrive/truckdrive_metainfo.json \
  paths.output_dir=output_truckdrive_qwen_scratch_5ep \
  task_name=alpamayo_1_5_sft_truckdrive_qwen_scratch_5ep
