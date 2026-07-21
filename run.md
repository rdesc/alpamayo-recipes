




# July 21st
Running eval again on an existing predictions file
```
python truckdrive/score_val_horizons.py
  /mnt/efs/users/rod/ckpts/Alpamayo-1.5-10B-A1-format/eval_20260721_201032/predictions.pt
```

Launching inference on TruckDrive val split with Alpamayo 1.5 zeroshot
```
runlog PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=4,5,6,7 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_HOME=/mnt/efs/users/rod/cuda-13.0 torchrun --nproc_per_node=4 --master_port=29541 -m alpamayo1_5_sft.train_hf --config-path pkg://alpamayo1_5_sft/configs --config-name sft_stage2_truckdrive data.train_dataset.data_root=s3://torc-data/datasets/TruckDrivePublic data.val_dataset.data_root=s3://torc-data/datasets/TruckDrivePublic data.train_dataset.backend=s3 data.val_dataset.backend=s3 data.train_dataset.metadata_cache=/mnt/efs/users/rod/truckdrive_cache/meta_5views.pkl data.val_dataset.metadata_cache=/mnt/efs/users/rod/truckdrive_cache/meta_5views.pkl data.train_dataset.split_metainfo=/mnt/efs/users/rod/repos/alpamayo-recipes/recipes/alpamayo1_5_sft/truckdrive/truckdrive_metainfo.json data.val_dataset.split_metainfo=/mnt/efs/users/rod/repos/alpamayo-recipes/recipes/alpamayo1_5_sft/truckdrive/truckdrive_metainfo.json model.stage1_vlm_checkpoint_path=/mnt/efs/users/rod/ckpts/alpamayo1.5_finetuned_truckdrive_full_checkpoint-8374 trainer.per_device_train_batch_size=2 trainer.gradient_accumulation_steps=1 trainer.num_train_epochs=2 trainer.learning_rate=1e-5 trainer.warmup_steps=500 trainer.save_steps=1000 trainer.dataloader_num_workers=8 paths.output_dir=/opt/dlami/nvme/rod/ckpts/alpamayo_truckdrive_finetuning/alpamayo1.5_truckdrive_STAGE2_expert viz.gen_batch_size=4 2>&1 | tee /mnt/efs/users/rod/logs/alpamayo1.5_truckdrive_STAGE2_expert.log
```

