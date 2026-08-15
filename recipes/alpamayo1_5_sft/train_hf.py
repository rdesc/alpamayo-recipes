# NOTE: modified in this fork -- see upstream NVlabs/alpamayo-recipes for the original.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import hydra
import hydra.utils as hyu
import torch
from omegaconf import DictConfig, OmegaConf

from alpamayo_r1.common import logging
from alpamayo.common import misc

from alpamayo1_5_sft.trainer import ReasoningVLA_Trainer
from alpamayo1_5_sft.trainer import TrainingArguments

from alpamayo.common import config_utils
from alpamayo.common import wandb_utils
from alpamayo_r1.common.logging import setup_logging

setup_logging()

logger = logging.RankedLogger("train", rank_zero_only=True)
logger.setLevel("INFO")


@hydra.main(version_base=None, config_path=None, config_name="config")
def train(cfg: DictConfig) -> None:
    """Main training entry point."""
    misc.seed_everything(42)

    training_args = TrainingArguments(**OmegaConf.to_container(cfg.trainer, resolve=True))
    logger.info("Configs:\n" + misc.pformat(OmegaConf.to_container(cfg, resolve=True)))

    model = hyu.instantiate(cfg.model, _convert_="partial")

    # Datasets/collate read `model.config`; instantiate them before LoRA injection
    # so they always see the underlying model config (injection is in place anyway).
    train_dataset = hyu.instantiate(
        cfg.data.train_dataset, _convert_="partial", model_config=model.config
    )
    # Trainer eval computes a teacher-forced loss -> needs generation_mode=False
    # (a generation-mode dataset masks all labels, giving zero loss). The viz
    # callback separately needs a generation_mode=True copy for autoregressive
    # rollout; that one is built in the viz block below.
    eval_ds_cfg = OmegaConf.merge(
        cfg.data.val_dataset,
        OmegaConf.create({"vla_preprocess_args": {"generation_mode": False}}),
    )
    eval_dataset = hyu.instantiate(
        eval_ds_cfg, _convert_="partial", model_config=model.config
    )

    collate_fn = hyu.instantiate(
        cfg.data.collate_fn, _convert_="partial", model_config=model.config
    )

    lora_callback = None
    if cfg.get("lora", None) is not None:
        from alpamayo1_5_sft.models.lora import (
            LoraSettings,
            apply_lora,
            make_lora_save_callback,
        )

        lora_settings = LoraSettings(**OmegaConf.to_container(cfg.lora, resolve=True))
        model = apply_lora(model, lora_settings)
        lora_callback = make_lora_save_callback(model, lora_settings)

    callbacks = []
    for cb_name, cb_cfg in cfg.callbacks.items():
        logger.info(f"Initializing callback {cb_name}")
        callbacks.append(hyu.instantiate(cb_cfg, _convert_="partial"))

    if lora_callback is not None:
        callbacks.append(lora_callback)

    # Viz and the val-minADE metric both need the generation-mode val set
    # (autoregressive rollout), distinct from the teacher-forced `eval_dataset`
    # used for eval/loss above. Build it once and share.
    viz_cfg = cfg.get("viz", None)
    valm_cfg = cfg.get("val_metric", None)
    need_gen_val = (viz_cfg is not None and viz_cfg.get("enabled", False)) or (
        valm_cfg is not None and valm_cfg.get("enabled", False)
    )
    gen_val_dataset = None
    if need_gen_val:
        gen_val_dataset = hyu.instantiate(
            cfg.data.val_dataset, _convert_="partial", model_config=model.config
        )

    if viz_cfg is not None and viz_cfg.get("enabled", False):
        from alpamayo1_5_sft.callbacks.viz_callback import TrajectoryVizCallback

        viz_kwargs = OmegaConf.to_container(viz_cfg, resolve=True)
        viz_kwargs.pop("enabled", None)
        callbacks.append(
            TrajectoryVizCallback(
                eval_dataset=gen_val_dataset,
                collate_fn=collate_fn,
                output_dir=cfg.paths.output_dir,
                **viz_kwargs,
            )
        )

    if valm_cfg is not None and valm_cfg.get("enabled", False):
        from alpamayo1_5_sft.callbacks.val_metric_callback import ValMinADECallback

        valm_kwargs = OmegaConf.to_container(valm_cfg, resolve=True)
        valm_kwargs.pop("enabled", None)
        callbacks.append(
            ValMinADECallback(
                eval_dataset=gen_val_dataset,
                collate_fn=collate_fn,
                **valm_kwargs,
            )
        )

    trainer = ReasoningVLA_Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        callbacks=callbacks,
    )

    if "deepspeed" in cfg.trainer and cfg.trainer.deepspeed is not None:
        # We should not cast the forward inputs to bfloat16 because our model is mixed
        # precision and trajectory encoder might require float32 input.
        ds_config = trainer.accelerator.state.deepspeed_plugin.hf_ds_config
        ds_config._dtype = torch.float32

    if cfg.get("wandb", None) is not None:
        wandb_utils.init_wandb(**cfg.wandb)

    if trainer.is_world_process_zero():
        config_utils.save_config(
            cfg,
            os.path.join(cfg.paths.output_dir, "config.yaml"),
            resolve_paths=True,
            include_hydra_config=True,
        )

    # Make every checkpoint self-describing: copy the run config into each
    # checkpoint-N/ as it is saved, so a checkpoint copied off the run dir still
    # carries the settings that produced it.
    from alpamayo1_5_sft.callbacks.config_snapshot_callback import (
        ConfigSnapshotCallback,
    )

    trainer.add_callback(ConfigSnapshotCallback(run_output_dir=cfg.paths.output_dir))

    # `training_args.resume_from_checkpoint` is an inherited HF field that was
    # previously never read anywhere in this recipe -- setting it on the CLI
    # (e.g. `trainer.resume_from_checkpoint=true` for the latest checkpoint in
    # output_dir, or `=<path>` for a specific one) had zero effect. Passing it
    # through here is what actually makes it resume: with `true`, HF resolves
    # it via `get_last_checkpoint(args.output_dir)` and reloads model, optimizer/
    # scheduler, RNG, and TrainerState (so global_step/epoch continue rather
    # than restart). Works for LoRA runs too -- `apply_lora` injects in place
    # rather than wrapping in a PeftModel, so each checkpoint-N/ already holds
    # a full state dict (LoRA weights included), not just a PEFT adapter, and
    # that's what this full-checkpoint load path expects.
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    # HF Trainer only saves at `save_steps` multiples -- if total_steps isn't
    # a multiple (e.g. short runs, or a batch size chosen so an epoch doesn't
    # divide evenly), training can finish with NO checkpoint ever written.
    # Always leave a usable final save regardless of that alignment.
    trainer.save_model()
    if trainer.is_world_process_zero():
        trainer.save_state()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train()
