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
    eval_dataset = hyu.instantiate(
        cfg.data.val_dataset, _convert_="partial", model_config=model.config
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

    viz_cfg = cfg.get("viz", None)
    if viz_cfg is not None and viz_cfg.get("enabled", False):
        from alpamayo1_5_sft.callbacks.viz_callback import TrajectoryVizCallback

        viz_kwargs = OmegaConf.to_container(viz_cfg, resolve=True)
        viz_kwargs.pop("enabled", None)
        callbacks.append(
            TrajectoryVizCallback(
                eval_dataset=eval_dataset,
                collate_fn=collate_fn,
                output_dir=cfg.paths.output_dir,
                **viz_kwargs,
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

    trainer.train()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train()
