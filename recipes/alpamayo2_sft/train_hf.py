"""Alpamayo 2 Super LoRA SFT entry point (trajectory-only, VLM-side CE loss).

Mirrors alpamayo1_5_sft/train_hf.py's structure: Hydra-instantiate the model,
build datasets/collate against the loaded model's config/tokenizer, optionally
inject LoRA, then hand everything to a custom HF Trainer (split-loss logging +
optional W&B trajectory viz).

MVP scope (see recipes/alpamayo2_sft/README.md for the full picture):
  - Trajectory-only supervision (no GT Chain-of-Causation text required).
  - VLM-side discrete-trajectory-token CE loss only -- the diffusion expert's
    own training loop (model.expert(...)) is a separate follow-up, not wired
    into this Trainer yet.
"""

import os

import hydra
import hydra.utils as hyu
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments

from alpamayo2_sft.trainer import Alpamayo2SFTTrainer


@hydra.main(version_base=None, config_path="configs", config_name="sft_base")
def train(cfg: DictConfig) -> None:
    """Main training entry point."""
    torch.manual_seed(42)
    print(OmegaConf.to_yaml(cfg))

    training_args = TrainingArguments(**OmegaConf.to_container(cfg.trainer, resolve=True))

    model = hyu.instantiate(cfg.model, _convert_="partial")

    if cfg.get("lora", None) is not None:
        from alpamayo2_sft.lora import LoraSettings, apply_lora, make_lora_save_callback

        lora_settings = LoraSettings(**OmegaConf.to_container(cfg.lora, resolve=True))
        model = apply_lora(model, lora_settings)
        lora_callback = make_lora_save_callback(model, lora_settings)
    else:
        lora_callback = None

    if training_args.gradient_checkpointing:
        model.vlm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs
        )
        # Trainer would otherwise also call this on `model` itself, which
        # Alpamayo2Super doesn't implement (only `model.vlm` supports it).
        training_args.gradient_checkpointing = False

    train_dataset = hyu.instantiate(
        cfg.data.train_dataset, model_config=model.config, tokenizer=model.tokenizer
    )
    eval_dataset = hyu.instantiate(
        cfg.data.val_dataset, model_config=model.config, tokenizer=model.tokenizer
    )
    collate_fn = hyu.instantiate(cfg.data.collate_fn, pad_token_id=model.tokenizer.pad_token_id)

    callbacks = [lora_callback] if lora_callback is not None else []

    viz_cfg = cfg.get("viz", None)
    if viz_cfg is not None and viz_cfg.get("enabled", False):
        from alpamayo2_sft.viz_callback import TrajectoryVizCallback

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

    valm_cfg = cfg.get("val_metric", None)
    if valm_cfg is not None and valm_cfg.get("enabled", False):
        from alpamayo2_sft.val_metric_callback import ValMinADECallback

        valm_kwargs = OmegaConf.to_container(valm_cfg, resolve=True)
        valm_kwargs.pop("enabled", None)
        callbacks.append(ValMinADECallback(eval_dataset=eval_dataset, **valm_kwargs))

    trainer = Alpamayo2SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        callbacks=callbacks,
    )

    if cfg.get("wandb", None) is not None:
        from alpamayo2_sft.wandb_utils import init_wandb

        os.makedirs(cfg.paths.output_dir, exist_ok=True)
        init_wandb(**OmegaConf.to_container(cfg.wandb, resolve=True))

    trainer.train()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train()
