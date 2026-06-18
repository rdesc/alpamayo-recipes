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

"""LoRA (PEFT) support for Alpamayo SFT.

The Alpamayo trainable models (``TrainableReasoningVLA`` / ``TrainableAlpamayoR1``)
are custom ``nn.Module`` wrappers around an HF VLM plus trajectory / diffusion
heads. Rather than wrap the whole thing in a ``PeftModel`` (which would change the
forward signature, break the custom gradient-checkpointing delegation, and break
the ``lr_multiplier`` prefix matching in the trainer), we inject LoRA adapters
*in place* with :func:`peft.inject_adapter_in_model`. The model keeps its original
class, so every existing code path (forward, sampling, eval, checkpoint loading)
is unchanged; only the targeted ``nn.Linear`` layers are swapped for LoRA layers.

Trainability is set explicitly here instead of relying on PEFT internals:

* LoRA adapter params (``lora_*``) are trainable.
* The frozen base weights under each adapter (``*.base_layer.*``) are frozen.
* Every other parameter keeps the ``requires_grad`` the model constructor gave it.

That last rule is what makes this compose with the expert recipe: the VLM is
already frozen (``cotrain_vlm=False``) while the action expert / projection heads
are trainable, so injecting LoRA into the VLM yields "LoRA on the frozen backbone
+ full-train the expert head" with no extra configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch

from alpamayo_r1.common import logging

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")

# Default modules to adapt for Qwen-VL language backbones: attention + MLP projections.
_DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

ADAPTER_NAME = "default"


@dataclass
class LoraSettings:
    """Hydra-friendly LoRA configuration.

    Attributes:
        r: LoRA rank.
        alpha: LoRA scaling (``lora_alpha``).
        dropout: LoRA dropout probability.
        target_modules: Linear layer name suffixes to adapt. ``None`` uses the
            Qwen-VL attention + MLP defaults. A string is treated as a regex by PEFT.
        modules_to_save: Extra (non-LoRA) module names to keep fully trainable and
            checkpoint. The expert recipe does not need this — its heads are already
            trainable — but it is exposed for completeness.
        bias: PEFT bias mode ("none" | "all" | "lora_only").
        target_submodule: Restrict adaptation to a named submodule of the model
            (e.g. "vlm"). ``None`` searches the whole model. Useful to guarantee
            LoRA only touches the VLM and never the diffusion expert.
    """

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] | str | None = None
    modules_to_save: list[str] | None = None
    bias: str = "none"
    target_submodule: str | None = "vlm"


def _resolve_target_modules(settings: LoraSettings) -> list[str] | str:
    if settings.target_modules is None:
        return list(_DEFAULT_TARGET_MODULES)
    return settings.target_modules


def build_lora_config(settings: LoraSettings) -> "Any":
    """Construct a :class:`peft.LoraConfig` from :class:`LoraSettings`."""
    from peft import LoraConfig  # imported lazily so peft stays an optional dep

    return LoraConfig(
        r=settings.r,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=_resolve_target_modules(settings),
        modules_to_save=settings.modules_to_save,
        bias=settings.bias,
        task_type=None,  # custom model, not a standard PEFT task type
    )


def apply_lora(model: torch.nn.Module, settings: LoraSettings) -> torch.nn.Module:
    """Inject LoRA adapters into ``model`` in place and fix up trainability.

    Args:
        model: A ``TrainableReasoningVLA`` / ``TrainableAlpamayoR1`` (or any
            ``nn.Module``). Modified in place.
        settings: LoRA configuration.

    Returns:
        The same ``model`` object, now containing LoRA layers.
    """
    from peft import inject_adapter_in_model

    lora_config = build_lora_config(settings)

    # Optionally restrict injection to a named submodule (e.g. the VLM) so the
    # diffusion expert is never adapted even if it shares projection names.
    target = model
    if settings.target_submodule:
        target = getattr(model, settings.target_submodule, None)
        if target is None:
            raise ValueError(
                f"target_submodule={settings.target_submodule!r} not found on "
                f"{type(model).__name__}"
            )

    # Snapshot pre-injection trainability so we can preserve full-train heads
    # (expert, projections) that PEFT may otherwise freeze.
    pre_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}

    inject_adapter_in_model(lora_config, target, adapter_name=ADAPTER_NAME)

    n_lora, n_base_frozen, n_preserved = 0, 0, 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            n_lora += param.numel()
        elif ".base_layer." in name:
            # Original weight wrapped by a LoRA layer: keep it frozen.
            param.requires_grad = False
            n_base_frozen += param.numel()
        else:
            # Untouched parameter: restore the constructor's intent.
            # Names are unchanged for non-adapted params, so the snapshot matches.
            param.requires_grad = pre_requires_grad.get(name, param.requires_grad)
            if param.requires_grad:
                n_preserved += param.numel()

    # With a frozen backbone + gradient checkpointing, inputs must require grad
    # for the graph to reach the LoRA params. Safe no-op if grad-ckpt is off.
    _enable_input_require_grads(model)

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Applied LoRA (r=%d, alpha=%d) to %s: %s trainable params "
        "(%s LoRA + %s preserved full-train); %s base params frozen.",
        settings.r,
        settings.alpha,
        f"{type(model).__name__}.{settings.target_submodule}"
        if settings.target_submodule
        else type(model).__name__,
        f"{total_trainable:,}",
        f"{n_lora:,}",
        f"{n_preserved:,}",
        f"{n_base_frozen:,}",
    )
    if n_lora == 0:
        raise ValueError(
            "LoRA injection added no trainable adapter params. Check that "
            "`target_modules` match Linear layers in the model "
            f"(searched submodule={settings.target_submodule!r})."
        )
    return model


def _enable_input_require_grads(model: torch.nn.Module) -> None:
    """Make input embeddings emit grad-requiring activations (for grad checkpointing)."""
    fn = getattr(model, "enable_input_require_grads", None)
    if callable(fn):
        fn()
        return
    vlm = getattr(model, "vlm", None)
    vlm_fn = getattr(vlm, "enable_input_require_grads", None)
    if callable(vlm_fn):
        vlm_fn()


# Substrings identifying adapter (and modules_to_save) tensors in the state dict.
# Because we inject in place rather than wrapping in a ``PeftModel``, the model has
# no ``peft_config`` for ``get_peft_model_state_dict`` to use, so we filter by name.
_ADAPTER_KEY_MARKERS: tuple[str, ...] = ("lora_", ".modules_to_save.")


def lora_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Extract only the LoRA adapter (and ``modules_to_save``) weights from ``model``.

    Keys are kept exactly as they appear in ``model.state_dict()`` so the adapter
    round-trips via :func:`load_lora_adapter` (apply LoRA, then ``load_state_dict``).
    """
    return {
        k: v
        for k, v in model.state_dict().items()
        if any(marker in k for marker in _ADAPTER_KEY_MARKERS)
    }


def load_lora_adapter(model: torch.nn.Module, adapter_dir: str) -> torch.nn.Module:
    """Load adapter weights saved by :func:`save_lora_adapter` into ``model``.

    ``model`` must already have LoRA injected (call :func:`apply_lora` first with
    matching settings). Returns ``model`` for convenience.
    """
    from safetensors.torch import load_file as load_safetensors_file

    state_dict = load_safetensors_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    result = model.load_state_dict(state_dict, strict=False)
    unexpected = list(result.unexpected_keys)
    if unexpected:
        raise ValueError(f"Adapter has keys not present in model: {unexpected[:5]}")
    logger.info("Loaded %d adapter tensors from %s", len(state_dict), adapter_dir)
    return model


def save_lora_adapter(
    model: torch.nn.Module,
    settings: LoraSettings,
    output_dir: str,
) -> None:
    """Write PEFT-compatible adapter artifacts (``adapter_config.json`` + weights).

    The saved directory can be loaded later with
    ``peft.inject_adapter_in_model`` + ``set_peft_model_state_dict``, or with the
    standard PEFT loaders.
    """
    from safetensors.torch import save_file

    os.makedirs(output_dir, exist_ok=True)
    state_dict = lora_state_dict(model)
    # Move to CPU and contiguous for safetensors.
    state_dict = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    save_file(state_dict, os.path.join(output_dir, "adapter_model.safetensors"))
    build_lora_config(settings).save_pretrained(output_dir)
    logger.info("Saved LoRA adapter (%d tensors) to %s", len(state_dict), output_dir)


# --- Trainer integration -----------------------------------------------------

# `TrainerCallback` is imported lazily inside the factory so this module can be
# imported without transformers present (e.g. for unit tests of the core logic).


def make_lora_save_callback(model: torch.nn.Module, settings: LoraSettings):
    """Create a ``TrainerCallback`` that writes adapter-only artifacts on each save.

    Because we inject in place (the model is not a ``PeftModel``), the HF Trainer
    saves a full checkpoint; this callback additionally drops a lightweight
    ``adapter/`` folder inside each checkpoint dir so the small LoRA delta can be
    shared/loaded on its own.
    """
    from transformers import TrainerCallback

    class LoraSaveCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
            if not state.is_world_process_zero:
                return
            ckpt_dir = os.path.join(
                args.output_dir, f"checkpoint-{state.global_step}", "adapter"
            )
            save_lora_adapter(model, settings, ckpt_dir)

    return LoraSaveCallback()
