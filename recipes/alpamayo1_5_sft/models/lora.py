# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
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
from dataclasses import asdict, dataclass, field
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
        freeze_non_adapted: Freeze every parameter inside ``target_submodule`` that
            LoRA did not adapt, yielding TRUE LoRA (adapters are the only trainable
            weights in the backbone).

            Default ``False`` preserves the Stage-2 contract: there the VLM is
            already frozen by the model constructor and the diffusion expert must
            stay trainable, so restoring the constructor's intent is correct.

            In Stage 1 nothing is pre-frozen, so the default leaves everything LoRA
            did not wrap -- embeddings, ``lm_head``, the whole vision tower -- fully
            trainable. On Alpamayo-1.5 that is 1.85B params of unintended full
            fine-tuning next to 43.6M of LoRA. Set this to ``true`` in Stage-1 LoRA
            configs. Params outside ``target_submodule`` (e.g. the expert head) are
            never touched by this flag.
    """

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] | str | None = None
    modules_to_save: list[str] | None = None
    bias: str = "none"
    target_submodule: str | None = "vlm"
    freeze_non_adapted: bool = False


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

    # Scope for `freeze_non_adapted`: only params inside the adapted submodule.
    submodule_prefix = f"{settings.target_submodule}." if settings.target_submodule else ""

    n_lora, n_base_frozen, n_preserved, n_frozen_non_adapted = 0, 0, 0, 0
    for name, param in model.named_parameters():
        if "lora_" in name or ".modules_to_save." in name:
            # Adapter weights, plus any `modules_to_save` copies PEFT made trainable.
            param.requires_grad = True
            n_lora += param.numel()
        elif ".base_layer." in name:
            # Original weight wrapped by a LoRA layer: keep it frozen.
            param.requires_grad = False
            n_base_frozen += param.numel()
        elif settings.freeze_non_adapted and name.startswith(submodule_prefix):
            # True-LoRA mode: nothing inside the adapted submodule trains except the
            # adapters themselves. `modules_to_save` entries are exempt -- PEFT gives
            # them a `.modules_to_save.` copy, which is matched by the branch above.
            param.requires_grad = False
            n_frozen_non_adapted += param.numel()
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
    if settings.freeze_non_adapted:
        logger.info(
            "freeze_non_adapted=True: froze %s additional non-adapted params inside %r "
            "(embeddings, lm_head, vision tower, norms).",
            f"{n_frozen_non_adapted:,}",
            settings.target_submodule,
        )
    elif n_preserved > 10 * n_lora:
        # Loud warning for the Stage-1 footgun: far more full-rank params are training
        # than LoRA params, so this is not meaningfully a LoRA run.
        logger.warning(
            "LoRA adapted %s params but %s NON-adapted params are still fully "
            "trainable (%.1fx more). If you intended true LoRA, set "
            "lora.freeze_non_adapted=true.",
            f"{n_lora:,}",
            f"{n_preserved:,}",
            n_preserved / max(n_lora, 1),
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


# Sidecar written next to ``adapter_config.json``. PEFT's own config cannot round-trip
# our custom ``target_submodule`` field, so we persist the full LoraSettings too.
SETTINGS_FILENAME = "alpamayo_lora.json"
ADAPTER_DIRNAME = "adapter"


def find_adapter_dir(checkpoint_path: str) -> str | None:
    """Return the ``adapter/`` dir inside a Trainer checkpoint, or None if absent.

    Used to auto-detect LoRA checkpoints so eval / merge paths can inject adapters
    before loading weights (see :func:`load_lora_settings`).
    """
    adapter_dir = os.path.join(checkpoint_path, ADAPTER_DIRNAME)
    if os.path.isfile(os.path.join(adapter_dir, "adapter_model.safetensors")):
        return adapter_dir
    return None


def load_lora_settings(adapter_dir: str) -> LoraSettings:
    """Reconstruct :class:`LoraSettings` from a saved adapter directory.

    Prefers the ``alpamayo_lora.json`` sidecar (exact round-trip). Falls back to
    PEFT's ``adapter_config.json`` for adapters saved before the sidecar existed,
    in which case ``target_submodule`` falls back to the dataclass default -- correct
    for every adapter this fork has produced, since only the default has been used.
    """
    import json

    sidecar = os.path.join(adapter_dir, SETTINGS_FILENAME)
    if os.path.isfile(sidecar):
        with open(sidecar, "r", encoding="utf-8") as f:
            return LoraSettings(**json.load(f))

    peft_config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.isfile(peft_config_path):
        raise FileNotFoundError(f"No LoRA settings found in {adapter_dir}")
    with open(peft_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    defaults = LoraSettings()
    logger.warning(
        "No %s in %s; reconstructing settings from adapter_config.json "
        "(target_submodule falls back to %r).",
        SETTINGS_FILENAME,
        adapter_dir,
        defaults.target_submodule,
    )
    return LoraSettings(
        r=cfg["r"],
        alpha=cfg["lora_alpha"],
        dropout=cfg["lora_dropout"],
        target_modules=cfg.get("target_modules"),
        modules_to_save=cfg.get("modules_to_save"),
        bias=cfg.get("bias", defaults.bias),
        target_submodule=defaults.target_submodule,
    )


def merge_lora_weights(model: torch.nn.Module) -> torch.nn.Module:
    """Fold LoRA deltas into their base weights, restoring original module names.

    After this, adapted ``lora.Linear`` layers are plain ``nn.Linear`` again, so the
    state dict uses the ORIGINAL key names (``...q_proj.weight``) rather than
    ``...q_proj.base_layer.weight`` + ``lora_A/lora_B``. That is what downstream
    consumers expect -- notably ``load_alpamayo1_vlm``, which matches by name and
    would silently skip every adapted projection otherwise.

    Returns ``model`` (modified in place).
    """
    from peft.tuners.lora import LoraLayer

    n_merged = 0
    for module in model.modules():
        if isinstance(module, LoraLayer):
            # PEFT computes delta_W = B @ A * scaling and adds it to base_layer.weight.
            module.merge()
            n_merged += 1

    # Replace each merged LoraLayer with its underlying base module so the wrapper
    # (and its `base_layer.` key prefix) disappears from the state dict entirely.
    def _unwrap(parent: torch.nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, LoraLayer):
                setattr(parent, name, child.get_base_layer())
            else:
                _unwrap(child)

    _unwrap(model)
    logger.info("Merged %d LoRA layers into their base weights.", n_merged)
    return model


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
    # PEFT's adapter_config.json has no field for our custom `target_submodule`, so
    # persist the full settings alongside it for an exact round-trip on load.
    import json as _json

    with open(os.path.join(output_dir, SETTINGS_FILENAME), "w", encoding="utf-8") as f:
        _json.dump(asdict(settings), f, indent=2)
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
