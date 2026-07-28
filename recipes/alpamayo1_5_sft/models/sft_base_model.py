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

from dataclasses import dataclass
from typing import Any, Mapping
import json
from collections import defaultdict
from pathlib import Path
import einops
import numpy as np
import torch
from transformers.utils import ModelOutput
from hydra.utils import instantiate
from safetensors.torch import load_file as load_safetensors_file

from alpamayo_r1.models.base_model import (
    ReasoningVLA,
    ReasoningVLAConfig,
    TrajectoryFusionMixin,
    tokenize_history_trajectory,
    replace_pad_token,
)
from alpamayo_r1.models.base_model import IGNORE_INDEX
from alpamayo_r1.models.token_utils import extract_traj_tokens, extract_text_tokens
from alpamayo_r1.common import logging

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")


def tokenize_future_trajectory(
    tokenizer, traj_data: Mapping[str, Any], start_idx: int = 0
) -> torch.Tensor:
    """Tokenize the future trajectory with prefix shape of (B, n_traj, ...).

    Args:
        tokenizer: BaseTrajectoryTokenizer
        traj_data: dict[str, Any], needs to at least contain:
            - "ego_history_xyz"
            - "ego_history_rot"
            - "ego_future_xyz"
            - "ego_future_rot"
        start_idx: int, start of token index of the future trajectory tokens

    Returns:
        torch.Tensor: [B, n_traj * tokens_per_future_traj]
    """
    assert "ego_future_xyz" in traj_data
    assert traj_data["ego_future_xyz"].ndim == 4, "ego_future_xyz must be 4D of [B, n_traj, T, 3]"

    B = traj_data["ego_future_xyz"].shape[0]
    hist_xyz = traj_data["ego_history_xyz"].flatten(start_dim=0, end_dim=1)
    hist_rot = traj_data["ego_history_rot"].flatten(start_dim=0, end_dim=1)
    fut_xyz = traj_data["ego_future_xyz"].flatten(start_dim=0, end_dim=1)
    fut_rot = traj_data["ego_future_rot"].flatten(start_dim=0, end_dim=1)

    fut_idx = (
        tokenizer.encode(hist_xyz=hist_xyz, hist_rot=hist_rot, fut_xyz=fut_xyz, fut_rot=fut_rot)
        + start_idx
    )
    fut_idx = einops.rearrange(fut_idx, "(b n_traj) n -> b (n_traj n)", b=B)
    return fut_idx


def load_alpamayo1_vlm(checkpoint_path: str, model: Any, strip_vlm_prefix: bool = False):
    """Load fine-tuned ``vlm.*`` weights from a Trainer checkpoint directory.

    Args:
        checkpoint_path: Trainer output dir with ``model.safetensors.index.json``.
        model: module the ``vlm.*`` tensors are loaded into. Two callers with
            different key expectations:
              * the full model (e.g. TrainableReasoningVLA) whose own params are
                already named ``vlm.*`` -> ``strip_vlm_prefix=False`` (default).
              * the ``vlm`` submodule itself (Stage-2 expert passes ``self.vlm``),
                whose params have NO ``vlm.`` prefix -> ``strip_vlm_prefix=True``,
                or every key mismatches and ``strict=False`` silently loads nothing.
    """
    checkpoint_dir = Path(checkpoint_path)
    index_path = checkpoint_dir / "model.safetensors.index.json"
    vlm_state_dict: dict[str, torch.Tensor] = {}

    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            weight_map: dict[str, str] = json.load(f).get("weight_map", {})

        shard_to_keys: defaultdict[str, list[str]] = defaultdict(list)
        for key, shard_name in weight_map.items():
            if key.startswith("vlm."):
                shard_to_keys[shard_name].append(key)

        for shard_name, keys in shard_to_keys.items():
            shard_path = checkpoint_dir / shard_name
            shard_sd = load_safetensors_file(str(shard_path), device="cpu")
            for key in keys:
                if key in shard_sd:
                    dst_key = key[len("vlm.") :] if strip_vlm_prefix else key
                    vlm_state_dict[dst_key] = shard_sd[key]

    if not vlm_state_dict:
        raise ValueError(f"No vlm.* tensors found in checkpoint: {checkpoint_dir}")

    load_result = model.load_state_dict(vlm_state_dict, strict=False, assign=True)
    logger.info(
        f"Loaded {len(vlm_state_dict)} VLM tensors from {checkpoint_dir} (missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)})",
    )
    # Guard against the silent prefix mismatch: if none of the checkpoint tensors
    # matched the target module, load_state_dict quietly kept the module's own
    # weights and we'd train on the wrong VLM.
    if len(load_result.unexpected_keys) == len(vlm_state_dict):
        raise ValueError(
            f"None of the {len(vlm_state_dict)} vlm tensors matched the target module "
            f"(all unexpected). Check strip_vlm_prefix (currently {strip_vlm_prefix})."
        )

    # Guard against a PARTIAL silent load. `from_alpamayo_checkpoint` builds the VLM
    # architecture without base weights, so any vlm.* param the checkpoint fails to
    # fill stays RANDOMLY INITIALIZED and the model emits noise. The all-unexpected
    # check above misses this: loading a LoRA checkpoint into a non-adapted model
    # matches embeddings/norms/vision-tower but leaves every adapted projection
    # behind as `...base_layer.weight` (missing=252, unexpected=756 in practice).
    # Only vlm.* misses matter -- non-VLM params (traj_tokenizer, heads) are
    # legitimately absent from the vlm.*-filtered state dict.
    vlm_prefix = "" if strip_vlm_prefix else "vlm."
    missing_vlm = [k for k in load_result.missing_keys if k.startswith(vlm_prefix)]
    if strip_vlm_prefix:
        # Every param of the target module is a VLM param in this case.
        missing_vlm = list(load_result.missing_keys)
    if missing_vlm:
        adapter_hint = ""
        if any(".base_layer." in k or "lora_" in k for k in load_result.unexpected_keys):
            adapter_hint = (
                " The checkpoint contains LoRA-format keys (`.base_layer.` / `lora_`), "
                "so this is a LoRA checkpoint being loaded into a model without adapters "
                "injected. Pass `lora_adapter_dir` (or merge the adapter first with "
                "truckdrive/merge_lora.py)."
            )
        raise ValueError(
            f"{len(missing_vlm)} VLM parameters were not found in {checkpoint_dir} and "
            f"would stay randomly initialized, e.g. {missing_vlm[:5]}."
            f"{adapter_hint}"
        )

    return model


class TrajectoryFusionWithFutureMixin(TrajectoryFusionMixin):
    def fuse_traj_tokens(
        self, input_ids: torch.Tensor, traj_data: dict[str, Any] | None = None
    ) -> torch.Tensor:
        """Fuse the trajectory tokens into the input ids.

        Args:
            input_ids: [B, n_token]
            traj_data: dict containing ego_history_xyz, ego_history_rot, etc.

        Returns:
            input_ids: [B, n_token] with trajectory tokens fused
        """
        if (
            traj_data is None
            or traj_data.get("ego_history_xyz") is None
            or traj_data.get("ego_history_rot") is None
        ):
            return input_ids

        has_future = "ego_future_xyz" in traj_data and traj_data["ego_future_xyz"] is not None
        attrs = self._validate_mixin_requirements(require_future=has_future)

        hist_idx = tokenize_history_trajectory(
            attrs["hist_traj_tokenizer"], traj_data, attrs["hist_token_start_idx"]
        )
        input_ids = replace_pad_token(
            input_ids, hist_idx, attrs["config"].traj_token_ids["history"]
        )

        if has_future:
            future_idx = tokenize_future_trajectory(
                attrs["traj_tokenizer"], traj_data, attrs["future_token_start_idx"]
            )
            input_ids = replace_pad_token(
                input_ids, future_idx, attrs["config"].traj_token_ids["future"]
            )

        return input_ids


@dataclass
class ReasoningVLAOutput(ModelOutput):
    """Output of the ReasoningVLA model."""

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    # Detached scalar components of `loss` for logging (trajectory-token loss vs
    # everything else). `loss` is their sum; these are diagnostics only.
    loss_future_traj: torch.FloatTensor | None = None
    loss_others: torch.FloatTensor | None = None


class TrainableReasoningVLA(ReasoningVLA, TrajectoryFusionWithFutureMixin):
    """Trainable ReasoningVLA model."""

    def __init__(
        self,
        config: ReasoningVLAConfig,
        pretrained_modules: dict[str, torch.nn.Module] | None = None,
        original_vocab_size: int | None = None,
        print_param_count: bool = True,
    ) -> None:
        super().__init__(config, pretrained_modules, original_vocab_size, print_param_count)

    @classmethod
    def from_alpamayo_checkpoint(
        cls,
        checkpoint_path: str,
        vlm_name_or_path: str,
        attn_implementation: str | None = None,
        lora_adapter_dir: str | None = None,
        **kwargs: Any,
    ) -> "ReasoningVLA":
        """Load the base ReasoningVLA from a full Alpamayo checkpoint.

        Creates the VLM architecture from config (without loading base weights),
        then loads the fine-tuned ``vlm.*`` weights from the Alpamayo checkpoint.
        The trajectory tokenizer is instantiated separately with its own pretrained
        weights.

        Unlike ``from_pretrained_submodules``, this avoids loading the base VLM
        weights from HuggingFace and the expensive ``resize_token_embeddings``
        call, since the checkpoint already contains correctly-sized embeddings.

        Args:
            checkpoint_path: Path to full Alpamayo checkpoint directory.
            vlm_name_or_path: Base VLM checkpoint used to build processor/tokenizer.
            attn_implementation: Optional override for the attention backend
                ("flash_attention_2", "sdpa", "eager"). Defaults to the
                checkpoint config (falling back to flash_attention_2). Use
                "sdpa" on GPUs whose flash-attn wheel lacks kernels (e.g.
                Blackwell with an sm_80/90-only build).
            lora_adapter_dir: If set, inject LoRA adapters (with the settings saved
                in that dir) before loading weights, so a LoRA-format checkpoint's
                ``base_layer``/``lora_*`` keys match. Required when ``checkpoint_path``
                is an unmerged LoRA checkpoint; ``evaluate_hf`` auto-detects it.
            **kwargs: Extra keyword arguments (currently unused).

        Returns:
            Initialized ``ReasoningVLA`` model with Alpamayo VLM weights.
        """
        checkpoint_config_path = Path(checkpoint_path) / "config.json"
        if not checkpoint_config_path.exists():
            raise FileNotFoundError(f"Missing config file: {checkpoint_config_path}")

        with checkpoint_config_path.open("r", encoding="utf-8") as f:
            checkpoint_config = json.load(f)

        config_kwargs = {
            "vlm_name_or_path": vlm_name_or_path,
            "vlm_backend": checkpoint_config.get("vlm_backend", "qwenvl3"),
            "traj_tokenizer_cfg": checkpoint_config.get("traj_tokenizer_cfg"),
            "hist_traj_tokenizer_cfg": checkpoint_config.get("hist_traj_tokenizer_cfg"),
            "traj_vocab_size": checkpoint_config.get("traj_vocab_size"),
            "tokens_per_history_traj": checkpoint_config.get("tokens_per_history_traj"),
            "tokens_per_future_traj": checkpoint_config.get("tokens_per_future_traj"),
            "model_dtype": checkpoint_config.get("model_dtype", "bfloat16"),
            "attn_implementation": attn_implementation
            or checkpoint_config.get("attn_implementation", "flash_attention_2"),
            "min_pixels": checkpoint_config.get("min_pixels"),
            "max_pixels": checkpoint_config.get("max_pixels"),
            "add_special_tokens": checkpoint_config.get("add_special_tokens", True),
        }
        config = instantiate(
            {
                "_target_": f"alpamayo_r1.models.base_model.{cls.config_class.__name__}",
                "_recursive_": False,
                "_convert_": "all",
                **config_kwargs,
            }
        )

        pretrained_modules = {}
        if config.traj_tokenizer_cfg is not None:
            pretrained_modules["traj_tokenizer"] = instantiate(config.traj_tokenizer_cfg)

        model = cls(config, pretrained_modules=pretrained_modules or None)

        # A LoRA checkpoint stores adapted projections as `...base_layer.weight` plus
        # separate `lora_A`/`lora_B` tensors, so the adapters must be injected BEFORE
        # loading -- otherwise none of those keys match and the projections are left
        # randomly initialized (the model architecture is built without base weights).
        if lora_adapter_dir is not None:
            from alpamayo1_5_sft.models.lora import apply_lora, load_lora_settings

            apply_lora(model, load_lora_settings(lora_adapter_dir))

        model = load_alpamayo1_vlm(checkpoint_path, model)

        return model

    def tie_weights(
        self,
        recompute_mapping: bool = False,
        missing_keys: list[str] = None,
        unexpected_keys: list[str] = None,
    ) -> None:
        """Delegate weight tying to the nested VLM model."""
        if hasattr(self.vlm, "tie_weights"):
            self.vlm.tie_weights()

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict[str, Any] | None = None
    ) -> None:
        """Enable gradient checkpointing for the model.

        Args:
            gradient_checkpointing_kwargs: Additional keyword arguments for gradient checkpointing.
        """
        if hasattr(self.vlm, "gradient_checkpointing_enable"):
            self.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        else:
            raise ValueError(
                f"{self.vlm.__class__.__name__} does not support gradient checkpointing."
            )

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing for the model."""
        if hasattr(self.vlm, "gradient_checkpointing_disable"):
            self.vlm.gradient_checkpointing_disable()
        else:
            raise ValueError(
                f"{self.vlm.__class__.__name__} does not support gradient checkpointing."
            )

    @torch._dynamo.disable
    def _compute_next_token_loss(
        self,
        outputs: ModelOutput,
        labels: torch.Tensor,
        labels_mask: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss for the next token prediction.

        Args:
            outputs: ModelOutput containing logits of shape (B, L, V)
            labels: [B, L]
            labels_mask: [B, L], indicates which tokens in the sequence are valid for loss
                computation
            token_mask: [V], indicates which tokens ids are valid for logits computation

        Returns:
            torch.Tensor: (,) loss value
        """
        if labels_mask is None:
            labels_mask = torch.ones_like(labels, dtype=torch.bool)
        if labels_mask[:, 1:].sum() == 0:
            return torch.tensor(0.0, device=labels.device)
        # Shift labels to the left by 1 position (predict next token)
        shift_labels = labels[..., 1:]
        # The logits should also be trimmed to match the shifted labels
        # NOTE: we clone the logits to avoid in-place operations if token_mask is present that will
        # modify the original
        shift_logits = outputs.logits[..., :-1, :].clone()

        shift_labels = shift_labels[labels_mask[:, 1:]].contiguous()
        shift_logits = shift_logits[labels_mask[:, 1:]].contiguous().float()
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        if token_mask is not None:
            shift_logits[..., ~token_mask] = torch.finfo(shift_logits.dtype).min
        loss = torch.nan_to_num(
            torch.nn.functional.cross_entropy(
                shift_logits, shift_labels, ignore_index=IGNORE_INDEX, reduction="mean"
            ),
            nan=0.0,
        )
        return loss

    def forward(
        self,
        tokenized_data: dict[str, Any],
        ego_history_xyz: torch.Tensor | None = None,
        ego_history_rot: torch.Tensor | None = None,
        ego_future_xyz: torch.Tensor | None = None,
        ego_future_rot: torch.Tensor | None = None,
        labels_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ReasoningVLAOutput:
        """Forward pass of the model."""
        # 1. tokenize trajectory and fuse into input_ids
        input_ids = tokenized_data.pop("input_ids")
        traj_data = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "ego_future_xyz": ego_future_xyz,
            "ego_future_rot": ego_future_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data)

        # 2. get labels
        labels = input_ids.clone()
        if labels_mask is not None:
            labels = torch.where(labels_mask, labels, IGNORE_INDEX)

        # 3. vlm forward pass
        outputs = self.vlm(input_ids=input_ids, labels=labels, **tokenized_data)

        losses = {}
        # Identify trajectory tokens (tokens between traj_future and next special token)
        traj_mask = (
            (
                (labels >= self.future_token_start_idx)
                & (labels < self.future_token_start_idx + self.config.traj_vocab_size)
            )
            | (labels == self.special_token_ids["traj_future_start"])
            | (labels == self.special_token_ids["traj_future_end"])
        )
        losses["future_traj"] = self._compute_next_token_loss(outputs, labels, traj_mask)
        #  * self.config.loss_weights.get("future_traj", 1.0)
        labels[traj_mask] = IGNORE_INDEX

        # Include all other tokens in the loss
        losses["others"] = self._compute_next_token_loss(outputs, labels, labels != IGNORE_INDEX)
        #  * self.config.loss_weights.get("others", 1.0)

        # Replace the original loss
        outputs.loss = sum(losses.values())

        return ReasoningVLAOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            loss_future_traj=losses["future_traj"].detach(),
            loss_others=losses["others"].detach(),
        )

    def sample_trajectories_from_data(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        last_component: str = "traj_future",
        *args: Any,
        **kwargs: Any,
    ) -> (
        tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, list[str]]]
    ):
        """Sample trajectories from the data.

        Args:
            data: The input data.
            top_p: The top-p value for sampling.
            top_k: The top-k value for sampling.
            temperature: The temperature for sampling.
            num_traj_samples: The number of trajectory samples.
            num_traj_sets: The number of trajectory sets.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pred_xyz: The predicted xyz.
            pred_rot: The predicted rotation.
            logprob: The log probability.
        """
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        B, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "Only one trajectory group is supported for inference."
        if "tokenized_data" not in data:
            tokenized_data = self._get_generation_mode_tokenized_data_online(
                data, last_component=last_component
            )
        else:
            tokenized_data = data["tokenized_data"]
        input_ids = tokenized_data.pop("input_ids")
        traj_data = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data)

        n_samples_total = num_traj_samples * num_traj_sets
        max_generation_length = kwargs.get(
            "max_generation_length", self.config.tokens_per_future_traj
        )
        assert max_generation_length >= self.config.tokens_per_future_traj
        generation_config = self.vlm.generation_config
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.do_sample = True
        generation_config.num_return_sequences = n_samples_total
        generation_config.max_new_tokens = max_generation_length
        generation_config.output_logits = True
        generation_config.return_dict_in_generate = True
        generation_config.top_k = top_k
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        generated = self.vlm.generate(
            input_ids=input_ids, **tokenized_data, generation_config=generation_config
        )
        # remove input ids from the generated sequences
        generated_tokens = generated.sequences[:, input_ids.shape[1] :]

        # extract trajectory tokens from generated sequences
        traj_token_ids = extract_traj_tokens(
            generated_tokens,
            self.special_token_ids,
            self.config.tokens_per_future_traj,
            self.future_token_start_idx,
            self.traj_tokenizer.vocab_size,
        )

        pred_xyz, pred_rot, _ = self.traj_tokenizer.decode(
            hist_xyz=einops.repeat(
                ego_history_xyz[:, -1],
                "b ... -> (b n) ...",
                n=n_samples_total,
            ),
            hist_rot=einops.repeat(
                ego_history_rot[:, -1],
                "b ... -> (b n) ...",
                n=n_samples_total,
            ),
            tokens=traj_token_ids,
        )
        pred_xyz = einops.rearrange(
            pred_xyz,
            "(b ns nj) ... -> b ns nj ...",
            ns=num_traj_sets,
            nj=num_traj_samples,
        )
        pred_rot = einops.rearrange(
            pred_rot,
            "(b ns nj) ... -> b ns nj ...",
            ns=num_traj_sets,
            nj=num_traj_samples,
        )

        # return additional information
        if kwargs.get("return_extra", False):
            extra = extract_text_tokens(self.tokenizer, generated_tokens)
            # rearrange text tokens to shape [B, ns, nj] to match trajectory shape
            for text_tokens in extra.keys():
                extra[text_tokens] = np.array(extra[text_tokens]).reshape(
                    [input_ids.shape[0], num_traj_sets, num_traj_samples]
                )
            return pred_xyz, pred_rot, extra
        return pred_xyz, pred_rot
