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

from contextlib import nullcontext
from typing import Any, Self

import einops
import numpy as np
import torch
from transformers import LogitsProcessorList, StoppingCriteriaList

from alpamayo1_5_sft.models.sft_base_model import ReasoningVLAOutput, load_alpamayo1_vlm
from alpamayo_r1.models.base_model import IGNORE_INDEX
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1, ExpertLogitsProcessor
from alpamayo_r1.models.token_utils import (
    StopAfterEOS,
    extract_text_tokens,
    replace_padding_after_eos,
    to_special_token,
)
from alpamayo_r1.config import AlpamayoR1Config
from alpamayo.common import misc
from alpamayo_r1.common import logging

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")


class TrainableAlpamayoR1(AlpamayoR1):
    def __init__(
        self,
        config: AlpamayoR1Config,
        pretrained_modules: dict[str, torch.nn.Module] | None = None,
        original_vocab_size: int | None = None,
        cotrain_vlm: bool = False,
        stop_grad_from_vlm: bool = True,
        stage1_vlm_checkpoint_path: str | None = None,
    ):
        if stage1_vlm_checkpoint_path is not None:
            raise ValueError(
                "stage1_vlm_checkpoint_path is only supported by "
                "TrainableAlpamayoR1.from_pretrained()"
            )

        super().__init__(config, pretrained_modules, original_vocab_size)

        self.cotrain_vlm = cotrain_vlm
        self.stop_grad_from_vlm = stop_grad_from_vlm

        self._set_vlm_trainability()
        # print the param count
        logger.info("Model parameter count:")
        param_count = misc.get_param_count(self)
        for key, value in param_count.items():
            logger.info(f"{key}: {value:,}")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> Self:
        stage1_vlm_checkpoint_path = kwargs.pop("stage1_vlm_checkpoint_path", None)
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )

        if stage1_vlm_checkpoint_path is not None:
            model.vlm = load_alpamayo1_vlm(
                stage1_vlm_checkpoint_path,
                model.vlm,
                preserve_model_device_and_dtype=True,
            )

        model._set_vlm_trainability()
        return model

    def _set_vlm_trainability(self) -> None:
        for param in self.vlm.parameters():
            param.requires_grad = self.cotrain_vlm

    def _process_traj_future_training(self, traj_data: dict[str, Any]) -> dict[str, Any]:
        """Process the trajectory future data for training."""
        ego_history_xyz = traj_data["ego_history_xyz"]
        ego_history_rot = traj_data["ego_history_rot"]
        ego_future_xyz = traj_data["ego_future_xyz"]
        ego_future_rot = traj_data["ego_future_rot"]
        action = self.action_space.traj_to_action(
            traj_history_xyz=ego_history_xyz,
            traj_history_rot=ego_history_rot,
            traj_future_xyz=ego_future_xyz,
            traj_future_rot=ego_future_rot,
        )
        action = action.reshape(-1, *self.action_space.get_action_space_dims())
        training_data: dict[str, Any] = self.diffusion.construct_training_data(action)
        return training_data

    def _process_position_ids_qwen2_5_vl(
        self, vlm_outputs: Any, batch_size: int, num_expert_tokens: int, device: torch.device
    ) -> torch.Tensor:
        """Process the position ids for the expert model.

        Qwen 2.5 VL has a special RoPE, so we need to process the position ids
        Args:
            vlm_outputs: The outputs of the VLM model.
            batch_size: The batch size.
            num_expert_tokens: The number of expert tokens.
            device: The device.
        Returns:
            The processed position ids.
        """
        position_ids = torch.arange(num_expert_tokens, device=device)
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=batch_size).clone()
        delta = vlm_outputs.rope_deltas + vlm_outputs.past_key_values.get_seq_length()
        position_ids += delta.to(position_ids.device)
        return position_ids

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
        batch_size = input_ids.shape[0]
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
        if self.cotrain_vlm:
            context = nullcontext()
        else:
            context = torch.no_grad()

        with context:
            vlm_outputs = self.vlm(
                input_ids=input_ids,
                labels=labels,
                use_cache=True,
                **tokenized_data,
            )

        future_start_token_id = self.config.traj_token_ids["future_start"]
        future_start_positions = (input_ids == future_start_token_id).nonzero(as_tuple=False)
        # The KV-cache below is a single batched tensor, so it can only be cropped at
        # one shared position, and position_ids are offset by that same shared scalar
        # for every row. That's only correct if <traj_future_start> lands at the same
        # absolute index in every sample -- true today because collate_fn left-pads
        # (right-flushing each sample's real tokens) and traj_future is the last
        # component in components_order (a fixed-length tail after the token for every
        # sample). Assert it explicitly so a future padding_side or components_order
        # change fails loudly here instead of silently corrupting gradients for all
        # rows but the last. See STAGE2_POSITION_ID_INVESTIGATION.md.
        assert future_start_positions.shape[0] == batch_size, (
            "expected exactly one <traj_future_start> token per sample"
        )
        future_start_idx_per_row = future_start_positions[:, 1]
        assert (future_start_idx_per_row == future_start_idx_per_row[0]).all(), (
            "<traj_future_start> index differs across the batch -- the shared "
            "KV-cache crop / position_ids below assume left-padding with traj_future "
            "last in components_order; that invariant just broke."
        )
        last_traj_future_start_idx = future_start_idx_per_row[0] + 1

        future_traj_data = self._process_traj_future_training(traj_data)
        # [B, n_token_per_future_traj, hidden_size]
        action_embeds = self.action_in_proj(
            future_traj_data["noisy_x"], future_traj_data["timesteps"]
        )
        # [B, n_token_per_history_traj + n_token_per_future_traj, hidden_size]
        expert_embeds = action_embeds
        # NOTE: we don't need to update the rope deltas as we assume after <traj_future_start> there
        # will be no more vision tokens.
        kv_cache = vlm_outputs.past_key_values
        # crop the kv cache to the last <traj_future_start> token
        kv_cache.crop(last_traj_future_start_idx)
        if self.stop_grad_from_vlm:
            for layer in kv_cache.layers:
                layer.keys = layer.keys.detach()
                layer.values = layer.values.detach()
        position_ids = self._process_position_ids_qwen2_5_vl(
            vlm_outputs, batch_size, expert_embeds.shape[1], expert_embeds.device
        )
        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False
        expert_outputs = self.expert(
            inputs_embeds=expert_embeds,
            position_ids=position_ids,
            past_key_values=kv_cache,
            attention_mask=None,
            use_cache=True,
            **forward_kwargs,
        )
        diffusion_out = expert_outputs.last_hidden_state[:, -action_embeds.shape[1] :]
        pred = self.action_out_proj(diffusion_out)
        pred = pred.view(-1, *self.action_space.get_action_space_dims())
        future_traj_loss = (
            self.diffusion.compute_loss_from_pred(
                training_data=future_traj_data,
                pred=pred,
            )
            # TODO: only support traj finetune for now, so no weight, add weight later when other losses added
            # * self.config.traj_loss_weight
        )
        loss = future_traj_loss
        if self.cotrain_vlm:
            loss += vlm_outputs.loss

        return ReasoningVLAOutput(
            loss=loss,
        )

    def sample_trajectories_from_data(  # type: ignore[override]
        self,
        data: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]
    ):
        """Sample trajectories from the data.

        Args:
            with_vlm_rollout: Whether to use VLM rollout.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        return self.sample_trajectories_from_data_with_vlm_rollout(
            data,
            *args,
            **kwargs,
        )

    def sample_trajectories_from_data_with_vlm_rollout(  # type: ignore[override]
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        diffusion_kwargs: dict[str, Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]
    ):
        """Sample trajectories from the data with VLM rollout.

        Fork override of ``AlpamayoR1.sample_trajectories_from_data_with_vlm_rollout``.
        The vendored version's expert attention mask only masks the trailing
        over-generation region after each row's own ``<traj_future_start>``; it never
        masks the batch's left-padding, so the diffusion expert cross-attends into
        pad-token KV states for any row shorter than its batch-mates. Confirmed by
        experiment (up to 1.39m of trajectory error at realistic padding widths) --
        see STAGE2_POSITION_ID_INVESTIGATION.md. This override adds the missing
        prefix mask, matching the fix already present in the canonical alpamayo1.5
        release (``Alpamayo1_5._build_expert_pos_ids_and_attn_mask``'s
        ``prefix_mask``).

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
        n_samples_total = num_traj_samples * num_traj_sets
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        B, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "Only one trajectory group is supported for inference."
        tokenized_data = data["tokenized_data"]
        input_ids = tokenized_data.pop("input_ids")
        traj_data_vlm = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)
        device = input_ids.device

        # Capture the prompt-level padding mask ([B, L], 0 = pad) before generate()
        # consumes tokenized_data -- needed below to mask left-padding out of the
        # expert's cross-attention.
        prefix_mask = tokenized_data.get("attention_mask")

        # 1) run autoregressive generation for the VLM
        max_generation_length = kwargs.get(
            "max_generation_length", self.config.tokens_per_future_traj
        )
        generation_config = self.vlm.generation_config
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.do_sample = True
        generation_config.num_return_sequences = num_traj_samples
        generation_config.max_new_tokens = max_generation_length
        generation_config.output_logits = True
        generation_config.return_dict_in_generate = True
        generation_config.top_k = top_k
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        # use custom stopping criteria to stop after EOS token + one more token,
        # because the KV cache is updated after the next token is generated
        eos_token_id = self.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
        stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])
        logits_processor = LogitsProcessorList(
            [
                ExpertLogitsProcessor(
                    traj_token_offset=self.config.traj_token_start_idx,
                    traj_vocab_size=self.config.traj_vocab_size,
                )
            ]
        )
        vlm_outputs = self.vlm.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            **tokenized_data,
        )
        vlm_outputs.rope_deltas = self.vlm.model.rope_deltas

        # manually replace padding after EOS token
        vlm_outputs.sequences = replace_padding_after_eos(
            token_ids=vlm_outputs.sequences,
            eos_token_id=eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_cache = vlm_outputs.past_key_values
        prefill_seq_len = prompt_cache.get_seq_length()

        # find <traj_future_start> token position for each sequence, use last token if not found
        b_star = vlm_outputs.sequences.shape[0]
        traj_future_start_mask = vlm_outputs.sequences == eos_token_id
        # [b_star], True if sequence has <traj_future_start>
        has_traj_future_start = traj_future_start_mask.any(dim=1)
        for i in range(b_star):
            if not has_traj_future_start[i]:
                logger.warning(
                    f"No <traj_future_start> token found in the generated sequences for sequence {i}"
                )
        # [b_star], first occurrence position
        traj_future_start_positions = traj_future_start_mask.int().argmax(dim=1)
        last_token_positions = torch.full(
            (b_star,), vlm_outputs.sequences.shape[1] - 1, device=device
        )
        valid_token_pos_id = torch.where(
            has_traj_future_start, traj_future_start_positions, last_token_positions
        )
        # note that vlm_outputs.sequences already include the input_ids,
        # so no need to add the input_ids length
        offset = valid_token_pos_id + 1

        # modify the position ids to remove padding tokens
        n_diffusion_tokens = self.action_space.get_action_space_dims()[0]
        position_ids = torch.arange(n_diffusion_tokens, device=device)
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b_star).clone()
        delta = vlm_outputs.rope_deltas + offset[:, None]
        position_ids += delta.to(position_ids.device)

        # modify the attention_masks to remove padding tokens
        attention_mask = torch.zeros(
            (b_star, 1, n_diffusion_tokens, prompt_cache.get_seq_length() + n_diffusion_tokens),
            dtype=torch.float32,
            device=device,
        )
        for i in range(b_star):
            attention_mask[i, :, :, offset[i] : -n_diffusion_tokens] = torch.finfo(
                attention_mask.dtype
            ).min

        # FIX: also mask each row's left-padding prefix (see docstring above and
        # STAGE2_POSITION_ID_INVESTIGATION.md). `prefix_mask` is [B, L]; expand with
        # repeat_interleave(num_traj_samples) to match generate()'s own row order for
        # num_return_sequences -- NOT n_samples_total, since b_star above is only
        # B * num_traj_samples (num_traj_sets multiplies the diffusion batch later,
        # after the VLM rollout, and isn't part of b_star here).
        if prefix_mask is not None:
            prefix_mask_expanded = torch.repeat_interleave(
                prefix_mask, num_traj_samples, dim=0
            ).to(device)
            assert prefix_mask_expanded.shape[0] == b_star
            pad_region = prefix_mask_expanded[:, None, None, :]
            L = pad_region.shape[-1]
            attention_mask[:, :, :, :L] = torch.where(
                pad_region == 0,
                torch.finfo(attention_mask.dtype).min,
                attention_mask[:, :, :, :L],
            )

        forward_kwargs = {}
        if self.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        # 2) Define denoising step that consumes noisy action and timestep
        def step_fn(
            x: torch.Tensor,
            t: torch.Tensor,
        ) -> torch.Tensor:
            # x: (B*, *action_dim)
            # t: broadcastable to x leading dims
            b_star = x.shape[0]
            # Project noisy action to expert token embeddings for the n future tokens
            # Expect shape (b*, n_token_per_traj, hidden_size)
            future_token_embeds = self.action_in_proj(x, t)
            if future_token_embeds.dim() == 2:
                future_token_embeds = future_token_embeds.view(b_star, n_diffusion_tokens, -1)

            # Run expert with cached prefill, only on the future tokens
            expert_out_base = self.expert(
                inputs_embeds=future_token_embeds,
                position_ids=position_ids,
                past_key_values=prompt_cache,
                attention_mask=attention_mask,
                use_cache=True,
                **forward_kwargs,
            )
            # crop the prompt cache to remove the newly added tokens
            prompt_cache.crop(prefill_seq_len)
            last_hidden = expert_out_base.last_hidden_state  # (b*, Tf, hidden_size)
            last_hidden = last_hidden[:, -n_diffusion_tokens:]
            pred = self.action_out_proj(last_hidden).view(
                -1, *self.action_space.get_action_space_dims()
            )  # (b*, Tf, C_action) -> noise/vector field
            return pred

        # 3) Diffusion sampling in action space with multiple samples per input
        total_batch = B * n_samples_total
        if diffusion_kwargs is None:
            diffusion_kwargs = {}

        sampled_action = self.diffusion.sample(
            batch_size=total_batch,
            step_fn=step_fn,
            device=device,
            return_all_steps=False,
            **diffusion_kwargs,
        )

        # Repeat history to align with num_traj_samples
        hist_xyz_rep = einops.repeat(
            ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )
        hist_rot_rep = einops.repeat(
            ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total
        )

        pred_xyz, pred_rot = self.action_space.action_to_traj(
            sampled_action, hist_xyz_rep, hist_rot_rep
        )

        # 4) Reshape to (B, num_traj_samples, n_traj, ...)
        pred_xyz = einops.rearrange(
            pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )
        pred_rot = einops.rearrange(
            pred_rot, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
        )

        # return the text tokens generated by the VLM
        if kwargs.get("return_extra", False):
            extra = extract_text_tokens(self.tokenizer, vlm_outputs.sequences)
            # rearrange text tokens to shape [B, ns, nj] to match trajectory shape
            for text_tokens in extra.keys():
                extra[text_tokens] = np.array(extra[text_tokens]).reshape(
                    [input_ids.shape[0], num_traj_sets, num_traj_samples]
                )
            return pred_xyz, pred_rot, extra
        return pred_xyz, pred_rot
