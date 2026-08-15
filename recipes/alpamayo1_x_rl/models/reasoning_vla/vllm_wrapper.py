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

"""vLLM wrapper for the ReasoningVLA VLM module."""

import logging
from typing import Iterable, Optional, Union

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from vllm.config import MultiModalConfig, VllmConfig
from vllm.model_executor.models.interfaces import SupportsMRoPE, SupportsMultiModal
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.v1.sample.metadata import SamplingMetadata  # vLLM >= 0.11

from alpamayo1_x_rl.models.reasoning_vla.weight_mapper import ReasoningVLAWeightMapper


class ReasoningVLAModelForVLLM(torch.nn.Module, SupportsMultiModal, SupportsMRoPE):
    """vLLM wrapper that hosts the VLM module of ReasoningVLA."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        orig_cfg = vllm_config.model_config.hf_config
        llm_cfg = orig_cfg.get_llm_config()

        self._orig_hf_config_for_mapper = orig_cfg
        self._hf_config_for_mapper = llm_cfg

        if hasattr(llm_cfg, "get_text_config"):
            llm_text_cfg = llm_cfg.get_text_config()
        else:
            llm_text_cfg = getattr(llm_cfg, "text_config", llm_cfg)

        ckpt_vocab = getattr(orig_cfg, "vocab_size", None)
        if ckpt_vocab is not None:
            if getattr(llm_cfg, "text_config", None) is not None:
                llm_cfg.text_config.vocab_size = ckpt_vocab
                llm_cfg.text_config.pad_vocab_size_multiple = 1
            llm_cfg.vocab_size = ckpt_vocab
            llm_cfg.pad_vocab_size_multiple = 1
            vllm_config.model_config.vocab_size = ckpt_vocab

        mc = vllm_config.model_config
        mc.hf_config = llm_cfg
        mc.hf_text_config = llm_text_cfg
        mc.task = "generate"
        mc.multimodal_config = getattr(mc, "multimodal_config", None) or MultiModalConfig()

        self.vlm = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "vlm"),
            architectures=["Qwen3VLForConditionalGeneration"],
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        """Return the placeholder token string for a given modality."""
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        raise ValueError("Only image or video modality is supported")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Forward pass delegated to the underlying VLM."""
        if input_ids is None and inputs_embeds is None:
            raise ValueError("input_ids and inputs_embeds cannot be None at the same time")
        return self.vlm(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def compute_logits(self, hidden_states, sampling_metadata: Optional[SamplingMetadata] = None):
        """Compute logits from hidden states, forwarding to the VLM head."""
        try:
            if sampling_metadata is None:
                return self.vlm.compute_logits(hidden_states)
            return self.vlm.compute_logits(hidden_states, sampling_metadata)
        except TypeError:
            return self.vlm.compute_logits(hidden_states)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return token embeddings, optionally merged with multimodal embeddings."""
        # NOTE: fork change. vllm 0.20 renamed this pair of hooks; 0.11 knows
        # only the old names and 0.20 only the new ones (verified: the two sets
        # are disjoint on Qwen3VLForConditionalGeneration in each venv). The
        # model runner calls whichever name its own version defines, so the
        # wrapper exposes BOTH and forwards to whichever the underlying VLM
        # actually has:
        #     vllm 0.11        ->  get_input_embeddings / get_multimodal_embeddings
        #     vllm 0.20        ->  embed_input_ids      / embed_multimodal
        # `is_multimodal` is 0.20-only and is dropped on the legacy path, which
        # matches the old signature (input_ids, multimodal_embeddings).
        if hasattr(self.vlm, "embed_input_ids"):
            return self.vlm.embed_input_ids(
                input_ids, multimodal_embeddings, is_multimodal=is_multimodal
            )
        return self.vlm.get_input_embeddings(input_ids, multimodal_embeddings)

    def embed_multimodal(self, **kwargs: object):
        """Compute multimodal (image/video) embeddings via the VLM encoder."""
        if hasattr(self.vlm, "embed_multimodal"):
            return self.vlm.embed_multimodal(**kwargs)
        return self.vlm.get_multimodal_embeddings(**kwargs)

    # Legacy (vllm 0.11) names for the two hooks above. That version's model
    # runner looks up these names, so they must exist on the wrapper even though
    # the 0.20 path never calls them. Both delegate to the version-dispatching
    # implementations above rather than duplicating the branch.
    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """vllm 0.11 alias for :meth:`embed_input_ids`."""
        return self.embed_input_ids(
            input_ids, multimodal_embeddings, is_multimodal=is_multimodal
        )

    def get_multimodal_embeddings(self, **kwargs: object):
        """vllm 0.11 alias for :meth:`embed_multimodal`."""
        return self.embed_multimodal(**kwargs)

    def get_mrope_input_positions(self, input_tokens, mm_features=None, **kwargs):
        """Compute M-RoPE positions, forwarding to the VLM's Qwen3-VL implementation.

        NOTE: fork change. vLLM changed this method's calling convention between
        the two versions this fork is run under, and the model runner calls it
        directly -- so the wrapper has to accept both shapes or the rollout dies
        on the first request:

            vllm 0.11.0 (gpu_model_runner.py:753) -- the pre-`mm_features` form,
                everything by keyword:
                    (prompt_token_ids, hf_config=..., image_grid_thw=...,
                     video_grid_thw=..., second_per_grid_ts=...,
                     audio_feature_lengths=..., use_audio_in_video=...)
                implemented in vllm/model_executor/models/qwen2_vl.py:1139

            vllm 0.20.0 (gpu_model_runner.py:1505) -- multimodal inputs collapsed
                into one positional argument:
                    (prompt_token_ids, mm_features)
                implemented in vllm/model_executor/models/qwen3_vl.py:2498

        Both are forwarded verbatim to the underlying VLM, which in each venv
        implements the convention its own vLLM uses -- so this shim only routes,
        it never translates between the two. Passing `mm_features` positionally
        keeps the 0.20 path identical to before.

        Observed 2026-08-14 as
        `TypeError: get_mrope_input_positions() got an unexpected keyword
        argument 'hf_config'` on the vllm 0.11.0 venv
        (logs_20260815-024109/rollout_0.log).
        """
        if mm_features is not None:
            return self.vlm.get_mrope_input_positions(input_tokens, mm_features)
        # vllm 0.11's Qwen3-VL does NOT implement get_mrope_input_positions at
        # all (it landed with the 0.20 refactor), so there is nothing to forward
        # to. That version's model runner handles such models in its `else`
        # branch (gpu_model_runner.py:762-773) by calling the shared helper with
        # these same kwargs -- so calling it here is exactly what vLLM would do
        # for us if the wrapper did not advertise SupportsMRoPE.
        from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding

        return MRotaryEmbedding.get_input_positions_tensor(input_tokens, **kwargs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load checkpoint weights into the VLM using the ReasoningVLA weight mapper."""
        mapper = ReasoningVLAWeightMapper(self._orig_hf_config_for_mapper)
        mapper.setup_rollout_backend("vllm")
        inplace_map, _ = mapper.rollout_prepare_recv(self.vlm)

        def normalize(hf_name: str) -> list[str]:
            name = hf_name.replace("language_model.", "")
            candidates: list[str] = [name]
            if name.startswith("vlm."):
                candidates.append(name.replace("vlm.", "", 1))
                candidates.append(name.replace("vlm.model.", "model."))
                if name.startswith("vlm.model.visual."):
                    candidates.append(name.replace("vlm.model.visual.", "visual."))
                if name.startswith("vlm.visual."):
                    candidates.append(name.replace("vlm.visual.", "visual."))
            if name.startswith("model."):
                candidates.append(f"vlm.{name}")
                candidates.append(name.replace("model.", "vlm.model."))
                if name.startswith("model.visual."):
                    candidates.append(name.replace("model.visual.", "visual."))
            if name.startswith("visual."):
                candidates.append(f"vlm.model.{name}")
                candidates.append(f"model.{name}")
                candidates.append(f"vlm.{name}")
            more: list[str] = []
            for n in candidates:
                idx = n.find("visual.")
                if idx != -1:
                    suffix = n[idx:]
                    more.append(f"vlm.{suffix}")
            for n in more:
                if n not in candidates:
                    candidates.append(n)
            extra: list[str] = []
            for n in list(candidates):
                if ".attn.proj." in n:
                    extra.append(n.replace(".attn.proj.", ".attn.out_proj."))
                if ".attn.out_proj." in n:
                    extra.append(n.replace(".attn.out_proj.", ".attn.proj."))
                if ".attn.qkv_proj." in n:
                    extra.append(n.replace(".attn.qkv_proj.", ".attn.qkv."))
                if ".attn.qkv." in n:
                    extra.append(n.replace(".attn.qkv.", ".attn.qkv_proj."))
            for n in extra:
                if n not in candidates:
                    candidates.append(n)
            seen = set()
            result = []
            for n in candidates:
                if n not in seen:
                    seen.add(n)
                    result.append(n)
            if "qkv" in name and "visual" in name:
                result.append(name.replace("vlm.model.visual.", "visual.", 1).replace("qkv", "q"))
                result.append(name.replace("vlm.model.visual.", "visual.", 1).replace("qkv", "k"))
                result.append(name.replace("vlm.model.visual.", "visual.", 1).replace("qkv", "v"))
            return result

        unused_ckpt_keys: set[str] = set()

        for raw_name, tensor in weights:
            copied = False
            for key in normalize(raw_name):
                dst = inplace_map.get(key, None)
                if dst is None:
                    continue
                target = dst if isinstance(dst, torch.Tensor) else dst()

                if "qkv" in raw_name and "qkv" not in key:
                    if ".q." in key:
                        target.data.copy_(
                            tensor[: tensor.shape[0] // 3].to(
                                dtype=target.dtype, device=target.device
                            )
                        )
                    elif ".k." in key:
                        target.data.copy_(
                            tensor[tensor.shape[0] // 3 : tensor.shape[0] // 3 * 2].to(
                                dtype=target.dtype, device=target.device
                            )
                        )
                    elif ".v." in key:
                        target.data.copy_(
                            tensor[tensor.shape[0] // 3 * 2 :].to(
                                dtype=target.dtype, device=target.device
                            )
                        )
                    copied = True
                    continue

                if target.shape != tensor.shape:
                    continue

                target.data.copy_(tensor.to(dtype=target.dtype, device=target.device))
                copied = True
                break

            if not copied:
                unused_ckpt_keys.add(raw_name)

        logging.info(
            "[ReasoningVLAModelForVLLM] Unused checkpoint keys: %s",
            unused_ckpt_keys,
        )

        loaded_param_names: set[str] = {name for name, _ in self.named_parameters()}
        return loaded_param_names


class ReasoningVLAProcessingInfo(Qwen3VLProcessingInfo):
    """Processing info that resolves the HF config for composite ReasoningVLA models."""

    def get_hf_config(self):
        """Return the underlying Qwen3VL config from the composite HF config."""
        try:
            return self.ctx.get_hf_config(Qwen3VLConfig)
        except TypeError:
            cfg = self.ctx.model_config.hf_config
            if hasattr(cfg, "get_llm_config"):
                return cfg.get_llm_config()
            return super().get_hf_config()


MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=ReasoningVLAProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)(ReasoningVLAModelForVLLM)
