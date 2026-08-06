"""True autoregressive rollout of the VLM's discrete future-trajectory tokens.

Alpamayo 2 Super's own release inference (``sample_trajectories_from_data``)
always masks discrete-trajectory-token generation out and routes the real
trajectory through the frozen diffusion expert -- there is no "generate the
discrete trajectory tokens autoregressively" path in the release code. This
module adds one, mirroring how Alpamayo-1.5's
``TrainableReasoningVLA.sample_trajectories_from_data`` decodes its own
``vlm.generate()`` output: each trajectory token is sampled conditioned on the
model's *own* previously generated tokens (not ground truth), so early errors
can propagate into later ones -- unlike the teacher-forced argmax decode this
replaces in ``viz_callback.py``.
"""

from __future__ import annotations

import copy
from typing import Any

import torch

from alpamayo2_super.chat_template.conversation import build_conversation
from alpamayo2_super.models.utils import fuse_traj_tokens


def build_generation_prompt(model: Any, data: dict[str, Any], processor: Any) -> dict[str, Any]:
    """Tokenize a prompt-only (no GT future) conversation ending at
    ``<|traj_future_start|>``, ready for ``model.vlm.generate()``.

    History is fused into the prompt (real context, same as release inference);
    the future span carries no placeholder tokens at all here (``ask_for_component=
    True`` on the last component emits only the start token -- see
    ``conversation.get_component_str``), so only ``ego_history_xyz``/``ego_history_rot``
    are passed to ``fuse_traj_tokens``.
    """
    messages = build_conversation(
        data=data,
        num_tokens_per_history_traj=model.config.tokens_per_history_traj,
        num_tokens_per_future_traj=model.config.tokens_per_future_traj,
        components_order=["image", "traj_history", "prompt", "traj_future"],
        components_prompt=["traj_future"],
        generation_mode=True,
        include_camera_ids=model.config.include_camera_ids,
        camera_ids=data["camera_indices"],
        include_frame_nums=model.config.frame_label == "frame_num",
    )
    has_assistant_content = messages[-1]["role"] == "assistant" and bool(messages[-1]["content"])
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not has_assistant_content,
        add_vision_id=False,
        continue_final_message=has_assistant_content,
    )
    images = data["image_frames"].flatten(0, 1)
    images = (images.float() / 255.0) if images.dtype == torch.uint8 else images.float()
    tokenized_data = dict(
        processor(
            text=text,
            images=images,
            videos=None,
            padding=False,
            return_tensors="pt",
            do_rescale=False,
        )
    )
    traj_data_hist_only = {
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    tokenized_data["input_ids"] = fuse_traj_tokens(
        model.history_traj_tokenizer,
        model.future_traj_tokenizer,
        tokenized_data["input_ids"],
        traj_data_hist_only,
        model.config.traj_ids,
    )
    return tokenized_data


def rollout_future_trajectory(
    model: Any,
    tokenized_data: dict[str, torch.Tensor],
    ego_history_xyz: torch.Tensor,
    ego_history_rot: torch.Tensor,
    num_traj_samples: int = 1,
    top_p: float = 0.98,
    temperature: float = 0.6,
    do_sample: bool = True,
) -> torch.Tensor:
    """Autoregressively generate discrete future-trajectory tokens and decode them.

    Unlike the teacher-forced decode this replaces, each generated token is
    conditioned on the model's own previously *generated* tokens (via
    ``model.vlm.generate()``'s normal KV-cache autoregression) -- not on
    ground truth -- so this reflects what actually happens when the model
    predicts on its own.

    Args:
        model: Alpamayo2Super (with LoRA already applied, if any).
        tokenized_data: Output of ``build_generation_prompt`` for ONE sample
            (batch dim of 1) -- history fused, ending at ``<|traj_future_start|>``.
        ego_history_xyz: [1, n_traj, T_hist, 3], as stored in traj_data.
        ego_history_rot: [1, n_traj, T_hist, 3, 3].
        num_traj_samples: Number of independent rollouts to sample.
        top_p, temperature: Nucleus sampling params (ignored if do_sample=False).
        do_sample: False for greedy decoding.

    Returns:
        Predicted future xyz, shape [1, num_traj_samples, T_future, 3].
    """
    device = next(model.parameters()).device
    input_ids = tokenized_data["input_ids"].to(device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    attention_mask = tokenized_data.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
    vision_keys = ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")
    vision_inputs = {
        key: tokenized_data[key].to(device) for key in vision_keys if key in tokenized_data
    }

    generation_config = copy.deepcopy(model.vlm.generation_config)
    generation_config.max_new_tokens = model.config.tokens_per_future_traj
    # Force exactly `tokens_per_future_traj` new tokens regardless of what the
    # model emits -- we decode a fixed-length trajectory, not a stop-token-
    # terminated string, so early-stopping on EOS would corrupt the decode
    # with pad tokens if the model happens to emit one early.
    generation_config.eos_token_id = None
    generation_config.do_sample = do_sample
    generation_config.top_p = top_p if do_sample else None
    generation_config.temperature = temperature if do_sample else None
    generation_config.num_return_sequences = num_traj_samples
    generation_config.pad_token_id = model.tokenizer.pad_token_id
    generation_config.return_dict_in_generate = True
    generation_config.output_logits = False

    with torch.no_grad():
        outputs = model.vlm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            **vision_inputs,
        )
    # generate() expands the batch via repeat_interleave(num_return_sequences),
    # so rows are grouped [b0_k0, b0_k1, ..., b1_k0, ...] -- verified empirically.
    generated = outputs.sequences[:, input_ids.shape[1] :]
    generated = generated[:, : model.config.tokens_per_future_traj]

    future_id0 = model.config.traj_ids["future_id0"]
    future_vocab_size = model.config.future_vocab_size
    pred_ids = (generated - future_id0).clamp(min=0, max=future_vocab_size - 1)

    batch_size = input_ids.shape[0]
    hist_xyz = ego_history_xyz.to(device).flatten(0, 1)  # [B, T_hist, 3]
    hist_rot = ego_history_rot.to(device).flatten(0, 1)
    hist_xyz_rep = hist_xyz.repeat_interleave(num_traj_samples, dim=0)
    hist_rot_rep = hist_rot.repeat_interleave(num_traj_samples, dim=0)

    fut_xyz, _fut_rot, _ = model.future_traj_tokenizer.decode(
        hist_xyz=hist_xyz_rep, hist_rot=hist_rot_rep, tokens=pred_ids
    )
    return fut_xyz.view(batch_size, num_traj_samples, *fut_xyz.shape[1:])
