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
import logging
from typing import Any

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from alpamayo2_super.chat_template.conversation import build_conversation
from alpamayo2_super.models.alpamayo2_super import (
    MaskDiscreteTrajectoryLogitsProcessor,
    _append_text_eos_mask,
)
from alpamayo2_super.models.token_utils import extract_text_tokens
from alpamayo2_super.models.utils import SPECIAL_TOKENS, fuse_traj_tokens

logger = logging.getLogger(__name__)


class _RestrictToFutureTrajVocab(LogitsProcessor):
    """Mask every logit outside the future-trajectory token range.

    We force exactly ``tokens_per_future_traj`` new tokens (``eos_token_id=None``
    below), so every generated position MUST be decodable as an action bin.
    Nothing in ``generate()`` guarantees that: the model can emit any token in
    the 155k-token vocab -- most importantly ``<|traj_future_end|>``, which it
    was trained to emit right after the 128th trajectory token and will emit
    early whenever it miscounts the span.

    Left unmasked, those tokens do not error out; they are silently *clamped*
    into the action range (see ``pred_ids`` below, and upstream's own
    ``token_utils.extract_traj_tokens``, which clamps the same way). The clamp
    is what makes the failure look like driving rather than like corruption:
    ``traj_future_end`` (155683) minus ``future_id0`` (152669) is 3014, which
    clamps to bin 2999 -- the *maximum* of both action channels, i.e. accel
    +6.84 m/s^2 and curvature +0.262 1/m (a 3.8 m turn radius). Since the
    unicycle decoder integrates curvature into heading and heading into
    position, a tail of those tokens renders as a smooth, plausible path that
    suddenly winds into tight circles. Masking makes the decode well-posed:
    the model can only choose among real action bins.

    Because masking makes out-of-vocab tokens unreachable, the clamp counter at
    the bottom of ``rollout_future_trajectory`` reads zero by construction and
    proves nothing about *why* the tokens were bad. So this processor also
    records what the model WOULD have emitted: the pre-mask argmax is free to
    look at, and whenever it lands outside the action range we log the first
    such step per sequence together with the offending token id. That is the
    evidence that distinguishes "the model emitted ``traj_future_end`` early"
    (id 155683) from "the model wandered into text" (id < 152669).
    """

    def __init__(
        self, future_id0: int, future_vocab_size: int, future_end_id: int | None = None
    ) -> None:
        self.lo = future_id0
        self.hi = future_id0 + future_vocab_size
        # Named only so the diagnostic can say "this was the end token" rather than
        # printing a bare id; read from the checkpoint's traj_ids, never assumed.
        self.future_end_id = future_end_id
        self.reset()

    def reset(self) -> None:
        """Clear per-``generate()``-call diagnostics (the instance is reused per chunk)."""
        self.step = 0
        self.first_bad_step: torch.Tensor | None = None
        self.first_bad_token: torch.Tensor | None = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
        # Pre-mask argmax: what the unconstrained decode would have picked here.
        # Temperature/top-p warpers are monotonic, so argmax is unaffected by
        # where in the processor list we sit.
        top = scores.argmax(dim=-1)
        bad = (top < self.lo) | (top >= self.hi)
        if self.first_bad_step is None:
            self.first_bad_step = torch.full_like(top, -1)
            self.first_bad_token = torch.full_like(top, -1)
        newly_bad = bad & (self.first_bad_step < 0)
        if newly_bad.any():
            self.first_bad_step = torch.where(newly_bad, self.step, self.first_bad_step)
            self.first_bad_token = torch.where(newly_bad, top, self.first_bad_token)
        self.step += 1

        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.lo : self.hi] = 0.0
        return scores + mask

    def log_summary(self, tokens_per_future_traj: int) -> None:
        """Warn about sequences whose unconstrained decode would have gone off-vocab."""
        if self.first_bad_step is None:
            return
        hit = self.first_bad_step >= 0
        if not bool(hit.any()):
            return
        steps = self.first_bad_step[hit].tolist()
        toks = self.first_bad_token[hit].tolist()
        logger.warning(
            "rollout: %d/%d sampled sequences would have left the trajectory vocab "
            "(suppressed). First offending (step_of_%d, token_id) pairs: %s. "
            "traj_future_end=%s; ids below %d are text.",
            int(hit.sum()),
            self.first_bad_step.numel(),
            tokens_per_future_traj,
            list(zip(steps, toks))[:8],
            self.future_end_id,
            self.lo,
        )


def build_generation_prompt(
    model: Any, data: dict[str, Any], processor: Any, with_cot: bool = False
) -> dict[str, Any]:
    """Tokenize a prompt-only (no GT future) conversation, ready for ``model.vlm.generate()``.

    History is fused into the prompt (real context, same as release inference);
    the last assistant component carries no placeholder tokens at all here
    (``ask_for_component=True`` on the last component emits only its start
    token -- see ``conversation.get_component_str``), so only
    ``ego_history_xyz``/``ego_history_rot`` are passed to ``fuse_traj_tokens``.

    Args:
        with_cot: If False (default), the prompt ends at ``<|traj_future_start|>``
            -- the model goes straight to trajectory tokens, as in release
            trajectory-only inference. If True, the prompt is byte-identical to
            the release CoC prompt (``helper.create_messages``): it ends at the
            empty assistant turn, priming the model to free-generate reasoning
            text and only THEN continue on its own into
            ``<|traj_future_start|>`` -- pair with
            ``rollout_future_trajectory_with_cot``, not the plain rollout.
    """
    if with_cot:
        # NOTE (fork): components_order must NOT contain "cot". The release CoC
        # prompt (``helper.create_messages``) ends at the *user* "prompt"
        # component, so the assistant turn is empty and the prompt ends at
        # ``<|im_start|>assistant\n`` -- the checkpoint then writes its reasoning
        # as plain text and never emits ``<|cot_start|>`` at all. Appending an
        # explicit ``<|cot_start|>`` as assistant content (what this used to do)
        # is off-distribution: measured over 12 val windows it made the model
        # continue mid-sentence (0/12 outputs started with a capital letter,
        # e.g. "because of a clear lane ahead.") instead of writing a whole
        # clause (12/12 with this ordering).
        components_order = ["image", "traj_history", "prompt"]
        components_prompt = ["cot", "traj_future"]
    else:
        components_order = ["image", "traj_history", "prompt", "traj_future"]
        components_prompt = ["traj_future"]
    messages = build_conversation(
        data=data,
        num_tokens_per_history_traj=model.config.tokens_per_history_traj,
        num_tokens_per_future_traj=model.config.tokens_per_future_traj,
        components_order=components_order,
        components_prompt=components_prompt,
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


def _prep_generate_inputs(
    model: Any, tokenized_data: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
    """Move a tokenized prompt onto the model's device, batch dims normalized."""
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
    return input_ids, attention_mask, vision_inputs


def _generate_restricted_trajectory_ids(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    vision_inputs: dict[str, torch.Tensor],
    num_traj_samples: int,
    top_p: float,
    temperature: float,
    do_sample: bool,
    max_gen_batch: int | None,
    restrict_to_traj_vocab: bool,
) -> torch.Tensor:
    """Force exactly ``tokens_per_future_traj`` new tokens, vocab-masked to action bins.

    Shared by ``rollout_future_trajectory`` and ``rollout_future_trajectory_with_cot``
    -- the trajectory-token decode step is identical regardless of what preceded it
    in ``input_ids`` (nothing, or a free-generated CoT span).

    Returns:
        pred_ids: [B * num_traj_samples, tokens_per_future_traj], the future-traj
            VOCAB-RELATIVE ids (i.e. already offset by ``-future_id0`` and clamped).
    """
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
    generation_config.pad_token_id = model.tokenizer.pad_token_id
    generation_config.return_dict_in_generate = True
    generation_config.output_logits = False

    # generate() expands the batch via repeat_interleave(num_return_sequences), so
    # asking for all K at once holds K copies of a long multi-image prompt's KV cache
    # simultaneously -- the peak that OOMs when eval runs alongside resident training
    # state. Draw the K samples in chunks instead: they are independent samples, so
    # chunking is mathematically identical, it just re-prefills the prompt once per
    # chunk (the cost of the fix -- lower `max_gen_batch` trades speed for headroom).
    future_id0 = model.config.traj_ids["future_id0"]
    future_vocab_size = model.config.future_vocab_size
    logits_processor = LogitsProcessorList()
    vocab_guard = None
    if restrict_to_traj_vocab:
        vocab_guard = _RestrictToFutureTrajVocab(
            future_id0, future_vocab_size, model.config.traj_ids.get("future_end")
        )
        logits_processor.append(vocab_guard)

    chunk = num_traj_samples if max_gen_batch is None else max(1, int(max_gen_batch))
    batch_size = input_ids.shape[0]
    prompt_len = input_ids.shape[1]
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        remaining = num_traj_samples
        while remaining > 0:
            k = min(chunk, remaining)
            generation_config.num_return_sequences = k
            if vocab_guard is not None:
                # Diagnostics are per-generate()-call: the row count changes with k
                # and the step counter restarts, so stale state would misattribute.
                vocab_guard.reset()
            outputs = model.vlm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                logits_processor=logits_processor,
                **vision_inputs,
            )
            # Rows are grouped [b0_k0, b0_k1, ..., b1_k0, ...] -- verified
            # empirically. Reshape to [B, k, L] before concatenating so the
            # per-batch grouping survives chunking (a flat cat would interleave
            # samples across batch elements when B > 1).
            seq = outputs.sequences[:, prompt_len:]
            seq = seq[:, : model.config.tokens_per_future_traj]
            chunks.append(seq.view(batch_size, k, -1))
            if vocab_guard is not None:
                vocab_guard.log_summary(model.config.tokens_per_future_traj)
            del outputs, seq
            remaining -= k
    generated = torch.cat(chunks, dim=1).flatten(0, 1)

    raw_ids = generated - future_id0
    # Never silently clamp: an out-of-range token means the decode is being fed
    # something that is not an action, and the clamp turns it into the *extreme*
    # action (max accel + a 3.8 m turn radius) rather than an obvious error --
    # that is the degenerate-loop failure mode. With restrict_to_traj_vocab this
    # count must be 0; if it is not, the mask is not being applied.
    n_bad = int(((raw_ids < 0) | (raw_ids >= future_vocab_size)).sum())
    if n_bad:
        logger.warning(
            "rollout: %d/%d generated tokens fell outside the future-trajectory vocab "
            "and were clamped to an extreme action bin (expect degenerate looping "
            "trajectories). restrict_to_traj_vocab=%s",
            n_bad,
            raw_ids.numel(),
            restrict_to_traj_vocab,
        )
    return raw_ids.clamp(min=0, max=future_vocab_size - 1)


def _decode_future_xyz(
    model: Any,
    pred_ids: torch.Tensor,
    ego_history_xyz: torch.Tensor,
    ego_history_rot: torch.Tensor,
    num_traj_samples: int,
    batch_size: int,
) -> torch.Tensor:
    device = pred_ids.device
    hist_xyz = ego_history_xyz.to(device).flatten(0, 1)  # [B, T_hist, 3]
    hist_rot = ego_history_rot.to(device).flatten(0, 1)
    hist_xyz_rep = hist_xyz.repeat_interleave(num_traj_samples, dim=0)
    hist_rot_rep = hist_rot.repeat_interleave(num_traj_samples, dim=0)

    fut_xyz, _fut_rot, _ = model.future_traj_tokenizer.decode(
        hist_xyz=hist_xyz_rep, hist_rot=hist_rot_rep, tokens=pred_ids
    )
    return fut_xyz.view(batch_size, num_traj_samples, *fut_xyz.shape[1:])


def rollout_future_trajectory(
    model: Any,
    tokenized_data: dict[str, torch.Tensor],
    ego_history_xyz: torch.Tensor,
    ego_history_rot: torch.Tensor,
    num_traj_samples: int = 1,
    top_p: float = 0.98,
    temperature: float = 0.6,
    do_sample: bool = True,
    max_gen_batch: int | None = 2,
    restrict_to_traj_vocab: bool = True,
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
        max_gen_batch: Max number of the ``num_traj_samples`` rollouts to generate in
            a single ``generate()`` call. Caps peak KV-cache memory at this many
            copies of the prompt instead of all ``num_traj_samples`` at once, at the
            cost of re-prefilling the prompt once per chunk. ``None`` disables
            chunking (original all-at-once behavior).
        restrict_to_traj_vocab: Mask sampling to the future-trajectory token range, so a
            forced fixed-length decode cannot emit a token that has no action bin. See
            ``_RestrictToFutureTrajVocab`` -- without it ``<|traj_future_end|>`` and any
            text token are silently clamped to an extreme action bin, which the unicycle
            decoder integrates into the degenerate loop-shaped trajectories. Set False
            only to reproduce the old (clamping) behaviour.

    Returns:
        Predicted future xyz, shape [1, num_traj_samples, T_future, 3].
    """
    input_ids, attention_mask, vision_inputs = _prep_generate_inputs(model, tokenized_data)
    pred_ids = _generate_restricted_trajectory_ids(
        model,
        input_ids,
        attention_mask,
        vision_inputs,
        num_traj_samples,
        top_p,
        temperature,
        do_sample,
        max_gen_batch,
        restrict_to_traj_vocab,
    )
    return _decode_future_xyz(
        model, pred_ids, ego_history_xyz, ego_history_rot, num_traj_samples, input_ids.shape[0]
    )


def rollout_future_trajectory_with_cot(
    model: Any,
    tokenized_data: dict[str, torch.Tensor],
    ego_history_xyz: torch.Tensor,
    ego_history_rot: torch.Tensor,
    num_traj_samples: int = 1,
    top_p: float = 0.98,
    temperature: float = 0.6,
    do_sample: bool = True,
    max_gen_batch: int | None = 2,
    restrict_to_traj_vocab: bool = True,
    max_cot_tokens: int = 256,
) -> tuple[torch.Tensor, str]:
    """Two-phase rollout: free-generate reasoning, then vocab-restricted trajectory tokens.

    ``tokenized_data`` must come from ``build_generation_prompt(..., with_cot=True)``
    (prompt ends at ``<|cot_start|>``, not ``<|traj_future_start|>``). Mirrors how
    alpamayo1_5_sft's ``ReasoningSampler(traj_only_generation=False)`` handles CoC:

      1. UNRESTRICTED decode (full vocab, normal sampling) until the model itself
         emits ``<|traj_future_start|>`` or ``max_cot_tokens`` is hit. This is the
         one part of the vocab restriction in ``rollout_future_trajectory`` that
         must NOT apply here -- reasoning is free text, not action bins.
      2. The vocab-RESTRICTED decode from ``rollout_future_trajectory``, continued
         from the reasoning-primed sequence, forcing exactly ``tokens_per_future_traj``
         more tokens. Branches into ``num_traj_samples`` here, so all K sampled
         trajectories condition on the SAME single reasoning pass (one CoT, K
         candidate futures) -- not K independent (reasoning, trajectory) pairs.

    Two independent ``generate()`` calls (no KV-cache carried from phase 1 into
    phase 2) -- simpler than splicing caches, at the cost of re-prefilling the
    (now longer) prompt once more. Correctness over speed for a first cut.

    Returns:
        (pred_xyz, cot_text): pred_xyz shaped [1, num_traj_samples, T_future, 3];
        cot_text is the single decoded reasoning string all K samples share.
    """
    input_ids, attention_mask, vision_inputs = _prep_generate_inputs(model, tokenized_data)
    if input_ids.shape[0] != 1:
        raise NotImplementedError(
            "rollout_future_trajectory_with_cot only supports batch size 1 "
            f"(got {input_ids.shape[0]}); phase 1 free-generates a single shared "
            "reasoning pass and batching would need per-row early-stop bookkeeping."
        )
    device = input_ids.device
    prompt_len = input_ids.shape[1]

    traj_future_start_id = model.tokenizer.convert_tokens_to_ids(
        SPECIAL_TOKENS["traj_future_start"]
    )

    # Phase 1: free-form reasoning decode, unrestricted vocab, stopping the instant
    # the model emits <|traj_future_start|> on its own (that's the signal it's done
    # reasoning and is handing off to the trajectory span).
    cot_config = copy.deepcopy(model.vlm.generation_config)
    cot_config.max_new_tokens = max_cot_tokens
    cot_config.eos_token_id = traj_future_start_id
    cot_config.do_sample = do_sample
    cot_config.top_p = top_p if do_sample else None
    cot_config.temperature = temperature if do_sample else None
    cot_config.pad_token_id = model.tokenizer.pad_token_id
    cot_config.return_dict_in_generate = True
    cot_config.output_logits = False
    cot_config.num_return_sequences = 1
    # NOTE (fork): the release CoC decode is not fully unrestricted -- it masks
    # the discrete trajectory vocab and the text EOS ids for the whole reasoning
    # span (``Alpamayo2Super.sample_trajectories_from_data``). Without these two
    # processors the checkpoint immediately emits ``<|im_end|>`` and stray
    # ``<i####>`` action tokens into the reasoning span; measured over 12 val
    # windows, 10/12 had ``<|im_end|>`` inside the first 8 generated tokens
    # (which ``skip_special_tokens=True`` then deletes, producing the truncated
    # "inal: Maintain Speed." strings) and 2/12 were pure action-token garbage.
    # With them, 0/12.
    cot_logits_processor = LogitsProcessorList(
        [
            MaskDiscreteTrajectoryLogitsProcessor(
                traj_token_offset=min(
                    model.config.traj_ids["history_id0"], model.config.traj_ids["future_id0"]
                ),
                traj_vocab_size=model.config.traj_vocab_size,
            )
        ]
    )
    _append_text_eos_mask(
        cot_logits_processor,
        model.vlm.generation_config.eos_token_id,
        preserved_token_id=traj_future_start_id,
    )

    with torch.no_grad():
        cot_outputs = model.vlm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=cot_config,
            logits_processor=cot_logits_processor,
            **vision_inputs,
        )
    cot_seq = cot_outputs.sequences  # [1, prompt_len + cot_len(+1 if it hit the eos id)]
    cot_new_ids = cot_seq[:, prompt_len:]
    del cot_outputs

    hit_traj_start = bool((cot_new_ids[0] == traj_future_start_id).any())
    if not hit_traj_start:
        logger.warning(
            "rollout_with_cot: reasoning ran the full max_cot_tokens=%d without the "
            "model emitting <|traj_future_start|> on its own; appending it manually "
            "so trajectory decoding can proceed. Consider raising max_cot_tokens.",
            max_cot_tokens,
        )
        traj_start_col = torch.full(
            (1, 1), traj_future_start_id, dtype=cot_seq.dtype, device=device
        )
        cot_seq = torch.cat([cot_seq, traj_start_col], dim=1)

    # Release-side extraction (splits off any trailing meta-action block and cuts
    # at <|traj_future_start|>/<|im_end|>) rather than a raw decode of the new ids.
    cot_text = extract_text_tokens(model.tokenizer, cot_seq)["cot"][0]

    # Phase 2: same restricted decode as the no-cot path, continued from the
    # reasoning-primed sequence. Fresh attention mask -- batch size 1, no padding.
    cot_attention_mask = torch.ones_like(cot_seq)
    pred_ids = _generate_restricted_trajectory_ids(
        model,
        cot_seq,
        cot_attention_mask,
        vision_inputs,
        num_traj_samples,
        top_p,
        temperature,
        do_sample,
        max_gen_batch,
        restrict_to_traj_vocab,
    )
    pred_xyz = _decode_future_xyz(
        model, pred_ids, ego_history_xyz, ego_history_rot, num_traj_samples, batch_size=1
    )
    return pred_xyz, cot_text
