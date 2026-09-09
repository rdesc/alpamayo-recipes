# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Phase-0 LLR diagnostic (langforce_readme.md Sec 4.2, action-direction) over the PAI-AV OOD
reasoning subset (``reasoning/ood_reasoning.parquet``) -- measures, with ZERO training, whether
the model's predicted future-trajectory tokens already depend on its chain-of-causation
reasoning, using NATURAL training order throughout (unlike ``phase0_llr_diagnostic.py``'s
Sec 4.1 language-direction measurement, which reorders the sequence to `[v, a*, cot]` and was
found -- see ``phase0_llr_ordering_control.py`` -- to be ~95% positional-novelty artifact: real
vs. wrong ground-truth trajectory teacher-forced into that untrained sequence position produced
almost identical cot log-probs, so the measured signal wasn't tracking content at all).

This script avoids that failure mode by construction: both passes below use the SAME sequence
order the model was actually trained on (``[image, traj_history, prompt, cot, traj_future]``,
matching e.g. ``recipes/alpamayo1_5_sft/configs/sft_truckdrive_qwen_scratch_coc.yaml``'s
``components_order``), and score the ``traj_future`` span rather than the ``cot`` span. Only the
CONTENT of the ``cot`` span differs between the two passes -- never its position or the sequence
length -- so nothing about token position changes between numerator and denominator.

    llr_act = logp_num - logp_den
    logp_num = mean_a[ log p(a* | v, ell) ]        -- Pass C: cot span = real gold CoC text
    logp_den = mean_a[ log p(a* | v, ell_blanked) ] -- Pass D: cot span content-blanked (same
                                                         position/length), rest of sequence
                                                         byte-identical to Pass C

Design decision -- content-blanking, not attention-mask zeroing (read before modifying):
the "natural" way to implement "mask reasoning out of context" would be to zero the 2D
``attention_mask`` at the cot token positions so the LM's causal-mask expansion treats them
like padding. That was deliberately NOT used here: Qwen3-VL (like Qwen2-VL) derives M-RoPE
``position_ids`` from ``input_ids``/``image_grid_thw`` and (when not explicitly passed)
``attention_mask``, via a `get_rope_index`-style routine designed for LEADING/TRAILING padding,
not an interior gap. Zeroing attention at a mid-sequence span risks silently corrupting the
position ids of every token after that span (shifting or duplicating positions), which would
reintroduce a confound of the same *character* as the ordering artifact this script exists to
avoid -- and worse, one that fails silently rather than producing an obviously-wrong sequence
length. Content-blanking (replacing the cot span's real token ids with the tokenizer's pad-token
id, holding position and sequence length fixed) sidesteps that risk entirely at the cost of the
blanked tokens still being att... to (with no information content) rather than hard-masked from
attention. That's an acceptable trade for a measurement diagnostic: the model can still `see`
that *something* occupies that span (which is arguably more faithful to "reasoning became
uninformative" than to "reasoning was never there"), it just can't read what it says.

Ground-truth future trajectory (`a*`) comes from the event's own `ego_future_xyz/rot`, fused
into the ``traj_future`` placeholder span exactly as Stage-1 SFT does every training step
(``TrainableReasoningVLA.fuse_traj_tokens`` -> ``tokenize_future_trajectory``). The `future_traj`
component of the model's own training loss (``losses["future_traj"]`` / returned as
``out.loss_future_traj``) is already `-logp(a* | context)` averaged over exactly the trajectory
token span (identified internally via the `traj_future_start`/`traj_future_end` special-token
ids and the `traj_vocab_size` range -- see ``sft_base_model.py``'s `forward`), REGARDLESS of what
``labels_mask``/``label_components`` was used to build the batch. So no custom span-gather code
is needed here either -- both passes just read ``out.loss_future_traj`` off two forward calls
that differ only in the cot span's token content.

Note on Sec 2.1's "gradient-trivial trap": that warning is about using the action-direction
formula as a *training* loss (stop-grad kills the denominator's gradient, so training on it
collapses to re-deriving the SFT loss). It does not apply to measurement -- nothing here is
backpropagated, so the concern is moot; Sec 4.2 explicitly lists this as a legitimate cross-check
measurement.

Use ``a1x_rl_b300`` on this B300 (sm_103) box -- plain ``a1x_rl`` bundles nvrtc 12.8, which
predates sm_103 and dies mid-run with ``nvrtc: invalid value for --gpu-architecture``. See the
root ``CLAUDE.md``.

Usage::

    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$((4+i)) a1x_rl_b300/bin/python scripts_fork/phase0_llr_action_direction.py \\
        --num-shards 4 --shard-idx $i \\
        --out /opt/dlami/nvme/rod/results/phase0_llr_act/llr.parquet &
    done; wait

Smoke test (single GPU, 8 events)::

    CUDA_VISIBLE_DEVICES=0 a1x_rl_b300/bin/python scripts_fork/phase0_llr_action_direction.py \\
      --limit 8 --out /tmp/phase0_llr_act_smoke.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

DEFAULT_MODEL_PATH = "/mnt/efs/users/rod/ckpts/Alpamayo-1.5-10B-rl-training"
DEFAULT_PARQUET = "/opt/dlami/nvme/rod/datasets/pai_av_dataset/reasoning/ood_reasoning.parquet"
# t0 must exceed the history window (16 * 0.1s = 1.6s); matches phase0_llr_diagnostic.py /
# eval_baseline_ood_reasoning.py's MIN_T0_US.
MIN_T0_US = 1_700_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--split", default="both", choices=["train", "val", "both"])
    p.add_argument("--first-event-only", action="store_true", help="One event/clip instead of all annotated events.")
    p.add_argument("--limit", type=int, default=None, help="Only this many events (post-shard).")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--out", required=True, help="Output .parquet path (sharded when --num-shards > 1).")
    return p.parse_args()


def load_events(parquet_path: str, split: str, all_events: bool) -> list[dict]:
    """Flatten ood_reasoning.parquet into one row per annotated event.

    Identical to phase0_llr_diagnostic.py's loader (see that file for provenance).
    """
    df = pd.read_parquet(parquet_path)
    if split != "both":
        df = df[df["split"] == split].copy()
    df["events"] = df["events"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    df = df.dropna(subset=["events"])
    rows = []
    for clip_id, row in df.iterrows():
        events = row["events"]
        idxs = range(len(events)) if all_events else [0]
        for ei in idxs:
            ev = events[ei]
            rows.append(
                {
                    "clip_id": clip_id,
                    "event_idx": ei,
                    "t0_us": max(int(ev["event_start_timestamp"]), MIN_T0_US),
                    "gold_coc": ev["coc"],
                    "event_cluster": row["event_cluster"],
                    "split": row["split"],
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    sys.path[:0] = ["../../../src", "../.."]
    import physical_ai_av
    from alpamayo_r1.helper import to_device
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo.processor.qwen_processor import (
        collate_fn_from_model_config,
        get_preprocess_data_fn_from_model_config,
        build_processor,
    )
    from alpamayo.utils.get_label_mask import get_label_mask

    from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA

    print(f"[phase0-act] loading model {args.model_path} (Stage-1 discrete-token class) ...", flush=True)
    model = TrainableReasoningVLA.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).cuda().eval()

    # Natural training order throughout: [image, traj_history, prompt, cot, traj_future].
    # Only the cot SPAN CONTENT differs between Pass C and Pass D below -- components_order,
    # sequence length, and every other token's position are identical.
    preprocess_fn = get_preprocess_data_fn_from_model_config(
        components_order=["image", "traj_history", "prompt", "cot", "traj_future"],
        components_prompt=["cot", "traj_future"],
        label_components=["cot"],
        generation_mode=False,
        include_camera_ids=True,
        include_frame_nums=True,
        model_config=model.config,
        chat_template_version="r1",
    )

    # Separate tokenizer handle (not exposed by the preprocess closure) purely to locate the
    # cot span's token positions for blanking -- same get_label_mask call the collate_fn uses
    # internally to build labels_mask, just invoked directly so we get the pure cot-only mask
    # (collate_fn's version additionally ORs in an assistant-eos mask we don't want here).
    processor = build_processor(
        vlm_name_or_path=model.config.vlm_name_or_path,
        traj_vocab_size=model.config.traj_vocab_size,
        min_pixels=model.config.min_pixels,
        max_pixels=model.config.max_pixels,
        include_camera_ids=True,
        include_frame_nums=True,
        chat_template_version="r1",
    )
    tokenizer = processor.tokenizer
    blank_token_id = tokenizer.pad_token_id
    if blank_token_id is None:
        blank_token_id = tokenizer.eos_token_id
    print(f"[phase0-act] cot-blank token id = {blank_token_id} ({tokenizer.convert_ids_to_tokens(blank_token_id)!r})", flush=True)

    events = load_events(args.parquet, args.split, all_events=not args.first_event_only)
    if args.num_shards > 1:
        events = events[args.shard_idx :: args.num_shards]
    if args.limit:
        events = events[: args.limit]
    print(
        f"[phase0-act] shard {args.shard_idx}/{args.num_shards}: {len(events)} events (split={args.split})",
        flush=True,
    )

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()  # streaming -- this root has no local clip media

    def _run_pass(tokenized_data: dict, data: dict) -> float:
        model_inputs = to_device({"tokenized_data": tokenized_data}, "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                tokenized_data=model_inputs["tokenized_data"],
                ego_history_xyz=data["ego_history_xyz"].cuda(),
                ego_history_rot=data["ego_history_rot"].cuda(),
                ego_future_xyz=data["ego_future_xyz"].cuda(),
                ego_future_rot=data["ego_future_rot"].cuda(),
                labels_mask=None,
            )
        return -out.loss_future_traj.item()

    rows: list[dict] = []
    t_start = time.time()
    for i, ev in enumerate(events):
        clip_id, t0_us = ev["clip_id"], ev["t0_us"]
        try:
            data = load_physical_aiavdataset(clip_id, t0_us=t0_us, avdi=avdi)
            data["gold_coc"] = ev["gold_coc"]

            sample = {
                "image_frames": data["image_frames"],
                "camera_indices": data["camera_indices"],
                "relative_timestamps": data["relative_timestamps"],
                "ego_history_xyz": data["ego_history_xyz"].squeeze(0),
                "ego_history_rot": data["ego_history_rot"].squeeze(0),
                "cot": ev["gold_coc"],
            }
            sample["tokenized_data"] = preprocess_fn(sample)
            batch = collate_fn_from_model_config(
                [sample], model_config=model.config,
                include_camera_ids=True, include_frame_nums=True, chat_template_version="r1",
            )
            td = batch["tokenized_data"]

            input_ids_real = td["input_ids"]
            cot_span_mask = get_label_mask(input_ids_real, tokenizer, ["cot"])  # [1, L], pure cot span
            n_cot_tokens = int(cot_span_mask.sum().item())
            if n_cot_tokens == 0:
                raise RuntimeError("cot span not found in tokenized sequence (0 tokens matched)")

            # --- Pass C: cot span = real gold CoC content ---
            td_real = dict(td)
            td_real["input_ids"] = input_ids_real.clone()
            logp_num = _run_pass(td_real, data)

            # --- Pass D: cot span content-blanked (same position/length), everything else
            # byte-identical to Pass C ---
            input_ids_blanked = input_ids_real.clone()
            input_ids_blanked[cot_span_mask] = blank_token_id
            td_blanked = dict(td)
            td_blanked["input_ids"] = input_ids_blanked
            logp_den = _run_pass(td_blanked, data)

            llr_act = logp_num - logp_den
            rows.append(
                {
                    "clip_id": clip_id,
                    "event_idx": ev["event_idx"],
                    "event_cluster": ev["event_cluster"],
                    "split": ev["split"],
                    "t0_us": int(data["t0_us"]),
                    "gold_coc": ev["gold_coc"],
                    "n_cot_tokens": n_cot_tokens,
                    "seq_len": int(input_ids_real.shape[1]),
                    "logp_num": logp_num,
                    "logp_den": logp_den,
                    "llr_act": llr_act,
                    "ok": True,
                }
            )
            if (i + 1) % 20 == 0 or (i + 1) == len(events):
                elapsed = time.time() - t_start
                recent = [r["llr_act"] for r in rows[-20:] if r.get("ok")]
                print(
                    f"[phase0-act] [{i + 1}/{len(events)}] {clip_id} split={ev['split']} "
                    f"llr_act_mean(last20)={np.mean(recent):.4f} nats "
                    f"({elapsed / 60:.1f} min elapsed)",
                    flush=True,
                )
        except Exception as e:  # keep going -- one bad event must not kill the shard
            rows.append(
                {
                    "clip_id": clip_id,
                    "event_idx": ev["event_idx"],
                    "split": ev["split"],
                    "ok": False,
                    "error": str(e),
                }
            )
            print(f"[phase0-act] [{i + 1}/{len(events)}] {clip_id} FAILED: {type(e).__name__}: {e}", flush=True)

    df = pd.DataFrame(rows)
    out_path = args.out
    if args.num_shards > 1:
        root, _, ext = args.out.rpartition(".")
        out_path = f"{root}.shard{args.shard_idx}-of-{args.num_shards}.{ext}"
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_parquet(out_path, index=False)

    ok = df[df.get("ok", False) == True]  # noqa: E712
    print(
        f"\n[phase0-act] shard {args.shard_idx}/{args.num_shards} done in "
        f"{(time.time() - t_start) / 60:.1f} min. {len(ok)}/{len(df)} events ok. Wrote {out_path}",
        flush=True,
    )
    if len(ok):
        print(
            f"[phase0-act] shard summary: llr_act mean={ok['llr_act'].mean():.4f}  "
            f"median={ok['llr_act'].median():.4f}  "
            f"logp_num mean={ok['logp_num'].mean():.4f}  logp_den mean={ok['logp_den'].mean():.4f}  "
            f"n_cot_tokens mean={ok['n_cot_tokens'].mean():.1f}",
            flush=True,
        )
        print(ok.groupby("event_cluster")["llr_act"].agg(["mean", "median", "count"]), flush=True)


if __name__ == "__main__":
    main()
