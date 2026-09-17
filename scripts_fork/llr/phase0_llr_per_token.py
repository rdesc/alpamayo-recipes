# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Phase-0 LLR resolved PER TRAJECTORY TOKEN -- where along the 6.4 s horizon, and in which
control channel, does the reasoning actually buy anything?

``phase0_llr_action_direction.py`` reports a single scalar per event because the model's own
``loss_future_traj`` is a ``reduction="mean"`` cross-entropy over the whole ``traj_future`` span
(``sft_base_model.py:623``). This script re-runs the SAME two passes (gold CoC vs. pad-blanked
CoC, identical length/position -- see that file for why blanking and not attention-masking) but
recomputes the span loss with ``reduction="none"``, giving

    llr_act[k] = log p(a*_k | a*_<k, v, ell_gold) - log p(a*_k | a*_<k, v, ell_blanked)

for each trajectory token k. Mean over k reproduces the headline scalar exactly, so this is a
decomposition of the existing number, not a new measurement.

Token layout (verified against the checkpoint config and ``UnicycleAccelCurvatureActionSpace``,
not assumed): the action space is unicycle accel/curvature with ``dt=0.1`` and
``n_waypoints=64``; ``traj_to_action`` returns ``stack([accel, kappa], dim=-1)`` of shape
(64, 2), which ``DiscreteTrajectoryTokenizer.encode`` flattens row-major to
``tokens_per_future_traj = 128``. So the span is INTERLEAVED, not one-token-per-timestep:

    token k  ->  horizon time t = (k // 2 + 1) * 0.1 s   (0.1 .. 6.4 s)
                 channel        = "accel" if k % 2 == 0 else "curvature"

which is what makes the two useful views possible: LLR vs. horizon time, and longitudinal
(accel) vs. lateral (curvature) control. A reasoning span that says "stop for the pedestrian"
should pay off in the accel channel; one that says "merge left" in the curvature channel.

The scored span also contains the ``traj_future_start``/``traj_future_end`` special tokens (the
model's ``traj_mask`` includes them), so they ARE part of the headline mean. They're emitted here
with ``channel="special"`` and ``traj_idx=-1`` rather than silently dropped -- filter on
``channel != "special"`` for the horizon plots, keep them to reconcile against the scalar.

Cost is two forward passes per event, same as the scalar run, so ``--split both`` over the full
OOD split is affordable; ``--events-json`` restricts it to the visualization showcase events.

Same venv/CWD rules as the rest of this directory (``a1x_rl_b300`` on this sm_103 box, invoked
from a recipe directory -- see ``README.md``).

Usage::

    # the 9 showcase events used by the artifact
    CUDA_VISIBLE_DEVICES=0 a1x_rl_b300/bin/python ../../scripts_fork/llr/phase0_llr_per_token.py \\
      --events-json <manifest.json> --out /tmp/llr_per_token_showcase.parquet

    # full split, sharded, for the aggregate horizon curve
    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i a1x_rl_b300/bin/python ../../scripts_fork/llr/phase0_llr_per_token.py \\
        --num-shards 4 --shard-idx $i --out /opt/dlami/nvme/rod/results/llr_per_token/pt.parquet &
    done; wait
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
import torch.nn.functional as F

DEFAULT_MODEL_PATH = "/mnt/efs/users/rod/ckpts/Alpamayo-1.5-10B-rl-training"
DEFAULT_PARQUET = "/opt/dlami/nvme/rod/datasets/pai_av_dataset/reasoning/ood_reasoning.parquet"
MIN_T0_US = 1_700_000
DT_S = 0.1  # action_space_cfg.dt -- one accel/curvature PAIR per 0.1 s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--split", default="both", choices=["train", "val", "both"])
    p.add_argument(
        "--events-json",
        default=None,
        help="Viz manifest (list of dicts with clip_id/event_idx) -- restrict to those events.",
    )
    p.add_argument(
        "--blank-mode",
        default="splice",
        choices=["splice", "interior", "span"],
        help=(
            "How the denominator suppresses the reasoning. 'splice' (default) REMOVES the prose "
            "between the markers, giving a genuine p(a*|v) -- the same empty-CoC sequence the "
            "validated `--no_coc` eval arm produces, and the only one of the three that is "
            "in-distribution. 'interior' overwrites the prose with pad tokens, holding length and "
            "position fixed but putting padding in a place training never does. 'span' overwrites "
            "everything get_label_mask returns for 'cot', which is INCLUSIVE of the markers "
            "(get_label_mask.py:45, `start : end + 1`) -- it therefore DELETES <|cot_end|>, "
            "leaving <|traj_future_start|> to follow an <|endoftext|>; its log p collapses ~23 "
            "nats and swamps any mean over the span. 'span' is what the original scalar "
            "measurement did and is kept only to reproduce that artifact."
        ),
    )
    p.add_argument("--first-event-only", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--out", required=True, help="Output .parquet: one row per (event, token).")
    return p.parse_args()


def load_events(parquet_path: str, split: str, all_events: bool) -> list[dict]:
    """Flatten ood_reasoning.parquet into one row per annotated event.

    Identical to phase0_llr_action_direction.py's loader (see that file for provenance).
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

    print(f"[per-token] loading model {args.model_path} ...", flush=True)
    model = TrainableReasoningVLA.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).cuda().eval()

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
    blank_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    traj_lo = model.future_token_start_idx
    traj_hi = traj_lo + model.config.traj_vocab_size
    sid = model.special_token_ids
    print(
        f"[per-token] traj value-token id range [{traj_lo}, {traj_hi}); "
        f"tokens_per_future_traj={model.config.tokens_per_future_traj}; dt={DT_S}s",
        flush=True,
    )

    events = load_events(args.parquet, args.split, all_events=not args.first_event_only)
    if args.events_json:
        with open(args.events_json) as f:
            man = json.load(f)
        if isinstance(man, dict):
            man = man.get("events", man.get("items", []))
        want = {(m["clip_id"], int(m["event_idx"])) for m in man}
        events = [e for e in events if (e["clip_id"], int(e["event_idx"])) in want]
        print(f"[per-token] --events-json: matched {len(events)}/{len(want)} requested events", flush=True)
    if args.num_shards > 1:
        events = events[args.shard_idx :: args.num_shards]
    if args.limit:
        events = events[: args.limit]
    print(f"[per-token] shard {args.shard_idx}/{args.num_shards}: {len(events)} events", flush=True)

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    def _per_token_logp(td: dict, data: dict, fused: torch.Tensor, traj_mask: torch.Tensor):
        """log p for each token in the traj span, in sequence order."""
        model_inputs = to_device({"tokenized_data": dict(td)}, "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                tokenized_data=model_inputs["tokenized_data"],
                ego_history_xyz=data["ego_history_xyz"].cuda(),
                ego_history_rot=data["ego_history_rot"].cuda(),
                ego_future_xyz=data["ego_future_xyz"].cuda(),
                ego_future_rot=data["ego_future_rot"].cuda(),
                labels_mask=None,
            )
        # Replicate sft_base_model._compute_next_token_loss's shift/gather exactly, but with
        # reduction="none" so each token keeps its own value. Mean of this == loss_future_traj.
        m = traj_mask[:, 1:]
        shift_labels = fused[:, 1:][m].contiguous()
        shift_logits = out.logits[..., :-1, :][m].contiguous().float()
        logp = -F.cross_entropy(shift_logits, shift_labels, reduction="none")
        # Sequence position each scored token predicts (the +1 undoes the left-shift), so history
        # and future trajectory tokens can be separated by position.
        seq_pos = m[0].nonzero().flatten() + 1
        return (
            logp.cpu().numpy(),
            shift_labels.cpu().numpy(),
            seq_pos.cpu().numpy(),
            -out.loss_future_traj.item(),
        )

    def _tokenize(data: dict, cot_text: str) -> dict:
        """Tokenize+collate one event at a given cot string. `cot_text=""` yields the empty-CoC
        sequence (markers kept, prose gone) that the validated `--no_coc` arm produces."""
        sample = {
            "image_frames": data["image_frames"],
            "camera_indices": data["camera_indices"],
            "relative_timestamps": data["relative_timestamps"],
            "ego_history_xyz": data["ego_history_xyz"].squeeze(0),
            "ego_history_rot": data["ego_history_rot"].squeeze(0),
            "cot": cot_text,
        }
        sample["tokenized_data"] = preprocess_fn(sample)
        batch = collate_fn_from_model_config(
            [sample], model_config=model.config,
            include_camera_ids=True, include_frame_nums=True, chat_template_version="r1",
        )
        return batch["tokenized_data"]

    def _prep_span(input_ids: torch.Tensor, data: dict):
        """Fuse trajectory tokens and build the model's own traj_mask plus the future delimiter
        position. Must be redone per pass whenever the sequence LENGTH can differ (splice mode)."""
        traj_data = {
            "ego_history_xyz": data["ego_history_xyz"].cuda(),
            "ego_history_rot": data["ego_history_rot"].cuda(),
            "ego_future_xyz": data["ego_future_xyz"].cuda(),
            "ego_future_rot": data["ego_future_rot"].cuda(),
        }
        fused = model.fuse_traj_tokens(input_ids.clone().cuda(), traj_data)
        # Replicates the model's own traj_mask (sft_base_model.py:660-667) EXACTLY, including its
        # quirk: history-trajectory tokens come from the same id block as future ones (both start
        # at traj_token_start_idx), so the id-range test catches all 48 tokens_per_history_traj as
        # well as the 128 future tokens. They're kept for reconciliation, then split out below.
        traj_mask = (
            ((fused >= traj_lo) & (fused < traj_hi))
            | (fused == sid["traj_future_start"])
            | (fused == sid["traj_future_end"])
        )
        fsp = (fused[0] == sid["traj_future_start"]).nonzero().flatten()
        if fsp.numel() == 0:
            raise RuntimeError("traj_future_start token not found in fused sequence")
        return fused, traj_mask, int(fsp[-1].item())

    rows: list[dict] = []
    t_start = time.time()
    for i, ev in enumerate(events):
        clip_id, t0_us = ev["clip_id"], ev["t0_us"]
        try:
            data = load_physical_aiavdataset(clip_id, t0_us=t0_us, avdi=avdi)
            td = _tokenize(data, ev["gold_coc"])
            input_ids_real = td["input_ids"]
            cot_span_mask = get_label_mask(input_ids_real, tokenizer, ["cot"])
            n_cot_tokens = int(cot_span_mask.sum().item())
            if n_cot_tokens == 0:
                raise RuntimeError("cot span not found in tokenized sequence")
            # get_label_mask's cot span is contiguous and INCLUSIVE of both markers
            # (get_label_mask.py:45), so its first and last set positions are <|cot_start|> and
            # <|cot_end|>. Overwriting those is what produced the 23-nat delimiter artifact.
            span_pos_all = cot_span_mask[0].nonzero().flatten()
            assert input_ids_real[0, span_pos_all[0]].item() == sid["cot_start"], (
                "first token of the cot span is not <|cot_start|> -- span layout changed"
            )
            assert input_ids_real[0, span_pos_all[-1]].item() == sid["cot_end"], (
                "last token of the cot span is not <|cot_end|> -- span layout changed"
            )

            # --- numerator: p(a* | v, ell_gold) ---
            fused, traj_mask, fut_start_pos = _prep_span(input_ids_real, data)
            td_real = dict(td)
            td_real["input_ids"] = input_ids_real.clone()
            lp_gold, labels_seq, seq_pos, scalar_gold = _per_token_logp(td_real, data, fused, traj_mask)

            # --- denominator ---
            if args.blank_mode == "splice":
                # A true p(a*|v): re-tokenize with an EMPTY cot, so the prose is gone and only the
                # markers remain. Sequence length shrinks, so this pass needs its own fuse/mask.
                # Length change is in-distribution here -- the model sees variable-length CoC every
                # training step, and the component ORDER is untouched (unlike the Sec 4.1 attempt,
                # whose artifact came from reordering components, not from a length change).
                td_den = _tokenize(data, "")
                fused_den, traj_mask_den, fut_start_pos_den = _prep_span(td_den["input_ids"], data)
                lp_blank, labels_den, seq_pos_den, scalar_blank = _per_token_logp(
                    td_den, data, fused_den, traj_mask_den
                )
                n_fut_num = int(
                    (
                        (labels_seq >= traj_lo) & (labels_seq < traj_hi) & (seq_pos > fut_start_pos)
                    ).sum()
                )
                n_fut_den = int(
                    (
                        (labels_den >= traj_lo) & (labels_den < traj_hi) & (seq_pos_den > fut_start_pos_den)
                    ).sum()
                )
                if n_fut_num != n_fut_den:
                    raise RuntimeError(
                        f"future-token count differs between passes ({n_fut_num} vs {n_fut_den}) "
                        "-- per-token pairing would be misaligned"
                    )
                # Both passes teacher-force the SAME ground-truth trajectory, so the k-th future
                # token is the same target in both and pairing by rank is exact.
                lp_blank = lp_blank[
                    (labels_den >= traj_lo) & (labels_den < traj_hi) & (seq_pos_den > fut_start_pos_den)
                ]
                fut_sel_num = (
                    (labels_seq >= traj_lo) & (labels_seq < traj_hi) & (seq_pos > fut_start_pos)
                )
                lp_gold_fut = lp_gold[fut_sel_num]
            else:
                blank_mask = cot_span_mask.clone()
                if args.blank_mode == "interior":
                    blank_mask[0, span_pos_all[0]] = False
                    blank_mask[0, span_pos_all[-1]] = False
                input_ids_blanked = input_ids_real.clone()
                input_ids_blanked[blank_mask] = blank_token_id
                td_blank = dict(td)
                td_blank["input_ids"] = input_ids_blanked
                lp_blank, _, _, scalar_blank = _per_token_logp(td_blank, data, fused, traj_mask)

            # Reconcile against the model's own mean -- catches any mask/shift drift immediately.
            recon_err = abs(float(lp_gold.mean()) - scalar_gold)
            if recon_err > 1e-3:
                raise RuntimeError(
                    f"per-token mean {lp_gold.mean():.6f} != loss_future_traj {scalar_gold:.6f} "
                    f"(err {recon_err:.2e}) -- span mask or shift is wrong"
                )

            # Map sequence order -> (horizon time, control channel). Only value tokens (inside the
            # traj vocab range) carry a timestep; the start/end specials are labelled separately.
            # History and future trajectory tokens share a token-id block, so "is this a future
            # token" is a question about POSITION, not id: only value tokens after the
            # traj_future_start delimiter are future ones. Ranking must be restricted to those, or
            # the 48 history tokens shift every future token's timestep (which they silently did
            # in the first version of this script -- they occupy the sequence BEFORE the cot span,
            # so causal attention makes their llr identically 0.0, and the 48 spurious leading
            # zeros masqueraded as "reasoning does nothing for t <= 2.4 s").
            is_value = (labels_seq >= traj_lo) & (labels_seq < traj_hi)
            is_future = is_value & (seq_pos > fut_start_pos)
            future_rank = np.cumsum(is_future) - 1  # 0..127 over FUTURE value tokens only

            if args.blank_mode == "splice":
                # The two passes have different sequence lengths, so history and delimiter tokens
                # cannot be paired positionally -- and there is nothing to learn from them here
                # anyway (history precedes the cot span; the delimiters are the artifact this mode
                # exists to avoid). Emit the 128 paired future tokens only.
                emit = [(int(future_rank[k]), k) for k in range(len(lp_gold)) if is_future[k]]
                gold_vals, blank_vals = lp_gold_fut, lp_blank
                pairs = [(kk, gi) for gi, (kk, _) in enumerate(emit)]
            else:
                gold_vals, blank_vals = lp_gold, lp_blank
                pairs = [(int(future_rank[k]) if is_future[k] else -1, k) for k in range(len(lp_gold))]

            for traj_idx, k in pairs:
                if traj_idx >= 0:
                    channel = "accel" if traj_idx % 2 == 0 else "curvature"
                    t_s = (traj_idx // 2 + 1) * DT_S
                elif args.blank_mode != "splice" and is_value[k]:
                    channel, t_s = "history", float("nan")
                else:
                    channel, t_s = "special", float("nan")
                rows.append(
                    {
                        "clip_id": clip_id,
                        "event_idx": ev["event_idx"],
                        "event_cluster": ev["event_cluster"],
                        "split": ev["split"],
                        "gold_coc": ev["gold_coc"],
                        "n_cot_tokens": n_cot_tokens,
                        "span_pos": k,
                        "traj_idx": traj_idx,
                        "channel": channel,
                        "t_s": t_s,
                        "blank_mode": args.blank_mode,
                        "logp_gold": float(gold_vals[k]),
                        "logp_blank": float(blank_vals[k]),
                        "llr_act": float(gold_vals[k] - blank_vals[k]),
                    }
                )
            if (i + 1) % 10 == 0 or (i + 1) == len(events):
                print(
                    f"[per-token] [{i + 1}/{len(events)}] {clip_id} "
                    f"scalar llr={scalar_gold - scalar_blank:+.4f} "
                    f"({(time.time() - t_start) / 60:.1f} min)",
                    flush=True,
                )
        except Exception as e:
            torch.cuda.empty_cache()
            print(f"[per-token] [{i + 1}/{len(events)}] {clip_id} FAILED: {type(e).__name__}: {e}", flush=True)

    df = pd.DataFrame(rows)
    out_path = args.out
    if args.num_shards > 1:
        root, _, ext = args.out.rpartition(".")
        out_path = f"{root}.shard{args.shard_idx}-of-{args.num_shards}.{ext}"
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(
        f"\n[per-token] done in {(time.time() - t_start) / 60:.1f} min. "
        f"{df['clip_id'].nunique()} events, {len(df)} token rows. Wrote {out_path}",
        flush=True,
    )
    if len(df):
        print(df.groupby("channel")["llr_act"].agg(["mean", "std", "count"]), flush=True)
        fut = df[df["channel"].isin(["accel", "curvature"])]
        n_ev = len(df[["clip_id", "event_idx"]].drop_duplicates())
        print(
            f"\n[per-token] mean llr_act, FUTURE tokens only = {fut['llr_act'].mean():+.4f} nats "
            f"({len(fut) // max(n_ev, 1)} tokens/event)\n"
            f"[per-token] mean llr_act over the model's full traj_mask = {df['llr_act'].mean():+.4f} "
            f"nats  <- what loss_future_traj averages, diluted by the "
            f"{(df['channel'] == 'history').sum() // max(n_ev, 1)} structurally-zero history tokens",
            flush=True,
        )
        # Coarse horizon profile, so the shape is visible without opening the parquet.
        bins = pd.cut(fut["t_s"], [0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6, 6.4])
        print(fut.groupby([bins, "channel"], observed=True)["llr_act"].mean().unstack(), flush=True)


if __name__ == "__main__":
    main()
