# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Phase-0 LLR *content* control: is the reasoning's CONTENT load-bearing, or merely its PRESENCE?

``phase0_llr_action_direction.py`` measured

    llr_act = log p(a* | v, ell_gold) - log p(a* | v, ell_blanked)          -> +0.227 nats (A1.5)

where the denominator's cot span is filled with pad tokens at identical length/position. That
controls the positional confound perfectly, but leaves a second one open: a pad-filled cot span
is itself a sequence the model NEVER saw in training. So part of +0.227 could be the model
reacting to a degenerate/unfamiliar span rather than to the loss of reasoning content.

This script closes that gap by adding a THIRD pass whose cot span holds *another event's real
gold CoC* -- real reasoning prose, in its normal position, at identical token length, just
semantically mismatched to the scene. Nothing about that sequence is out of distribution; only
the content is wrong. All three passes run on the SAME event so the decomposition is exact:

    logp_gold   = log p(a* | v, ell_gold)      Pass C  -- correct reasoning
    logp_wrong  = log p(a* | v, ell_wrong)     Pass W  -- another event's real CoC (in-distribution)
    logp_blank  = log p(a* | v, ell_blanked)   Pass D  -- pad-filled span (out-of-distribution)

    llr_content  = logp_gold  - logp_wrong   <- "is this reasoning RIGHT for this scene?"
    llr_presence = logp_wrong - logp_blank   <- "is there readable text here at all?"
    llr_act      = logp_gold  - logp_blank   = llr_content + llr_presence  (reproduces the
                                               headline number on identical events, as a check)

``llr_content`` is the cleaner quantity, and the one that should be believed over ``llr_act``:
both of its passes sit inside the model's trained sequence distribution. If
``llr_content`` ~= ``llr_act``, the pad-blank measurement was sound. If ``llr_content`` is much
smaller, then most of the headline +0.227 was pad-token novelty, not reasoning.

Donor selection (holding the positional control intact -- read before modifying): swapping in a
donor CoC of a DIFFERENT token length would move the ``traj_future`` span and reintroduce exactly
the positional-novelty artifact that invalidated Sec 4.1 (see
``phase0_llr_ordering_control.py``). So donors are restricted to events whose tokenized CoC is at
least as long as the target's cot span and then TRUNCATED to exactly ``n_cot_tokens`` -- sequence
length and every token position stay byte-identical across all three passes. Truncation can cut
the donor mid-sentence, which is a far milder distribution shift than a pad-filled span (the model
sees truncated/continuing prose constantly; it never sees interior padding). Donors are drawn from
a different ``clip_id`` and, where possible, a different ``event_cluster``, so the mismatch is
semantic and not just lexical. Selection is seeded, so a re-run reproduces the same pairing.

Same venv/CWD rules as the rest of this directory -- ``a1x_rl_b300`` on this B300 (sm_103) box,
invoked from a recipe directory (see ``README.md``).

Usage::

    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$((4+i)) a1x_rl_b300/bin/python ../../scripts_fork/llr/phase0_llr_wrong_coc.py \\
        --num-shards 4 --shard-idx $i \\
        --out /opt/dlami/nvme/rod/results/phase0_llr_wrong_coc/llr.parquet &
    done; wait

Smoke test (single GPU, 8 events)::

    CUDA_VISIBLE_DEVICES=0 a1x_rl_b300/bin/python ../../scripts_fork/llr/phase0_llr_wrong_coc.py \\
      --limit 8 --out /tmp/phase0_llr_wrong_smoke.parquet
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
# t0 must exceed the history window (16 * 0.1s = 1.6s); matches phase0_llr_action_direction.py.
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
    p.add_argument("--seed", type=int, default=42, help="Seeds donor pairing (identical across shards).")
    p.add_argument("--out", required=True, help="Output .parquet path (sharded when --num-shards > 1).")
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
    torch.manual_seed(args.seed)

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

    print(f"[wrong-coc] loading model {args.model_path} (Stage-1 discrete-token class) ...", flush=True)
    model = TrainableReasoningVLA.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).cuda().eval()

    # Natural training order throughout; only the cot SPAN CONTENT differs between passes.
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
    blank_token_id = tokenizer.pad_token_id
    if blank_token_id is None:
        blank_token_id = tokenizer.eos_token_id
    traj_lo = model.future_token_start_idx
    traj_hi = traj_lo + model.config.traj_vocab_size
    sid = model.special_token_ids
    print(
        f"[wrong-coc] cot-blank token id = {blank_token_id} "
        f"({tokenizer.convert_ids_to_tokens(blank_token_id)!r})",
        flush=True,
    )

    # ---- Donor pool: every annotated event's gold CoC, pre-tokenized once. Built from the FULL
    # unsharded event list so donor pairing is identical no matter how the work is sharded.
    all_events = load_events(args.parquet, args.split, all_events=not args.first_event_only)
    donor_ids = [
        np.asarray(tokenizer(e["gold_coc"], add_special_tokens=False)["input_ids"], dtype=np.int64)
        for e in all_events
    ]
    donor_len = np.asarray([len(d) for d in donor_ids])
    donor_clip = [e["clip_id"] for e in all_events]
    donor_cluster = [e["event_cluster"] for e in all_events]
    print(
        f"[wrong-coc] donor pool: {len(donor_ids)} CoCs, token len "
        f"min={donor_len.min()} median={int(np.median(donor_len))} max={donor_len.max()}",
        flush=True,
    )

    def pick_donor(n_needed: int, clip_id: str, cluster, rng: np.random.Generator):
        """Real CoC prose, from other clip(s), trimmed to exactly n_needed tokens.

        Prefers a single donor at least n_needed long, from a different clip and (where possible)
        a different event_cluster. Gold CoCs are short (median 15 tokens, max 33), so a long cot
        span can exceed every single donor -- in that case donors are CONCATENATED until the
        length is reached, which keeps the span real prose rather than falling back to padding.
        Returns (token ids of exactly n_needed, primary donor index, is_cross_cluster, n_donors).
        """
        not_self = np.asarray([c != clip_id for c in donor_clip])
        cross = np.asarray([c != cluster for c in donor_cluster])
        single = np.flatnonzero((donor_len >= n_needed) & not_self & cross)
        if single.size == 0:
            single = np.flatnonzero((donor_len >= n_needed) & not_self)
        if single.size:
            j = int(rng.choice(single))
            return donor_ids[j][:n_needed], j, bool(donor_cluster[j] != cluster), 1

        # No single donor is long enough -- concatenate distinct donors until it is.
        pool = np.flatnonzero(not_self & cross)
        if pool.size == 0:
            pool = np.flatnonzero(not_self)
        if pool.size == 0:
            return None
        order = rng.permutation(pool)
        chunks, total, used = [], 0, []
        for j in order:
            chunks.append(donor_ids[j])
            used.append(int(j))
            total += len(donor_ids[j])
            if total >= n_needed:
                break
        if total < n_needed:  # whole pool is shorter than the span -- cycle it
            while total < n_needed:
                j = int(order[len(used) % order.size])
                chunks.append(donor_ids[j])
                used.append(j)
                total += len(donor_ids[j])
        cat = np.concatenate(chunks)[:n_needed]
        j0 = used[0]
        return cat, j0, bool(donor_cluster[j0] != cluster), len(used)

    events = list(all_events)
    if args.num_shards > 1:
        events = events[args.shard_idx :: args.num_shards]
    if args.limit:
        events = events[: args.limit]
    print(
        f"[wrong-coc] shard {args.shard_idx}/{args.num_shards}: {len(events)} events (split={args.split})",
        flush=True,
    )

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()  # streaming -- this root has no local clip media

    def _run_pass(tokenized_data: dict, data: dict, fused, traj_mask, fut_start_pos) -> dict:
        """Score a pass, SPLIT BY TOKEN ROLE rather than as one mean.

        ``out.loss_future_traj`` is a single ``reduction="mean"`` over the model's ``traj_mask``,
        which is NOT 128 future-trajectory tokens -- it is 178: 48 history-trajectory tokens
        (same token-id block, so the id-range test catches them), the 2
        ``traj_future_start``/``traj_future_end`` delimiters, and only then the 128 real future
        tokens. Those three roles behave completely differently under CoC ablation, so averaging
        them together is not a measurement of anything:

          * history tokens precede the cot span, so causal attention pins their LLR to EXACTLY 0 --
            pure dilution of whatever the real effect is;
          * the delimiters go from log p = 0.0 (certainty) with gold CoC to about -18 / -28 nats
            when the span is blanked, because the model no longer knows the prose has ended and a
            trajectory begins. At ~+23.6 nats each, 2/178 of them contribute +0.265 nats to the
            mean -- MORE than the entire +0.227 that the scalar version of this measurement
            reported. That is sequence formatting, not reasoning informing driving.

        So ``logp_future`` (the 128 real tokens) is the quantity to believe. The other two are
        returned alongside it so the decomposition stays visible instead of silently re-entering
        the headline number.
        """
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
        # Replicate _compute_next_token_loss's shift/gather with reduction="none".
        m = traj_mask[:, 1:]
        shift_labels = fused[:, 1:][m].contiguous()
        shift_logits = out.logits[..., :-1, :][m].contiguous().float()
        logp = -torch.nn.functional.cross_entropy(shift_logits, shift_labels, reduction="none")
        seq_pos = m[0].nonzero().flatten() + 1
        is_value = (shift_labels >= traj_lo) & (shift_labels < traj_hi)
        is_future = is_value & (seq_pos > fut_start_pos)
        is_hist = is_value & ~is_future
        is_delim = ~is_value
        return {
            "logp_future": float(logp[is_future].mean().item()),
            "logp_hist": float(logp[is_hist].mean().item()) if bool(is_hist.any()) else float("nan"),
            "logp_delim": float(logp[is_delim].mean().item()) if bool(is_delim.any()) else float("nan"),
            "logp_maskmean": float(logp.mean().item()),  # == -out.loss_future_traj, for reconciliation
            "n_future": int(is_future.sum().item()),
            "scalar_check": -out.loss_future_traj.item(),
        }

    rows: list[dict] = []
    t_start = time.time()
    for i, ev in enumerate(events):
        clip_id, t0_us = ev["clip_id"], ev["t0_us"]
        try:
            data = load_physical_aiavdataset(clip_id, t0_us=t0_us, avdi=avdi)

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

            # Donor pairing is keyed on the event itself, not on iteration order, so a shard's
            # results are independent of --num-shards / --shard-idx.
            rng = np.random.default_rng(
                abs(hash((args.seed, clip_id, ev["event_idx"]))) % (2**32)
            )
            picked = pick_donor(n_cot_tokens, clip_id, ev["event_cluster"], rng)
            if picked is None:
                raise RuntimeError(f"no donor CoC available from another clip for {clip_id}")
            donor_tokens, donor_j, donor_is_cross_cluster, n_donors = picked

            # The traj span's ids/positions are identical in all three passes (only the cot span's
            # CONTENT differs), so fuse once and share the mask.
            traj_data = {
                "ego_history_xyz": data["ego_history_xyz"].cuda(),
                "ego_history_rot": data["ego_history_rot"].cuda(),
                "ego_future_xyz": data["ego_future_xyz"].cuda(),
                "ego_future_rot": data["ego_future_rot"].cuda(),
            }
            fused = model.fuse_traj_tokens(input_ids_real.clone().cuda(), traj_data)
            traj_mask = (
                ((fused >= traj_lo) & (fused < traj_hi))
                | (fused == sid["traj_future_start"])
                | (fused == sid["traj_future_end"])
            )
            fsp = (fused[0] == sid["traj_future_start"]).nonzero().flatten()
            if fsp.numel() == 0:
                raise RuntimeError("traj_future_start token not found in fused sequence")
            fut_start_pos = int(fsp[-1].item())

            # --- Pass C: real gold CoC ---
            td_real = dict(td)
            td_real["input_ids"] = input_ids_real.clone()
            r_gold = _run_pass(td_real, data, fused, traj_mask, fut_start_pos)

            # --- Pass W: another event's real CoC, truncated to the SAME token count so every
            # downstream position is unchanged. In-distribution prose, wrong content. ---
            input_ids_wrong = input_ids_real.clone()
            input_ids_wrong[cot_span_mask] = torch.from_numpy(donor_tokens).to(input_ids_wrong.dtype)
            td_wrong = dict(td)
            td_wrong["input_ids"] = input_ids_wrong
            r_wrong = _run_pass(td_wrong, data, fused, traj_mask, fut_start_pos)

            # --- Pass D: pad-blanked span (reproduces phase0_llr_action_direction.py) ---
            input_ids_blanked = input_ids_real.clone()
            input_ids_blanked[cot_span_mask] = blank_token_id
            td_blanked = dict(td)
            td_blanked["input_ids"] = input_ids_blanked
            r_blank = _run_pass(td_blanked, data, fused, traj_mask, fut_start_pos)

            rows.append(
                {
                    "clip_id": clip_id,
                    "event_idx": ev["event_idx"],
                    "event_cluster": ev["event_cluster"],
                    "split": ev["split"],
                    "t0_us": int(data["t0_us"]),
                    "gold_coc": ev["gold_coc"],
                    "donor_clip_id": donor_clip[donor_j],
                    "donor_event_cluster": donor_cluster[donor_j],
                    "donor_is_cross_cluster": donor_is_cross_cluster,
                    "donor_len_tokens": int(donor_len[donor_j]),
                    "n_donors": n_donors,
                    "n_cot_tokens": n_cot_tokens,
                    "seq_len": int(input_ids_real.shape[1]),
                    "n_future_tokens": r_gold["n_future"],
                    # --- future-trajectory tokens only: the quantity to believe ---
                    "logp_gold": r_gold["logp_future"],
                    "logp_wrong": r_wrong["logp_future"],
                    "logp_blank": r_blank["logp_future"],
                    "llr_content": r_gold["logp_future"] - r_wrong["logp_future"],
                    "llr_presence": r_wrong["logp_future"] - r_blank["logp_future"],
                    "llr_act": r_gold["logp_future"] - r_blank["logp_future"],
                    # --- the delimiter formatting effect, kept separate and visible ---
                    "delim_gold": r_gold["logp_delim"],
                    "delim_wrong": r_wrong["logp_delim"],
                    "delim_blank": r_blank["logp_delim"],
                    "llr_delim_content": r_gold["logp_delim"] - r_wrong["logp_delim"],
                    "llr_delim_act": r_gold["logp_delim"] - r_blank["logp_delim"],
                    # --- the old, role-mixed scalar, so the correction is auditable ---
                    "llr_act_maskmean_OLD": r_gold["logp_maskmean"] - r_blank["logp_maskmean"],
                    "ok": True,
                }
            )
            if (i + 1) % 20 == 0 or (i + 1) == len(events):
                elapsed = time.time() - t_start
                recent = [r for r in rows[-20:] if r.get("ok")]
                print(
                    f"[wrong-coc] [{i + 1}/{len(events)}] {clip_id} split={ev['split']} "
                    f"last20: content={np.mean([r['llr_content'] for r in recent]):.4f} "
                    f"presence={np.mean([r['llr_presence'] for r in recent]):.4f} "
                    f"act={np.mean([r['llr_act'] for r in recent]):.4f} nats "
                    f"({elapsed / 60:.1f} min elapsed)",
                    flush=True,
                )
        except Exception as e:  # keep going -- one bad event must not kill the shard
            # Free the allocator before continuing: a swallowed CUDA OOM otherwise leaves the
            # arena fragmented and poisons every subsequent event (this bit the A2S run).
            torch.cuda.empty_cache()
            rows.append(
                {
                    "clip_id": clip_id,
                    "event_idx": ev["event_idx"],
                    "split": ev["split"],
                    "ok": False,
                    "error": str(e),
                }
            )
            print(f"[wrong-coc] [{i + 1}/{len(events)}] {clip_id} FAILED: {type(e).__name__}: {e}", flush=True)

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
        f"\n[wrong-coc] shard {args.shard_idx}/{args.num_shards} done in "
        f"{(time.time() - t_start) / 60:.1f} min. {len(ok)}/{len(df)} events ok. Wrote {out_path}",
        flush=True,
    )
    if len(ok):
        for col in ("llr_content", "llr_presence", "llr_act", "llr_delim_act", "llr_act_maskmean_OLD"):
            print(
                f"[wrong-coc] {col:13s} mean={ok[col].mean():+.4f}  median={ok[col].median():+.4f}  "
                f"std={ok[col].std():.4f}  frac>0={(ok[col] > 0).mean():.3f}",
                flush=True,
            )
        print(
            f"[wrong-coc] logp_gold={ok['logp_gold'].mean():.4f}  "
            f"logp_wrong={ok['logp_wrong'].mean():.4f}  logp_blank={ok['logp_blank'].mean():.4f}  "
            f"cross-cluster donors={ok['donor_is_cross_cluster'].mean():.3f}",
            flush=True,
        )
        print(ok.groupby("event_cluster")[["llr_content", "llr_presence"]].agg(["mean", "count"]), flush=True)


if __name__ == "__main__":
    main()
