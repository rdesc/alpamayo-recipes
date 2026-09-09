# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Phase-0 LLR diagnostic (langforce_readme.md Sec 4.1, language-direction) over the PAI-AV OOD
reasoning subset (``reasoning/ood_reasoning.parquet``) -- measures, with ZERO training, whether
the model's chain-of-causation reasoning already depends on the ground-truth trajectory, or is
decorative.

For each annotated event: gold CoC text (`ell`) comes from the dataset's real human annotation
(`ev["coc"]`) -- NOT the TruckDrive fork's unsupervised pseudo-labels (see
``recipes/alpamayo1_5_sft/configs/sft_truckdrive_qwen_scratch_coc.yaml`` for why those don't
support this measurement: CoC there is conditioning-only, never supervised). Ground-truth future
trajectory (`a*`) comes from the same event's `ego_future_xyz/rot`, exactly as Stage-1 SFT
training discretizes it every step (``TrainableReasoningVLA.fuse_traj_tokens`` ->
``tokenize_future_trajectory``).

    llr_lang = logp_num - logp_den
    logp_num = mean_ell[ log p(ell | v, a*) ]   -- reordered pass, sequence: [v, traj_history, a*, cot]
    logp_den = mean_ell[ log p(ell | v) ]       -- normal pass,    sequence: [v, traj_history, cot]

Implementation shortcut (load-bearing, read before modifying): rather than hand-rolling per-token
log-prob extraction, this script calls ``TrainableReasoningVLA.forward`` -- the exact Stage-1 SFT
training forward pass -- twice per event, with ``label_components=["cot"]`` both times and
``components_order`` differing only in whether ``traj_future`` precedes ``cot``. That forward
pass already computes ``labels_mask``-restricted next-token cross-entropy internally
(``_compute_next_token_loss``) and, because our label mask covers only the ``cot`` span (the
``future_traj`` sub-mask it also computes internally is empty either way -- there are no
traj_future labels among cot-only labels), the returned ``loss_others`` **is already**
``-logp(cot | ...)`` averaged over the cot span. So:

    logp_num = -out_num.loss_others
    logp_den = -out_den.loss_others

No custom masking/gather code needed -- this reuses the training loss path byte-for-byte, which
is also why label masking is guaranteed consistent with how the model was actually trained
(see ``get_label_mask`` / ``SPECIAL_TOKENS["cot_start"/"cot_end"]``).

For the denominator pass, ``ego_future_xyz/rot`` are NOT passed to ``forward`` (left ``None``):
``fuse_traj_tokens`` only fuses future-trajectory tokens when ``has_future`` is true AND the
placeholder text contains a ``traj_future`` span; the denominator's ``components_order`` never
includes ``traj_future``, so there is no placeholder to fill and no reason to risk a
shape/count mismatch in ``replace_pad_token`` by feeding future data it doesn't need.

Loads the Stage-1 model class directly (``TrainableReasoningVLA``, not the RL eval scripts'
``RLWrapperReasoningVLA``) because it is what exposes ``fuse_traj_tokens``/the per-span
next-token-loss machinery this diagnostic depends on. Per prior investigation in this session:
Stage 1 trains real discrete AR trajectory tokens with a real next-token CE loss (this is the
half of Sec 3's CONFIRM-1 fork-in-the-road this script exercises); Stage 2's deployed
diffusion/flow-matching head has no trajectory-token logits at all and this script's approach
does not apply there (see Sec 7/Sec 8 of langforce_readme.md).

Use ``a1x_rl_b300`` on this B300 (sm_103) box -- plain ``a1x_rl`` bundles nvrtc 12.8, which
predates sm_103 and dies mid-run with ``nvrtc: invalid value for --gpu-architecture``. See the
root ``CLAUDE.md``.

Usage::

    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$((4+i)) a1x_rl_b300/bin/python scripts_fork/phase0_llr_diagnostic.py \\
        --num-shards 4 --shard-idx $i \\
        --out /opt/dlami/nvme/rod/results/phase0_llr/llr.parquet &
    done; wait

Smoke test (single GPU, 8 events)::

    CUDA_VISIBLE_DEVICES=0 a1x_rl_b300/bin/python scripts_fork/phase0_llr_diagnostic.py \\
      --limit 8 --out /tmp/phase0_llr_smoke.parquet
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
# t0 must exceed the history window (16 * 0.1s = 1.6s); matches eval_baseline_ood_reasoning.py's
# MIN_T0_US / eval_common.py's floor.
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

    Identical to eval_baseline_ood_reasoning.py's loader (see that file for provenance).
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
    )

    from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA

    print(f"[phase0] loading model {args.model_path} (Stage-1 discrete-token class) ...", flush=True)
    model = TrainableReasoningVLA.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).cuda().eval()

    # Numerator: [v, traj_history, prompt, a*(traj_future), cot] -- cot scored WITH ground-truth
    # trajectory tokens already in context.
    preprocess_num = get_preprocess_data_fn_from_model_config(
        components_order=["image", "traj_history", "prompt", "traj_future", "cot"],
        components_prompt=["cot"],
        label_components=["cot"],
        generation_mode=False,
        include_camera_ids=True,
        include_frame_nums=True,
        model_config=model.config,
        chat_template_version="r1",
    )
    # Denominator: [v, traj_history, prompt, cot] -- cot scored from vision/history only.
    preprocess_den = get_preprocess_data_fn_from_model_config(
        components_order=["image", "traj_history", "prompt", "cot"],
        components_prompt=["cot"],
        label_components=["cot"],
        generation_mode=False,
        include_camera_ids=True,
        include_frame_nums=True,
        model_config=model.config,
        chat_template_version="r1",
    )

    events = load_events(args.parquet, args.split, all_events=not args.first_event_only)
    if args.num_shards > 1:
        events = events[args.shard_idx :: args.num_shards]
    if args.limit:
        events = events[: args.limit]
    print(
        f"[phase0] shard {args.shard_idx}/{args.num_shards}: {len(events)} events (split={args.split})",
        flush=True,
    )

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()  # streaming -- this root has no local clip media

    def _make_sample(data: dict, preprocess_fn) -> dict:
        sample = {
            "image_frames": data["image_frames"],
            "camera_indices": data["camera_indices"],
            "relative_timestamps": data["relative_timestamps"],
            "ego_history_xyz": data["ego_history_xyz"].squeeze(0),
            "ego_history_rot": data["ego_history_rot"].squeeze(0),
            "cot": data["gold_coc"],
        }
        sample["tokenized_data"] = preprocess_fn(sample)
        return sample

    rows: list[dict] = []
    t_start = time.time()
    for i, ev in enumerate(events):
        clip_id, t0_us = ev["clip_id"], ev["t0_us"]
        try:
            data = load_physical_aiavdataset(clip_id, t0_us=t0_us, avdi=avdi)
            data["gold_coc"] = ev["gold_coc"]

            # --- numerator: cot scored WITH ground-truth future trajectory tokens ---
            sample_num = _make_sample(data, preprocess_num)
            batch_num = collate_fn_from_model_config(
                [sample_num], model_config=model.config,
                include_camera_ids=True, include_frame_nums=True, chat_template_version="r1",
            )
            model_inputs_num = to_device(batch_num, "cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out_num = model(
                    tokenized_data=model_inputs_num["tokenized_data"],
                    ego_history_xyz=data["ego_history_xyz"].cuda(),
                    ego_history_rot=data["ego_history_rot"].cuda(),
                    ego_future_xyz=data["ego_future_xyz"].cuda(),
                    ego_future_rot=data["ego_future_rot"].cuda(),
                    labels_mask=model_inputs_num["labels_mask"],
                )
            logp_num = -out_num.loss_others.item()

            # --- denominator: cot scored from vision/history only (no future traj in context) ---
            sample_den = _make_sample(data, preprocess_den)
            batch_den = collate_fn_from_model_config(
                [sample_den], model_config=model.config,
                include_camera_ids=True, include_frame_nums=True, chat_template_version="r1",
            )
            model_inputs_den = to_device(batch_den, "cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out_den = model(
                    tokenized_data=model_inputs_den["tokenized_data"],
                    ego_history_xyz=data["ego_history_xyz"].cuda(),
                    ego_history_rot=data["ego_history_rot"].cuda(),
                    ego_future_xyz=None,
                    ego_future_rot=None,
                    labels_mask=model_inputs_den["labels_mask"],
                )
            logp_den = -out_den.loss_others.item()

            llr_lang = logp_num - logp_den
            rows.append(
                {
                    "clip_id": clip_id,
                    "event_idx": ev["event_idx"],
                    "event_cluster": ev["event_cluster"],
                    "split": ev["split"],
                    "t0_us": int(data["t0_us"]),
                    "gold_coc": ev["gold_coc"],
                    "logp_num": logp_num,
                    "logp_den": logp_den,
                    "llr_lang": llr_lang,
                    "ok": True,
                }
            )
            if (i + 1) % 20 == 0 or (i + 1) == len(events):
                elapsed = time.time() - t_start
                recent = [r["llr_lang"] for r in rows[-20:] if r.get("ok")]
                print(
                    f"[phase0] [{i + 1}/{len(events)}] {clip_id} split={ev['split']} "
                    f"llr_lang_mean(last20)={np.mean(recent):.4f} nats "
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
            print(f"[phase0] [{i + 1}/{len(events)}] {clip_id} FAILED: {type(e).__name__}: {e}", flush=True)

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
        f"\n[phase0] shard {args.shard_idx}/{args.num_shards} done in "
        f"{(time.time() - t_start) / 60:.1f} min. {len(ok)}/{len(df)} events ok. Wrote {out_path}",
        flush=True,
    )
    if len(ok):
        print(
            f"[phase0] shard summary: llr_lang mean={ok['llr_lang'].mean():.4f}  "
            f"median={ok['llr_lang'].median():.4f}  "
            f"logp_num mean={ok['logp_num'].mean():.4f}  logp_den mean={ok['logp_den'].mean():.4f}",
            flush=True,
        )
        print(ok.groupby("event_cluster")["llr_lang"].agg(["mean", "median", "count"]), flush=True)


if __name__ == "__main__":
    main()
