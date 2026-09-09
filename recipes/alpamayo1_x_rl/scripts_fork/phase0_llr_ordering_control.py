# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Ordering-artifact control for phase0_llr_diagnostic.py.

phase0_llr_diagnostic.py's smoke test found llr_lang strongly negative (mean -3.47 nats over 8
events): teacher-forcing the real ground-truth future trajectory into context *lowers* the
model's log-prob on the gold CoC text, relative to a vision-only pass. That is confounded with
sequence-order novelty -- the numerator pass uses ``[v, traj_history, prompt, traj_future, cot]``,
an ordering the model was never trained on (training order has ``cot`` BEFORE ``traj_future``).

This script disambiguates with a third pass per event: same reordered structure as the
numerator (``[v, ..., traj_future, cot]``), same token-count/discretization machinery, but with
a WRONG trajectory substituted -- another event's real ground-truth future (paired via
``i -> (i+1) % n`` within the loaded batch, so it's real discretized data, not synthetic noise;
isolates "content is wrong" from "token distribution is unrealistic").

    logp_den  = log p(cot | v)                      -- vision only (no future traj in context)
    logp_num  = log p(cot | v, a*_real)              -- reordered, REAL future traj
    logp_shuf = log p(cot | v, a*_wrong)              -- reordered, WRONG (another event's) future traj

Reuses phase0_llr_diagnostic.py's model class, dataset loader, and preprocess/collate calls
verbatim (see that file's docstring for why ``TrainableReasoningVLA.forward``'s own
``loss_others`` on a ``cot``-only ``labels_mask`` already equals ``-logp(cot | context)``).

Smoke test only (single GPU, 8 events)::

    CUDA_VISIBLE_DEVICES=0 a1x_rl_b300/bin/python scripts_fork/phase0_llr_ordering_control.py \\
      --limit 8 --out /tmp/phase0_llr_ordering_control_smoke.parquet
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
MIN_T0_US = 1_700_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--split", default="both", choices=["train", "val", "both"])
    p.add_argument("--first-event-only", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_events(parquet_path: str, split: str, all_events: bool) -> list[dict]:
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

    print(f"[ctrl] loading model {args.model_path} ...", flush=True)
    model = TrainableReasoningVLA.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).cuda().eval()

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
    if args.limit:
        events = events[: args.limit]
    n = len(events)
    print(f"[ctrl] {n} events (split={args.split})", flush=True)

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

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

    def _run_pass(data: dict, preprocess_fn, future_xyz, future_rot):
        sample = _make_sample(data, preprocess_fn)
        batch = collate_fn_from_model_config(
            [sample], model_config=model.config,
            include_camera_ids=True, include_frame_nums=True, chat_template_version="r1",
        )
        model_inputs = to_device(batch, "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                tokenized_data=model_inputs["tokenized_data"],
                ego_history_xyz=data["ego_history_xyz"].cuda(),
                ego_history_rot=data["ego_history_rot"].cuda(),
                ego_future_xyz=future_xyz.cuda() if future_xyz is not None else None,
                ego_future_rot=future_rot.cuda() if future_rot is not None else None,
                labels_mask=model_inputs["labels_mask"],
            )
        return -out.loss_others.item()

    print("[ctrl] loading all event data up front (needed to pair i -> (i+1)%n for the swap) ...", flush=True)
    loaded = []
    for ev in events:
        try:
            data = load_physical_aiavdataset(ev["clip_id"], t0_us=ev["t0_us"], avdi=avdi)
            data["gold_coc"] = ev["gold_coc"]
            loaded.append((ev, data))
        except Exception as e:
            print(f"[ctrl] load FAILED {ev['clip_id']}: {type(e).__name__}: {e}", flush=True)
            loaded.append((ev, None))

    rows: list[dict] = []
    t_start = time.time()
    m = len(loaded)
    for i, (ev, data) in enumerate(loaded):
        if data is None:
            rows.append({"clip_id": ev["clip_id"], "event_idx": ev["event_idx"], "ok": False, "error": "load_failed"})
            continue
        wrong_data = loaded[(i + 1) % m][1]
        if wrong_data is None:
            rows.append({"clip_id": ev["clip_id"], "event_idx": ev["event_idx"], "ok": False, "error": "wrong_pair_load_failed"})
            continue
        try:
            logp_den = _run_pass(data, preprocess_den, None, None)
            logp_num = _run_pass(data, preprocess_num, data["ego_future_xyz"], data["ego_future_rot"])
            logp_shuf = _run_pass(data, preprocess_num, wrong_data["ego_future_xyz"], wrong_data["ego_future_rot"])
            rows.append(
                {
                    "clip_id": ev["clip_id"],
                    "event_idx": ev["event_idx"],
                    "event_cluster": ev["event_cluster"],
                    "split": ev["split"],
                    "wrong_pair_clip_id": loaded[(i + 1) % m][0]["clip_id"],
                    "logp_den": logp_den,
                    "logp_num": logp_num,
                    "logp_shuf": logp_shuf,
                    "llr_num": logp_num - logp_den,
                    "llr_shuf": logp_shuf - logp_den,
                    "ok": True,
                }
            )
            print(
                f"[ctrl] [{i + 1}/{m}] {ev['clip_id']} "
                f"logp_den={logp_den:.3f} logp_num={logp_num:.3f} logp_shuf={logp_shuf:.3f}",
                flush=True,
            )
        except Exception as e:
            rows.append({"clip_id": ev["clip_id"], "event_idx": ev["event_idx"], "ok": False, "error": str(e)})
            print(f"[ctrl] [{i + 1}/{m}] {ev['clip_id']} FAILED: {type(e).__name__}: {e}", flush=True)

    df = pd.DataFrame(rows)
    parent = os.path.dirname(args.out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_parquet(args.out, index=False)

    ok = df[df.get("ok", False) == True]  # noqa: E712
    print(f"\n[ctrl] done in {(time.time() - t_start) / 60:.1f} min. {len(ok)}/{len(df)} ok. Wrote {args.out}", flush=True)
    if len(ok):
        print(
            f"[ctrl] mean logp_den={ok['logp_den'].mean():.4f}  logp_num={ok['logp_num'].mean():.4f}  "
            f"logp_shuf={ok['logp_shuf'].mean():.4f}",
            flush=True,
        )
        print(
            f"[ctrl] mean llr_num (real-vs-vision)={ok['llr_num'].mean():.4f}  "
            f"llr_shuf (wrong-vs-vision)={ok['llr_shuf'].mean():.4f}  "
            f"gap (num-shuf, real-vs-wrong)={(ok['logp_num'] - ok['logp_shuf']).mean():.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
