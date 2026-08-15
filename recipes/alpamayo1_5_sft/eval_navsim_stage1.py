# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Batch inference over a NAVSIM split using a Stage-1 (VLM-only) checkpoint.

alpamayo1.5/eval_navsim.py (the existing navhard eval pipeline, see
navsim/docs/alpamayo_navsim.md) loads checkpoints via the released
`Alpamayo1_5` class, whose `__init__` unconditionally does
`hyu.instantiate(config.diffusion_cfg)` / `hyu.instantiate(config.action_space_cfg)`
(alpamayo1.5/src/alpamayo1_5/models/alpamayo1_5.py). A Stage-1 checkpoint from
this recipe (`TrainableReasoningVLA`) has neither key in its config.json --
Stage 1 never trains a diffusion/flow-matching action expert, only the VLM +
discrete trajectory-token head -- so loading it through `Alpamayo1_5` crashes
at construction time, before any weights load. No config patch fixes this: the
class requires an action expert to exist at all.

This script sidesteps that entirely by loading the checkpoint with its own
native class (`TrainableReasoningVLA.from_pretrained`, zero config surgery)
and calling its `sample_trajectories_from_data` (models/sft_base_model.py),
which does exactly "autoregressively generate discrete trajectory tokens ->
decode via the frozen traj_tokenizer" -- no diffusion involved. This is not
new/experimental: callbacks/viz_callback.py uses the exact same method for
Stage-1's own per-eval-step trajectory visualization during training.

Output is the same parquet schema alpamayo1.5/eval_navsim.py writes, so the
existing (unmodified) scoring pipeline works as-is:

    python scripts/navsim_submission_from_parquet.py --parquet ... --out submission.pkl
    python navsim/planning/script/run_pdm_score_from_submission.py \\
        train_test_split=<split> submission_file_path=submission.pkl ...

Preprocessing note
-------------------
NAVSIM's per-token inference `.npz` (from navsim/scripts/export_navsim_npz.py,
loaded here via alpamayo1.5's `load_navsim_npz` -- a pure numpy/torch/PIL
function with no alpamayo1_5-model dependency, safe to import cross-repo) is
NOT the same schema `NavsimDataset` trains on (that reads the finetune-cache
pickle instead). This script re-packs `load_navsim_npz`'s output into the
same per-item dict shape `NavsimDataset.__getitem__` produces and feeds it
through the SAME preprocessing/collate functions training uses
(`alpamayo.processor.qwen_processor`), with `generation_mode=True` so the
chat template ends the prompt at `<|traj_future_start|>` instead of filling in
ground truth -- verified by hand that the resulting prompt cuts off exactly
there, ready for `model.vlm.generate()` to continue.

One shape gotcha: `load_navsim_npz` bakes in an extra leading "batch of 1" dim
(e.g. `ego_history_xyz` is `(1, 1, 16, 3)`), sized for eval_navsim.py's own
manual `torch.cat`-based batching. `NavsimDataset`'s collate_fn instead
*stacks* raw per-item tensors shaped `(1, 16, 3)` (n_traj_group, T, 3) with no
pre-existing batch dim. That extra dim must be squeezed off before collation,
or `sample_trajectories_from_data`'s `B, n_traj_group, _, _ = shape` unpack
gets a 5D tensor and throws.

The `components_order`/`components_prompt`/`label_components`/
`chat_template_version`/`include_camera_ids`/`include_frame_nums` defaults
below match `data.val_dataset.vla_preprocess_args` in
configs/sft_stage1_navsim*.yaml as of 2026-08-11. If Stage-1 training's
preprocessing config changes, update the defaults here (or pass overrides) to
match -- a mismatch produces a syntactically valid but off-distribution
prompt, not an error.

Usage
-----
    # Single-GPU smoke test:
    a1_5_sft_b300/bin/python -m alpamayo1_5_sft.eval_navsim_stage1 \\
        --checkpoint /path/to/merged_stage1_checkpoint \\
        --npz_dir /opt/dlami/nvme/rod/datasets/navsim_npz/navhard_two_stage \\
        --limit 8 --out /tmp/navhard_stage1_smoke.parquet

`--checkpoint` must be a LoRA-*merged* checkpoint if Stage 1 was trained with
LoRA (`truckdrive/merge_lora.py`) -- `TrainableReasoningVLA.from_pretrained`
loads by name and will leave adapted projections randomly initialized
otherwise (see models/lora.py's ADAPTER_KEY guard in load paths that check
for it; this script does not check for it, so merge first).

This script is intentionally single-GPU / single-process (`--num_shards`/
`--shard_idx` only control which slice of `--npz_dir` it reads, matching
eval_navsim.py's shard-file-naming convention for `run_eval_multigpu.py`-style
merging). For navhard's ~5912 tokens, run one process per free GPU with
`CUDA_VISIBLE_DEVICES` + distinct `--shard_idx` set, then concatenate/rename
shards the same way eval_navsim.py's shards are combined.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ALPAMAYO1_5_SRC = "/mnt/efs/users/rod/repos/alpamayo1.5/src"
if _ALPAMAYO1_5_SRC not in sys.path:
    sys.path.insert(0, _ALPAMAYO1_5_SRC)
from alpamayo1_5.load_navsim import load_navsim_npz  # noqa: E402 -- pure fn, no alpamayo1_5 model deps

from alpamayo.processor.qwen_processor import (  # noqa: E402
    collate_fn_from_model_config,
    get_preprocess_data_fn_from_model_config,
)
from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA  # noqa: E402

# Without this, stdout is fully block-buffered (not line-buffered) whenever it's
# redirected to a file/pipe rather than a TTY -- progress prints (including the
# very first "[shard N/M] ... tokens" line) sit invisible in the buffer for the
# lifetime of the process instead of appearing as they're printed. Don't rely on
# callers remembering `python -u`.
sys.stdout.reconfigure(line_buffering=True)

# Model prediction grid: tokens_per_future_traj waypoints at 10 Hz -> 0.1, 0.2, ... s
_NAVSIM_TIMES_2HZ = np.arange(1, 9) * 0.5

_COMMAND_NAMES = {0: "left", 1: "straight", 2: "right", 3: "unknown"}

_DEFAULT_COMPONENTS_ORDER = ["image", "traj_history", "route", "prompt", "traj_future"]
_DEFAULT_COMPONENTS_PROMPT = ["traj_future"]
_DEFAULT_LABEL_COMPONENTS = ["traj_future"]


def nav_text_from_command(driving_command: np.ndarray) -> str | None:
    """Build a nav_text instruction from NAVSIM's route command one-hot.

    Copied from alpamayo1.5/eval_navsim.py so both eval paths condition on
    route intent identically. Only turns are conditioned; straight/unknown are
    left unconditioned (matches eval_wod_e2e.py's behaviour).
    """
    direction = _COMMAND_NAMES.get(int(np.argmax(driving_command)))
    if direction in ("left", "right"):
        return f"Turn {direction}"
    return None


def to_navsim_poses(pred_xy: np.ndarray, pred_times_10hz: np.ndarray) -> np.ndarray:
    """Convert one 10 Hz (Tf, 2) prediction to NAVSIM's (8, 3) x/y/heading grid.

    Copied from alpamayo1.5/eval_navsim.py (heading derived from path tangent;
    near-stationary steps carry the previous heading forward).
    """
    xy = np.stack(
        [np.interp(_NAVSIM_TIMES_2HZ, pred_times_10hz, pred_xy[:, d]) for d in range(2)],
        axis=-1,
    )  # (8, 2)

    pts = np.concatenate([np.zeros((1, 2)), xy], axis=0)  # (9, 2)
    deltas = np.diff(pts, axis=0)  # (8, 2)
    headings = np.arctan2(deltas[:, 1], deltas[:, 0])

    moving = np.linalg.norm(deltas, axis=-1) > 1e-3
    last = 0.0
    for i in range(len(headings)):
        if moving[i]:
            last = headings[i]
        else:
            headings[i] = last

    return np.concatenate([xy, headings[:, None]], axis=-1).astype(np.float32)  # (8, 3)


def _build_sample(npz_path: Path, no_nav_text: bool) -> dict:
    data = load_navsim_npz(npz_path)
    nav_text = None if no_nav_text else nav_text_from_command(data["driving_command"])
    return {
        "image_frames": data["image_frames"],
        "camera_indices": data["camera_indices"],
        "relative_timestamps": data["relative_timestamps"],
        # Squeeze load_navsim_npz's extra leading batch-of-1 dim -- see module
        # docstring's "shape gotcha" note.
        "ego_history_xyz": data["ego_history_xyz"].squeeze(0),
        "ego_history_rot": data["ego_history_rot"].squeeze(0),
        "nav_text": nav_text,
        "stage": data["stage"],
        "record_key": data["record_key"],
    }


def _run_multigpu_launcher(args: argparse.Namespace) -> None:
    """Re-invoke this same module once per GPU in --gpus, wait for all, merge.

    Mirrors alpamayo1.5/run_eval_multigpu.py's shard-launch-and-merge pattern,
    kept self-contained here (rather than reusing that script directly) so this
    stays a single `-m alpamayo1_5_sft.eval_navsim_stage1 ...` invocation with
    no cross-repo/cross-venv coupling -- the launcher and the workers it spawns
    are the exact same script, same interpreter, same venv.
    """
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    n = len(gpus)
    if n == 0:
        sys.exit("--gpus given but empty.")

    # Forward everything from the original invocation except --gpus itself
    # (each child gets a real --num_shards/--shard_idx instead).
    passthrough = []
    skip_next = False
    for tok in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if tok == "--gpus":
            skip_next = True
            continue
        if tok.startswith("--gpus="):
            continue
        passthrough.append(tok)

    root, _, ext = args.out.rpartition(".")
    log_dir = Path(args.out).parent / "shard_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Launching {n} shards across GPUs {gpus}")
    procs = []
    t0 = time.time()
    for shard_idx, gpu in enumerate(gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        cmd = [
            sys.executable, "-m", "alpamayo1_5_sft.eval_navsim_stage1",
            *passthrough,
            "--num_shards", str(n),
            "--shard_idx", str(shard_idx),
        ]
        log_path = log_dir / f"shard{shard_idx}-of-{n}.log"
        log_f = open(log_path, "w")
        p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((shard_idx, gpu, p, log_f, log_path))
        print(f"  shard {shard_idx} -> GPU {gpu}  (pid {p.pid}, log {log_path})")

    failed = []
    for shard_idx, gpu, p, log_f, log_path in procs:
        rc = p.wait()
        log_f.close()
        status = "ok" if rc == 0 else f"EXIT {rc}"
        print(f"  shard {shard_idx} (GPU {gpu}) finished: {status}  [{log_path}]  "
              f"({(time.time() - t0) / 60:.1f} min elapsed)")
        if rc != 0:
            failed.append((shard_idx, log_path))
    if failed:
        print(f"\nWARNING: {len(failed)} shard(s) failed; merged results will be incomplete.")
        for shard_idx, log_path in failed:
            print(f"  shard {shard_idx}: see {log_path}")

    shard_paths = [Path(f"{root}.shard{i}-of-{n}.{ext}") for i in range(n)]
    existing = [p for p in shard_paths if p.exists()]
    if not existing:
        sys.exit("No shard outputs found to merge.")
    read_fn = pd.read_csv if ext == "csv" else pd.read_parquet
    merged = pd.concat([read_fn(p) for p in existing], ignore_index=True)
    write_fn = merged.to_csv if ext == "csv" else merged.to_parquet
    write_fn(args.out, index=False)

    n_failed = int((merged["stage"] == -1).sum())
    print(f"\nAll shards done in {(time.time() - t0) / 60:.1f} min. "
          f"Merged {len(merged)} rows ({n_failed} failed) -> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True,
                        help="Stage-1 TrainableReasoningVLA checkpoint dir (LoRA-merged if applicable)")
    parser.add_argument("--npz_dir", required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N tokens (for quick tests)")
    parser.add_argument("--num_traj_samples", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=8,
                         help="Number of distinct npz tokens generated together per "
                              "model.generate() call (cross-scene batching, left-padded "
                              "via collate_fn_from_model_config -- same pattern as "
                              "callbacks/val_metric_callback.py's ValMinADECallback). "
                              "Raise until GPU memory or diminishing returns.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_generation_length", type=int, default=None,
                        help="Defaults to the checkpoint's tokens_per_future_traj")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn_implementation", default="eager",
                        choices=["eager", "sdpa", "flash_attention_2"],
                        help="Applies to the HF from_pretrained() attention dispatch check on the "
                             "TrainableReasoningVLA *wrapper* class, which declares neither "
                             "_supports_sdpa nor _supports_flash_attn (only its self.vlm submodule "
                             "does) -- sdpa/flash_attention_2 both raise ValueError at construction "
                             "time. 'eager' is the only choice that loads; self.vlm's own attention "
                             "backend is governed separately by the checkpoint's saved "
                             "config.attn_implementation field (flash_attention_2 for this recipe's "
                             "checkpoints), unaffected by this flag.")
    parser.add_argument("--out", default=None,
                        help="Defaults to <checkpoint>/eval_navsim/navsim_<npz_dir_name>.parquet")
    parser.add_argument("--no_nav_text", action="store_true")
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated physical GPU ids (e.g. '4,5,6,7'). When set, this "
                             "process becomes a launcher: it re-invokes itself once per GPU (each "
                             "pinned via CUDA_VISIBLE_DEVICES, one --shard_idx of len(gpus) total "
                             "shards), waits for all of them, then merges the per-shard parquets "
                             "into --out -- the multi-process for-loop, in one command. Mutually "
                             "exclusive with --num_shards/--shard_idx (which this sets itself).")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--chat_template_version", default="r1_5")
    parser.add_argument("--components_order", nargs="+", default=_DEFAULT_COMPONENTS_ORDER)
    parser.add_argument("--components_prompt", nargs="+", default=_DEFAULT_COMPONENTS_PROMPT)
    parser.add_argument("--label_components", nargs="+", default=_DEFAULT_LABEL_COMPONENTS)
    args = parser.parse_args()

    if args.out is None:
        split_name = Path(args.npz_dir.rstrip("/")).name
        args.out = str(Path(args.checkpoint) / "eval_navsim" / f"navsim_{split_name}.parquet")
        print(f"--out not given; defaulting to {args.out}")

    if args.gpus:
        _run_multigpu_launcher(args)
        return

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    npz_files = sorted(Path(args.npz_dir).glob("*.npz"))
    if args.limit:
        npz_files = npz_files[: args.limit]
    if args.num_shards > 1:
        npz_files = npz_files[args.shard_idx :: args.num_shards]
    n = len(npz_files)
    print(f"[shard {args.shard_idx}/{args.num_shards}] {n} tokens from {args.npz_dir}")

    print(f"Loading checkpoint {args.checkpoint} (attn={args.attn_implementation}) ...")
    model = TrainableReasoningVLA.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, attn_implementation=args.attn_implementation
    ).to("cuda")
    model.eval()

    max_generation_length = args.max_generation_length or model.config.tokens_per_future_traj

    preprocess_fn = get_preprocess_data_fn_from_model_config(
        components_order=args.components_order,
        components_prompt=args.components_prompt,
        label_components=args.label_components,
        generation_mode=True,
        include_camera_ids=True,
        include_frame_nums=True,
        model_config=model.config,
        chat_template_version=args.chat_template_version,
    )

    def _failed_result(stem: str) -> dict:
        return {
            "token": stem, "stage": -1, "poses": np.full(24, np.nan),
            "pred_xy_10hz": np.full(0, np.nan), "num_samples": 0,
            "medoid_idx": -1, "nav_text": "", "coc_text": "",
        }

    def _extract_result(sample: dict, pred_xyz_j) -> dict:
        # pred_xyz_j: [num_traj_sets=1, K, Tf, 3]. Tf is the decoded WAYPOINT
        # count (64, at a fixed 10 Hz/0.1s cadence), not tokens_per_future_traj
        # (128 -- discrete tokens per waypoint pair, e.g. accel+curvature bins;
        # decode() collapses 2 tokens -> 1 waypoint). Derive the time grid from
        # the actual decoded length rather than the token count, or
        # to_navsim_poses's np.interp gets mismatched array lengths.
        pred_xy = pred_xyz_j[0].float().cpu().numpy()[:, :, :2]  # (K, Tf, 2)
        pred_times_10hz = np.arange(1, pred_xy.shape[1] + 1) * 0.1

        if pred_xy.shape[0] > 1:
            dists = np.linalg.norm(
                pred_xy[:, None, :, :] - pred_xy[None, :, :, :], axis=-1
            ).mean(axis=-1)
            medoid_idx = int(dists.sum(axis=1).argmin())
        else:
            medoid_idx = 0

        poses = to_navsim_poses(pred_xy[medoid_idx], pred_times_10hz)

        return {
            "token": sample["record_key"],
            "stage": sample["stage"],
            "poses": poses.reshape(-1),
            "pred_xy_10hz": pred_xy.reshape(-1),
            "num_samples": pred_xy.shape[0],
            "medoid_idx": medoid_idx,
            "nav_text": sample["nav_text"] or "",
            "coc_text": "",  # Stage 1 has no chain-of-causation reasoning to record.
        }

    results = []
    t_start = time.time()
    done = 0
    # Wall-clock breakdown so a slow run can be diagnosed as CPU-bound (disk
    # read + PIL resize of the raw ~1080p frames, done serially with no
    # prefetch/overlap against the GPU) vs GPU-bound (the generate() call
    # itself) instead of guessed at.
    data_time = 0.0
    gpu_time = 0.0

    for batch_start in range(0, n, args.batch_size):
        chunk = npz_files[batch_start: batch_start + args.batch_size]

        # Build + preprocess each scene individually so one corrupt npz only
        # drops that scene, not the whole batch (matches the old per-token
        # fault isolation). Successfully-built scenes are collated together
        # below for a single cross-scene generate() call -- same pattern as
        # ValMinADECallback._build_cache/_run (callbacks/val_metric_callback.py).
        t0 = time.time()
        samples = []
        for npz_path in chunk:
            try:
                sample = _build_sample(npz_path, args.no_nav_text)
                sample["tokenized_data"] = preprocess_fn(sample)
                samples.append(sample)
            except Exception as exc:  # keep going; one bad token shouldn't kill a multi-hour run
                print(f"  FAILED {npz_path.stem}: {type(exc).__name__}: {exc}")
                results.append(_failed_result(npz_path.stem))
            done += 1
        data_time += time.time() - t0

        if samples:
            t1 = time.time()
            try:
                batch = collate_fn_from_model_config(
                    samples, model_config=model.config, chat_template_version=args.chat_template_version
                )
                batch = {
                    k: (v.to("cuda") if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                batch["tokenized_data"] = {
                    k: (v.to("cuda") if isinstance(v, torch.Tensor) else v)
                    for k, v in batch["tokenized_data"].items()
                }

                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_xyz, _pred_rot = model.sample_trajectories_from_data(
                        data=batch,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        temperature=args.temperature,
                        num_traj_samples=args.num_traj_samples,
                        num_traj_sets=1,
                        max_generation_length=max_generation_length,
                    )
                torch.cuda.synchronize()  # generate() is async; sync before stopping the clock
                gpu_time += time.time() - t1
                # pred_xyz: [B, num_traj_sets=1, K, Tf, 3], B == len(samples).
                for j, sample in enumerate(samples):
                    try:
                        results.append(_extract_result(sample, pred_xyz[j]))
                    except Exception as exc:
                        print(f"  FAILED {sample['record_key']}: {type(exc).__name__}: {exc}")
                        results.append(_failed_result(sample["record_key"]))
            except Exception as exc:  # keep going; one bad batch shouldn't kill a multi-hour run
                gpu_time += time.time() - t1
                stems = ", ".join(s["record_key"] for s in samples)
                print(f"  FAILED batch [{stems}]: {type(exc).__name__}: {exc}")
                for sample in samples:
                    results.append(_failed_result(sample["record_key"]))

        # One print per batch (rather than "every 20" as when this looped
        # per-token) -- each generate() call already takes long enough that
        # this is a reasonable cadence regardless of --batch_size.
        elapsed = time.time() - t_start
        rate = done / elapsed
        print(f"  [{done}/{n}] {rate:.2f} tok/s  eta {(n - done) / max(rate, 1e-9) / 60:.1f} min "
              f"(data {data_time:.0f}s / gpu {gpu_time:.0f}s of {elapsed:.0f}s total)")

    df = pd.DataFrame(results)
    n_failed = int((df["stage"] == -1).sum())
    print(f"Done: {len(df)} tokens, {n_failed} failed, {time.time() - t_start:.0f}s")

    out_path = args.out
    if args.num_shards > 1:
        root, _, ext = args.out.rpartition(".")
        out_path = f"{root}.shard{args.shard_idx}-of-{args.num_shards}.{ext}"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".csv":
        df.to_csv(out, index=False)
    else:
        df.to_parquet(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
