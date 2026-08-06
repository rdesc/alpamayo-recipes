#!/usr/bin/env python
# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Ego-status-only trajectory baselines on TruckDrive (MLP + constant-velocity).

Answers "how much of the finetuned Alpamayo number is actually coming from the
cameras?" by predicting the future trajectory from the ego history ALONE -- no
images, no language, no VLM.

Comparability is the whole point, so the windows come from the *same*
``TruckDriveDataset`` used for SFT/eval: this module instantiates it with the
identical args (``camera_views`` included -- the index skips scenes lacking a
requested view, so it affects window numbering) and then reads windows through
``_ego_traj``, which is the exact ego trajectory ``__getitem__`` feeds the model
but without the image loads. Val index i here == val index i in ``evaluate_hf``.

Two baselines:

* ``mlp``  ego history -> flat MLP -> 64 future XY waypoints (deterministic).
* ``cv``   constant velocity: extrapolate the last history velocity. No fitting;
           the "is the model doing anything at all" floor.

Both are deterministic (K=1), so minADE == ADE and minFDE == FDE for them --
whereas the Alpamayo numbers are best-of-K=6. Compare ``ade``/``fde`` against
``ade``/``fde``; the K=6 ``min_ade`` column flatters the generative model.

Eval writes a ``predictions.pt`` in the same per-sample record format
``evaluate_hf.py`` dumps, so the shared scorer applies unchanged::

    python truckdrive/score_val_horizons.py <out_dir>/predictions.pt

Usage (from the recipe root, e.g. recipes/alpamayo1_5_sft)::

    export TD_ROOT=s3://torc-data/datasets/TruckDrivePublic
    export TD_CACHE=/mnt/efs/users/rod/truckdrive_cache/meta_5views.pkl
    export TD_SPLIT=$PWD/truckdrive/truckdrive_metainfo.json

    # 1. cache the pose windows once (train+val), then train + eval the MLP
    a1_5_sft/bin/python truckdrive/mlp_baseline.py train \
        --data-root $TD_ROOT --backend s3 --metadata-cache $TD_CACHE \
        --split-metainfo $TD_SPLIT \
        --window-cache /mnt/efs/users/rod/truckdrive_cache/windows.npz \
        --out-dir output_mlp_baseline

    # 2. constant-velocity floor on the same windows (no training)
    a1_5_sft/bin/python truckdrive/mlp_baseline.py eval --model cv \
        --data-root $TD_ROOT --backend s3 --metadata-cache $TD_CACHE \
        --split-metainfo $TD_SPLIT \
        --window-cache /mnt/efs/users/rod/truckdrive_cache/windows.npz \
        --out-dir output_cv_baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Run as a script from the recipe root, so this file's directory is on sys.path.
from horizon_metrics import DEFAULT_HORIZONS_S, displacement_metrics_from_records  # noqa: E402

from alpamayo.data.truckdrive import TruckDriveDataset  # noqa: E402

# The five views sft_truckdrive.yaml trains on. Must match, since scenes missing
# any requested view are dropped from the index (see _build_index).
DEFAULT_VIEWS = (
    "forward_center_medium",
    "sideward_left_front_wide",
    "sideward_right_front_wide",
    "rearward_left_bottom_medium",
    "rearward_right_bottom_medium",
)


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #
def _yaw_from_rot(rot: np.ndarray) -> np.ndarray:
    """Planar yaw (rad) from (T, 3, 3) rotation matrices."""
    return np.arctan2(rot[:, 1, 0], rot[:, 0, 0])


def extract_windows(args, split: str) -> dict[str, np.ndarray | list]:
    """Build the pose-only window arrays for ``split``.

    Returns ``hist_xyz`` (M, H, 3), ``hist_yaw`` (M, H), ``fut_xyz`` (M, T, 3),
    ``scene_id`` (M,) and ``t0_us`` (M,) -- in dataset index order, so element i
    corresponds to ``TruckDriveDataset[i]`` of the same split.
    """
    ds = TruckDriveDataset(
        data_root=args.data_root,
        backend=args.backend,
        region=args.region,
        camera_views=list(args.views),
        num_frames=args.num_frames,
        image_time_step=args.image_time_step,
        num_history_steps=args.num_history_steps,
        num_future_steps=args.num_future_steps,
        time_step=args.time_step,
        t0_stride=args.t0_stride,
        filter_reverse=True,
        metadata_cache=args.metadata_cache,
        split_metainfo=args.split_metainfo,
        split=split,
        val_fraction=args.val_fraction,
        seed=args.seed,
        # No tokenizer pre-filter: it needs a model_config, and it is a property
        # of the discrete trajectory tokenizer, not of this baseline. Val is
        # unfiltered in sft_truckdrive.yaml either way, so eval windows match
        # exactly; the train split here is a superset of the SFT train split.
        tok_recon_max_xy_m=None,
    )

    n = len(ds)
    H, T = ds.num_history_steps, ds.num_future_steps
    hist_xyz = np.zeros((n, H, 3), dtype=np.float32)
    hist_yaw = np.zeros((n, H), dtype=np.float32)
    fut_xyz = np.zeros((n, T, 3), dtype=np.float32)
    scene_ids: list[str] = []
    t0_us = np.zeros((n,), dtype=np.int64)

    t_start = time.time()
    for i in range(n):
        scene_id, t0 = ds._samples[i]
        poses = ds._load_scene(scene_id)["poses"]
        h_xyz, h_rot, f_xyz, _ = ds._ego_traj(poses, t0)
        hist_xyz[i] = h_xyz
        hist_yaw[i] = _yaw_from_rot(h_rot)
        fut_xyz[i] = f_xyz
        scene_ids.append(scene_id)
        t0_us[i] = int(round(t0 * 1e6))
        if (i + 1) % 5000 == 0:
            print(f"  [{split}] {i + 1}/{n} windows ({time.time() - t_start:.0f}s)", flush=True)

    print(f"[{split}] {n} windows in {time.time() - t_start:.0f}s", flush=True)
    return {
        "hist_xyz": hist_xyz,
        "hist_yaw": hist_yaw,
        "fut_xyz": fut_xyz,
        "scene_id": np.array(scene_ids),
        "t0_us": t0_us,
    }


def load_windows(args, splits: tuple[str, ...]) -> dict[str, dict]:
    """Load windows for ``splits``, using ``--window-cache`` when it exists.

    Pose extraction is cheap per window but pays the dataset-init cost, so the
    npz cache makes repeated experiments (loss/architecture sweeps) instant.
    """
    cache = args.window_cache
    if cache and os.path.exists(cache):
        blob = np.load(cache, allow_pickle=True)
        cached_meta = json.loads(str(blob["meta"]))
        want = _window_meta(args)
        if cached_meta != want:
            raise SystemExit(
                f"window cache {cache} was built with different window settings:\n"
                f"  cached: {cached_meta}\n  wanted: {want}\n"
                "Delete it or pass a different --window-cache."
            )
        missing = [s for s in splits if f"{s}/fut_xyz" not in blob]
        if not missing:
            print(f"loaded windows from cache {cache}")
            return {
                s: {k: blob[f"{s}/{k}"] for k in
                    ("hist_xyz", "hist_yaw", "fut_xyz", "scene_id", "t0_us")}
                for s in splits
            }
        print(f"cache {cache} lacks split(s) {missing}; rebuilding")

    out = {s: extract_windows(args, s) for s in splits}
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)) or ".", exist_ok=True)
        flat = {f"{s}/{k}": v for s, d in out.items() for k, v in d.items()}
        flat["meta"] = np.array(json.dumps(_window_meta(args)))
        np.savez(cache, **flat)
        print(f"wrote window cache {cache}")
    return out


def _window_meta(args) -> dict:
    """The settings that determine window identity/ordering (cache key)."""
    return {
        "data_root": args.data_root,
        "views": list(args.views),
        "num_history_steps": args.num_history_steps,
        "num_future_steps": args.num_future_steps,
        "time_step": args.time_step,
        "t0_stride": args.t0_stride,
        "split_metainfo": os.path.basename(args.split_metainfo or ""),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
    }


# --------------------------------------------------------------------------- #
# Features / model
# --------------------------------------------------------------------------- #
def make_features(w: dict) -> np.ndarray:
    """Ego history -> flat feature vector (M, H*6).

    Per history step, in the t0 ego frame (so the last step is x=y=yaw=0):
    position (x, y), heading (cos, sin), and the per-step velocity (vx, vy)
    in m/s. Velocity is redundant with position for an MLP in principle, but
    handing it over explicitly is what makes the constant-velocity solution
    trivially reachable -- the baseline should not have to learn to differentiate.
    """
    xy = w["hist_xyz"][:, :, :2]                      # (M, H, 2)
    yaw = w["hist_yaw"]                               # (M, H)
    vel = np.diff(xy, axis=1, prepend=xy[:, :1])      # (M, H, 2), first step = 0
    feats = np.concatenate(
        [xy, np.cos(yaw)[..., None], np.sin(yaw)[..., None], vel], axis=-1
    )
    return feats.reshape(len(xy), -1).astype(np.float32)


class TrajMLP(nn.Module):
    """Flat MLP: ego-history features -> (T, 2) future XY, in normalized space."""

    def __init__(self, in_dim: int, num_steps: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        blocks: list[nn.Module] = []
        d = in_dim
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.GELU()]
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
            d = hidden
        blocks.append(nn.Linear(d, num_steps * 2))
        self.net = nn.Sequential(*blocks)
        self.num_steps = num_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, self.num_steps, 2)


def constant_velocity(w: dict, num_steps: int, time_step: float) -> np.ndarray:
    """Extrapolate the last history velocity -> (M, T, 2) future XY.

    Velocity is the mean over the last few history steps rather than a single
    finite difference: pose noise at 10 Hz makes the one-step estimate jittery,
    which would understate a fair CV floor.
    """
    xy = w["hist_xyz"][:, :, :2]
    k = min(5, xy.shape[1] - 1)
    v = (xy[:, -1] - xy[:, -1 - k]) / (k * time_step)  # (M, 2) m/s
    t = np.arange(1, num_steps + 1, dtype=np.float32) * time_step  # future offsets
    return (v[:, None, :] * t[None, :, None]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Metrics / dumping
# --------------------------------------------------------------------------- #
def build_records(pred_xy: np.ndarray, w: dict) -> list[dict]:
    """Per-sample records in evaluate_hf.py's ``predictions.pt`` format.

    Shapes follow the generative path: ``pred_xyz`` is [N=1, K=1, T, 3] (these
    baselines are deterministic) and ``ego_future_xyz`` is [1, T, 3]. z is left
    at 0 -- every metric in horizon_metrics is BEV (XY) only.
    """
    m, t, _ = pred_xy.shape
    pred_xyz = np.zeros((m, 1, 1, t, 3), dtype=np.float32)
    pred_xyz[..., :2] = pred_xy[:, None, None]
    return [
        {
            "pred_xyz": torch.from_numpy(pred_xyz[i]),
            "ego_future_xyz": torch.from_numpy(w["fut_xyz"][i][None]),
            "ego_history_xyz": torch.from_numpy(w["hist_xyz"][i][None]),
            "scene_id": str(w["scene_id"][i]),
            "clip_id": str(w["scene_id"][i]),
            "t0_us": int(w["t0_us"][i]),
        }
        for i in range(m)
    ]


def report(records: list[dict], horizons, time_step: float, out_dir: str, tag: str) -> dict:
    metrics = displacement_metrics_from_records(records, horizons_s=horizons, time_step=time_step)
    print(f"\n{tag}: records={len(records)}  (BEV XY L2, K=1 -> minADE==ADE)\n")
    header = f"{'horizon':>8} | {'ADE':>8} {'FDE':>8}"
    print(header)
    print("-" * len(header))
    for sec in horizons:
        key = f"by_t={sec:.1f}"
        if f"ade/{key}" not in metrics:
            continue
        print(f"{sec:>6.1f}s | {metrics[f'ade/{key}']:>8.4f} {metrics[f'fde/{key}']:>8.4f}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save(records, os.path.join(out_dir, "predictions.pt"))
    print(f"\nwrote {out_dir}/predictions.pt and metrics.json")
    return metrics


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_train(args) -> None:
    windows = load_windows(args, ("train", "val"))
    tr, va = windows["train"], windows["val"]

    x_tr, x_va = make_features(tr), make_features(va)
    y_tr = tr["fut_xyz"][:, :, :2]
    y_va = va["fut_xyz"][:, :, :2]

    # Standardize inputs and targets with TRAIN statistics only.
    x_mean, x_std = x_tr.mean(0), x_tr.std(0) + 1e-6
    y_mean, y_std = y_tr.mean(0), y_tr.std(0) + 1e-6

    device = torch.device(args.device)
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)  # noqa: E731
    xt, yt = to((x_tr - x_mean) / x_std), to((y_tr - y_mean) / y_std)
    xv = to((x_va - x_mean) / x_std)
    yv_m = to(y_va)  # val target kept in metres for the reported ADE

    y_mean_t, y_std_t = to(y_mean), to(y_std)

    torch.manual_seed(args.seed)
    model = TrajMLP(xt.shape[1], y_tr.shape[1], args.hidden, args.layers, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(xt) // args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nMLP: {n_params / 1e6:.2f}M params  in_dim={xt.shape[1]}  "
          f"train={len(xt)}  val={len(xv)}\n")

    best_ade = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        total = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * args.batch_size : (b + 1) * args.batch_size]
            pred = model(xt[idx])
            if args.loss == "l2":
                # Mean per-waypoint L2 in NORMALIZED space -- the direct
                # surrogate for ADE (MSE would over-weight the far horizon).
                loss = torch.linalg.norm(pred - yt[idx], dim=-1).mean()
            else:
                loss = nn.functional.mse_loss(pred, yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()

        model.eval()
        with torch.no_grad():
            pv = model(xv) * y_std_t + y_mean_t  # back to metres
            ade = torch.linalg.norm(pv - yv_m, dim=-1).mean().item()
        flag = ""
        if ade < best_ade:
            best_ade, flag = ade, "  *"
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch + 1:3d}/{args.epochs}  train_loss={total / steps_per_epoch:.4f}  "
              f"val_ADE(6.4s)={ade:.4f} m{flag}", flush=True)

    model.load_state_dict(best_state)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std,
            "arch": {"in_dim": int(xt.shape[1]), "num_steps": int(y_tr.shape[1]),
                     "hidden": args.hidden, "layers": args.layers, "dropout": args.dropout},
        },
        os.path.join(args.out_dir, "mlp_baseline.pt"),
    )
    print(f"\nbest val ADE {best_ade:.4f} m -> {args.out_dir}/mlp_baseline.pt")

    model.eval()
    with torch.no_grad():
        pred_xy = (model(xv) * y_std_t + y_mean_t).cpu().numpy()
    report(build_records(pred_xy, va), args.horizons, args.time_step, args.out_dir, "MLP (val)")


def cmd_eval(args) -> None:
    va = load_windows(args, ("val",))["val"]

    if args.model == "cv":
        pred_xy = constant_velocity(va, args.num_future_steps, args.time_step)
        tag = "constant-velocity (val)"
    else:
        ckpt_path = args.checkpoint or os.path.join(args.out_dir, "mlp_baseline.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        device = torch.device(args.device)
        model = TrajMLP(**ckpt["arch"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        x = (make_features(va) - ckpt["x_mean"]) / ckpt["x_std"]
        with torch.no_grad():
            out = model(torch.from_numpy(x).to(device)).cpu().numpy()
        pred_xy = out * ckpt["y_std"] + ckpt["y_mean"]
        tag = f"MLP (val) [{ckpt_path}]"

    report(build_records(pred_xy, va), args.horizons, args.time_step, args.out_dir, tag)


# --------------------------------------------------------------------------- #
def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-root", required=True, help="s3://bucket/prefix or local dir")
    p.add_argument("--backend", choices=("s3", "local"), default="s3")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--metadata-cache", default=None, help="scene-metadata pickle (build_cache)")
    p.add_argument("--split-metainfo", default=None, help="truckdrive_metainfo.json (recommended)")
    p.add_argument("--views", default=",".join(DEFAULT_VIEWS),
                   help="comma-separated views; MUST match the SFT config for index parity")
    p.add_argument("--num-frames", type=int, default=4)
    p.add_argument("--image-time-step", type=float, default=0.2)
    p.add_argument("--num-history-steps", type=int, default=16)
    p.add_argument("--num-future-steps", type=int, default=64)
    p.add_argument("--time-step", type=float, default=0.1)
    p.add_argument("--t0-stride", type=int, default=10)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--window-cache", default=None, help="npz cache of extracted pose windows")
    p.add_argument("--horizons", type=float, nargs="+", default=list(DEFAULT_HORIZONS_S))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train the ego-status MLP and score it on val")
    _add_data_args(t)
    t.add_argument("--out-dir", default="output_mlp_baseline")
    t.add_argument("--hidden", type=int, default=512)
    t.add_argument("--layers", type=int, default=3)
    t.add_argument("--dropout", type=float, default=0.0)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--loss", choices=("l2", "mse"), default="l2")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", help="score a trained MLP, or the constant-velocity floor")
    _add_data_args(e)
    e.add_argument("--model", choices=("mlp", "cv"), default="mlp")
    e.add_argument("--checkpoint", default=None, help="mlp_baseline.pt (default: <out-dir>/)")
    e.add_argument("--out-dir", default="output_mlp_baseline")
    e.set_defaults(func=cmd_eval)

    args = ap.parse_args()
    args.views = [v.strip() for v in args.views.split(",") if v.strip()]
    args.func(args)


if __name__ == "__main__":
    main()
