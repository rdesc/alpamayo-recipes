#!/usr/bin/env python
"""Recompute ADE / minADE at 1..6 s horizons from a saved predictions.pt dump.

evaluate_hf.py saves per-sample records (all K trajectory samples + GT), so any
horizon can be scored offline without re-running inference. Metrics are BEV (XY)
L2, matching alpamayo.metrics.distance_metrics (only_xy=True).

Definitions (K = number of sampled trajectories per window):
  ADE@t      mean over K of (mean L2 over steps 0..t)   -- expected error of the set
  minADE@t   min  over K of (mean L2 over steps 0..t)   -- best-of-K, re-selected per t
  ADE0@t     L2 of sample 0 averaged over 0..t          -- matches metrics.json "ade"
                                                           (its dummy logprob picks K=0)

Usage:
  a1_5_sft/bin/python truckdrive/score_val_horizons.py <predictions.pt> [--time-step 0.1]
"""

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", help="path to predictions.pt")
    ap.add_argument("--time-step", type=float, default=0.1, help="seconds per trajectory step")
    ap.add_argument(
        "--horizons",
        type=float,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6],
        help="horizons in seconds",
    )
    args = ap.parse_args()

    records = torch.load(args.predictions, map_location="cpu")
    # pred_xyz per record: [N, K, T, 3]; ego_future_xyz: [n_traj_group, T, 3] (use last group).
    pred = torch.stack([r["pred_xyz"] for r in records]).float()  # [B, N, K, T, 3]
    gt = torch.stack([r["ego_future_xyz"][-1] for r in records]).float()  # [B, T, 3]
    B, N, K, T, _ = pred.shape

    # per-step BEV L2: [B, N, K, T]
    l2 = torch.linalg.norm((pred - gt[:, None, None])[..., :2], dim=-1)

    print(f"records={B}  N={N}  K={K}  T={T} ({T * args.time_step:.1f}s @ {args.time_step}s/step)\n")
    header = f"{'horizon':>8} | {'ADE@t':>8} {'minADE@t':>9} {'ADE0@t':>8}"
    print(header)
    print("-" * len(header))

    for sec in args.horizons:
        t = int(round(sec / args.time_step))
        if t > T:
            print(f"{sec:>6.1f}s | (skipped: {t} > T={T})")
            continue
        cum = l2[..., :t].mean(dim=-1)  # [B, N, K] mean displacement up to t
        ade = cum.mean(dim=2).mean(dim=1).mean().item()  # mean over K, N, batch
        min_ade = cum.min(dim=2).values.mean(dim=1).mean().item()  # best-of-K
        ade0 = cum[:, :, 0].mean(dim=1).mean().item()  # sample 0 only
        print(f"{sec:>6.1f}s | {ade:>8.4f} {min_ade:>9.4f} {ade0:>8.4f}")

    # Full-horizon minADE (matches metrics.json "min_ade").
    cum_full = l2.mean(dim=-1)
    print(
        f"\nfull {T * args.time_step:.1f}s | minADE={cum_full.min(dim=2).values.mean().item():.4f} "
        f" ADE(meanK)={cum_full.mean(dim=2).mean().item():.4f} "
        f" ADE0={cum_full[:, :, 0].mean().item():.4f}"
    )


if __name__ == "__main__":
    main()
