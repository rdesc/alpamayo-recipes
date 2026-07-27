#!/usr/bin/env python
# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Recompute displacement metrics at fixed horizons from a saved predictions.pt.

evaluate_hf.py saves per-sample records (all K trajectory samples + GT), so any
horizon can be scored offline without re-running inference. This uses the SAME
shared implementation as metrics.json (truckdrive/horizon_metrics.py) so the two
always agree -- BEV (XY) L2, convention-B minADE (best mode chosen over the full
horizon, then truncated), sample-0 ADE/FDE, best-of-K minFDE.

Usage:
  a1_5_sft/bin/python truckdrive/score_val_horizons.py <predictions.pt> \
      [--horizons 3 6 6.4] [--time-step 0.1]
"""

import argparse

import torch

# Run as a script from the recipe root (`truckdrive/score_val_horizons.py`), so
# the script's own directory is on sys.path and horizon_metrics is importable.
from horizon_metrics import DEFAULT_HORIZONS_S, displacement_metrics_from_records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", help="path to predictions.pt")
    ap.add_argument("--time-step", type=float, default=0.1, help="seconds per trajectory step")
    ap.add_argument(
        "--horizons",
        type=float,
        nargs="+",
        default=list(DEFAULT_HORIZONS_S),
        help="horizons in seconds (default: 3.0 6.0 6.4)",
    )
    args = ap.parse_args()

    records = torch.load(args.predictions, map_location="cpu", weights_only=False)
    metrics = displacement_metrics_from_records(
        records, horizons_s=args.horizons, time_step=args.time_step
    )

    print(f"records={len(records)}  (BEV XY L2)\n")
    header = f"{'horizon':>8} | {'minADE':>8} {'ADE':>8} {'minFDE':>8} {'FDE':>8}"
    print(header)
    print("-" * len(header))
    for sec in args.horizons:
        key = f"by_t={sec:.1f}"
        if f"min_ade/{key}" not in metrics:
            print(f"{sec:>6.1f}s | (skipped: exceeds trajectory length)")
            continue
        print(
            f"{sec:>6.1f}s | "
            f"{metrics[f'min_ade/{key}']:>8.4f} {metrics[f'ade/{key}']:>8.4f} "
            f"{metrics[f'min_fde/{key}']:>8.4f} {metrics[f'fde/{key}']:>8.4f}"
        )


if __name__ == "__main__":
    main()
