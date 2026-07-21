# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Re-select the fixed viz set from an existing score_val_ade manifest.

The per-window minADE lives in the manifest's ``per_window``, so changing the
number of bins / examples-per-bin needs no re-scoring -- just re-run this.

    python -m alpamayo1_5_sft.truckdrive.rebin_val_ade \
        /mnt/efs/users/rod/truckdrive_val_ade_selection.json \
        --num-bins 5 --per-bin 4
"""
import argparse
import json

from alpamayo1_5_sft.truckdrive._ade_binning import select_stratified


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", help="score_val_ade output JSON")
    ap.add_argument("--num-bins", type=int, default=5)
    ap.add_argument("--per-bin", type=int, default=4)
    ap.add_argument("--out", default=None, help="rewrite manifest here (default: in place)")
    args = ap.parse_args()

    m = json.load(open(args.manifest))
    records = m["per_window"]
    sample_indices, bins_summary = select_stratified(records, args.num_bins, args.per_bin)

    m["num_bins"] = args.num_bins
    m["per_bin"] = args.per_bin
    m["bins"] = bins_summary
    m["sample_indices"] = sample_indices
    out = args.out or args.manifest
    with open(out, "w") as f:
        json.dump(m, f, indent=2)

    print(f"re-binned {len(records)} windows -> {args.num_bins} bins x {args.per_bin}")
    for bs in bins_summary:
        print(f" bin {bs['bin']}  ADE [{bs['ade_lo']:.2f}, {bs['ade_hi']:.2f}] m  "
              f"(n={bs['n_in_bin']}):")
        for p in bs["picked"]:
            print(f"    idx {p['idx']:>5}  {p['scene_id']}  minADE={p['min_ade']}m")
    print(f"\n viz.sample_indices: {sample_indices}")
    print(f" wrote -> {out}")


if __name__ == "__main__":
    main()
