# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Turn an evaluate_hf CoC dump into a TruckDriveDataset ``cot_labels`` pickle.

TruckDrive ships no chain-of-causation ground truth, so CoC targets for the
pretraining-ablation experiments are *distilled*: run ``evaluate_hf`` with a
CoT-generating config (e.g. ``sft_truckdrive_cot``) against a checkpoint that
can actually reason, then convert the resulting ``predictions.pt`` into the
``f"{scene_id}|{t0_us}" -> cot`` mapping ``TruckDriveDataset(cot_labels=...)``
expects.

Usage:
    python truckdrive/build_cot_labels.py \
        --predictions /path/to/eval_dump/predictions.pt \
        --out /path/to/cot_train.pkl
"""

import argparse
import pickle

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True, help="evaluate_hf predictions.pt dump")
    ap.add_argument("--out", required=True, help="destination .pkl")
    ap.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="which generated sample to keep when the dump has several per window",
    )
    args = ap.parse_args()

    obj = torch.load(args.predictions, map_location="cpu", weights_only=False)
    records = obj if isinstance(obj, list) else list(obj.values())

    lut: dict[str, str] = {}
    for rec in records:
        if "gen_text/cot" not in rec:
            raise KeyError(
                "predictions.pt has no 'gen_text/cot' -- the dump must come from a run with "
                "return_extra=true and traj_only_generation=false (e.g. sft_truckdrive_cot)."
            )
        # gen_text/* is [num_traj_sets][num_traj_samples]; one set, N samples per window.
        lut[f"{rec['scene_id']}|{int(rec['t0_us'])}"] = rec["gen_text/cot"][0][
            args.sample_index
        ].strip()

    if len(lut) != len(records):
        raise ValueError(
            f"{len(records)} records collapsed to {len(lut)} keys -- (scene_id, t0_us) is "
            "supposed to be unique; refusing to write a lossy mapping."
        )

    with open(args.out, "wb") as f:
        pickle.dump(lut, f)

    empty = sum(1 for v in lut.values() if not v)
    print(f"wrote {args.out}: {len(lut)} entries ({empty} empty)")


if __name__ == "__main__":
    main()
