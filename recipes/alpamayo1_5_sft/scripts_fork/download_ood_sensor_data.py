# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Pre-download the sensor data (egomotion + cameras) for every clip referenced by the PAI-AV
OOD reasoning subset (``reasoning/ood_reasoning.parquet``), so eval runs read from the local HF
cache instead of streaming over HTTP each time (streaming trips the account-wide HF Hub rate
quota -- see eval_baseline_ood_reasoning_actionhead.py's module docstring).

This populates ``physical_ai_av.PhysicalAIAVDatasetInterface``'s own cache (``HF_HOME``, unless
``--cache-dir`` is given), which is exactly what a bare ``PhysicalAIAVDatasetInterface()`` in the
eval scripts resolves to -- so no eval-script changes are needed after running this once.

Usage:
  a1_5_sft_b300/bin/python scripts_fork/download_ood_sensor_data.py \\
    --parquet /opt/dlami/nvme/rod/datasets/pai_av_dataset/reasoning/ood_reasoning.parquet
"""

import argparse

import pandas as pd
import physical_ai_av

DEFAULT_PARQUET = "/opt/dlami/nvme/rod/datasets/pai_av_dataset/reasoning/ood_reasoning.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--split", default="both", choices=["train", "val", "both"])
    p.add_argument(
        "--cache-dir",
        default=None,
        help="HF Hub cache dir. Default: None, i.e. whatever HF_HOME/huggingface_hub resolves to "
        "-- must match what the eval scripts' bare PhysicalAIAVDatasetInterface() will use.",
    )
    p.add_argument("--max-workers", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_parquet(args.parquet)
    if args.split != "both":
        df = df[df["split"] == args.split]
    clip_ids = sorted(set(df.index))
    print(f"[download] {len(clip_ids)} unique clip_ids ({args.split=}) from {args.parquet}")

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(
        cache_dir=args.cache_dir,
        confirm_download_threshold_gb=float("inf"),  # non-interactive
    )

    features = {
        avdi.features.LABELS.EGOMOTION,
        avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
    }
    chunks = sorted(set(avdi.clip_index.loc[clip_ids, "chunk"]))
    print(f"[download] spans {len(chunks)} chunks; downloading features={sorted(features)}")

    avdi.download_chunk_features(chunks, features, max_workers=args.max_workers)
    print("[download] done.")


if __name__ == "__main__":
    main()
