# Ego-status-only baselines on TruckDrive

<!-- NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes. -->

How much of the finetuned Alpamayo trajectory number comes from the *cameras*,
and how much from simply extrapolating the ego history? These baselines predict
the 6.4 s future from the ego history alone — no images, no language, no VLM.

Both live in `recipes/alpamayo1_5_sft/truckdrive/mlp_baseline.py`:

| Baseline | What it is |
|---|---|
| `cv` | Constant velocity — extrapolate the last history velocity. No fitting. |
| `mlp` | 3×512 GELU MLP, 96-dim ego-history features → 64 XY waypoints. 640,640 params. |

## Why the comparison is apples-to-apples

The script instantiates the **same `TruckDriveDataset`** the SFT/eval configs
use (same `camera_views`, history/future steps, `t0_stride`, reverse filter,
official metainfo split) and reads windows through `_ego_traj` — the exact ego
trajectory `__getitem__` feeds the model, minus the image loads. Val index *i*
here is val index *i* in `evaluate_hf.py`; both produce **2474 records**.

Eval dumps a `predictions.pt` in `evaluate_hf.py`'s record format, so the shared
scorer works unchanged:

```bash
python truckdrive/score_val_horizons.py output_mlp_baseline/predictions.pt
```

Two caveats when reading the table:

- The baselines are **deterministic (K=1)**, so their minADE == ADE. Alpamayo is
  scored best-of-K=6. Compare **ADE against ADE**; `min_ade` flatters the
  generative model by letting it pick its best of six after seeing the GT.
- The baselines' train split carries **no tokenizer-reconstruction pre-filter**
  (it needs a `model_config` and is a property of the discrete trajectory
  tokenizer), so it is a slight superset of the SFT train split. Val is
  unfiltered in `sft_truckdrive.yaml` too, so the eval windows match exactly.

## Numbers (official val split, 2474 windows, BEV XY L2, metres)

| Model | K | ADE 3.0s | ADE 6.0s | ADE 6.4s | FDE 6.4s | minADE 6.4s |
|---|---|---|---|---|---|---|
| constant velocity | 1 | 0.621 | 2.025 | 2.266 | 6.128 | 2.266 |
| **ego-only MLP** | 1 | **0.373** | **1.260** | **1.425** | **4.064** | 1.425 |
| Alpamayo Stage-1 (ckpt-8238) | 6 | 0.395 | 1.445 | 1.641 | 4.798 | **0.688** |
| Alpamayo Stage-2 expert (ckpt-8238) | 6 | 0.549 | 1.774 | 1.986 | 5.386 | 1.134 |

Reading of this: **on single-sample accuracy the ego-only MLP is currently ahead
of both finetuned checkpoints** (1.425 vs 1.641 / 1.986 m at 6.4 s), so the
camera stack is not yet paying for itself on the average window. Where Alpamayo
clearly wins is *multimodality* — its best-of-6 minADE (0.688 m for Stage-1) is
well below anything a deterministic regressor can reach, which is the expected
signature of a model that proposes several plausible futures rather than
regressing to their mean. The MLP's advantage on mean ADE is partly structural:
L2-trained regression to the conditional mean is the ADE-optimal point estimate,
and TruckDrive is highly extrapolable (mostly highway cruising). The interesting
diagnostic is the per-regime split — check whether Alpamayo beats the MLP on the
low-speed turning scenes where extrapolation must fail (`score_val_ade.py`,
`score_val_interesting.py`).

## Reproducing

```bash
cd recipes/alpamayo1_5_sft
export TD_ROOT=s3://torc-data/datasets/TruckDrivePublic
export TD_CACHE=/mnt/efs/users/rod/truckdrive_cache/meta_5views.pkl
export TD_SPLIT=$PWD/truckdrive/truckdrive_metainfo.json
DATA="--data-root $TD_ROOT --backend s3 --metadata-cache $TD_CACHE \
  --split-metainfo $TD_SPLIT \
  --window-cache /mnt/efs/users/rod/truckdrive_cache/baseline_windows.npz"

a1_5_sft/bin/python truckdrive/mlp_baseline.py train $DATA --out-dir output_mlp_baseline
a1_5_sft/bin/python truckdrive/mlp_baseline.py eval  $DATA --model cv --out-dir output_cv_baseline
```

The `--window-cache` npz holds the extracted pose windows (train + val), so
architecture/loss sweeps skip dataset init entirely and train in ~1 min on CPU
or GPU. It is keyed on the window-defining settings and refuses to load if those
changed.
