# PAI dataset download — run notes

Working log of how we downloaded the Physical AI Autonomous Vehicles (PAI) dataset
subset used for the Stage-1 SFT navigation smoke test. Complements the recipe
instructions in [`recipes/alpamayo1_5_sft/README.md`](../recipes/alpamayo1_5_sft/README.md)
(section "Prepare dataset → PAI dataset (for navigation)") with the exact commands,
paths, and sizes from an actual run.

## What we downloaded

The **nav-annotation subset**: the 19 chunks that contain the 20 navigation demo
clips in [`nav_demo_samples.json`](https://github.com/NVlabs/alpamayo1.5/blob/main/notebooks/nav_demo_samples.json),
for 4 cameras + calibration + egomotion.

- **Output dir:** `~/datasets/pai_nav_subset_19chunks` (symlink → `/opt/dlami/nvme/rod/datasets/pai_nav_subset_19chunks`)
- **Final size:** ~100 GB (99 GB camera, 752 MB egomotion, rest is metadata/index)
- **Files:** 137 (19 chunks × 4 cameras + calibration + egomotion + mandatory index files)

### Command

```bash
cd ~/repos/alpamayo-recipes
python scripts/download_pai.py \
  --chunk-ids "214 224 276 317 420 727 728 968 982 1519 1657 1984 2277 2368 2372 2447 2599 2634 2868" \
  --camera camera_front_wide_120fov camera_cross_left_120fov camera_cross_right_120fov camera_front_tele_30fov \
  --calibration camera_intrinsics sensor_extrinsics \
  --labels egomotion \
  --output-dir ~/datasets/pai_nav_subset_19chunks
```

The repo has no project venv yet, so we ran the script with a venv from a sibling
repo that already has `huggingface_hub` + `pandas`:
`~/repos/alpasim/.venv/bin/python` (huggingface_hub 0.36.2). Any env with those two
deps works.

The 19 chunk IDs are not arbitrary — they are the `chunk` values from
`clip_index.parquet` for the 20 nav demo `clip_id`s (two samples share a clip, so
20 clips → 19 unique clip_ids → 19 chunks). Verified by joining `nav_demo_samples.json`
against `clip_index.parquet`.

## Gotchas worth remembering

- **`hf download` URL form does not work.** `hf download hf://datasets/<repo>/<path>`
  fails. Use repo id + file path + `--repo-type`:
  ```bash
  hf download nvidia/PhysicalAI-Autonomous-Vehicles --repo-type dataset reasoning/ood_reasoning.parquet
  ```
  The `hf://datasets/...` URI form only works with the Python `HfFileSystem` API
  (e.g. `pandas.read_parquet("hf://datasets/.../ood_reasoning.parquet")`).

- **`--only-reasoning-chunks` without `--num-reasoning-clips` is huge.** That mode
  downloads every chunk that contains any OOD-reasoning clip with non-empty events.
  We measured it: **1,084 chunks → ~6.2 TB** (camera tars are ~99% of it, ~5.7 GB/chunk).
  Use `--num-reasoning-clips N` to downsample, or use the fixed nav chunk list above.

- **Rough sizing rule:** cost scales with the number of distinct chunks; budget
  ~5.7 GB per chunk for the 4-camera + calibration + egomotion component set.

- **Resuming works.** A cancelled `snapshot_download` re-run with the same args
  skips already-fetched files. Re-running the command above to the same `--output-dir`
  fills in the rest.

- **HF hub cache is separate from `--output-dir`.** A standalone `hf download`
  (no `--local-dir`) lands in `~/.cache/huggingface/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles/`
  (symlinks into `blobs/`). The `download_pai.py` script writes real files into
  `--output-dir`. Clean up the cache separately if you need the space.

## Verification

```bash
du -sh ~/datasets/pai_nav_subset_19chunks                 # ~100G
for c in ~/datasets/pai_nav_subset_19chunks/camera/*/; do echo "$(basename "$c"): $(ls "$c" | wc -l)"; done
# each of the 4 cameras should list 19 files (one tar per chunk)
```
