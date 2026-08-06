# Running inference with a TruckDrive-SFTed Alpamayo-1.5 checkpoint

How to run eval / inference on the TruckDrive val split with a fine-tuned
checkpoint — Stage-1 (discrete trajectory tokens) or Stage-2 (diffusion action
expert), with chain-of-causation (CoC) reasoning on or off.

Nothing here trains anything: `evaluate_hf.py` loads a checkpoint, rolls out the
val split, prints aggregate metrics, and (by default) dumps per-sample
predictions for offline analysis.

## 0. Prerequisites

- **Venv**: `recipes/alpamayo1_5_sft/a1_5_sft/` — python at
  `a1_5_sft/bin/python`. All commands below are run from `recipes/alpamayo1_5_sft/`.
- **Data**: TruckDrivePublic, either streamed from S3
  (`backend=s3`) or a staged local dir of `scene_*/` subdirs (`backend=local`).
- **Metadata cache** (optional): a pickle of pose text + per-view
  frame listings. Without it, dataset init spends its time on S3 metadata (one
  pose GET + one LIST per view per scene); with it, only image bytes are
  streamed. It is ~140 MB for the 5-view set, so it is **not** in the repo —
  build it once (takes a few minutes):

  ```bash
  python -m alpamayo1_5_sft.truckdrive.build_cache \
    --data-root s3://torc-data/datasets/TruckDrivePublic \
    --cache /path/on/shared/storage/meta_5views.pkl \
    --views forward_center_medium,sideward_left_front_wide,sideward_right_front_wide,rearward_left_bottom_medium,rearward_right_bottom_medium \
    --threads 32
  ```

  The view list must match the config's `camera_views` (the five above are what
  `sft_truckdrive` uses). The cache is also available at `/mnt/efs/users/rod/truckdrive_cache/meta_5views.pkl`.
- **Split metainfo**: `truckdrive/truckdrive_metainfo.json` (in this repo) — the
  official devkit train/val/test scene lists. 

Set these once and reuse them:

```bash
export TD_ROOT=s3://torc-data/datasets/TruckDrivePublic
export TD_CACHE=/mnt/efs/users/rod/truckdrive_cache/meta_5views.pkl
export TD_SPLIT=$PWD/truckdrive/truckdrive_metainfo.json
DATA_ARGS="data.val_dataset.data_root=$TD_ROOT \
  data.val_dataset.backend=s3 \
  data.val_dataset.metadata_cache=$TD_CACHE \
  data.val_dataset.split_metainfo=$TD_SPLIT"
```

## 1. Pick the config

The stage and the CoC on/off switch are both *config* choices — there is no CLI
flag for them. Four combinations:

| Config | Stage | CoC | What produces the trajectory |
|---|---|---|---|
| `sft_truckdrive` | 1 | off | VLM generates discrete trajectory tokens, decoded by the frozen unicycle tokenizer |
| `sft_truckdrive_cot` | 1 | **on** | VLM free-runs reasoning → `<cot_end>` → trajectory tokens |
| `sft_stage2_truckdrive` | 2 | off | Frozen VLM's KV-cache cropped at `<traj_future_start>`; diffusion expert emits continuous actions |
| `sft_stage2_truckdrive_cot` | 2 | **on** | Same, but the VLM reasons first, so the expert conditions on a KV-cache containing the reasoning |

How the CoC variants work: in generation mode only the **last** entry of
`components_order` is "asked for". The `_cot` configs end it at `cot`, which
primes the assistant with `<cot_start>` and nothing else, so the model free-runs
reasoning before continuing into the trajectory. They also raise
`max_generation_length` (reasoning + trajectory must both fit) and set
`return_extra=true` so the decoded reasoning is captured.


## 2. Point at the checkpoint

Pass `+evaluate.eval_ckpt=<dir>` (the leading `+` is required) and point it at the `checkpoint-N/` output dir.
That single flag is enough for either stage: `evaluate_hf.py` routes it to the
right model field, and a Stage-2 checkpoint already contains both the
fine-tuned VLM and the trained expert.

## 3. Commands

Stage 1, traj-only (the reference ADE/minADE run):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node 8 --master_port 29513 -m alpamayo1_5_sft.evaluate_hf \
  --config-path pkg://alpamayo1_5_sft/configs --config-name sft_truckdrive \
  +evaluate.eval_ckpt=/path/to/stage1_run/checkpoint-4119 \
  evaluate.max_eval_steps=-1 \
  $DATA_ARGS
```

Stage 1, CoC on:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node 8 --master_port 29513 -m alpamayo1_5_sft.evaluate_hf \
  --config-path pkg://alpamayo1_5_sft/configs --config-name sft_truckdrive_cot \
  +evaluate.eval_ckpt=/path/to/stage1_run/checkpoint-4119 \
  evaluate.max_eval_steps=-1 \
  $DATA_ARGS
```

Stage 2 (expert), traj-only:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node 8 --master_port 29541 -m alpamayo1_5_sft.evaluate_hf \
  --config-path pkg://alpamayo1_5_sft/configs --config-name sft_stage2_truckdrive \
  +evaluate.eval_ckpt=/path/to/stage2_run/checkpoint-8238 \
  evaluate.save_predictions=true \
  $DATA_ARGS
```

Stage 2, CoC on — same as above with `--config-name sft_stage2_truckdrive_cot`.

Useful overrides:

- `evaluate.max_eval_steps=20` — smoke-test on a few batches before committing
  the full val split.
- `trainer.deepspeed=null` — the Stage-1 configs inherit ZeRO-2 from
  `sft_base`; drop it if DeepSpeed init complains or you are on a single GPU.
  (The Stage-2 configs already set it to `null`.)
- `model.attn_implementation=sdpa` — required on Blackwell (B300) boxes; the
  venv's flash-attn wheel has no `sm_100` kernels. A100 boxes can keep the
  checkpoint default.
- `--master_port` — bump it if you get an address-in-use error.

## 4. Outputs

Rank 0 **prints** the streamed aggregate metrics at the end of the run:
`val/metric/ade`, `min_ade`, their `by_t=3.0` variants, and corner distances.
Note that FDE/minFDE are *not* in this printout.

With `evaluate.save_predictions=true` (the default), two files land in
`<eval_ckpt>/eval/` — or `eval_<YYYYmmdd_HHMMSS>/` if that dir already exists,
so a re-run never clobbers an earlier eval (override with
`evaluate.predictions_dir=...`):

- **`metrics.json`** — the printed metrics *plus* `fde/*` and `min_fde/*` and
  the 6.0 s horizon, recomputed over the deduped records. This is the complete
  number set; prefer it over the console output.
- **`predictions.pt`** — per-sample records (per-rank shards are merged, deduped
  on `(scene_id, t0_us)`, and removed).

Each record carries `scene_id` / `t0_us`, `pred_xyz`, `ego_future_xyz`,
`ego_history_xyz`, the per-sample `metric/*` values, and — on CoC runs —
`gen_text/cot`. **A non-empty `gen_text/cot` is the reliable marker that a run
was actually CoC-on.**

## 5. Offline analysis

These all work off a saved `predictions.pt`:

```bash
# Re-score with shared-horizon metrics
python truckdrive/score_val_horizons.py /path/to/checkpoint-N/eval/predictions.pt

# Per-scene minADE, worst-first, coloured by TruckDrive chunk
python -m alpamayo1_5_sft.truckdrive.plot_scene_ade \
  --predictions /path/to/checkpoint-N/eval/predictions.pt \
  --out /tmp/scene_ade.png
  --out /tmp/scene_ade.png
```

`truckdrive/render_scene_video.py` renders an MP4 walkthrough of a whole val
scene (camera inputs + BEV overlay of GT history / GT future / sampled
predictions, plus the generated reasoning when the run was CoC-on). It re-loads
camera images, so it does need dataset access:

```bash
python -m alpamayo1_5_sft.truckdrive.render_scene_video \
  --config-path pkg://alpamayo1_5_sft/configs --config-name sft_truckdrive \
  +video.predictions=/path/to/checkpoint-N/eval/predictions.pt \
  +video.scenes=[scene_28_24,scene_42_12] \
  +video.out_dir=/tmp/truckdrive_val_videos \
  $DATA_ARGS
```

Add `+video.list_scenes=true` to list the scenes in a predictions file,
worst mean-minADE first, and exit. Other knobs: `+video.fps=2`,
`+video.show_cot=false`, `+video.overlay_sample=N` (which sample is drawn on the
camera tiles; default 0).

### Comparing two models in one video

Pass a second predictions file to overlay both models on the same frames — e.g.
the Stage-2 diffusion action head against the Stage-1 discrete trajectory
tokens. The two files are joined per window on `(scene_id, t0_us)`; windows
missing from B simply get no B curves.

```bash
python -m alpamayo1_5_sft.truckdrive.render_scene_video \
  --config-path pkg://alpamayo1_5_sft/configs --config-name sft_truckdrive \
  +video.predictions=/path/to/stage1-checkpoint/eval/predictions.pt \
  +video.predictions_b=/path/to/stage2-checkpoint/eval/predictions.pt \
  +video.label=stage1-tokens +video.label_b=stage2-actionhead \
  +video.scenes=[scene_28_24] \
  +video.out_dir=/tmp/truckdrive_val_videos_cmp \
  $DATA_ARGS
```

Colour encoding is **hue = model, shade = sample within model**: set A solid in
shades of blue, set B dashed in one orange, GT future red, GT history grey. The
best-of-K draw in each set is heavier and is the one named in the legend. Both
minADEs appear in the BEV title. Only one sample per model is projected onto the
camera tiles (all six overlap into a smear there); the full fan stays in BEV.

Note the CoT strip comes from set A only — Stage-2 action-head predictions carry
no generated text. If B's eval is still running, snapshot its `predictions.pt`
first so the render is not a moving target.
