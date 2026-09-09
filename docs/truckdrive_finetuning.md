# TruckDrive fine-tuning — working notes & TODO

Fine-tuning Alpamayo-1.5 on the TruckDrivePublic dataset
(`s3://torc-data/datasets/TruckDrivePublic/`). This doc tracks decisions,
open tasks, and tabled ideas so context isn't lost between sessions.

## Status

- **Loader**: `src/alpamayo/data/truckdrive.py` — reads camera + poses, emits the
  Alpamayo sample dict, derives ego heading from the position track. Verified to
  tokenize through the **real** checkpoint tokenizer at ~cm (XY) for forward
  driving.
- **Integration tooling**: `recipes/alpamayo1_5_sft/truckdrive/`
  - `validate.py` — real tokenizer via `--checkpoint`, XY-only reconstruction
    metric, inputs + GT/recon viz (`python -m truckdrive.validate ...`).
  - `projection.py` — reusable: load `K` + TF extrinsics, project an ego
    trajectory onto a camera image (FLU→FRU handled).
  - `render.py` — scene inspection renders: `projection`, `allviews`,
    `filmstrip` (`python -m truckdrive.render <cmd> --scene ...`).
- **Reverse filtering**: on by default in the loader (`filter_reverse`). Windows
  whose future travels backward relative to body orientation are dropped at
  index-build time. Verified: an all-reverse scene → 0 samples, a forward scene
  → all kept.
- Nothing committed yet; all changes are in the working tree.

## Key decisions & findings (durable)

- **First target = unconditioned trajectory prediction** (`[image, traj_history,
  traj_future]`). It's the only task fully supported by the data we have.
- **Real future tokenizer** = `DiscreteTrajectoryTokenizer` +
  `UnicycleAccelCurvatureActionSpace` (accel±9.8 m/s², curvature±0.33 /m, 3000
  bins, 128 tokens). It's the paper's scheme. History uses the simpler
  `DeltaTrajectoryTokenizer`. Always validate with `--checkpoint` (the class
  defaults are a different tokenizer).
- The unicycle tokenizer is a **2-D ground-plane path model**: it encodes (x, y)
  + travel heading; **z/elevation is not modeled** → measure reconstruction in
  XY only.
- **Heading must come from the position tangent, not the pose quaternions.** The
  gt_trajectory quaternions encode *body orientation*, which only equals travel
  heading during forward driving. Position-tangent → ~cm; quaternions → 0.5–9 m.
  Low-speed windows need a distance-adaptive (arc-length) tangent; per-step is
  noisy near standstill.
- **Skip reverse / low-speed maneuvers.** They aren't well-posed (image→future)
  examples for a forward-camera model (body ≠ travel direction), the unicycle
  model can't represent the sideslip, and there's no nav signal for them.
- **No language annotations in TruckDrive.** Only 3-D detection/tracking boxes
  (class labels + geometry), poses, images, lidar/radar/depth. No nav text, CoT,
  VQA, or captions.
- **SDPA, not Flash-Attention, on the B300 (Blackwell) box.** The venv's
  flash-attn 2.8.3 wheel has no sm_100 kernels ("no kernel image is available"),
  so training there runs with `model.attn_implementation=sdpa` (torch/cuDNN
  fused attention — near-FA2 speed on Blackwell, and attention isn't the
  bottleneck for this workload anyway). The checkpoint default
  (flash_attention_2) still applies on A100 boxes. Revisit if/when a
  Blackwell-capable flash-attn build is installed (would need an sm_80+sm_100
  fat build or a per-box venv, since the venv is shared on EFS).
- **NVRTC must be ≥ 12.9 on B300 — a *second, independent* Blackwell gap**
  (fixed 2026-08-26). `torch==2.8.0+cu128` bundles NVRTC **12.8**, which
  supports `sm_100`/`sm_101`/`sm_120` but **not `sm_103`** — and B300 is
  exactly `sm_103` (`torch.cuda.get_device_capability()` → `(10, 3)`). Every
  op needing a runtime **JIT-compiled** kernel then dies with
  `nvrtc: error: invalid value for --gpu-architecture (-arch)`. One-line
  repro on any new box:
  `python -c "import torch; print(torch.arange(1,5,device='cuda').prod())"`.
  Fix: `uv pip install --python <venv>/bin/python "nvidia-cuda-nvrtc-cu12==12.9.86" --no-deps`
  (12.9's arch list is a strict superset of 12.8's, so it can't regress other
  GPUs). Done for `a1_5_sft_b300`.
  Two traps worth knowing: (1) most kernels are **precompiled**, so a venv can
  import fine, use all 8 GPUs and run forward/backward while still being
  fatally broken — only jiterator ops (here a `.prod()` reduction on the
  trajectory-token **generation** path) fail, which is why it surfaces in the
  viz/`ValMinADE` callbacks; (2) those callbacks log
  `skipped at step 0: RuntimeError(...)` and look survivable, but the error
  escapes and kills the whole job via `ChildFailedError`. Fixing SDPA does not
  fix this and vice-versa — they are unrelated.
- **Venv fixes do NOT propagate between `a1_5_sft` and `a1_5_sft_b300`.** They
  are independent installs on EFS. `a1_5_sft_b300` was still on the buggy
  **wandb 0.26.0** long after `a1_5_sft` had been upgraded to 0.28.2; it
  crashes rank 0 in `wandb.init()` (`json.loads(None)` in `_get_username`),
  which then shows up as misleading `NCCL error / ncclRemoteError` failures on
  the other ranks. Always check the venv you're actually launching with.

## Technical notes (mechanism — the non-obvious stuff)

### Trajectory tokenizer internals
- **Two tokenizers by design** (observe vs. generate): history is a faithful
  *input* → simple delta-xyz (3-D, keeps z); future is a *sampled output* →
  unicycle accel/curvature, which guarantees any sampled token sequence decodes
  to a smooth, feasible path and is ~speed-invariant.
- **The unicycle state's yaw = travel (velocity) heading, not body orientation.**
  The math: `Δp_t = (dt/2)(v_t·u_t + v_{t+1}·u_{t+1})` with `u=(cosθ,sinθ)` and a
  *scalar* speed `v`. So motion is constrained **collinear** with θ — no
  perpendicular/sideslip term. θ must be the direction the position moves.
- **Negative `v` is allowed** → pure straight reverse reconstructs (~3 mm). What
  breaks is **non-collinearity** (sideslip/turning-while-reversing): the
  perpendicular component is unrepresentable for any sign of `v`. Also
  `κ = dθ/(v·dt)` is ill-conditioned as `|v|→0`.
- **Reconstruction floor** is ~1–6 cm (XY) for clean forward driving; that's the
  ceiling on model trajectory accuracy, and it's negligible.
- The earlier "±4 m per-step clipping" concern was an **artifact of the wrong
  (delta) tokenizer**; the real accel/curvature bounds are speed-invariant, so
  highway speed is fine.

### Frame conventions
- TruckDrive `gt_trajectory` poses are **FLU** (x-forward, y-left, z-up), same as
  Alpamayo — no lateral flip needed for the model. `flip_lateral` in the loader
  is an off-by-default escape hatch for a future y-right dataset.
- The unicycle decoder assumes **θ = 0 at t0**, i.e. the trajectory is in the ego
  frame at t0 (loader enforces this by zeroing t0 heading).

### Camera calibration & trajectory projection
- We **have intrinsics + extrinsics**: per-camera pinhole `K` (+ `D`, all zeros,
  `plumb_bob`) in `calibrations/calib_camera_*.json`; extrinsics via the TF tree
  `calib_tf_tree_full.json` (`vehicle → cab → camera`). So we can project the
  trajectory onto any camera image (pinhole, **not** the f-theta model in
  Alpamayo's `viz.py`).
- **Gotcha (handled in `projection.py`)**: the rig's TF *vehicle* frame is
  y-right (FRU), not ROS-FLU — a naive pinhole projector comes out mirrored in y.
  `projection.py` negates y (FLU→FRU) before applying the extrinsics; the
  loader/poses stay correct FLU. Verified against the path's frame-independent
  turn direction (overlay tracks lanes + painted turn arrows).

### Dataset shape & size
- 3829 scenes, ~16 s each. Per scene ~11.6 GB all-in; **lidar is ~80%**.
- Camera + poses (+ labels) ≈ 1.8 GB/scene → ~7 TB for the full RGB subset
  (44 TB all-in). Too big to stage whole → stage a scene subset (camera+poses).
- Cameras ~5 Hz (80 frames/scene, 15 views); poses ~10 Hz (~160/scene). Files
  share a `SYNC_KEY` prefix; join camera frames to the pose timeline by sync key.

### Official devkit split (adopted)

- The TruckDrive devkit's `metainfo.json` (vendored at
  `recipes/alpamayo1_5_sft/truckdrive/truckdrive_metainfo.json`) defines the
  official split: 141 val / 167 test scenes held out as **whole collection
  batches** (scene groups = drives), 3,521 train. Use it via
  `split_metainfo=...` on the dataset (disables the random val_fraction split).
- The three training lists (sequentially / non-sequentially / un-labelled)
  differ only in 3-D **box** annotation style — all trainable for trajectory.
- Measured with the 5-view config: train 38,561 windows / 3,463 scenes,
  val 2,474 / 139, test 3,000 / 167 (never trained or tuned on).
- `scene_<G>_<N>` decodes as collection **batch G**, scene N (internal naming:
  `su-input-batch-<G>-<x>-<y>_scene_id_<N>`).

### gt_trajectory provenance & the two pose-file formats (from the data owner)

- `poses/gt_trajectory.txt` is the **best-selected trajectory from multiple lidar
  SLAM approaches** (GPS alone isn't label-grade). SLAM drifts; for drifty scenes
  the owner re-optimized poses using **static objects (traffic signs) as anchors**
  and overwrote the file in place (bucket keeps `__pose_overwrite_backups__/`).
- Two formats in the public export (verified across all 3,769 scenes; no others):
  - **refined (2,681 scenes)**: `# SYNC_KEY, TIMESTAMP, ...` header + a
    `refined_gt_trajectory ...` METHOD_NAME line (in internal files this line
    carries a global anchor pose; the public export replaces the 7 globals with
    column labels). ~16 s clips.
  - **raw (1,088 scenes)**: same schema, header **without** `#`, no method line.
    Never refined. **~26 s clips**, different scene groups (g17/18/22/26/27/31/32...).
- Data rows are identical in both: `SYNC_KEY t_sec x y z qx qy qz qw` (quat is
  scipy `[x,y,z,w]` order — confirmed against the owner's own loader).
- **Raw scenes are training-grade**: measured on 400-scene samples per format,
  raw is actually *smoother* at window scale (p95 |acc| 1.23 vs 1.59 m/s²,
  p95 |jerk| 10.1 vs 13.1 m/s³) — refinement fixes *global drift*, which cancels
  in our 8 s ego-relative windows, at the cost of small local anchor-correction
  wiggle. Both ~100% forward-window rate. The loader's per-line parser handles
  both formats (the bare header once aborted whole scenes: `float('TIMESTAMP,')`).
- Caveat: raw scenes have worse *global* accuracy — irrelevant for trajectory
  training, relevant if ever projecting labels across long horizons/scenes.

### Reverse diagnostic (scene_10_1)
- Confirmed reversing: x goes 0 → −48 m while body points +x;
  `angle(velocity, body-forward) > 90°` for 100% of moving samples (≈114°).
  Use that angle as the filter/diagnostic signal. The ~114° (not 180°) may be
  amplified by the pose reference point / articulation — see TODO #3.

## TABLED — synthesize nav commands from the trajectory

Deferred (not needed for the first unconditioned trajectory fine-tune). Revisit
if/when we want nav-conditioned training.

- **Idea**: TruckDrive has no nav/instruction text, so derive a discrete
  high-level command per window from the GT future trajectory and feed it via
  the r1.5 chat template `route` component.
- **Classes** (starting point): `{go straight, turn left, turn right,
  lane change left, lane change right}`, from net heading change + lateral offset
  over the future horizon.
- **Nice side effect**: reverse maneuvers don't fit any forward class → they get
  excluded for free, consistent with the reverse-skip decision.
- **Open questions**: thresholds for each class; class balance across scenes;
  whether to also use the 3-D boxes for object-aware context ("pedestrian
  ahead-left"); whether the pretrained model's `route` conditioning transfers.

## Open TODOs (next steps)

1. **Scene sweep** — reverse filtering is implemented (defaults:
   `reverse_angle_deg=90`, `max_reverse_fraction=0.2`, `min_speed_mps=0.5`).
   Still to do: run the real-tokenizer round-trip across many scenes to (a)
   quantify the clean-forward fraction vs. reverse/low-speed tail, (b) tune the
   filter thresholds, and (c) tune the distance-adaptive heading baseline.
2. **Wire into training** — add `configs/data/truckdrive.yaml` + a
   `vla_processor` variant so `train_hf.py` can instantiate the dataset
   (mirror the PAI/LingoQA configs). This turns the loader into a runnable
   fine-tune.
3. **Pose reference point** — confirm which point `gt_trajectory` is logged at
   (rear axle vs. sensor rig vs. tractor/trailer). A large body-vs-tangent gap in
   maneuvers points at an offset reference point; may want to re-express the
   trajectory at the rear axle. (Ties to reverse feasibility if ever revisited.)
4. **Camera→Alpamayo-slot mapping** — the 15-view→7-slot mapping in
   `DEFAULT_VIEW_TO_ALPAMAYO` is a heuristic; verify before multi-view training.
5. **Data access** — training runs as `rod`; the S3 FUSE mount is owned by
   `ec2-user` without `--allow-other`. Either remount with `--allow-other`, use
   `s3torchconnector`, or stage a scene subset to local NVMe (camera+poses only
   ≈ tens of GB for a few hundred scenes).
6. ~~Regenerate validation gallery with real-tokenizer XY numbers~~ — done
   (scene_10_5: projection overlay, all-views, filmstrip, 2.1 cm XY round-trip).

## Tabled / out of scope

- **Reverse-view fine-tuning** (rear/side cameras + reverse trajectories):
  interesting future capability, but requires resolving the pose-reference-point
  issue, handling negative-velocity distribution shift from the forward-only
  pretrained model, and building a reverse nav signal. Not now.
- lidar / radar / depth modalities (RGB only for this work).
