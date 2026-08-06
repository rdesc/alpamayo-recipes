# Navigation-command conditioning for TruckDrive

NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.

Goal: give the model a coarse driving command (`straight` / `left` / `right`,
possibly a few more classes) as *input* conditioning, so the trajectory decoder
is no longer forced to guess intent from pixels + pose history alone.

Motivation: per `truckdrive-val-failure-profile`, val minADE is dominated by 8
low-speed turning scenes, and the observed loop failures on `scene_28_24`
correlate with the CoT asserting the *wrong turn direction*. Direction is
genuinely ambiguous from a single forward-camera window in a yard; a nav command
removes that ambiguity from the part of the problem we don't care about
measuring.

## 1. What already exists

### TruckDrive devkit / dataset — nothing usable
Searched `/mnt/efs/users/rod/repos/TruckDrive` (devkit) for
`nav_command|navigation|driving_command|maneuver|turn_left|intent|route`:
**zero hits** outside two unrelated uses of the word "intentionally".

Per-scene modalities are `camera / lidar / radar / poses / calibration /
annotations / accumulated-gt-depth`. Annotations are 2D/3D boxes, tracks, and
lane lines (rendered by `dataset_viewer/entrypoint.py`) — object-level, no map,
no route, no ego-intent labels. `generate_training_data/` only builds
sync-info and mmdet3d box annotations. The paper framing is
perception/long-range highway, not navigation-conditioned planning.

**Conclusion: we must derive the labels ourselves, from `poses/gt_trajectory.txt`.**
There is no lane-graph or route to project against, so the labels will be
*behavioral* (what the ego actually did), not *navigational* (what a router
would have said). This distinction matters — see §4.

### The model side is already plumbed
Good news: nav conditioning is a first-class component in this stack and needs
no model surgery.

- `alpamayo_r1.models.base_model.SPECIAL_TOKENS_KEYS` includes
  `route_start`, `route_pad`, `route_end` — these tokens exist in the released
  Alpamayo-1.5 checkpoint's vocabulary.
- `src/alpamayo/chat_template/components.py:244` `construct_route()` wraps
  `data["nav_text"]` in `<|route_start|>…<|route_end|>`, and **returns `[]` when
  `nav_text` is absent** — so adding `route` to `components_order` is backward
  compatible with unlabeled samples.
- `src/alpamayo/chat_template/r1_5.py:51` dispatches the `route` component
  (r1_5 template is what `sft_base.yaml` selects).
- `src/alpamayo/data/pai_nav.py` is the reference pattern: a plain dataset that
  merges a **sideband JSON** of `{clip_id, t0_relative, nav_text}` into the
  sample dict. Its `nav_text` examples are free-form (`"Turn left in 11m"`).

So the work is entirely: (a) generate labels, (b) attach them in the
TruckDrive loader, (c) add `route` to `components_order`.

## 2. Label-generation plan

### Geometry
The loader already computes exactly the quantity we need, in the right frame:
`TruckDriveDataset._ego_traj()` (`src/alpamayo/data/truckdrive.py:1155`)
returns `fut_xyz`/`fut_rot` **in the ego frame at t0**. Defaults are
`num_future_steps=64`, `time_step=0.1` → a 6.4 s horizon, `t0_stride=10` → one
window per second.

Per window, compute from `fut_xyz` (and `fut_rot` for the yaw source):

- `heading_change` = yaw at the last future step minus yaw at t0 (signed, deg).
  Since the trajectory is already t0-framed, this is just
  `atan2(R[-1][1,0], R[-1][0,0])`.
- `lateral_offset` = `fut_xyz[-1, 1]` (signed, m) — final lateral displacement.
- `path_length` = cumulative `‖Δxy‖` (m), and mean speed = `path_length / 6.4`.

### Classifier — threshold on CURVATURE, not net heading
The first cut used a net-heading threshold (straight < 8°, turn > 12°). Measured
on all 38,927 windows, that was **wrong**: 38% of the resulting `left`/`right`
labels were highway windows with a median 15.7° heading change — i.e. ~170 m of
travel through a 15.7° arc, a **620 m-radius bend**. That is lane-following, not
a turn, and it made the turn classes mean two unrelated things.

The fix is to threshold on curvature, expressed as equivalent turn radius
`R = path_length / |heading_change|` (radians), which is directly interpretable:

| geometry | R |
|---|---|
| yard / intersection turn | 15–30 m |
| freeway ramp | 50–150 m |
| highway alignment curve | > 500 m |

Rule as implemented in `truckdrive/nav_labels.py`:

- **turn** if `R <= turn_radius_m` (200) **and** `|heading| >= min_turn_deg` (10) —
  both, so neither a tight-but-tiny wiggle nor a long gentle drift qualifies;
- **or** the same holds for the tightest 1.5 s sub-window anywhere in the horizon
  (`min_radius_m`), which catches turn *onset* near the end of the window that
  the whole-window average dilutes into "straight";
- **and in both cases** the bend must be physically possible:
  `v^2 / R <= max_lat_acc_mps2` (3.0) — see below;
- **straight** if nothing anywhere in the window bends appreciably
  (`R >= straight_radius_m` = 300);
- **unlabeled** in between (the dead band), and for gate-rejected windows.

### The lateral-acceleration gate (do not remove)
The sub-window clause is short (1.5 s) and therefore noise-sensitive, and pose
yaw jitter at highway speed manufactures a spurious tight radius. Ungated, it
recovered 174 "highway turns" at a **median 7.26 m/s² implied lateral
acceleration, 94 % of them above 3 m/s²** — a loaded semi rolls over near
3-4 m/s², so those were all fake. The control that made this obvious: the
whole-window turns over the same data sit at a median **0.95 m/s²** with 0.5 %
above 3.

`min_radius_speed_mps` is recorded over the same stretch as `min_radius_m` so
the check uses matched speed rather than the window average. Gate-rejected
windows become **unlabeled, not `straight`** — if the geometry is untrustworthy
we should not assert either way.

After gating, the turn set sits at median 1.25 m/s², p95 2.70, max 4.85. The
1.6 % above 3.0 are windows admitted by the *whole-window* clause (gated on mean
speed against whole-window radius) that happen to contain a tighter sub-bend;
the two clauses gate on their own matched (v, R) pairs, so this is expected
rather than a leak.

Guard rails:
- The **dead band** avoids coin-flip labels at the boundary. Unlabeled windows
  carry no `nav_text`, and `construct_route` no-ops on absent nav text, so they
  train unconditioned rather than on a guess.
- Very low speed (`path_length < 2 m`) makes the position tangent unstable →
  unlabeled. Sub-windows with < 1 m of travel are skipped for the same reason.
- Sub-windows rather than single steps, because per-step yaw differences are
  noise-dominated at low speed.
- `filter_reverse=true` is already on in the config, so reversing windows are
  out of the index; no `reverse` class needed for now.

### Measured label distribution (all 38,927 windows, 3,192 scenes)

Final labels: `/mnt/efs/users/rod/truckdrive_cache/nav_labels_all_v3.json`.

| class | heading-based (rejected) | whole-window curvature | **+ sub-window, gated (final)** |
|---|---|---|---|
| straight | 32,753 (84.1%) | 36,124 (92.8%) | **35,015 (90.0%)** |
| left | 1,430 (3.7%) | 849 (2.2%) | **1,184 (3.0%)** |
| right | 1,581 (4.1%) | 650 (1.7%) | **1,059 (2.7%)** |
| unlabeled | 3,163 (8.1%) | 1,304 (3.3%) | **1,669 (4.3%)** |

Going from heading to curvature dropped highway contamination of the turn class
from 38.0 % to 2.2 %. Adding the gated sub-window clause then recovered **751
turn-onset windows** the whole-window average had diluted away (median net
heading 7.5°, median sub-radius 57 m, median speed 7.3 m/s — real bends starting
late in the horizon). The gate removed 180 implausible windows to unlabeled, of
which the highway recoveries collapsed 174 → 9.

Sign convention verified on `scene_28_24` (known left yard turn, 109.6° mean):
16 `left` + 2 `right`, the latter as the truck counter-steers out at t0 ≥ 18 s.
`scene_35_1` (highway) is `straight` on every window. **No `flip_lateral`
correction needed: +heading = left.**

Turns are 5.7 % of windows and ~9.6 % of scenes contain one; 37 of 40 chunks
have turns, so they are spread across the dataset rather than confined to the
yard chunks.

### Optional v2 classes (decide after the histogram)
Highway data means class balance will be brutal — likely > 90 % `straight`.
Two directions to consider:
- **Split `straight`** into `follow lane` vs `lane change left/right` using
  `lateral_offset` with small `heading_change` (a lane change is ≈ 3.7 m lateral,
  ~0° net heading). This makes the command informative on the highway majority
  instead of a constant, and lane-line annotations exist if we ever want to
  verify against actual lane boundaries.
- **Add a magnitude/distance qualifier** matching the pai_nav free-form style
  (`"Turn left in 20m"`), computed as arc-length to the point where cumulative
  heading change crosses the threshold. Richer, but harder to evaluate and
  harder to supply at inference. Recommend deferring; ship categorical first.

Prompt strings: keep them short and natural-language, matching the pai_nav
style, e.g. `"Continue straight"` / `"Turn left"` / `"Turn right"`. Avoid bare
enum tokens — the route content is plain text through the tokenizer, so
in-distribution English is likelier to land on useful pretrained embeddings.

## 3. Implementation steps

1. **`recipes/alpamayo1_5_sft/truckdrive/nav_labels.py`** (new). Reuses
   `TruckDriveDataset`'s index + `_ego_traj` (pose-only, **no image loads** —
   same trick as the `tok_recon` pre-filter, so it runs over all 3.8 k scenes
   cheaply) and writes a sideband JSON:
   `[{"scene_id": ..., "t0_us": ..., "nav_text": ..., "heading_change_deg": ...,
   "lateral_offset_m": ..., "path_length_m": ...}, …]`.
   Keep the raw geometry in the file so thresholds can be re-cut without a
   recompute. Point it at the EFS metadata cache
   (`/mnt/efs/users/rod/truckdrive_cache/meta_5views.pkl`) to skip S3 init.
2. **Analysis pass** before wiring anything into training: histogram of
   `heading_change_deg`, class counts overall and per split, and per-chunk
   (expect chunks 41/42/28 to hold nearly all turns, 35/36 nearly none).
   Confirm the threshold and check the val turning scenes get the right sign.
3. **Sanity-check the sign convention.** Poses are FLU, so +y is left and
   positive yaw is left — but `flip_lateral` exists in the loader for a reason.
   Verify on `scene_28_24 @ t0=2.0 s`, which is *known* from the failure profile
   to turn **left** (109.6° mean |turn|). If that window comes out `right`, the
   sign is flipped. This is the single highest-value check in the whole plan.
4. **Loader hook**: `TruckDriveDataset(nav_labels=<json path>)`; in
   `__getitem__` set `sample_data["nav_text"]` when `(scene_id, t0_us)` is in
   the map. Absent → key omitted → `construct_route` no-ops. Mirrors
   `pai_nav.py:113`.
5. **Config**: new `configs/sft_truckdrive_nav.yaml` inheriting
   `sft_truckdrive`, overriding `vla_preprocess_args.components_order` to
   `["image", "traj_history", "route", "prompt", "traj_future"]` (route before
   prompt, so the instruction is in context when the ask is made — confirm
   against how PAI-nav orders it). `label_components` unchanged
   (`["traj_future"]`) — the command is **input only**, never supervised.
6. **Eval**: at inference the command has to come from somewhere. For offline
   ADE scoring, feed the GT-derived command (this is the standard
   nuScenes/CARLA protocol) and report it as *command-conditioned* ADE — it is
   not comparable to the existing unconditioned numbers, so keep both.
   Recompute per-scene ADE on the 8 turning scenes; that's where the win, if
   any, has to show up.
7. **Dropout for robustness**: train with the route randomly dropped (~10–20 %
   of samples) so the model does not become unusable without a command and so we
   keep an unconditioned comparison path in the same checkpoint.

## 4. Known caveats

- **The label is derived from the same future we regress.** A GT-derived command
  leaks the answer's coarse direction. That is the intended semantics of nav
  conditioning, but it means ADE improvements are *not* evidence of better
  driving — only of better use of a signal a real deployment would get from a
  router. Say this explicitly in any results write-up.
- **Behavioral ≠ navigational.** Without a map, "turn left" here means "the ego
  did in fact turn left", including lane-follow curvature on a highway ramp that
  no router would call a turn. Horizon-length choice (6.4 s) makes gentle
  highway curvature a real contaminant of the `left`/`right` classes; the
  hysteresis band is the mitigation.
- **Class imbalance is real.** Turns are 5.7 % of windows (2,243) and ~9.6 % of
  scenes contain any turn at all. Concentration is fine (37 of 40 chunks have
  turns), but at that rate **turn oversampling in the train sampler is expected
  to be necessary**. That is a config change, not a label change.
- **Two plausible-looking rules in a row produced physically nonsensical labels
  that the class counts did not reveal** (heading-based highway contamination;
  then the ungated sub-window clause). Both were caught only by checking a
  *physical* quantity — turn radius, then lateral acceleration — never by label
  balance. Before training on these labels, render a sample of turn-labeled
  windows against the camera frames (`truckdrive/render_scene_video.py`) as a
  visual check that the label matches what the truck is doing.
- **Route embeddings may be untrained.** The `route_*` tokens exist in the
  checkpoint vocab, but whether Alpamayo-1.5 was pretrained with meaningful
  route conditioning is unverified. If the embeddings are effectively random,
  the model must learn the mapping from our fine-tune data alone — feasible with
  3 classes, but a reason to prefer plain-English content strings and a reason
  not to expect immediate gains at low LR / LoRA rank.

## 5. Suggested first slice

Steps 1–3 only (generate + analyze + sign check). They're pose-only, cost
minutes, and the histogram decides whether the class scheme in §2 survives
contact with the data before any training config is written.
