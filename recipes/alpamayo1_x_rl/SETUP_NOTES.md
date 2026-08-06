# RL recipe setup notes — gaps not covered by SKILL.md / README.md

NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.

Everything below was hit and verified while bringing up the motion-only
(`alpamayo_rvla_rl_local_test.toml`) local run on 2026-07-30, on an 8x A100-80GB
box. It is *not* a replacement for [`SKILL.md`](SKILL.md) — that remains
authoritative for the pipeline. This records the things `SKILL.md` and
`README.md` do not mention, ordered by how much time they cost.

Contents:
1. [Hard prerequisites that are undocumented](#1-hard-prerequisites-that-are-undocumented)
2. [Dataset gotchas](#2-dataset-gotchas)
3. [Benign startup warnings (do not chase these)](#3-benign-startup-warnings-do-not-chase-these)
4. [There is no video logging on the reasoning_vla path](#4-there-is-no-video-logging-on-the-reasoning_vla-path)
5. [Reward reference (motion)](#5-reward-reference-motion)
6. [Run-length arithmetic](#6-run-length-arithmetic)
7. [Checkpoint conversion caveats](#7-checkpoint-conversion-caveats)

---

## 1. Hard prerequisites that are undocumented

Both fail roughly **80 s into launch**, after model imports and W&B init, which
is what makes them irritating rather than trivial.

### 1a. A `redis-server` **binary** on `PATH` (fatal)

The controller uses Redis as its rollout dispatch queue. The TOML's
`redis = "12800"` is only the *port* — nothing in the docs says a server binary
is required, and the Python `redis` client (installed as a transitive dep) is
**not** sufficient.

```text
/bin/sh: line 1: redis-server: command not found
RuntimeError: Failed to start redis server with command: redis-server ... with return code 127
```

Grep confirms `redis` appears in **no** `.md` or `.sh` file in this repo.

Rootless install (no sudo on this box; `gcc`/`make` are present):

```bash
curl -sSLO https://download.redis.io/releases/redis-7.2.5.tar.gz
tar xzf redis-7.2.5.tar.gz && cd redis-7.2.5
make MALLOC=libc -j8
mkdir -p ~/.local/bin && cp src/redis-server src/redis-cli ~/.local/bin/
redis-server --version   # verify it is on PATH
```

Verify end to end with `redis-server --port 12801 --daemonize yes --save ''`
then `redis-cli -p 12801 ping` → `PONG`.

Notes:
- Amazon Linux's `sudo dnf install -y redis6` also works, but installs the
  binary as **`redis6-server`**; Cosmos-RL invokes `redis-server` literally, so
  that route needs a symlink.
- `~/.local/bin` works because `launch_controller.sh` inherits the parent
  environment. A SLURM/systemd launch that does not source the profile will
  lose it — put redis somewhere on the system `PATH` before scaling to cluster.
- Redis is started with `--save ""`, so its `vm.overcommit_memory` warning
  (about background saves) does not apply.

### 1b. `WANDB_ENTITY` (silent failure — worse)

If the W&B account has no default entity, init fails with a **400**:

```text
wandb: Currently logged in as: <user>
ERROR - Failed to initialize wandb: ... 400: {"errors":[{"message":"entityName required for model query (user has no default)"}]}
```

Cosmos-RL **catches this and continues**, so training runs to completion with
W&B silently dead. Fix:

```bash
export WANDB_ENTITY=<your-entity>
```

`SKILL.md`'s failure table covers only the *403* `upsertBucket permission
denied` case; `WANDB_ENTITY` appears nowhere in it.

---

## 2. Dataset gotchas

### 2a. `include_extr_intr: true` requires `vehicle_dimensions`

`hydra_configs/alpamayo1_5_rvla_rl_pai.yaml` sets `include_extr_intr: true`.
That single flag gates **three** calibration features in
`src/alpamayo/data/pai.py:124-134` — `sensor_extrinsics`, `camera_intrinsics`,
**and** `vehicle_dimensions` (used to build `ego_lwh` and `ego_length_offset`).

A dataset downloaded without `--calibration vehicle_dimensions` (e.g. one
curated for the Stage-1 nav SFT recipe, which omits it) throws on the first
`__getitem__`. It is a tiny download — no footage:

```bash
python scripts/download_pai.py \
  --chunk-ids "<your chunk ids>" \
  --calibration vehicle_dimensions \
  --output-dir "$ALPAMAYO_PAI_LOCAL_DIR"
```

Worth knowing: **no shipped reward reads those two tensors.** `ego_lwh`,
`ego_length_offset`, `egomotion_road_boundaries`, `egomotion_lanelines` and
`obstacle_bbox_*` are all packed into `get_reference_answer()`
(`base_dataset.py:71-79`) but unused by both aggregators — there is no collision
or off-road term. So this failure is pure plumbing, and the fields are the
hook if you want a geometric safety reward later.

### 2b. `curate_pai_samples.py` does **not** filter on `split`

The script filters on `--chunk` only (`chunk_subset = clip_index`, then chunk
filtering, then `.sample(random_state=11)`). It does not look at `split` or
`clip_is_valid`. `PAIDataset` has **no split handling either** — grep for
`split` in `src/alpamayo/data/pai.py` returns nothing, so the TOML's
`[train.train_policy] dataset.split = "train"` does not protect you.

Result: a curated mini index silently mixes train/val/test clips. A 16-sample
draw on one local subset came out 9 train / 6 val / 1 test.

Harmless for a wiring smoke test, but it is test-set contamination for any run
you intend to compare against held-out numbers. Build the index explicitly:

```python
import pandas as pd
df = pd.read_parquet("clip_index.parquet")
sub = df[(df["chunk"].isin(CHUNKS_WITH_FOOTAGE))
         & (df["split"] == "train")
         & (df["clip_is_valid"] == True)]
sub.sample(frac=1.0, random_state=11).to_parquet("clip_index_mini.parquet")
```

Then verify every sampled chunk actually has camera zips on disk — curation
reads the *full* index, which covers chunks you may not have downloaded.

Related: in a 19-chunk nav subset, only 9 chunks contained `split == "train"`
clips at all (~100 each); the rest were val/test. Check before assuming a
chunk count implies training data.

---

## 3. Benign startup warnings (do not chase these)

All of these print on every launch of this recipe and none touch the training
path. **Do not "fix" them by moving pinned versions** — `torch==2.8.0`,
`vllm==0.11.0` and `flash-attn` are ABI-coupled, which is why the install is a
two-pass `uv sync`.

| Warning | Why it is fine |
|---|---|
| `xformers is not compatible with Cosmos-RL now, please uninstall it` | Cosmos-RL **already fixed it** — `cosmos_rl/utils/diffusers_utils.py:6-19` monkeypatches `importlib.util.find_spec` to hide xformers, then logs the warning. Do **not** uninstall: xformers is a hard declared dep of `vllm==0.11.0` (`xformers==0.0.32.post1` on linux/x86_64), so removing it breaks vllm and `uv sync` reinstalls it anyway. |
| `Skipping import of cpp extensions due to incompatible torch version 2.8.0+cu128 for torchao version 0.15.0` | `torchao` is a transitive dep of `cosmos-rl`. Its prebuilt CUDA extensions do not match torch 2.8.0, so it falls back to pure Python. Irrelevant with `[rollout].quantization = "none"` and `param_dtype = "bfloat16"`; the recipe's only fp8 code uses native `torch.scaled_mm`. (A100s are SM80 and have no fp8 tensor cores regardless.) |
| `grouped_gemm is not available` | MoE-only kernel. |
| `transformer_engine.pytorch is not available. DeepSeek model will not work.` | DeepSeek-only. |
| `imageio is not installed, video logging will not work` | Red herring — see [section 4](#4-there-is-no-video-logging-on-the-reasoning_vla-path). |
| `The pynvml package is deprecated` | Emitted by torch itself. |

Cold `import vllm` takes ~2 minutes. A 2-minute timeout on an import smoke test
will look like a hang when it is not.

---

## 4. There is no video logging on the reasoning_vla path

Installing `imageio` will **not** produce rollout videos here. Its consumers in
`cosmos_rl` are all other pipelines:

- `policy/trainer/diffusers_trainer/nft_trainer.py` — diffusion/NFT trainer
- `simulators/utils.py` — **closed-loop simulator** rollout videos
- `tools/dataset/wfm/…`, `utils/wfm/io/…` — world-foundation-model tooling

This run is `trainer_type = "reasoning_vla_grpo"` +
`backend = "reasoning_vla_vllm_rollout"`, and the recipe's own `models/`,
`rewards/` and `launcher.py` contain **zero** references to `imageio`,
`wandb.Video`, `mimsave` or `.mp4`, and never import `simulators` or
`diffusers`.

That follows from what this pipeline is: **open-loop**. A "rollout" is a decoded
trajectory scored against ground truth — there is no simulated episode to
record. W&B gets scalars only.

To get videos, render offline from an exported checkpoint:
`convert_cosmos_rl_checkpoint.py` → then
[`notebooks/rl_checkpoint_inference.ipynb`](notebooks/rl_checkpoint_inference.ipynb).
A base-vs-RL-step comparison on the same clips is the informative version — but
that needs early checkpoints, so raise `[train.ckpt].max_keep` (default `10`,
with `save_freq = 50`) *before* launching a long run, or they get pruned.

---

## 5. Reward reference (motion)

`SKILL.md` names the weights but not the formula.
`rewards/aggregated_reward.py`:

```text
if ADE < 3.0:  reward = -traj_l2_weight * (ADE / 3.0) + comfort_weight * (comfort - 1)
else:          reward = -1.0
```

With the shipped `traj_l2_weight = 0.4`, `comfort_weight = 0.1`, reward lands in
**[-1, 0]** (0 is best), consistent with the doc's expected `-0.28 -> -0.21`.

- `traj_L2` is ADE over **XY only** — `traj_reward.py` slices `[..., :2]`, Z is
  ignored.
- `comfort_reward` (`comfort_reward.py`) finite-differences the predicted
  trajectory at 10 Hz and checks 6 quantities (lon/lat accel, jerk, lon jerk,
  yaw accel, yaw rate) against fixed thresholds. Each check is **binary and
  all-or-nothing across the whole horizon** (`torch.all` over time), averaged
  over the 6, then shifted by -1 → `[-1, 0]`. So it takes only 7 distinct
  values and, at weight 0.1 against a continuous ADE term at 0.4, acts as a
  tiebreak. Do not over-read its movement.

**The ADE gate is a GRPO group-collapse hazard.** Every rollout with
`ADE >= 3.0` scores *exactly* `-1.0`. If all `n_generation` completions in a
group are beyond the gate, within-group reward variance is zero and that group
contributes no learning signal. Symptom: reward pinned at `-1`. The base
Alpamayo 1.5 sits near 1.66 m ADE on PAI so this is usually fine — but it is
the first thing to check on a fine-tune whose ADE is worse, or on a new dataset.

The joint reward (`aggregated_reward_with_reasoning.py`) adds a Lingo-Judge CoT
term and uses a stricter triple gate (CoT decoded **and**
`reasoning_score > -0.4` **and** `ADE < 3.0`). Its TOML also reweights heavily:
`traj_l2_weight = 0.2`, **`comfort_weight = 0.0`**, `reasoning_weight = 0.3`.

---

## 6. Run-length arithmetic

Not stated anywhere, and easy to get wrong by a factor of `n_generation`.

**`train_batch_per_replica` counts ROLLOUTS, not prompts.** Each prompt produces
`rollout.n_generation` completions, so:

```text
prompts_per_step = train_batch_per_replica / rollout.n_generation   (x n_policy_replicas)
steps_per_epoch  = num_curated_clips / prompts_per_step
total_steps      = steps_per_epoch * epoch
```

With the shipped `train_batch_per_replica = 48` and `n_generation = 12`, that is
only **4 prompts per step** — not 48.

Verified against a real run: 891 clips, `epoch = 50` → controller reported
`train/total_steps: 11137` (= 891 * 50 / 4, exact). Measured **22.4 s/step** on
1 policy replica (4x A100-80GB) + 1 rollout replica (1x A100), i.e. **~69 hours**
for that config. Budget from measured step time, not from epoch count:

| Clips | Steps/epoch | `epoch` for ~10 h at 22.4 s/step |
|---|---|---|
| 891 | ~223 | ~7 (≈1600 steps) |

Use `[train].max_num_steps` to bound wall clock without re-deriving `epoch`.
Note `[train].epoch` alone gives no hint of the real length: the shipped
`epoch = 15` with the doc's suggested `--num-samples 16` is only
16 * 15 / 4 = **60 steps**.

Also: `[custom.alpamayo].prefetch.capacity` ships at `16` while
`train_batch_per_replica` is `48`. The doc's own rule is
`capacity = train_batch_per_replica * replicas_per_node`, so the default is
undersized — and prefetch is cited as the single biggest lever against
`pending rollouts` growth (44 s -> 5 s per step). Fix it before a long run.

---

## 7. Checkpoint conversion caveats

`scripts/convert_release_config_to_training.py`:

- **Writes its ~17 files directly into `--output-dir`**, with no
  model-named subdirectory. Pointing it at an existing checkpoints *parent*
  dir litters that dir rather than creating a checkpoint inside it. Give it a
  fresh leaf path.
- **Weights are symlinks**, not copies — the output dir is a few hundred KB
  pointing into the HF cache. Consequences: clearing the HF cache dangles the
  "checkpoint", and a **multi-node** run needs the symlink targets (and ideally
  the dir itself) on shared storage, not node-local disk.
- Step 5 imports `alpamayo1_x_rl` and instantiates the model, so **the `a1x_rl`
  venv must exist first**. Provision order is venv → conversion.
- It is cheap and repeatable (no GPU, no training) — rerunning after a mistake
  costs nothing.
