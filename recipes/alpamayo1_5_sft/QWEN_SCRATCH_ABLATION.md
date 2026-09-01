# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
# Qwen-from-scratch pretraining ablation

**Goal:** measure how much of Alpamayo-1.5's TruckDrive performance comes from
NVIDIA's physical-AI VLM pretraining vs. the TruckDrive fine-tune itself, by
running the *same* Stage-1 TruckDrive SFT recipe but starting from a vanilla
HF `Qwen/Qwen3-VL-*-Instruct` checkpoint instead of the Alpamayo-1.5-pretrained
one. Stage 1 (discrete-trajectory-token VLM SFT) only -- no Stage-2 action
head, to keep the ablation simple.

**STATUS (2026-08-20): running and training on 8xA100.** The B300 hardware
wall is gone, the venv is repaired, and the run is past first loss and
learning. See "Current run" below. The rest of this doc is now a record of
what was fixed plus what the *next* person needs to watch.

> **UPDATE (2026-08-26): "the B300 hardware wall is gone" was FALSE.** It was
> never fixed -- the work simply moved to A100. The wall is real and it is
> `sm_103` vs. torch's bundled NVRTC 12.8; see
> "The B300 blocker: ACTUALLY fixed" near the end of this doc. It **is** now
> genuinely fixed (NVRTC 12.9.86 in `a1_5_sft_b300`), so B300 nodes are usable.
> A 3-epoch A100 run finished at `checkpoint-12357`
> (`output_truckdrive_qwen_scratch/`); a **5-epoch, batch-4 B300 run** is the
> current one -- see "Running on B300" below.

---

## Current run

- **Node:** `ip-10-225-141-207` (8x A100-SXM4-80GB, compute cap 8.0, system
  CUDA at `/usr/local/cuda`, 1121 GB host RAM).
- **Backbone:** `Qwen/Qwen3-VL-8B-Instruct`, deepspeed **on** (recipe default).
- **wandb:** https://wandb.ai/rdesc1-milaquebec/alpamayo/runs/fbutalhs
- **Output dir:** `output_truckdrive_qwen_scratch/`
- **Log:** `logs/qwen_scratch_20260819_234502.log`
- **Launch script (exact command used):**
  `/tmp/claude-1001/.../scratchpad/launch_qwen_scratch.sh` -- reproduced under
  "How to launch" below; copy it somewhere durable, the scratchpad is
  session-scoped.

Health at ~step 80:

| metric | value |
|---|---|
| step-0 `ValMinADE` (K=6, n=248) | mean **76.135 m**, median 87.416, p95 96.246 |
| step-0 `eval_loss` | 23.338 |
| first logged `loss` (step 5) | 22.638, `grad_norm` 732 |
| `loss` at step ~80 | **~4.9-5.6**, `grad_norm` ~18 |
| throughput | ~6-8 s/it |
| host RAM | ~291 GB / 1121 GB, stable |

The step-0 numbers are the *untrained-backbone baseline* and are exactly what
you want for this ablation: a from-scratch Qwen emits essentially random
trajectory tokens (76 m minADE). Step 0 reproduced bit-identically across two
separate launches, so it's deterministic and safe to quote. The fast
22.6 -> 5.0 loss drop in 80 steps is the model learning the trajectory-token
format, not yet driving skill -- judge the ablation on `ValMinADE`, not loss.

**Full run is 12357 steps (3 epochs) ≈ 24-27 h wall clock.** `save_steps=500`,
`eval_steps=500`.

---

## What's built

1. **`models/sft_base_model.py`** -- added
   `TrainableReasoningVLA.from_scratch_vlm()`. Unlike
   `from_alpamayo_checkpoint` (builds an empty VLM shell and overwrites it
   with Alpamayo's fine-tuned weights), this calls the upstream
   `ReasoningVLA.from_pretrained_submodules()` path, which loads **real
   vanilla HF Qwen weights** via `Qwen3VLForConditionalGeneration.from_pretrained(vlm_name_or_path, ...)`
   -- no Alpamayo pretraining anywhere in the loop. It reads the
   trajectory-tokenizer *codec* (a fixed kinematic binning scheme --
   `DiscreteTrajectoryTokenizer` / `DeltaTrajectoryTokenizer`, not a learned
   network) from an existing Alpamayo checkpoint's `config.json` so the label
   format matches the rest of the recipe; that checkpoint's actual weights
   are never loaded.

2. **`configs/models/ar1_5_base_qwen_scratch.yaml`** -- Hydra model config
   wiring `from_scratch_vlm`. `vlm_name_or_path` defaults to
   `Qwen/Qwen3-VL-8B-Instruct` and is freely overridable on the CLI for
   smaller/larger Qwen3-VL siblings, e.g.
   `model.vlm_name_or_path=Qwen/Qwen3-VL-2B-Instruct`.

3. **`configs/sft_truckdrive_qwen_scratch.yaml`** -- top-level training
   config. Identical to `sft_truckdrive.yaml` (same data/viz/val_metric/
   trainer wiring), just overrides `/models@model: ar1_5_base_qwen_scratch`.

**Verified correctness** (CPU, with `Qwen/Qwen3-VL-2B-Instruct`): instantiated
`from_scratch_vlm` and confirmed vocab size, every entry in
`special_token_ids`, and the trajectory tokenizer class all match byte-for-byte
what `from_alpamayo_checkpoint` produces from the real Alpamayo checkpoint. So
the TruckDrive dataset/collate/viz/val_metric code needs zero further changes
-- the scratch-Qwen model is a drop-in swap.

**Verified end-to-end on A100:** checkpoint load, token-embedding resize,
S3 dataset load, deepspeed init, step-0 eval + viz, and sustained
forward/backward with a decreasing loss.

---

## How to launch

```bash
cd recipes/alpamayo1_5_sft
source a1_5_sft/bin/activate
export CUDA_HOME=/usr/local/cuda
export WANDB_ENTITY=models
R=$PWD

torchrun --nproc_per_node 8 \
  -m alpamayo1_5_sft.train_hf \
  --config-path pkg://alpamayo1_5_sft/configs \
  --config-name sft_truckdrive_qwen_scratch \
  trainer.dataloader_num_workers=2 \
  data.train_dataset.data_root=s3://torc-data/datasets/TruckDrivePublic \
  data.train_dataset.backend=s3 \
  data.train_dataset.metadata_cache=/mnt/efs/users/rod/truckdrive_cache/meta_5views.pkl \
  data.train_dataset.split_metainfo=$R/truckdrive/truckdrive_metainfo.json \
  data.val_dataset.data_root=s3://torc-data/datasets/TruckDrivePublic \
  data.val_dataset.backend=s3 \
  data.val_dataset.metadata_cache=/mnt/efs/users/rod/truckdrive_cache/meta_5views.pkl \
  data.val_dataset.split_metainfo=$R/truckdrive/truckdrive_metainfo.json
```

`trainer.dataloader_num_workers=2` is **required**, not optional -- see
"Host RAM OOM" below. Keep deepspeed **on** (the recipe default) on A100.

To compare against a smaller backbone, override
`model.vlm_name_or_path=Qwen/Qwen3-VL-2B-Instruct` (or `-4B-Instruct`).

### Running on B300 (added 2026-08-26)

The command above is **A100-specific**. On a B300 node four things change:

| | A100 node | B300 node (`ip-10-225-130-85`) |
|---|---|---|
| venv | `a1_5_sft` | **`a1_5_sft_b300`** |
| `CUDA_HOME` | `/usr/local/cuda` | **`/opt/pytorch/cuda`** (no `/usr/local/cuda` exists) |
| attention | recipe default (flash-attn) | **`model.attn_implementation=sdpa`** (no `sm_10x` flash-attn kernels) |
| NVRTC | fine (`sm_80`) | must be **>= 12.9** -- see the B300 blocker section |

`dataloader_num_workers=2` is an **A100-node** constraint (1121 GB host RAM).
The B300 node has 3.9 TB and 192 cores, so it is not the same landmine there --
but raising it is still untested; the recipe default of 16 x 8 ranks = 128
worker processes is what caused the original OOM, so raise deliberately.

A ready-to-run script for this node is checked in as
**`launch_qwen_scratch_5ep.sh`** (5 epochs, per-epoch checkpoints, batch 4).
Unlike the scratchpad script referenced above, it lives in the repo and will
not disappear with the session.

---

## Traps hit on the A100 node, and their fixes

### 1. Host RAM OOM -- the big one

The first A100 launch died with all 8 ranks `SIGKILL`ed (exitcode -9) about
45 min in, during the step-0 eval. Not GPU memory -- the **kernel OOM killer**
(confirmed in `journalctl -k`: `Out of memory: Killed process ... (pt_data_worker)`,
26.9 GB anon-rss each).

Cause: the recipe default `dataloader_num_workers: 16`, times 8 ranks, is
**128 worker processes**, each forked from a ~21 GB-RSS parent. That exceeds
1121 GB of host RAM. `dataloader_num_workers=2` (16 workers total) holds
steady at ~291 GB and costs nothing in throughput here, because the loop is
S3-latency-bound rather than worker-bound.

**This is a landmine for any 8-rank run of this recipe on this box, not just
the ablation.** If a TruckDrive run ever dies with exitcode -9 and no Python
traceback, this is why.

### 2. `wandb.util.generate_id` is gone in wandb 0.28.x

`src/alpamayo/common/wandb_utils.py` calls `wandb.util.generate_id()` when
starting a *new* run; that attribute was removed in the wandb 0.28 line, so a
fresh run dies with
`AttributeError: module 'wandb.util' has no attribute 'generate_id'`.
Existing runs are unaffected because they take the resume branch.

**Workaround used (no upstream edit):** `init_wandb` reads the run id from
`<output_dir>/.wandb_id` if present and calls `wandb.init(..., resume="allow")`,
so pre-seeding that file with any unused 8-char id skips `generate_id`
entirely and wandb creates the run on first contact:

```bash
mkdir -p output_truckdrive_qwen_scratch
python -c "import random,string;print(''.join(random.choices(string.ascii_lowercase+string.digits,k=8)))" \
  > output_truckdrive_qwen_scratch/.wandb_id
```

**Do this for every fresh output dir.** A permanent fix would be patching
`wandb_utils.py` to fall back when `generate_id` is missing (needs the fork
NOTE header per repo convention) -- not done, to keep the upstream file clean.

### 3. venv corruption -- fixed, and the fix is on EFS so it follows you

The corruption described in the previous handoff was real and did follow to
this node (`a1_5_sft/` is on EFS): `deepspeed` was missing `__init__.py` and
`comm/__init__.py`, and `wandb` was missing `wandb_promise` *and had no
dist-info at all*, so no package manager considered it installed.

`uv sync --active` could not repair this (the broken deepspeed dist-info has
no `RECORD`, so uv can't uninstall it). What worked:

```bash
# quarantine the broken trees rather than deleting them
mv a1_5_sft/lib/python3.12/site-packages/{deepspeed,deepspeed-0.18.2.dist-info,wandb} /some/scratch/
CUDA_HOME=/usr/local/cuda uv pip install --python a1_5_sft/bin/python "deepspeed==0.19.1" "wandb>=0.25.0"
```

Result: `deepspeed 0.19.1`, `wandb 0.28.2`, both importing cleanly.
`s3torchconnector`/`s3torchconnectorclient` 1.5.0 were **already installed**
in this venv -- no action needed. Verify with:

```bash
CUDA_HOME=/usr/local/cuda a1_5_sft/bin/python -c "import torch,transformers,deepspeed,wandb,s3torchconnector; print('ok')"
```

### 4. Things the previous handoff warned about that turned out fine

- **wandb 0.26.0 login bug** (`json.loads(None)` in `_get_username`): gone on
  0.28.2. `wandb.Api()` authenticates from `~/.netrc` as `rdesc1`. No
  `WANDB_MODE=disabled` workaround needed.
  **CAVEAT (2026-08-26): "gone on 0.28.2" is a statement about a *version*, not
  about every venv.** That upgrade was only ever applied to `a1_5_sft` (the
  A100 venv). `a1_5_sft_b300` was still on **wandb 0.26.0** months later and
  died on this exact bug at `wandb.init()` -- and because it kills rank 0, the
  other 7 ranks then fail with confusing downstream
  `NCCL error / ncclRemoteError: Connection closed by remote peer` messages
  that look like a network or hardware fault. Fixed by
  `uv pip install --python a1_5_sft_b300/bin/python "wandb>=0.28.0"`.
  **The venvs are independent installs on EFS -- a fix applied to one does NOT
  propagate to the other. Check the venv you are actually launching with.**
- **HF weights needing a re-pull:** they don't. `HF_HOME` is
  `/mnt/efs/users/rod/hf_cache` (**EFS, not `~/.cache`**), and both
  Qwen3-VL-8B and -2B are already there -- 4 shards load in under a second.
  The previous handoff's "local to this box, re-pull on A100" note was wrong;
  the 11 MB under `~/.cache/huggingface` is just stray config files.
- **`CUDA_HOME` on EFS:** unnecessary here, the node has a real system toolkit
  at `/usr/local/cuda` (plus `/usr/local/cuda-13.0`).
  **Node-specific -- verify, don't assume.** On the B300 node
  `ip-10-225-130-85` there is **no `/usr/local/cuda`** at all; the toolkit is
  at **`/opt/pytorch/cuda`** (CUDA 13.0). With `CUDA_HOME` unset or wrong,
  `import deepspeed` fails outright with
  `MissingCUDAException: CUDA_HOME does not exist` (or a `FileNotFoundError`
  on `.../bin/nvcc`). Find it with `find / -maxdepth 4 -name nvcc` per box.

### 5. It looks stalled for ~40 minutes at startup. It isn't.

With `eval_on_start`, HF runs a full 155-iteration eval pass at ~11.5 s/it
(~36 min) *before* step 1, and tqdm's `0/12357` training bar is printed up
front the whole time. GPUs sit near 0% because the loop is S3-fetch-bound on
5 camera views. Don't kill it; wait for `{'eval_loss': ...}` and then the
first `{'loss': ...}`. Also ignore the flood of
`Invalid token ids found` / `Number of tokens is not equal to the expected
number` warnings at step 0 -- an untrained backbone emitting non-trajectory
tokens is the expected baseline behavior, and they stop as it learns the
format.

---

## The B300 blocker: ACTUALLY fixed (2026-08-26)

**This section previously read "Resolved: the B300 blocker" and claimed nothing
further was needed. That was wrong and it cost a later session real time.** The
only thing that had happened was a move to A100 hardware; the venv was never
repaired, so the wall was still standing the moment anyone came back to a B300.

The symptom, on any Blackwell B300 node:

```
RuntimeError: nvrtc: error: invalid value for --gpu-architecture (-arch)
```

**Root cause.** `torch==2.8.0+cu128` bundles NVRTC **12.8**, whose supported-arch
list is `[50,52,53,60,61,62,70,72,75,80,86,87,89,90,100,101,120]`. B300 is
compute capability **(10, 3) = `sm_103`**, which is *not* in that list --
`sm_103` first appears in CUDA **12.9**. (Note `sm_100`/B200 *is* supported,
so this bites B300 specifically.) Any op that needs a **runtime JIT-compiled**
kernel therefore fails. Query the list yourself:

```python
import ctypes
lib = ctypes.CDLL('<venv>/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12')
n = ctypes.c_int(); lib.nvrtcGetNumSupportedArchs(ctypes.byref(n))
arr = (ctypes.c_int * n.value)(); lib.nvrtcGetSupportedArchs(arr); print(list(arr))
```

**One-line repro** (no model, no data needed -- use this to test any new box):

```bash
python -c "import torch; print(torch.arange(1,5,device='cuda').prod())"
```

**THE FIX** -- install an NVRTC that knows `sm_103`, into the B300 venv:

```bash
CUDA_HOME=/opt/pytorch/cuda uv pip install --python a1_5_sft_b300/bin/python \
  "nvidia-cuda-nvrtc-cu12==12.9.86" --no-deps
```

Done on 2026-08-26; `a1_5_sft_b300` now carries 12.9.86 (arch list gains `103`
and `121`). This is strictly a **superset** of 12.8's archs, so it cannot
regress any other GPU sharing the venv. Verified after install, with no
`LD_PRELOAD` shim: `prod`, `matmul`, and `scaled_dot_product_attention` all
work, and a Stage-1 run's generation callbacks (viz + `ValMinADE`) execute
instead of erroring.

**Why this hid for so long.** Most of training uses *precompiled* kernels, so a
venv can import cleanly, allocate on all 8 GPUs, load checkpoints, and run
forward/backward while still being fatally broken. Only the ops that hit
torch's **jiterator** blow up -- here `reduction_prod_kernel` (a `.prod()`
reduction) reached via the discrete-trajectory-token **generation** path. That
is why the failure surfaces in the viz / `ValMinADE` callbacks rather than in
the training step.

**It is not a cosmetic, callback-only failure.** The callbacks log
`[TrajectoryVizCallback] skipped at step 0: RuntimeError(...)` and look
survivable, but the error escapes and takes the whole job down with a
`ChildFailedError` across all ranks. If you see a flood of `skipped at step`
warnings on a Blackwell box, fix NVRTC -- do not just ignore the warnings.

Related but **separate** Blackwell issue, still true: keep
`model.attn_implementation=sdpa` (the venv's flash-attn 2.8.3 wheel has no
`sm_100`/`sm_103` kernels). NVRTC and flash-attn are two independent Blackwell
gaps; fixing one does not fix the other.

---

## Results: the pretraining question is answered

Both Stage-1 runs evaluated over the full official val split (2474 windows,
XY plane, K=6), scored against **identical** ground truth. The pretrained
baseline was re-evaluated on 2026-08-24 because the original numbers predated
`standstill_snap_mps` (commit `fb26d81`, 2026-08-06), which changes val GT on
parked windows -- the old figures are not comparable and should not be quoted.

| metric | scratch-Qwen `ckpt-12357` | Alpamayo-pretrained `ckpt-4119` | gap |
|---|---|---|---|
| minADE_6 @3s | 0.3076 | **0.1930** | 1.59x |
| minADE_6 @6s | 1.0680 | **0.6032** | 1.77x |
| ADE @3s | 0.6187 | **0.3816** | 1.62x |
| ADE @6s | 2.1018 | **1.1773** | 1.79x |
| FDE @3s | 1.6298 | **0.9417** | 1.73x |
| FDE @6s | 5.7803 | **3.1196** | 1.85x |
| minADE_6 full (6.4s) | 1.2049 | **0.6730** | 1.79x |

**NVIDIA's physical-AI pretraining is worth ~1.79x on minADE**, and it reaches
that in 1 epoch where the from-scratch run needed 3. The gap widens
monotonically with horizon (1.59x at 3s -> 1.77x at 6s), so what pretraining
buys is long-horizon prediction; short-horizon "continue current motion" is
nearly a tie.

Metric conventions, which matter when comparing to other tables:

- `ADE`/`FDE` here are **sample 0**, not a mean or a best-of-K. Upstream
  `metric_api.py` selects via `argmax` over an all-zeros dummy logprob, so it
  always returns index 0. This is the convention every eval in this repo has
  used since the original Alpamayo-1 commit; it is unbiased for i.i.d. samples
  (2.3597 vs 2.3628 mean-over-6), just noisier.
- `minADE_6 @T` above is the standard definition (min over 6 of ADE on 0..T).
  The codebase's own `by_t=` variant picks the best sample once on the FULL
  horizon then slices, so its @3s reads higher (0.3448 vs 0.3076). Both are
  defensible; do not mix them in one table.

---

## Conditioning experiments: does extra context help?

Two follow-ups ask whether the scratch model improves when given information
the images alone do not carry. Both supervise **trajectory only** -- the extra
context is input, never a loss target.

### Chain-of-causation (CoC) conditioning -- DONE

`configs/sft_truckdrive_qwen_scratch_coc.yaml`. TruckDrive ships no CoC ground
truth, so labels are pseudo-labels distilled from `ckpt-4119` (32946 train +
2474 val windows, ~8 h on 8xA100) via `truckdrive/build_cot_labels.py`, loaded
by `TruckDriveDataset(cot_labels=..., cot_dropout=...)`.

Trained 3 epochs at `cot_dropout=0.5`, so ONE model is evaluable both ways.
Paired over the same 2474 windows (`checkpoint-12357/eval_cocON` vs
`eval_cocOFF`):

| minADE | CoC-ON | CoC-OFF | benefit | 95% CI (bootstrap) | win rate |
|---|---|---|---|---|---|
| full 6.4s | 1.2540 | 1.3323 | **+0.078 m** | [+0.045, +0.115] | 53.6% |
| @3s | 0.3250 | 0.3441 | +0.019 m | [+0.010, +0.029] | 53.7% |
| @6s | 1.1149 | 1.1849 | +0.070 m | [+0.038, +0.102] | 54.6% |

**CoC conditioning helps, but only slightly.** Every CI excludes zero and the
benefit grows with horizon, mirroring the pretraining gap. Two caveats: the
win rate is barely above a coin flip (the mean is carried by a tail), and
CoC-ON (1.2540) is still WORSE than the plain traj-only run (1.2049) --
plausibly the cost of spending half of training on blanked CoC. A
`cot_dropout=0.0` run would isolate that.

**Do not compare across runs to measure this.** The cross-run comparison
(1.225 CoC vs 1.240 traj-only, mean over matched steps >= 6000) buries the
effect in seed and data-order noise. The dropout trick exists precisely so the
comparison is within one set of weights.

### Navigation conditioning -- RUNNING

`configs/sft_truckdrive_qwen_scratch_nav.yaml`, launched 2026-09-01,
`output_truckdrive_qwen_scratch_nav/`, wandb run `ob1u6uvh`.

TruckDrivePublic ships no route or ego-intent annotation, so at an interchange
the route is genuinely ambiguous from pixels -- the model must guess which way
ego intends to go. Evidence this is real, measured on A2S's CoC over val:

- "turn right" statements are correct **30%** of the time (n=107) against a
  **64%** base rate; "turn left" is 96% correct (n=133).
- **50%** of turn-mentioning windows have samples that disagree left vs right.
- Under ambiguity the model confabulates supporting detail: two audited
  interchange scenes contain no traffic light at all, yet were justified with
  "since the right-turn traffic light is green".
- The commands derived from GT poses by `truckdrive/nav_labels.py` are correct
  on **106/106** clear turns and on **40/40** of the audited mislabeled
  windows. The information is fully determined by the poses; it was simply
  never given to the model.

`TruckDriveDataset(nav_labels=..., nav_dropout_prob=...)` emits `nav_text`,
consumed by the already-plumbed `route` component
(`<|route_start|>Turn left<|route_end|>`). Coverage is 99.2% of val windows
(2283 straight / 122 left / 49 right; the labeller's dead band stays
unlabeled). `route` is a USER component, so it cannot enter `label_components`
-- it is pure input, like the images.

Note nav dropout removes the **key**, whereas CoC dropout blanks the
**string**. `construct_route` no-ops on absent `nav_text`, so a dropped window
trains genuinely unconditioned rather than on an empty route block -- matching
how dead-band windows behave.

**Read the result on the turning windows, not the headline.** Only ~7% of val
windows are turns (171 of 2474); the rest carry "Continue straight" and cannot
benefit. Aggregate minADE will barely move even if conditioning works well;
score the 171 turning windows separately.

Evaluate the finished run both ways on one set of weights:

```bash
# nav-ON
... -m alpamayo1_5_sft.evaluate_hf --config-name sft_truckdrive_qwen_scratch_nav \
  +evaluate.eval_ckpt=$CKPT data.val_dataset.nav_dropout_prob=0.0 \
  evaluate.predictions_dir=$CKPT/eval_navON  ...
# nav-OFF: same, nav_dropout_prob=1.0, predictions_dir=$CKPT/eval_navOFF
```

### Generating CoC with a stronger annotator (Alpamayo 2 Super)

Not started. A2S produces markedly richer CoC than the 1.5 labels (whose top
two phrases cover 58% of the corpus), and an existing 2474-window val corpus
lives at
`/mnt/efs/users/rod/repos/alpamayo2/outputs/truckdrive_zero_shot_eval_results/`
(0 empty, 0 trajectory-token leaks, 83% unique). The train split would need
~20-30 h to generate.

`recipes/alpamayo2_sft/rollout.py` had a real bug here, now fixed: its CoC
phase-1 decode skipped the release path's `MaskDiscreteTrajectoryLogitsProcessor`
and text-EOS mask and forced an off-distribution `<|cot_start|>` prefix, so the
checkpoint emitted `<|im_end|>` and stray `<i####>` action tokens into the
reasoning span (10/12 windows truncated, 2/12 pure garbage; 0/12 after the
fix). This was NOT the missing LoRA adapter -- the broken path fails
identically with `ckpt-1030` loaded, and the fixed path is clean without it.
Also note `+eval.use_cot=true` needs the leading `+`.

If this is revisited, generate **with** nav conditioning so the direction
defect above does not get baked into the labels.

---

## Next steps

- [ ] Let the run reach step 500 for the first mid-training `ValMinADE` +
      checkpoint, and confirm minADE is falling well below the 76.1 m
      step-0 baseline.
- [ ] Run to completion (~24-27 h) or until `ValMinADE` plateaus.
- [ ] **Answer the research question:** compare this run's `ValMinADE` curve
      against the existing Alpamayo-pretrained TruckDrive Stage-1 run
      (`output_stage1_nav`-style checkpoints / wandb). The gap between the two
      curves *is* the value of NVIDIA's physical-AI pretraining. Also worth
      putting both against the ego-only MLP baseline (ADE 1.425 m), which
      already beats the finetuned Alpamayo on K=1 ADE -- if scratch-Qwen lands
      near the MLP, that says the VLM contributes little on this dataset.
- [ ] Optional: repeat with `model.vlm_name_or_path=Qwen/Qwen3-VL-2B-Instruct`
      for a backbone-size sweep (weights already cached on EFS).

Conditioning follow-ups (see the section above):

- [ ] **Nav run**: when `output_truckdrive_qwen_scratch_nav/` finishes, eval the
      same checkpoint at `nav_dropout_prob=0.0` and `1.0`, then score the 171
      turning val windows separately -- the aggregate will hide the effect.
- [ ] **Score the existing runs on turning windows only.** Costs nothing:
      predictions for traj-only, CoC-ON and CoC-OFF are already on disk. If the
      +0.078 m CoC benefit is concentrated in turns, that is prior evidence nav
      conditioning will help; if it is not, expect little from either.
- [ ] `cot_dropout=0.0` run, to separate "CoC does not help much" from "half of
      training was spent on blanked CoC".
