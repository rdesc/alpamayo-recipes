# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.

# Phase 2 -- does the reasoning change what the car DOES?

Companion docs: [`FM_HEAD_STATUS.md`](FM_HEAD_STATUS.md) (Phase 1 results),
[`fm_llr_method.md`](fm_llr_method.md) (Phase 1 method + derivation),
[`README.md`](README.md) (token-head study).

**This doc is self-contained enough to start a fresh session from.** Read section 5 before
writing any code -- it is the list of things that cost hours to find.

---

## 1. The question, and why Phase 1 does not answer it

Phase 1 measured **density at the ground-truth trajectory**:

    log p(a* | v, gold_coc) - log p(a* | v, donor_coc)   ~=  0

on both heads, by three independent readouts. But the deployed system **samples**; it does not
evaluate a density at `a*`. Two conditionings can assign the *same* density at one point and
still produce *different* trajectories, because our measurement pins down the density at exactly
one location in a 128-dimensional space.

So Phase 1 established: *the reasoning does not change how likely the true trajectory is.*
It did NOT establish: *the reasoning does not change what the car does.*

That is the hole Phase 2 closes.

---

## 2. What Phase 1 settled (do not re-litigate)

| | result |
|---|---|
| token head, content | ~0 (donor Wilcoxon p=0.99) |
| FM surrogate (reweighted ELBO), content | +0.45% of separation, p<1e-5 -- **later shown to be slack** |
| **FM exact** (10-step map likelihood, fp32, gated), content | **-0.31 +/- 0.38, p=0.42** (n=106) |
| cross-head per-event agreement | **r=0.021** against a 0.81 ceiling; **49.6% sign agreement** |
| per-event reliability | token **+0.818**, FM **+0.872** -- neither head is noise |
| bf16 KV contribution | paired **-0.11 +/- 0.17, p=0.49** -- ruled out |
| surrogate vs exact | surrogate inflates every arm **6-8x** (variational slack) |

Two conventions that are settled and should not be re-opened:

1. **Normalise each head by its OWN `log p(correct) - log p(wrong)`.** Absolute levels are not
   comparable across heads -- a 3000-bin softmax per token and a continuous density on R^128 have
   no reason to share a scale. FM exact separation ~+0.45 nats/cell vs the token head's +0.775 is
   **expected and must not be chased.** The one precondition -- that both yardsticks be the same
   measurement -- is satisfied (`--donor-traj-min-ade 5.0` on both).
2. **Quote ratios and trimmed means, never surrogate levels.**

---

## 3. Experiment A -- paired sampling displacement (do this first)

Assumption-free, and cheap. The deployed sampler is

    x_0 ~ N(0, I);  for k = 0..9:  x_{k+1} = x_k + 0.1 * v_theta(x_k, k/10, KV(v, c))

Sample under gold and under donor **from the same `x_0`** (common random numbers, as everywhere
in this project), then measure the displacement between the two rollouts in metres.

    displacement(gold, donor) = ADE( rollout(x_0, gold), rollout(x_0, donor) )

Scale it against the same ladder as Phase 1 -- `wrongtraj` is not meaningful here (it varies the
target, not the conditioning), so the yardstick is `blankvision`:

| arm | what it measures |
|---|---|
| `null` | same conditioning twice -- MUST be exactly 0 |
| `donor` | the headline: does swapping the prose move the car? |
| `shuffled` | own words permuted |
| `blankvision` | the yardstick: how far does removing the cameras move it? |

**Cost:** 10 forward passes per condition, **no Jacobians, no inversion, no Newton** -- seconds
per event, against ~24 min for the exact density. The full split is 1-2 GPU-h.

**Sharp prediction.** If content is genuinely inert, `sample(gold) ~= sample(donor)` under matched
noise, and `donor` displacement should be a tiny fraction of `blankvision`. If instead the two
rollouts diverge substantially, then the reasoning changes behaviour *without* changing the
likelihood of the true trajectory -- a subtle and important result, and directly relevant to the
LangForce premise (a regulariser would then have something to reinforce).

---

## 4. Experiment B -- CFG amplification

Classifier-free guidance, but between two REAL conditionings rather than against a null:

    v_cfg = v_donor_avg + alpha * ( v_gold - v_donor_avg ) ,   v_donor_avg = (1/K) sum_k v(., donor_k)

Sweep `alpha` in {0, 1, 2, 4}. `alpha=1` is the ordinary gold-conditioned field; `alpha=0` is the
donor-average field; `alpha>1` amplifies whatever is specific to gold.

Two things this buys that Experiment A does not:

- **Amplification.** If the gold-vs-donor difference is real but small, alpha>1 makes it
  measurable, and reveals whether the direction is *systematic across events* or noise.
- **Direction, not just magnitude.** Does amplifying "what is specific to gold" move the rollout
  TOWARD the ground truth (content is being used, just weakly) or in an arbitrary direction
  (the difference is nuisance structure)? Project the displacement onto the
  `gold_rollout -> a*` direction.

Averaging over donors is deliberate: it marginalises donor-specific idiosyncrasy and leaves only
what is specific to gold.

### Why this variant is better-behaved than textbook CFG on this checkpoint

`FlowMatching.__init__` declares `train_ignore_guidance_rate = 0.1` and
`inference_guidance_weight = 1.0`, but `construct_training_data` sets `"is_drop_guidance": None`
and **nothing in the installed `alpamayo_r1` package consumes either field** (verified by grep).
So this checkpoint has no trained null condition, and textbook CFG (cond vs null) would evaluate
the model off-distribution.

Interpolating between gold and donor-average avoids that: both endpoints are real, in-distribution
conditionings. `alpha > 1` still extrapolates, but from two valid anchors rather than from an
untrained null.

`sample_trajectories_from_data_with_vlm_rollout_cfg_nav` exists only in the separate
`alpamayo1_5` package (`/mnt/efs/users/rod/repos/alpamayo1.5/`, see
`notebooks/inference_nav.ipynb`), NOT in the `alpamayo_r1` package this study uses. Implement it
here instead -- the velocity-field plumbing already exists in the scratchpad `dmap_model.py`
(`make_vfield`) and in `phase0_llr_flow_matching.py` (`score_fast`).

---

## 5. Practical notes -- read before writing code

**Venv and box.** Always `recipes/alpamayo1_x_rl/a1x_rl_b300/bin/python`, run from
`recipes/alpamayo1_x_rl/`. The box is **8x A100 (sm_80)**. The two venvs differ by ~0.01 nats on
identical inputs, which is the same order as the effects being measured -- see the root
`CLAUDE.md`.

**Gotchas that cost hours:**

1. **`pgrep -f <script>` matches your own shell wrapper.** A driver that waits on
   `pgrep -f "dmap_model"` never exits, because the tool-call wrapper that created it carries that
   string in its argv. Hit **three times** in this project, including once after "fixing" it by
   moving the driver into a file. **Wait on actual GPU occupancy instead:**
   `while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -gt 0 ]; do sleep 120; done`
   Also: `pkill -f <pattern>` will kill the shell running it (exit 144).

2. **`torch_dtype=` is silently ignored.** `config.model_dtype = bfloat16` with
   `keep_same_dtype = True` makes the constructor cast expert/action_in_proj/action_out_proj to
   bf16 regardless, while ~half the VLM stays fp32 -- producing
   `mat1 and mat2 must have the same dtype`. Cast explicitly AFTER construction. This is what
   killed the ODE fp32 arm for weeks; it was never a deep problem.

3. **Full fp32 OOMs at 79 GB** (weights are only ~41 GB; the fp32 multi-camera vision prefill adds
   ~37 GB). Fix: VLM on `cuda:0`, expert on `cuda:1`, cast the KV cache at the boundary
   (`--device-split` in `dmap_model.py`). They run strictly sequentially, so this costs nothing
   but a GPU. Measured: 32.8 GiB + 8.5 GiB.

4. **Forward-mode AD** needs `sdpa_kernel([SDPBackend.MATH])` (unimplemented for efficient SDPA),
   `.contiguous()` on `expand`ed inputs (rejects stride-0 views), and **must** be wrapped in
   `torch.no_grad()` or it also builds a reverse graph and OOMs regardless of chunk size.
   *Not needed for Phase 2* -- sampling requires no derivatives at all.

5. **Argument-name collisions.** `--split` was already the dataset split; adding a device-split
   flag of the same name raises `conflicting option string` at launch. It is now `--device-split`.

6. **Diagnostics measured at a diverged iterate are meaningless.** `cond(G) ~ 1e20` and
   `det < 0` on 25% of rows looked like the map being degenerate; both were artefacts of an
   unguarded Newton step. Always report diagnostics at the iterate you actually return.

**Existing results:**

| what | where |
|---|---|
| surrogate ladder (ungated) | `/mnt/efs/users/rod/results/llr_fm_ladder/` |
| noise-grid replicate | `/mnt/efs/users/rod/results/llr_fm_ladder_seed7/` |
| nav arms (parked) | `/mnt/efs/users/rod/results/llr_fm_nav/` |
| exact map, bf16 KV | `/mnt/efs/users/rod/results/llr_fm_dmap/` |
| exact map, fp32, gated | `/mnt/efs/users/rod/results/llr_fm_dmap_fp32/` |
| directive flip (running) | `/mnt/efs/users/rod/results/llr_fm_flip/` |
| token-head donor term | `/mnt/efs/users/rod/results/llr_gap_donor_a15/` |

---

## 6. Status

| | |
|---|---|
| **Experiment A** (sampling displacement) | not started |
| **Experiment B** (CFG sweep) | not started |
| directive flip (Phase 1 tail) | running -- `flipdir` arm, 828/2077 events flippable, reproduces 12/12 hand-verified flips |
