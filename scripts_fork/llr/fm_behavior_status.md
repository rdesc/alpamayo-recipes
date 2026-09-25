# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.

# Phase 2 -- does the reasoning change what the car DOES?

Companion docs: [`FM_HEAD_STATUS.md`](FM_HEAD_STATUS.md) (Phase 1 results),
[`fm_llr_method.md`](fm_llr_method.md) (Phase 1 method + derivation),
[`README.md`](README.md) (token-head study).

**This doc is self-contained enough to start a fresh session from.** Read section 5 before
writing any code -- it is the list of things that cost hours to find.

> **Experiment A is done (2026-09-25, n=2,071) and the answer is YES: the reasoning changes what
> the car does.** Section 6 has the result; sections 1-3 are the framing that led to it and are
> left as written. Phase 1 is not overturned -- it measured the density at `a*`, and that stays
> inert. The car does not sample from that density at that point.

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

**RESOLVED -- the second branch. See section 6.** `donor` displacement is 23.3% of
`blankvision` and 51.7% of the sampler's own spread, and substituting another clip's reasoning
costs +0.26 m of ADE against the truth (p=8.7e-42). The prediction above was framed as
displacement versus `blankvision`; in the event the sampler's OWN spread turned out to be the
more informative denominator, and both are reported.

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
| exact map, fp32, gated | `/mnt/efs/users/rod/results/llr_fm_dmap_fp32/` (COMPLETE, n=208) |
| directive flip | `/mnt/efs/users/rod/results/llr_fm_flip/` -- **empty, the launch died; see section 7** |
| token-head donor term | `/mnt/efs/users/rod/results/llr_gap_donor_a15/` |
| **Phase 2 Experiment A** | `/mnt/efs/users/rod/results/llr_fm_sampdisp/` |
| Experiment A, fp32 control | `/mnt/efs/users/rod/results/llr_fm_sampdisp_fp32/` |
| Phase-1 per-event LLR, per arm | `/mnt/efs/users/rod/results/llr_fm_ladder/per_event_fm.parquet` -- readout 7 needs it; it existed only in a session scratchpad under `/tmp` until 2026-09-25 |

---

## 6. EXPERIMENT A: RESULT (full split, n=2,071, 16 draws/event, bf16)

**The reasoning IS behaviourally live, on the same head and the same events where Phase 1 found
it likelihood-inert.** Phase 1 is not wrong; it measured a different thing, and that thing turns
out not to be the thing the car does.

Code: `phase2_sample_displacement.py`, analysis `analyze_phase2_displacement.py` (11 readouts,
gated on readout 1). Results: `/mnt/efs/users/rod/results/llr_fm_sampdisp/`. bf16 end to end,
which is what the car runs -- see the caveats for the fp32 control. 6 events skipped of 2,077.

### 6.1 Displacement -- swapping the prose moves the car

`null` is **exactly 0.000e+00** on all 2,071 events, so there is no numerical floor being read as
signal. Everything below is against that gate.

| arm | mean (m) | trim10 | median | % of `blankvision` | % of sampler's own spread |
|---|---|---|---|---|---|
| `null` | **0.0000** | 0.0000 | 0.0000 | 0% | 0% |
| `donor` | 0.8571 | 0.5860 | 0.4221 | **23.3%** | **51.7%** |
| `cluster` | 0.7345 | 0.4918 | 0.3524 | 19.9% | 44.3% |
| `shuffled` | 0.3761 | 0.2479 | 0.1782 | 10.2% | 22.7% |
| `blankvision` | 3.6812 | 3.2231 | 2.8589 | 100% | 221.9% |
| *(sampler spread)* | 1.6588 | 1.4870 | 1.3575 | 45.1% | -- |
| *(gold vs GT)* | 2.4619 | 2.1410 | 1.9090 | -- | -- |

`spread` is the mean pairwise ADE among the 16 gold rollouts -- the sampler's own diversity, and
the honest denominator. Swapping the chain-of-causation moves the car **half as far as re-rolling
the dice**, and on 13.8% of events further than that.

Against Phase 1's **0.27-0.46% of separation**, this is two orders of magnitude larger relative
to its own yardstick. The two measurements are not in contradiction -- see 6.5.

### 6.2 The three content arms come APART, which they never did on the likelihood

| pair | median diff | mean | Wilcoxon p | Phase-1 p (likelihood) |
|---|---|---|---|---|
| `donor` vs `shuffled` | +0.1758 m | +0.4810 | **1.6e-148** | 0.52 |
| `cluster` vs `shuffled` | +0.1114 m | +0.3584 | **1.6e-101** | 0.19 |
| `donor` vs `cluster` | +0.0359 m | +0.1226 | **3.8e-08** | 0.20 |

Phase 1's headline was that these three are indistinguishable, and that indistinguishability *was*
the occupancy result. Behaviourally they are ordered, cleanly and with enormous margins:
**donor > cluster > shuffled**. Permuting gold's own words moves the car less than half as far as
substituting someone else's prose.

**Not length.** The regression of displacement on |Δprefix_len| gives `donor` slope +0.0133
(p=0.085, r=0.038) against an intercept of 0.7938 versus a mean of 0.8571 -- 93% intercept, the
same pattern Phase 1 found on the likelihood. And on the **404 events where the donor happens to
be length-matched to within 1 token** (as `shuffled` always is), donor − shuffled is still
**+0.4575 m, p=4.1e-25**, donor further on 72.3%. The confound is dead.

### 6.3 Accuracy -- and this is the part that bears on LangForce

Displacement says the rollouts *differ*. It does not say the difference is a *degradation*:
under matched noise a displacement is only damage if it moves the car away from the truth.

| arm | Δ ADE vs GT | p | Δ minADE | p | worse on |
|---|---|---|---|---|---|
| `donor` | **+0.2648 m** | 8.7e-42 | +0.1178 | 1.0e-15 | 62.5% |
| `cluster` | +0.1723 | 7.7e-21 | +0.0812 | 2.0e-09 | 58.4% |
| `shuffled` | **+0.0013** | **0.56** | −0.0032 | 0.22 | 51.4% |
| `blankvision` | +1.6000 | 1.0e-170 | +0.8101 | 2.8e-107 | 80.0% |

Substituting another clip's reasoning costs **0.26 m of ADE, ~17% of what blinding the cameras
costs**. Permuting gold's own words costs **nothing at all** (p=0.56) despite moving the car
0.38 m. Both readouts agree, and minADE agrees with mean ADE, so this is not a sampling artefact.

### 6.4 Is the damage just "the sampler got pushed off its own mode"?

The deflationary reading, and the one worth the most care: under gold the rollout sits at the
model's own best guess, so *any* displacement should raise ADE and 6.3 would say nothing about
content. Damage per metre moved:

| arm | displacement | Δ ADE | **Δ ADE per metre** |
|---|---|---|---|
| `donor` | 0.8571 | +0.2648 | **0.309** |
| `cluster` | 0.7345 | +0.1723 | 0.235 |
| `shuffled` | 0.3761 | +0.0013 | **0.004** |
| `blankvision` | 3.6812 | +1.6000 | 0.435 |

Not constant, and the slope of damage on displacement says the same thing about the whole
relation: `donor` **+0.498**, `cluster` +0.423, `blankvision` **+0.468**, `shuffled` **−0.030**.
Displacement caused by foreign prose costs *what displacement through the cameras costs*.
Displacement caused by scrambling correct content costs **nothing**.

Binned on each row's own displacement, so the arms are compared at matched push, and
residualised on two arm-independent difficulty measures (gold's own ADE, the sampler's spread),
because it takes a harder scene for `shuffled` to displace as far as a donor does:

| displacement bin | `donor` | `cluster` | `shuffled` | p |
|---|---|---|---|---|
| 0.294-0.457 m | −0.0610 | −0.1315 | −0.1304 | 4.3e-04 |
| 0.457-0.738 m | −0.0203 | −0.0401 | **−0.1547** | 7.8e-04 |
| 0.738-1.327 m | +0.0192 | +0.0424 | **−0.1736** | 1.9e-03 |
| 1.327+ m | +1.0025 | +0.6908 | −0.0852 | 3.0e-05 |

A gap of **0.13-0.19 m at matched displacement and matched difficulty**. Note the confound runs
*against* the effect: within every bin the `shuffled` rows are the harder scenes (gold ADE 3.22
vs 2.60 at 0.457-0.738 m) and they still degrade less.

**The honest range is 0.03-0.19 m.** Differencing the two arms *within* each event and
controlling the displacement difference puts the arm gap at only ~0.03-0.05 m. That estimator is
recorded in the code and should not be re-run as though it were the conservative answer: it has
to extrapolate to "equal displacement within one event", which almost never occurs (the donor
nearly always displaces further), across a visibly convex relation -- and its three pairwise
estimates do not compose, `(d−s) ≠ (d−c)+(c−s)`, which is the diagnostic of mis-specification
rather than of caution.

### 6.5 Why this does not contradict Phase 1

Three independent signs that Phase 2 is measuring something Phase 1 could not see, rather than
contradicting it:

1. **Horizon.** Displacement grows monotonically -- 0.0003 m at 0.1 s to 2.52 m at 6.4 s -- which
   is a divergence compounding, exactly the shape a behavioural effect should have. Phase 1's
   content arms spiked at 0.8 s and decayed, which `FM_HEAD_STATUS.md` flags as never having made
   physical sense.
2. **Channel signature replicates by a different route.** lat/lon is 0.20 (`donor`), 0.22
   (`cluster`), 0.29 (`shuffled`) against **0.46** for `blankvision`: content is far more
   longitudinal than vision. That is Phase 1's "content moves acceleration, vision moves
   curvature" recovered from rollout geometry rather than from a loss decomposition.
3. **Per-event, the two phases disagree about WHICH scenes** -- and the positive control proves
   the test has power. Against an attenuation ceiling of sqrt(0.872 × 0.992) = 0.93:

   | arm | Pearson r vs Phase-1 LLR | p | Spearman | p |
   |---|---|---|---|---|
   | `blankvision` (control) | +0.1702 | 6.4e-15 | **+0.2220** | 1.5e-24 |
   | `donor` | +0.0992 | 6.1e-06 | +0.0854 | 1.0e-04 |
   | `cluster` | +0.1937 | 6.0e-19 | +0.1134 | 2.3e-07 |
   | `shuffled` | +0.2350 | 2.2e-27 | +0.0393 | 0.074 |

   All far below the ceiling, and `shuffled` is Pearson +0.235 against Spearman +0.039, i.e.
   tail-driven. Displacement is per-event reliable at **0.99** (split-half over draws,
   Spearman-Brown) for every arm, so none of this is attenuation on the Phase-2 side.

**Reading.** The deployed head uses the reasoning's *bag of words*, not its syntax. A different
bag is misinformation and costs ADE at roughly the per-metre rate that losing the cameras does; the
same bag permuted moves the sampler but carries no misinformation and costs nothing. Phase 1's
density-at-`a*` readout was blind to all of it. For the LangForce premise: there **is** something
for a regulariser to reinforce -- with the caveat that what is being read looks lexical, so a
regulariser could be satisfied by keyword presence rather than by correct reasoning.

### 6.6 Caveats

- **bf16.** Deployment-faithful and therefore the right default, but `null = 0` is a determinism
  check, not a bound on bf16 chaos amplifying a tiny KV difference. The `--fp32` control arm
  (paired, same events, same x_0) is what bounds it.
- **One noise seed.** x_0 is drawn per event from (noise_seed, clip_id, event_idx) -- this does
  *not* repeat Phase 1's shared-`eps` defect, where one grid was reused across all events -- but a
  seed replicate is still the analogue of Phase 1's seed-7 check and has not been run.
- **Displacement is pointwise, not distributional.** Two conditionings could in principle have
  identical pushforwards and still show CRN displacement. 6.3 is what makes the claim
  distributional; 6.1 on its own would not.
- **`cluster` ≈ `donor`.** Topic-matching a donor buys only 0.12 m of the 0.86 m. Whatever is
  being read is not well captured by `event_cluster`.

---

## 7. Status

| | |
|---|---|
| **Experiment A** (sampling displacement) | **DONE** -- n=2,071, see section 6 |
| Experiment A, fp32 precision control | running -- 4 workers x 2 GPUs, ~128 events, paired |
| Experiment A, seed replicate | not started |
| **Experiment B** (CFG sweep) | not started -- section 4 |
| directive flip (Phase 1 tail) | **NOT running.** The queued launch died at 17:19 on 2026-09-25, every shard, with `unknown arm 'flipdir'`: the arm was fully implemented but never added to the validation whitelist in `phase0_llr_flow_matching.py`. Fixed; needs a relaunch. |
