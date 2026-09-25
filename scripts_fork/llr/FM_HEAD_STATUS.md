# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.

# Flow-matching head: running status

> **Phase 1 (likelihood) is essentially complete.** The behavioural follow-up -- does the
> reasoning change what the car actually *does*, as opposed to how likely the true trajectory is
> -- is tracked separately in [`fm_behavior_status.md`](fm_behavior_status.md).

Working log for porting the discrete-token LLR analysis to the **deployed Stage-2 flow-matching
action expert**. Companion docs: [`README.md`](README.md) (token head),
[`fm_llr_method.md`](fm_llr_method.md) (the estimator, derived step by step).

**Goal:** reproduce, on the head that actually drives, what `README.md` established on the token
head -- above all the content term (does the trajectory use what the reasoning SAYS?) with a
control ladder that makes a null interpretable.

**Quoting rules, learned the hard way.** (1) Quote the **trimmed** mean as well as the mean --
several arms are tail-driven and the two differ by 2x. (2) Quote every figure with its
**`lambda_max`**; nothing has plateaued. (3) Quote content as a **share of the in-run
`wrongtraj` yardstick**, never as an absolute nats level.

> ### WARNING (2026-09-24): the surrogate's ABSOLUTE levels are inflated ~6-8x
>
> The exact 10-step discrete-map likelihood (see IN FLIGHT) measures the same arms on the same
> events with **no variational slack**, and puts every one of them 6-8x BELOW the reweighted-ELBO
> surrogate: `donor` +1.90 -> **-0.30 +/- 0.39 (p=0.24)**, `blankvision` +82.6 -> **+13.2**,
> `wrongtraj` +418 -> **+54.9**. This is the slack-asymmetry mechanism predicted in
> `fm_llr_method.md` section 9, now measured rather than hypothesised.
>
> **Consequence for everything below: read the RATIOS, not the levels.** Every absolute nats
> figure in this file is a surrogate figure and is biased upward by roughly that factor. The
> occupancy-vs-content conclusion is unaffected -- it is a ratio, and the exact measure
> *strengthens* it, turning the surrogate's small-but-significant content term (p<1e-5) into a
> true zero (p=0.24).

---

## HEADLINE (full split, n=2,071, all six arms, replicated on two noise grids)

**The flow-matching head's answer is the same as the token head's: occupancy, not content.**

At `lambda <= 9.19` (t <= 0.99, the sampling cap):

| arm | mean | trimmed 10% | % of sep (mean) | % of sep (trim) | Wilcoxon p |
|---|---|---|---|---|---|
| `wrongtraj` (separation) | +3.26605 | +2.93209 | 100% | 100% | <1e-5 |
| `blankvision` | +0.64566 | +0.26353 | 19.8% | **9.0%** | <1e-5 |
| `donor` (random CoC) | +0.01486 | +0.00803 | 0.46% | **0.27%** | <1e-5 |
| `donor_cluster` (same topic) | +0.01394 | +0.00602 | 0.43% | 0.21% | <1e-5 |
| `shuffled` (own words permuted) | +0.01420 | +0.00834 | 0.43% | 0.28% | <1e-5 |
| `null` | **+0.00000** | +0.00000 | 0% | 0% | -- |

**The three content arms are indistinguishable.** Paired Wilcoxon between them:
donor-cluster p=0.20, donor-shuffled p=0.52, cluster-shuffled p=0.19. That is the whole result.
They are designed to come apart if the head reads content:

- `cluster` holds the topic fixed, so a same-genre donor should be a BETTER substitute than a
  random one. It is not.
- `shuffled` keeps gold's exact bag of words and destroys only syntax. If the head read the
  sentence, this should hurt less than unrelated prose; if it read keywords, not at all. It
  costs the same as unrelated prose.

So any prose that is not gold costs the same, regardless of topic, regardless of whether the
words are gold's own. **That is the occupancy signature the token head found, reproduced on the
deployed head.**

Content is ~0.27-0.46% of separation here vs ~0.05% on the token head. Larger in relative
terms, still under half a percent, against vision at 9-20%.

**Channel asymmetry is sharp and interpretable** (lambda<=6). Vision is mostly about *steering*;
content touches *only* acceleration:

| arm | accel | curvature |
|---|---|---|
| `blankvision` | +0.13765 | **+0.71892** |
| `wrongtraj` | +1.73885 | +1.88950 |
| `donor` | +0.01711 | **-0.00012** |
| `shuffled` | +0.01584 | -0.00598 |

Curvature is *exactly zero* for the content arms. A2S's token head showed the same acceleration
bias and attributed it to acceleration being the harder prediction, which gains from extra prefix
compute -- i.e. a presence effect, not content. This is consistent with that.

**Length is not the driver.** Regressing per-event LLR on signed prefix-length difference:
slope +0.00082 (p=0.036), intercept +0.00854 against a total of +0.00849 -- essentially all
intercept. `shuffled` is *nearly* length-matched (mean |d_prefix| = **0.38** tokens, vs 3.89 for
`donor`; permuting words usually but not always preserves the token count) and its slope is
p=0.94. Both arms give the same answer.

**The early-horizon spike is less alarming than it looked.** `wrongtraj`, which is unambiguously
real, shows the same shape (+6.51 at 0.8 s decaying to +0.47), so an early spike is partly a
property of the metric. `blankvision` is flatter. The content arms look like `wrongtraj`'s shape,
which is still unexplained.

---

## NOISE-GRID REPLICATE (seed 7) -- the shared-eps defect is real but small

**The defect.** `eps_all`/`t_all` are drawn ONCE outside the event loop
(`phase0_llr_flow_matching.py:362-363`), so all 2,071 events reuse the same 32 eps draws. That
grid's Monte-Carlo error is perfectly correlated across events and does **not** shrink as
1/sqrt(n) -- yet every SE was computed as `std(per_event)/sqrt(n_events)`. `null == 0` is no
reassurance: it is zero *by construction* (deterministic re-score), a determinism check, not a
noise floor.

**The test.** Re-ran the entire ladder with `--noise-seed 7`, same events, same donors, same
`t` grid (the `t` grid is deterministic quadrature -- a bias, not a variance). 2,071 events,
8 GPUs, ~59 s/event.

| arm | seed 0 | seed 7 | delta | SE(seed0) | sigma_grid | SE_total | inflation | r per-event |
|---|---|---|---|---|---|---|---|---|
| `blankvision` | +0.64566 | +0.62856 | +0.01710 | 0.06099 | 0.01209 | 0.06218 | 1.02x | **0.993** |
| `donor` | +0.01486 | +0.01505 | -0.00018 | 0.00397 | 0.00013 | 0.00397 | **1.00x** | **0.787** |
| `donor_cluster` | +0.01394 | +0.01825 | -0.00431 | 0.00375 | 0.00305 | 0.00483 | 1.29x | 0.801 |
| `shuffled` | +0.01420 | +0.01581 | -0.00161 | 0.00272 | 0.00114 | 0.00294 | 1.08x | 0.937 |
| `wrongtraj` | +3.26605 | +3.57960 | -0.31355 | 0.13306 | 0.22171 | 0.25858 | **1.94x** | 0.524 |

`sigma_grid = |delta|/sqrt(2)` (2 independent grids, 1 dof -- crude). `r per-event` is the
correlation of per-event values across the two grids.

**Three conclusions.**

1. **The content arms are safe.** `donor` moves by 0.00018 nats (1.2% of its value); the grid
   contributes essentially nothing and the reported SE needs no inflation. The headline null
   stands.
2. **The yardstick is the noisiest arm.** `wrongtraj` moves 0.31 nats (9.6%) and its SE is
   understated ~2x. So every "% of separation" figure carries ~10% wobble in its *denominator*:
   read "0.46%" as 0.42-0.46%. No conclusion turns on this.
3. **The per-event FM signal is reliable, not noise.** `donor` reproduces at r=0.787 across
   *completely independent* noise grids (consistent with the earlier within-run split-half of
   0.681 after Spearman-Brown: 2*0.681/1.681 = 0.810). This matters for the cross-head section
   below.

**Still to fix in code:** move the `eps` draw inside the event loop so each event gets its own
grid. This does not invalidate anything above -- it only means absolute SEs for `wrongtraj`
should be widened ~2x until it is fixed.

---

## NAV ARM (full split, n=2,071; navswap n=187 turn rows) -- the dissociation replicates

| arm | nats | median | frac>0 | Wilcoxon p |
|---|---|---|---|---|
| `navnone` (gold nav vs none) | +0.00304 | +0.00095 | 53.9% | **0.052** |
| `navswap` (gold nav vs inverted) | +0.09957 | +0.00120 | 52.9% | **0.011** |
| `null` | +0.00000 | -- | -- | -- |

**Channel split is the result -- but it is truncation-dependent, so both cuts are given.**
(An earlier version of this table printed the lambda<=9.2 numbers under a lambda<=6 heading.
Corrected 2026-09-23.)

| arm | cut | accel | p | curvature | p | n |
|---|---|---|---|---|---|---|
| `navswap` | lambda<=6 | +0.00021 | 0.155 | **+0.07142** | 0.055 | 187 |
| `navswap` | lambda<=9.2 | -0.00195 | 0.019 | **+0.20108** | **9.4e-4** | 187 |
| `navnone` | lambda<=9.2 | -0.00154 | 0.406 | +0.00762 | 5.9e-5 | 2071 |
| `donor` (CoC content) | lambda<=6 | **+0.01711** | -- | **-0.00012** | -- | 2071 |

**What is robust:** the channel *ordering*. Inverting the nav direction moves curvature far more
than acceleration at every truncation (100x the magnitude at the cap, opposite signs), while CoC
content does the reverse -- accel only, curvature exactly zero. Two language channels, opposite
channel signatures, same estimator, same head. An artefact should smear across both; this does
not. It mirrors the token head's "turns -> curvature only" (p=4.3e-3).

**What is NOT robust:** the significance. Inside training support (lambda<=6) neither navswap
channel reaches p<0.05 (curvature 0.055, accel 0.155). The p=9.4e-4 on curvature only appears at
the lambda=9.19 cap, i.e. it leans on the top bin where training mass is 0.085%. And at that cap
acceleration is **not** zero -- it is -0.00195 at p=0.019. So "curvature and nothing else" is
too strong; "curvature-dominated, by two orders of magnitude" is what the data support.

The horizon profile corroborates it: `navswap` is slightly NEGATIVE at 0.8 s, peaks at 1.6-2.4 s
and decays -- the right shape for a maneuver unfolding over seconds. The CoC content arms spike
at 0.8 s and decay, which never made physical sense.

**Caveats, which are substantial:**
- `navnone` at p=0.052 is marginal. Nav *presence* does NOT clearly transfer; the token head had
  p=6e-16 for the same quantity.
- `navswap` is n=187 and heavily tail-driven: mean +0.0996 against median +0.0012, and the top 5
  events carry **67%** of the sum. The channel ORDERING is robust across truncations; the
  significance and the magnitude are not -- do not quote +0.0996 as an effect size, and do not
  quote p=9.4e-4 without saying it is a lambda<=9.19 figure.
- A wrong direction hurts more than no direction, which is sensible (a false command misleads;
  silence is merely uninformative).
- **No in-run yardstick.** The nav run did not carry `wrongtraj`/`blankvision` under the `r1_5`
  template, so the nav effect cannot yet be expressed as a share of separation. That is the
  single cheapest thing to fix (~2 GPU-h).

**Status: PARKED -- not being pursued (user decision, 2026-09-24).** The channel-specificity is
striking and matches a strong prior from the token head, but it rests on 187 events with a skewed
distribution, its significance appears only at the untrained lambda tail, and -- decisively -- it
rests ENTIRELY on the surrogate, which the exact map has since shown to be inflated 6-8x. It was
never validated against the exact likelihood. The data and the code path (`--nav`) remain; if it
is ever revived, the exact-map nav arm is the prerequisite, not more surrogate events.

---

## Cross-head comparison: RE-INSTATED, with the reasoning fixed

This section has flip-flopped twice; the measurement below settles it.

1. Originally written up as "the strongest single piece of evidence" that the FM content term is
   noise, justified as *"this is what two near-zero NOISE measurements look like."*
2. **Retracted**, on the grounds that the token head's donor term is an established zero
   (Wilcoxon p=0.99), so correlating anything against it gives r ~= 0 by construction and the
   low r carries no information.
3. **The retraction's premise was wrong, and is now itself retracted.** A mean of zero does not
   imply a noisy *per-event* measurement. Measured directly by split-half over the 128
   (waypoint x channel) cells, Spearman-Brown corrected:

| | per-event reliability of the content term |
|---|---|
| token head | **+0.818** |
| FM surrogate | **+0.872** (cf. +0.787 across independent noise grids) |

**Neither head is noise at the event level.** Both produce a reproducible per-event number. So
the cross-head correlation IS informative, and the correct yardstick for it is the attenuation
bound `sqrt(rel_tok * rel_fm) = 0.845` -- what we would see if the two heads were measuring the
same per-scene quantity.

Observed, on the 2,062 matched events:

| | |
|---|---|
| Pearson r | **+0.021** |
| Spearman rho | +0.052 |
| disattenuated r (r / 0.845) | **+0.025** |
| sign agreement | **49.6%** -- a coin flip |
| fraction positive | token 37.9%, FM 56.5% |

**Both heads reliably measure something per event, and those somethings are completely
unrelated.** Against a ceiling of 0.845 we observe 0.021. If CoC content genuinely mattered for
particular scenes, two readouts of the same prose on the same scene through the same VLM trunk
should agree about which scenes. They agree at chance.

So the per-event structure in each head's content term is **idiosyncratic readout structure, not
shared signal** -- which is the strongest form of the occupancy-not-content result, and stronger
than either of the two earlier versions of this section.

**The ceiling is robust to how reliability is estimated.** Split-half over CELLS cannot see noise
shared by all cells within an event -- donor choice above all, and the two heads drew different
donors. Re-estimating FM reliability by splitting across DONORS instead gives +0.802 (vs +0.872
from cells, +0.787 across noise grids), so the ceiling moves 0.845 -> 0.810 and the observed
r=0.021 goes from 2.5% to 2.6% of it. Nothing turns on the choice. (The donor split is slightly
approximate: this run carried 3 donors, so the halves are 2-vs-1 rather than equal.) And
**49.6% sign agreement is chance-level regardless of any ceiling.**

(Computed on the SURROGATE FM values, where n=2,062. The exact discrete-map donor values exist
for only 111 events so far; repeating this against them is worthwhile but underpowered.)

**On the separation LEVELS differing across heads -- this is expected, not a finding.** FM exact
separation is ~+0.45 nats/cell against the token head's +0.775. The two heads parameterise their
output completely differently (a 3000-bin softmax per token vs a continuous density on R^128), so
there is no reason `log p` should share a scale. **Do not chase this gap.** The agreed protocol is
that each head is normalised by its OWN `log p(correct) - log p(wrong)`, and only the resulting
ratios are compared.

That protocol has exactly one precondition: the two yardsticks must be the SAME measurement. It
was briefly violated -- the token head gates its wrong-trajectory donor at ADE >= 5 m and the
flow-matching arm did not -- and is now satisfied (`--donor-traj-min-ade 5.0` in both;
`donor_ade` recorded per row, median 15.3 m, 98% >= 5 m).

The comparison that IS in comparable units is the camera share: **20-22% (FM exact) vs 34%
(token head)**. Same sign, same order, same qualitative story; a quantitative difference of this
size between two genuinely different heads is unremarkable.

## Remaining caveat: still not plateaued in lambda

Sampling is capped at lambda=9.19 (t<=0.99, inside training support), so the lambda<=11 and
<=13.9 columns just repeat the cap and the "plateau" is trivial. Within the sampled range the
content term is still climbing:

| arm | lam<=2 | lam<=4 | lam<=6 | lam<=9.2 |
|---|---|---|---|---|
| `donor` | +0.00533 | +0.00802 | +0.00849 | +0.01486 |
| `shuffled` | -0.00036 | +0.00134 | +0.00493 | +0.01420 |
| `wrongtraj` | +0.50793 | +1.06400 | +1.81418 | +3.26605 |

The top sampled bin (lambda 6..9, ~4 of 32 draws) contributes **43%** of the donor term. **Quoted
figures are upper bounds at the cap, not converged values.** The occupancy-vs-content conclusion
does not depend on this: all three content arms move together under any truncation, and the
content/separation RATIO is far more stable than either level (0.29% -> 0.47% across the full
truncation sweep, versus a 2.8x move in the level).

---

## STATE: the estimator is settled and has a working control ladder

**The measurement is the reweighted flow-matching loss gap, in nats:**

    LLR = E_{t,eps}[ (t/(1-t)) * ( ||v(x_t,t,ctrl) - u||^2 - ||v(x_t,t,gold) - u||^2 ) ]

with shared (t, eps) across arms (common random numbers) and t on a uniform log-SNR grid, which
makes the per-draw weight `(lam_hi-lam_lo)*t^2/2` -- bounded. Positive = the gold reasoning makes
the true trajectory more likely, matching the token head's sign convention. Derivation and
citations in [`fm_llr_method.md`](fm_llr_method.md).

---

## DONE

| # | experiment | n | outcome |
|---|---|---|---|
| 1 | **Step-0 reproduction** | 12 events | Token-head `llr_gold` reproduces **bit-exactly** (max abs 0). FM fast path verified against the real `model.forward`. |
| 2 | **`--verify` on the CONTRAST** | 1 event, 16 draws | Falsified the assumption that fast-path error is common-mode: level errors 1.5e-3 + 1.7e-3 **ADD** to 2.6e-3. Bias +9.8e-5 +/- 1.5e-4 -- negligible against a 2e-3 target. |
| 3 | **Full-resolution residual** | -- | Scorer stores the `(64,2)` squared residual per draw, so horizon and channel splits are recoverable. ~113 MB/run. |
| 4 | **Pilot** | 1,077 events, 2 arms | Both content arms significantly positive; effect concentrated at 0.8-1.6 s and in acceleration. Flagged a possible length confound. |
| 5 | **Length-confound test** | 2,071 events | **Hypothesis rejected.** Slope +0.0008, intercept +0.00854 vs total +0.00849. `shuffled` slope p=0.94. |
| 6 | **Full control ladder** | 2,071 events, 6 arms | **THE HEADLINE.** Occupancy, not content. Three content arms indistinguishable at 0.27-0.46% of separation. |
| 7 | **Nav arm** | 2,071 events (187 turns) | Curvature-dominated (100x accel at the cap), opposite signature to CoC content's accel-only. Significant only at lambda<=9.19. **Provisional.** |
| 8 | **Channel + horizon splits** | 2,071 events | Content = accel only, curvature exactly 0. Vision = mostly curvature. Nav = curvature only. |
| 9 | **lambda truncation sweep** | 2,071 events | Nothing plateaus; top bin = 43% of donor. Ratios far more stable than levels. |
| 10 | **Noise-grid replicate (seed 7)** | 2,071 events | Content arms unaffected (1.00-1.08x SE inflation); `wrongtraj` SE understated 1.94x; per-event reliability r=0.787. |
| 11 | **`ode_core.py` analytic self-test** | -- | Integrator + divergence validated in closed form (`v=ax`, `v=c`, `v=a*t*x`). Caught a real sign error. |
| 12 | **Channel variance normalisation** | -- | Accel residual is ~1.8x curvature (the raw 3.2x was mostly target scale). Split reported separately because a flat mean over 128 cells weights accel ~3:1 by accident. |
| 13 | **Split composition** | -- | Curvature `var(x) = 0.19` against a normaliser expecting 1.0 -- independent confirmation of README's ~92% "Continue straight". |
| 15 | **ADE gate on `wrongtraj`** | 72 (surrogate), 194 (exact) | **Splits by estimator.** On the SURROGATE the gate does nothing: paired, -0.82, Wilcoxon p=0.12, wrong direction. On the EXACT map it clearly helped -- `wrongtraj` went from mean +31 / trimmed +53 (a 1.7x spread signalling heavy negative outliers) to mean +58 / trimmed +61, essentially converged. That makes sense: an exact density is far more sensitive to a near-duplicate target than an MSE gap is. Caveat: the PAIRED difference is n.s. (n=44), so part of the shift is which events pass the filter. Gate retained in both, since the yardsticks must match. |
| 16 | **Per-event reliability, both heads** | 2,062 | Token head **+0.818**, FM **+0.872** (cells) / **+0.802** (donors) / +0.787 (noise grids). Neither head is noise at the event level -- which is what makes the cross-head test informative, and what refuted the earlier retraction. |
| 17 | **`flipdir` builder** | -- | Reproduces 12/12 hand-verified directive flips exactly. Only the FIRST, directive occurrence is flipped (case n2 keeps "the left lane" intact); a global swap would be a weaker, different manipulation. Token-count guard drops any edit that changes prefix length. |
| 14 | **`CLAUDE.md` rewrite** | -- | "Check which box you are on first", with the verified torch/nvrtc inventory per venv. The prior B300 claim was stale; this is an 8x A100 box. |

---

## THE EXACT MEASUREMENT: 10-step discrete-map likelihood (COMPLETE, n=208)

The deployed sampler is a deterministic 10-step Euler map, and a deterministic diffeomorphism has
an exact log-density by change of variables at each step -- **no variational slack, no Monte
Carlo**. This is the measurement that supersedes the surrogate's levels.

    log p(y) = log N(x_0; 0, I) - sum_k log|det( I + dt * J_v(x_k, t_k) )| ,   x_0 = T^-1(y)

Rows are filtered on round-trip < 1e-3 and neg_sign = 0; **58% pass** (475/824).

| arm | n | mean | SE | trimmed | per cell | surrogate x128 | p |
|---|---|---|---|---|---|---|---|
| `donor` | 111 | **-0.16** | 0.35 | -0.32 | -0.001 | +1.90 | **0.35** |
| `blankvision` | 101 | +12.50 | 1.96 | +12.29 | +0.098 | +82.60 | 1.0e-7 |
| `wrongtraj` | 73 | +31.25 | 28.85 | +52.81 | +0.244 | +418.00 | 4.4e-7 |

**1. The content term is a TRUE ZERO.** p=0.35, and the surrogate's +1.90 is excluded at 5.7 SE.
The one place the two heads appeared to disagree -- the surrogate's small-but-highly-significant
content term against the token head's robust zero -- was variational slack. The heads agree.

**2. Normalised by each head's own yardstick, the heads line up.** This is the only cross-head
comparison that is valid, since absolute levels are not calibrated:

| | token head | FM exact | FM surrogate |
|---|---|---|---|
| cameras / `blankvision` | 34% | **23-40%** | 19.8% |
| CoC content | ~0% | **-0.5% (n.s.)** | +0.45% |

**3. The surrogate inflates every arm 6-8x**, consistently -- the slack-asymmetry mechanism from
`fm_llr_method.md` section 9, measured rather than hypothesised.

**Caveats.** `wrongtraj` is badly behaved: mean +31 but trimmed +53, SE 28.9, i.e. large negative
outliers (events where a wrong trajectory scores *higher* than the true one). That is what makes
the camera share a range (23-40%) rather than a number, and it is the main thing still worth
understanding. 42% of rows fail the convergence filter; a selection check on `|x0|` came back
clean (Mann-Whitney p = 0.79-0.88).

---

## IN FLIGHT

| what | status |
|---|---|
| **fp32 end-to-end discrete map**, same 208 events, ADE-gated `wrongtraj` | launched 2026-09-24 18:45, 4 workers x 2 GPUs, ~23 h. VLM fp32 on cuda:0 (32.8 GiB), expert fp32 on cuda:1 (8.5 GiB). Paired against the bf16 run above, so it measures the bf16 KV contribution directly. `donor` is the arm most exposed to it. |
| **`flipdir` arm** (single-word directive flip) | **built and tested, not yet run.** Reproduces 12/12 hand-verified flips from `flip_cases_clean.json` exactly; **828 of 2,077 events (40%) are flippable** (622 lateral, 206 longitudinal) -- 69x the hand-verified set. Launch when the fp32 run frees the GPUs. |

---

## ABANDONED, and why (do not redo these)

- **Exact continuous-time probability-flow ODE likelihood.** Built, debugged, and dropped.
  1. **It targets the wrong distribution.** The deployed sampler is `num_inference_steps = 10`,
     `int_method = euler` (verified in the checkpoint config). The continuous-time density is the
     n->infinity limit -- a distribution the car never samples from. Our own numbers put the two
     >16 nats apart. No solver fixes this.
  2. **The contrast is noise-limited, not solver-limited.** RK4 converged the LEVEL to ~24-25
     with as few as 8 divergence evaluations, but the contrast scattered -0.30 to +3.55 with sign
     flips and no trend. Run-to-run spread at fixed config: 0.029-0.036 nats, ~36x the effects of
     interest. `null` on the surrogate, by contrast, is exactly 0.
  3. Measured convergence orders 1.0 / 1.9 / 3.9 for euler / heun / rk4 -- Euler needs 2,560
     divergence evals for 1e-4 where RK4 needs 20.
  4. The literature never uses Euler for likelihood (standard is dopri5 @ rtol=atol=1e-5,
     100-1000 NFE), and nobody does it in bf16.
- **Hutchinson's trace estimator** -- rejected for variance, correctly, but moot now.
- **fp32-vs-bf16 ODE precision check** -- bf16 repeats gave 0.97390 / 1.00307 (spread 0.029 nats,
  confirming ODE noise). The fp32 arm failed on a dtype mismatch and was not pursued once the ODE
  left the critical path.

**Bugs found along the way, worth not repeating:** a sign error in the divergence accumulation
(the loop walks t: 1->0 so every dt is negative; the integral must be SUBTRACTED) that produced
clean, plausible, wrong numbers for hours; an autograd-graph leak from not detaching the running
divergence sum (78 GB); `use_cache=True` materialising the image-heavy prefix once per batch row;
forward AD rejecting stride-0 `expand` views (needs `.contiguous()`) and being unimplemented for
efficient SDPA (needs `SDPBackend.MATH`). Only the memory ones crashed. The sign error did not --
which is why `ode_core.py` now carries an analytic self-test that would have caught it in seconds.

---

## NEXT

1. **Exact 10-step discrete-map likelihood** (in flight). 100-200 events, fp32, arms
   `gold`/`donor`/`wrongtraj`/`blankvision`. This is the only measurement that has **no
   variational slack** and targets the density the car actually samples from. If it reproduces
   the ratios, the surrogate is vindicated; if separation collapses toward ~0.8, we have found
   the inflation.
2. **Solver yield on the exact arm.** 41% of rows fail the convergence filter, and rows needing 8
   Newton iterations cost 1184 s against 232 s for rows needing 0. Improving convergence is worth
   more than more events on BOTH axes -- it is ~2x cheaper per usable row AND removes the main
   residual bias worry. (A selection check on `|x0|` came back clean: Mann-Whitney p = 0.79-0.88.)
3. **Matched donor pools.** Exact separation is +0.429 +/- 0.080 nats/cell against the token
   head's +0.775 -- a 1.8x gap at 4.3 SE. Leading suspect: our `wrongtraj` pool is a rolling
   buffer over a ~92%-straight split, so many "wrong" trajectories are near-duplicates. No amount
   of n answers this; matching the pools does.
4. **Scale the exact arm** only after 2 and 3. Measured cost is 0.39 GPU-h/event, so the full
   2,071 events is ~815 GPU-h (4.2 days on 8 GPUs). Both headline claims are already decided at
   current n (the surrogate's donor value is excluded at 5.7 SE), so this buys precision, not
   conclusions.
5. **Fix the shared eps grid**: move the draw inside the event loop.
6. ~~Nav in-run yardstick~~ -- **dropped**, see the nav section.
7. ~~lambda convergence~~ -- largely moot for the headline now that the exact arm supersedes the
   surrogate's levels; still relevant if a surrogate number is ever quoted on its own.

## Caveats to write into any writeup

- A difference of two ELBO bounds is not a bound on the difference; the slack differs per
  condition, and it plausibly differs in the SAME direction as the effect (worse conditioning =>
  worse denoiser => looser bound => inflated gap). Song et al. (2101.09258) Table 2 puts the
  ELBO-vs-exact gap at 0.10-0.13 bpd on CIFAR-10. **NEXT item 1 bounds this empirically.**
- The bound is on the SDE model, never the ODE one (Albergo et al. 2303.08797, Remark 2.11).
- Cross & Ragni (2409.06364) find conditional-diffusion likelihood largely agnostic to
  conditioning, i.e. it may measure nuisance structure. The `shuffled` arm is the pre-emption.
- Absolute nats levels are not trustworthy (lambda non-convergence + slack asymmetry). Ratios to
  the in-run yardstick are.
