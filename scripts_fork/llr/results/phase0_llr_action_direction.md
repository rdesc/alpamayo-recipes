# Phase 0 — action-direction LLR (langforce_readme.md §4.2), full OOD reasoning split

> ## ⚠️ RETRACTED: the headline `+0.227 nats` was an artifact of our own ablation
>
> This document previously reported `llr_act = +0.227 nats` for Alpamayo 1.5 (and
> `+0.037` for Alpamayo 2 Super) as evidence that Chain-of-Causation reasoning is
> load-bearing for trajectory prediction, at 84σ from zero with 94.8% of events
> positive. **That conclusion does not survive.** Roughly `+0.27` of the `+0.227`
> came from two delimiter tokens whose structural markers the ablation had deleted.
> With a denominator that is actually in distribution, the effect is **~0 nats**.
>
> The retracted numbers are preserved in "Appendix: the retracted run" below, because
> the decomposition of *how* they arose is the useful part. The corrected measurement
> is `phase0_llr_per_token.py --denominator empty`; this script,
> `phase0_llr_action_direction.py`, is superseded and kept only as the record.

## What this measures

`langforce_readme.md` §4 asks, before any training: is the model's Chain-of-Causation
reasoning already load-bearing for its trajectory prediction, or decorative? The
language-direction form (§4.1) was tried first and found broken — see
`phase0_llr_ordering_control.py` — its `[v, a*, cot]` reordering put the trajectory in
a sequence position the model was never trained on, and a wrong-trajectory control
showed the resulting "signal" was ~95% positional-novelty artifact.

The action-direction form (§4.2) needs no reordering, keeping the model's natural
trained order `[v, cot, traj_future]` in both passes:

```
llr_act = mean_a[log p(a* | v, ℓ)] − mean_a[log p(a* | v)]
```

- **Numerator:** real gold CoC (`ℓ`) teacher-forced, score the ground-truth trajectory
  tokens (`a*`).
- **Denominator:** the CoC *prose removed*, markers retained — the empty-CoC sequence
  the validated `--no_coc` eval arm produces. This is a genuine `p(a*|v)`.

## The correction, in three parts

### 1. The scored span is not the trajectory

`loss_future_traj` is a single `reduction="mean"` over the model's `traj_mask`
(`sft_base_model.py:623`). That mask holds **178** tokens per event, not 128, in three
roles that behave nothing alike:

| count | role | why |
|---|---|---|
| 48 | **history** trajectory tokens | History and future trajectory tokens are drawn from the *same* token-id block (both start at `traj_token_start_idx`), so the mask's id-range test (`sft_base_model.py:660`) catches all `tokens_per_history_traj` too. |
| 2 | `traj_future_start` / `traj_future_end` | Explicitly OR'd into the mask at lines 665–666. |
| 128 | the actual future trajectory | The only ones we meant to score. |

The name is misleading: `loss_future_traj` is not a loss over the future trajectory.

### 2. The old ablation deleted the reasoning block's delimiters

`get_label_mask(input_ids, tokenizer, ["cot"])` returns a span that is **inclusive of
both markers** — `get_label_mask.py:45` is `labels_mask[start : end + 1] = True`, and
the NOTE above it says so explicitly. Pad-blanking "the cot span" therefore overwrote
`<|cot_start|>` and `<|cot_end|>` with `<|endoftext|>`, leaving
`<|traj_future_start|>` to follow a *document-boundary* token — a sequence the model has
never seen.

Measured per token role, the consequence is unambiguous:

| blanked span | delimiter LLR | future-token LLR |
|---|---|---|
| whole span (markers deleted) — **the old ablation** | **+23.63** | −0.169 |
| prose only (markers kept) | **+0.000004** | +0.0053 |
| prose removed (`empty`) — true `p(a*\|v)` | n/a (lengths differ) | **−0.0072** |

With gold CoC the delimiters score `log p` of *exactly* 0.0 — probability 1.000 — while
blanking drops them to −18 and −28 nats. At ~+23.6 each, 2 tokens out of 178 contribute
`2 × 23.63 / 178 = +0.265` nats, **more than the entire reported effect**.

### 3. Where the old spread came from

The retracted run had mean `+0.227`, std `0.124`, 94.8% of events positive. Decomposing
per event (9 events, from `phase0_llr_per_token.py --blank-mode span` — **that mode has since been
deleted from the script**, so this decomposition is a historical record and is no longer
re-runnable; it overwrote the marker-inclusive cot span, destroying `<|cot_end|>`):

| component | mean | std |
|---|---|---|
| delimiter contribution | **+0.2656** | **0.0123** |
| future-token contribution | −0.1213 | 0.5492 |
| total (the old metric) | +0.1443 | 0.5564 |

The delimiters were **saturated** — `logp_gold ≡ 0.0` exactly, `logp_blank` wobbling only
between −18 and −28 — so scaled by 2/178 they acted as a near-constant **+0.27 offset**
(std 0.012). *All* dispersion came from the 128 future tokens being scored against a
corrupted prefix; that variance measured how badly the broken context happened to hurt
each event, not anything about reasoning. This also explains the 94.8%-positive figure:
on the full split the future-token part had std ≈0.12 about mean ≈−0.04, so a constant
+0.27 lifted nearly the whole distribution above zero. A near-universal positive result,
manufactured by an offset.

(The 9 events above show std 0.55 rather than 0.12 only because they were min/mean/max-selected
*by the broken metric*.)

### Why blanking was chosen, and why that was wrong

The original rationale — written into this doc and the script docstring as a considered
trade-off — was that numerator and denominator must have byte-identical token positions,
since §4.1 had died of a 3.3–3.5 nat *positional* confound. Padding was the only way to
neutralize the CoC at fixed length.

Both halves of that were wrong:

- **The constraint didn't apply.** §4.1's artifact came from **reordering components** —
  placing the trajectory *before* the reasoning, an ordering that occurs zero times in
  training. Splicing the CoC out preserves component order and only shortens one span.
  The model sees variable-length CoC on every training step, so `traj_future` legitimately
  appears at many absolute positions; there is no fixed positional expectation to violate.
- **The cure was worse.** Because the span includes the markers, padding injected a 23-nat
  artifact — roughly 7× the confound it was avoiding.

**Corrected guidance:** for *scoring*, use the empty denominator (`--denominator empty`), which is
in-distribution. If a fixed-length denominator is needed for some other reason, blank only
the prose *interior* and never the markers, and score the 128 future tokens only.

## Full-split result, and why the estimator matters

`phase0_llr_per_token.py`, full OOD reasoning split (train+val), 2,071 of 2,077 events,
265,088 future trajectory tokens, per-event means. Two denominators over an **identical event
set**; `logp_gold` is bit-identical between the runs (max |diff| = 0.000e+00 over all 265,088
tokens), so the two columns are directly comparable.

| | 1.5 `gold − empty` (PMI) | 1.5 `gold − donor` (content) | A2S `gold − empty` |
|---|---|---|---|
| mean | +0.0010 (0.88σ) | **−0.0037 (−3.92σ)** | +0.0179 (15.41σ) |
| median | +0.0021 | +0.0009 | +0.0152 |
| 10% trimmed mean | **+0.0029** | **−0.0004** | +0.0171 |
| bootstrap 95% CI | [−0.0012, +0.0031] | [−0.0055, −0.0018] | [+0.0339, +0.0401]\* |
| fraction > 0 | 54.4% (sign p=**6.3e−5**) | 52.4% (sign p=0.028) | 69.2% (sign p=3.0e−70) |
| Wilcoxon signed-rank | p=**6.2e−5** | p=**0.99** | p=2.8e−72 |
| skew | −0.87 | −1.37 | **+0.04** |

\* A2S CI computed on the superseded per-event scalars; its corrected per-token mean/median/
trimmed figures are the ones in the row above.

**Read the mean and the robust estimators together — they disagree, in opposite directions.**

- **`gold − donor` is zero, not negative.** The mean says −0.0037 at −3.9σ, which would read as
  "gold reasoning is *worse* than another event's". It is an artifact of a left tail: the median
  is **positive**, the trimmed mean is −0.0004, and Wilcoxon gives **p = 0.99** — no
  distributional shift at all. Quote the mean here and you publish a spurious 4σ effect.
- **`gold − empty` is small but not zero.** The published "0.9σ, no measurable dependence" rests
  on the mean, and the *same* left skew understates it. By rank tests the center is robustly
  positive: median +0.0021, trimmed mean +0.0029, and 54.4% of 2,071 events positive at
  **p = 6.3e−5**.
- **A2S needs no such care.** Its distribution is symmetric (skew +0.04) and every estimator
  agrees, which is independent support that its effect is real rather than tail-driven.

### What this actually establishes for 1.5

Having *something* in the reasoning slot is worth a small, statistically solid amount. **What the
reasoning says is worth nothing.** Another event's prose does exactly as well as the correct
prose (`gold − donor`: Wilcoxon p=0.99, trimmed −0.0004). So the measurable part of `llr_act` is
**occupancy, not content** — which is what the n=4 pad probe suggested, now confirmed at
n=2,071.

> **Later qualification (2026-09-21).** "Occupancy, not content" survives — indeed the filler
> sweep strengthens it, since meaningless filler scores at least as well as gold at matched
> length. What does **not** survive is the natural reading of *why*. The filler dose-response is
> an inverted U, not a slope, and one arbitrary token already recovers about twice the whole
> gold-vs-empty effect. Most of the "occupancy" term is the cost of an empty block the model
> treats as near-impossible (−14.41 nats), not a benefit that accrues with positions. See
> "Three later designs" below.

In scale terms the occupancy term is ~0.15% of the −1.932 baseline NLL and ~1% of what the
cameras are worth (+0.266 wrong-vision). The correct one-line summary is therefore not "reasoning
does nothing" but "**the reasoning's content does nothing; the slot's presence is worth ~0.1% of
the baseline**".

### A2S: measured at scale, and it is also occupancy

The donor arm has now been run on Alpamayo 2 Super too (n=2,070, K=2, same script logic). **A2S's
content term is zero as well.**

| A2S | mean | median | 10% trimmed | frac>0 | sign p | Wilcoxon p | skew |
|---|---|---|---|---|---|---|---|
| `gold − empty` | +0.0179 | +0.0152 | +0.0171 | 69.2% | 3.0e−70 | 2.8e−72 | +0.04 |
| `gold − donor` | −0.0026 (−2.7σ) | **+0.0010** | **+0.0003** | 51.6% | 0.15 | **0.58** | −2.05 |

Content is ~2% of the total. The mean's −2.7σ is a skew artifact for the third time in this study
(skew −2.05, Wilcoxon p=0.58) — quoting it would say A2S's gold reasoning is significantly
*worse* than a stranger's.

**The two models differ in magnitude, not in kind:**

| | presence (robust) | content |
|---|---|---|
| 1.5 | +0.0029 (p=6.3e−5) | 0 (p=0.99) |
| A2S | +0.0171 (p=3.0e−70) | 0 (p=0.58) |

**The ~63% content figure is now refuted, not merely unestablished.** It came from n=4; a 6-event
smoke agreed at ~72%; the full split says ~2%. Two small samples pointing the same way were not
corroboration — they were drawn the same way and wrong the same way.

**The channel asymmetry is an occupancy effect, not a reasoning one.** A2S's accel/curvature split
(+0.0265 / +0.0093) was read as the reasoning's longitudinal content being used. On the donor arm
the channels are −0.0043 / −0.0010, so the asymmetry lives entirely in the presence term. The
likelier account is that acceleration is the harder prediction and benefits more from extra prefix
compute.

**And the flip probe was right.** Inverting "left"→"right" cost nothing on either model (retained
~104%/105%, n=4). That was treated as a curiosity against the donor probe's ~63%; at scale the
flip probe was the honest signal and the donor n=4 was noise.


## Is the presence term even about the trajectory?

The presence term survives every content control, so the remaining question is whether it is a
*prediction improvement* at all, or just a generic likelihood shift. Measured by holding a
**wrong** trajectory fixed as the teacher-forced target and varying only the conditioning
(`phase0_llr_per_token.py --target donor`, n=2,062, donor futures gated to ADE >= 5 m; achieved
median ADE 17.2 m, max 106.7 m):

| | trimmed | Wilcoxon |
|---|---|---|
| presence on the **GT** target | +0.0028 | 9.9e-05 |
| presence on a **wrong** target | **+0.0017** | **0.014** |
| paired difference (GT - wrong) | +0.0014 | **0.106 (n.s.)** |

Sign test p=0.50, 50.8% of events positive. **A filled reasoning span raises the likelihood of a
trajectory the scene does not support**, significantly, and we cannot distinguish that from its
effect on the correct trajectory.

**This is underpowered, not proven zero.** The paired per-event SD is 0.048 against a difference
of 0.0014, so resolving it at 80% power needs ~9,000 events -- 4.5x what the split provides. The
honest statement is "no detectable target-specificity", not "none exists". Point estimate of the
generic share is 60%, bootstrap 95% CI [5%, 116%] -- uninformative on its own.

Distinct from the `wrong_traj` control (+1.07), which varies the TARGET at fixed conditioning.
This varies the CONDITIONING at a fixed wrong target.

**Taken with the other two results, the presence term looks like a generic likelihood shift:**

| property | evidence |
|---|---|
| not content | donor arm zero on both models (p=0.99 / p=0.58) |
| scales with token count | +0.0012 (1.5) / +0.0016 (A2S) nats per token, 7.0 / 8.8 sigma |
| not demonstrably about the trajectory | wrong-target effect +0.0017 (p=0.014); paired diff n.s. |


### How the model actually discriminates, and why per-token means mislead

The `--target donor` run also gives the **separation** -- `log p(true traj) - log p(foreign traj)`
under fixed conditioning. This is the model doing its job, and it is the scale against which every
reasoning effect should be read.

Mean separation is **+0.81 nats/token** (a 2.25x probability ratio, or 10^45 over the full
128-token sequence). **But no token is typical of that mean:**

| position | separation | ratio |
|---|---|---|
| first token (0.1 s) | **10.16** | ~26,000x |
| 0.0-0.8 s | 3.98 | |
| 0.8-1.6 s | 0.98 | |
| 1.6-2.4 s | 0.38 | |
| 5.6-6.4 s | 0.18 | |
| **median position** | **0.29** | 1.33x |

The top 10 of 128 positions carry **56%** of the total (uniform would be 8%). By channel,
acceleration separates at 1.15 nats and curvature at 0.47.

The mechanism is straightforward: the first trajectory token is the immediate acceleration, which
is pinned by the current ego state the model already has from the history. A foreign trajectory
starts at a different speed, so it is rejected almost instantly. Past a second or two the real
future is genuinely uncertain, and the gap collapses to 0.1-0.2 nats.

### Everything on one scale

The separation is the natural denominator: it is what the model gains from knowing which
trajectory is actually right. Expressing every measured effect as a fraction of it (1.5's
separation, trimmed: **+0.7748** overall, +1.1044 accel, +0.4557 curvature):

| effect | nats/token | % of separation | status |
|---|---|---|---|
| wrong vision (n=10) | +0.2661 | **34%** | the cameras |
| CoC presence, A2S | +0.0171 | 2.2% | **OOD denominator -- suspect** |
| nav, turns, curvature channel | +0.0030 | 0.66% | vs. the *curvature* separation |
| CoC presence, 1.5 | +0.0029 | 0.37% | **OOD denominator -- suspect** |
| nav, overall | +0.0018 | 0.23% | clean |
| sharpening from reasoning | +0.0014 | 0.18% | n.s., p=0.11 |
| CoC content, A2S | +0.0003 | 0.04% | clean, zero |
| CoC content, 1.5 | -0.0004 | -0.05% | clean, zero |

Read as a sentence: **knowing which trajectory is right is worth 1.0; the cameras are worth about
a third of that; the navigation command about a fifth of a percent; and the reasoning's content
about a twentieth of a percent, i.e. nothing.** The two presence rows sit between them but are
measured against a denominator the model considers impossible, so they are not safe to quote.

Caveat: the separation is 1.5's, measured on the `--target donor` run. No equivalent run exists
for A2S, so the A2S row borrows 1.5's denominator and should be read as indicative only.

**Two consequences.**

1. **Do not describe +0.81 as a per-token property.** It is a sum divided by 128, useful for
   arithmetic, not a description of a typical token. The honest statement is "the model rejects a
   foreign trajectory almost entirely on its first token, and is nearly indifferent by 5 s out."
   Sequence-level totals are unaffected -- sums do not care about the distribution.
2. **Reasoning does not touch this structure.** The separation curves with and without the
   reasoning span lie on top of each other at every horizon (see `separation_horizon.png`).
   Overall sharpening is +0.0014, p=0.11. The only visible deviation is a small *negative* dip at
   the very first token (-0.08), i.e. if anything the span slightly blunts the one place where
   discrimination actually happens.

## Sanity controls

A null result only means something if the instrument demonstrably responds to things it
should. `phase0_llr_sanity_controls.py`, 10 events, all through the identical scoring path:

| control | mean (nats) | expected | verdict |
|---|---|---|---|
| `null_identical` — same CoC, scored twice | **0.0000** | == 0 | PASS |
| `null_retokenize` — fresh tokenize, same inputs | **0.0000** | == 0 | PASS |
| `history_llr` — the 48 history tokens | **0.0000** | == 0 | PASS |
| **`empty_llr` — remove reasoning** | **−0.0062** (std 0.0185) | — | the result |
| `wrong_coc_llr` — another event's real prose | −0.0081 | ? | ~same as removing it |
| `no_vision_llr` — blanked camera frames | **+0.1017** | > 0 | 16× reasoning |
| `wrong_vision_llr` — another scene's frames | **+0.2661** | > 0 | **43× reasoning** |
| `wrong_traj_llr` — another event's trajectory | **+0.7748** | >> 0 | 125× reasoning |

> The wrong-trajectory row quotes the **full-split separation** (n=2,062, 10% trimmed; raw
> +0.8094). The n=10 ladder's own figure was +1.0680 with std 0.71 and one event at −0.21 —
> unstable because an ungated donor is often a near-duplicate on a split that is ~92%
> "Continue straight". **Quote +0.7748, not +1.0680.**

The three nulls landing at *exactly* 0.0 rules out position/shift misalignment anywhere in
the scoring — the history-token null especially, where causal attention makes zero a
structural invariant (those tokens precede the cot span, so dependence is impossible). And
misleading vision costs 0.27 nats through the same code path that scores reasoning at
−0.006, so the probe is not insensitive: reasoning is simply not something it responds to.

Reference `log p(a*)` per future token is **−1.788**, not the −7.3 previously quoted — that
figure was inflated by the high-entropy history tokens, so the old "LLR is 3% of baseline
NLL" calibration was wrong in both numerator and denominator.

## Result

`phase0_llr_per_token.py --denominator empty`, full OOD reasoning split (train+val), 8 shards,
2,071 / 2,077 events ok (6 per-clip streaming/interpolation failures, same class as before),
265,088 scored future-trajectory tokens.

| | retracted | **corrected** |
|---|---|---|
| mean `llr_act` | +0.227 | **+0.0010** |
| median | +0.248 | +0.0021 |
| std | 0.124 | 0.0509 |
| min → max | −0.586 → +0.751 | −0.3269 → +0.2594 |
| fraction > 0 | 94.8% | **54.4%** |
| distance from 0 | 84σ | **0.9σ** |

**0.9σ on n = 2,071 — statistically indistinguishable from zero.** The fraction of events with
positive LLR is a coin flip. For scale, the effect is 0.05% of the −1.932 baseline NLL per future
token, and **240× smaller than the wrong-vision control** measured through the same code path.

### By split

| split | mean | median | std | n |
|---|---|---|---|---|
| train | +0.0006 | +0.002 | 0.0532 | 1,722 |
| val | +0.0030 | +0.003 | 0.0377 | 349 |

### By event cluster

Previously *all nine* clusters were positive at 0.161–0.295. They now straddle zero:

| cluster | mean | median | n |
|---|---|---|---|
| ROAD_DEBRIS_OR_SAFETY_TRACES | −0.0131 | −0.0078 | 22 |
| ANIMALS_BIRDS_ROADKILL | −0.0075 | +0.0014 | 29 |
| SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR | −0.0061 | −0.0002 | 303 |
| PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY | −0.0032 | +0.0027 | 433 |
| OTHER_LONGTAIL | −0.0029 | +0.0005 | 42 |
| COMPLEX_INTERSECTION_INTERACTION | −0.0007 | +0.0003 | 57 |
| EMERGENCY_INCIDENT_SCENE | +0.0001 | −0.0001 | 32 |
| WORK_ZONES_TEMP_TRAFFIC_CONTROL | +0.0048 | +0.0032 | 1,054 |
| CYCLISTS_AND_MICROMOBILITY_COMPLEX | +0.0093 | +0.0046 | 99 |

No cluster is meaningfully non-zero, and the sign is not ordered the way a real effect would be
(the largest cluster, work zones at n=1,054, sits at +0.005).

### Horizon profile and control channel

Flat. Every 0.8 s bin is within ±0.007 nats, with no trend toward the long horizon — so there is
no long-horizon payoff hidden inside the average:

| t (s) | accel | curvature |
|---|---|---|
| 0.0–0.8 | +0.0037 | +0.0024 |
| 0.8–1.6 | −0.0036 | −0.0032 |
| 1.6–2.4 | +0.0031 | +0.0011 |
| 2.4–3.2 | +0.0003 | +0.0020 |
| 3.2–4.0 | −0.0044 | −0.0018 |
| 4.0–4.8 | −0.0020 | +0.0043 |
| 4.8–5.6 | −0.0017 | +0.0040 |
| 5.6–6.4 | +0.0034 | +0.0073 |

Channel totals: accel **−0.0001** (n=132,544), curvature **+0.0021** (n=132,544). So reasoning
does not preferentially inform longitudinal control ("stop for the pedestrian") over lateral
("merge left"), which was the obvious hypothesis to check — neither channel shows an effect.

### Alpamayo 2 Super, re-measured: the opposite answer

A2S was re-run on the same corrected basis, and it does **not** follow 1.5. Full details in
`~/repos/alpamayo2/examples/llr/README.md`.

| | Alpamayo 1.5 | Alpamayo 2 Super |
|---|---|---|
| corrected `llr_act` | +0.0010 | **+0.0179** |
| distance from 0 | 0.9σ | **15.4σ** |
| fraction > 0 | 54.4% | 69.2% |
| accel / curvature | −0.0001 / +0.0021 | **+0.0265 / +0.0093** |
| content vs. presence | no content signal (n=10) | ~63% content, **n=4 — not established** (+0.0110 of +0.0176) |
| vs. its own wrong-vision control | 0.4% | 6% |

**A2S never had either of 1.5's defects**, which is why its published number was roughly the right
order to begin with. Verified against the loaded model, not inferred: its history and future
trajectory tokens occupy *disjoint* id blocks (`[151669,152669)` vs `[152669,155669)`), so its mask
holds **0** history tokens rather than 48; and its ablation blanked only the span strictly between
the markers, so `<|cot_start|>`/`<|cot_end|>` survived. Its mask is 130 tokens, dilution 1.016
against 1.5's 1.391.

What A2S *did* have is a third contamination, in the delimiters, which 1.5 did not exhibit once its
markers were preserved: they move up to 3.95 nats under interior pad-blanking and ~−1.1 nats under
splicing — enough that the 130-token mask-mean reads ~+0.000 while the 128 real trajectory tokens
sit at +0.018. The lesson generalizes: **exclude the delimiters regardless of denominator**, rather
than trying to pick a denominator they tolerate.

So the two models genuinely differ: 1.5's reasoning is decorative, A2S's is load-bearing at a
small but unambiguous level, and on A2S about two thirds of that comes from the reasoning being
*correct for the scene* rather than merely occupying the span. Caveat: A2S uses a 6-camera profile
and a different training recipe, so each is measured against its own baseline and the ratio between
them is not controlled.

### Reading against the §4.4 decision gate

The honest Phase-0 answer is that Alpamayo 1.5's trajectory prediction shows **no
measurable dependence on its Chain-of-Causation reasoning** — `|llr| < 0.01` nats against
controls that register 0.1–1.1 nats. The horizon profile is flat as well: no accel-vs-curvature
asymmetry and nothing beyond ±0.11 in any 0.8 s bin, so there is no long-horizon payoff
hiding inside the average.

This is a *stronger* case for Phase 1 than a positive result would have been. If reasoning
currently buys ~0 nats, then `L_total = L_SFT − β·L_LLR` would be **creating** a dependence
rather than reinforcing one that already exists — which is what the regularizer is for, and
the corrected `llr_act` is exactly the quantity it should move. Note §2.1's gradient-trivial
trap applies to the training objective, not to this measurement.

Caveat carried forward: all of this reaches Stage 1's discrete-token pathway. The *deployed*
head is Stage 2's diffusion/flow-matching expert, which exposes no trajectory-token logits;
reaching it needs §8's consistency term.

---

## Three later designs (2026-09-21)

The full-split runs compare gold prose against a few random donors per event. That design cannot
separate scene difficulty, CoC main effect, and content, and it cannot tell a length effect from a
composition effect. Three designs since then address those, and **two revised earlier conclusions
in this file.**

### 1. Within-scene donor sweep — the length slope does not survive

`phase0_llr_donor_sweep.py`, 9 scenes × 100 donors, scene and target trajectory held fixed.

| scene | min1 | min2 | min3 | mean1 | mean2 | mean3 | max1 | max2 | max3 |
|---|---|---|---|---|---|---|---|---|---|
| slope, nats/token | +0.0043 | +0.0171 | −0.0277 | −0.0003 | −0.0000 | −0.0004 | +0.0023 | +0.0240 | +0.0007 |

Six of nine are individually significant, and they **disagree in sign**. Pooled with scene fixed
effects: +0.00253, t=+3.29, **R² = 1.2%** — and unstable, since on the first four scenes it was
−0.0007. The across-scene +0.00122 therefore reflected *which scenes get long annotations*, not
tokens buying likelihood.

The same run gives the regression-to-the-mean control:

| bucket | own reasoning | mean over 100 donors | shift |
|---|---|---|---|
| min | −0.3077 | −0.1998 | **+0.1079** |
| mean | +0.0010 | −0.0000 | −0.0010 |
| max | +0.2465 | +0.1810 | **−0.0655** |

Arbitrary prose lifts the bottom, drops the top, leaves the middle alone — **more** than the
directive flips did at either end. Any conclusion drawn from tail-selected events is measuring
this.

### 2. Filler dose-response — the presence term is the empty control's cost

`phase0_llr_filler_dose.py`, 70 scenes × 14 doses × 2 filler kinds, scene and target fixed.
Content is zero **by construction**: filler cannot say more as it lengthens.

| n filler | 1 | 8 | 16 | 24 | 32 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|
| repeated word | +0.0041 | +0.0056 | +0.0084 | **+0.0106** | +0.0100 | −0.0017 | −0.0253 | −0.0546 |
| pad token | −0.0001 | −0.0033 | +0.0020 | +0.0043 | +0.0043 | −0.0087 | −0.0319 | −0.0621 |

**An inverted U.** Gold on the same scenes is +0.0022 (se 0.0061, i.e. not itself distinguishable
from zero here). Filler at 16–32 tokens is significantly above the empty block (Wilcoxon p=0.005,
p=0.014); at matched length gold vs filler is a coin flip (−0.0059, p=0.48; +0.0013, p=0.14).

The decisive number is the first: **one arbitrary token is worth +0.0041, about twice the entire
gold-vs-empty effect.** With `log p(cot_end|cot_start)` at −14.41 nats (≈5×10⁻⁷) for an empty
block, the presence term reads as the cost of an alien empty state, not a benefit of reasoning.

Two quoting hazards. A straight line through the curve gives **−0.00039/token at t=−16.5**, a
confident *negative* slope that is pure misspecification — no single slope describes an inverted
U, including the +0.00122 quoted earlier in this file. And real CoCs are 14.9 ± 4.2 tokens, lying
entirely on the rising limb, which is why the across-scene regression looked clean.

The pad arm dips **below** zero at 4–8 tokens before recovering — the in-distribution penalty for
padding inside a reasoning block, and a reason to prefer the repeated-word filler as the control.

### 3. Token-clean directive flips — and a tokenizer trap

The flip probe was written at the string level, but `deceleration` is 3 tokens
(`dec`·`eler`·`ation`) while `acceleration` is 1, so two of the original four flips changed length
as well as meaning. Re-scanning for inversions that are clean by construction — equal token count,
exactly one differing position — only **33 of the 120 most-negative events admit one**.

| token-clean flips | n | mean gold − flipped | true directive won |
|---|---|---|---|
| negative tail, lateral | 9 | −0.013 | 3/9 |
| negative tail, longitudinal | 3 | **−0.118** | 0/3 |
| negative tail, all | 12 | **−0.039** | 3/12 (Wilcoxon p=0.021) |
| positive tail | 4 | −0.015 | 2/4 |

Taken alone this says the model prefers the *false* directive at both ends. It does not: §1's
control shows donors move the same tails further (+0.108 / −0.066). **The flip test has no power
on tail-selected events.** All 12 reproduce their full-split `llr_gold` exactly — under
`a1x_rl_b300`; under the other venv none of them do (see README trap 6).

### What survives

The **content null** is the most robust result here, agreeing across three independent designs:
full-split donor (p=0.99 / 0.58), matched-length gold-vs-filler (p=0.48 / 0.14), and the
within-scene sweep. What does *not*
survive as stated is the length mechanism and the presence term's interpretation.

## Appendix: the retracted run

For provenance. Produced by `phase0_llr_action_direction.py` (superseded) against the full
PAI-AV OOD reasoning subset (`reasoning/ood_reasoning.parquet`, train+val) on
`/mnt/efs/users/rod/ckpts/Alpamayo-1.5-10B-rl-training` via `TrainableReasoningVLA`.

| | Alpamayo 1.5 | Alpamayo 2 Super |
|---|---|---|
| mean `llr_act` | +0.227 | +0.037 |
| median | +0.248 | +0.036 |
| std | 0.124 | — |
| range | −0.586 → +0.751 | −0.325 → +0.532 |
| n (ok / total) | 2,071 / 2,077 | 2,031 / 2,077 |
| by split | train 0.227 / val 0.230 | — |
| event clusters | all 9 positive, 0.161–0.295 | all positive |

Every one of these figures is dominated by the delimiter offset described above. The
"all 9 clusters positive" and "84σ from zero" claims are consequences of a constant
+0.27 shift, not of reasoning content. The 6 failures were streaming/decode errors on
individual clips, unrelated.
