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
> is `phase0_llr_per_token.py --blank-mode splice`; this script,
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
| prose spliced out — true `p(a*\|v)` | n/a (lengths differ) | **−0.0072** |

With gold CoC the delimiters score `log p` of *exactly* 0.0 — probability 1.000 — while
blanking drops them to −18 and −28 nats. At ~+23.6 each, 2 tokens out of 178 contribute
`2 × 23.63 / 178 = +0.265` nats, **more than the entire reported effect**.

### 3. Where the old spread came from

The retracted run had mean `+0.227`, std `0.124`, 94.8% of events positive. Decomposing
per event (9 events, from `phase0_llr_per_token.py --blank-mode span`):

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

**Corrected guidance:** for *scoring*, use the splice (`--blank-mode splice`), which is
in-distribution. If a fixed-length denominator is needed for some other reason, blank only
the prose *interior* and never the markers, and score the 128 future tokens only.

## Sanity controls

A null result only means something if the instrument demonstrably responds to things it
should. `phase0_llr_sanity_controls.py`, 10 events, all through the identical scoring path:

| control | mean (nats) | expected | verdict |
|---|---|---|---|
| `null_identical` — same CoC, scored twice | **0.0000** | == 0 | PASS |
| `null_retokenize` — fresh tokenize, same inputs | **0.0000** | == 0 | PASS |
| `history_llr` — the 48 history tokens | **0.0000** | == 0 | PASS |
| **`splice_llr` — remove reasoning** | **−0.0062** (std 0.0185) | — | the result |
| `wrong_coc_llr` — another event's real prose | −0.0081 | ? | ~same as removing it |
| `no_vision_llr` — blanked camera frames | **+0.1017** | > 0 | 16× reasoning |
| `wrong_vision_llr` — another scene's frames | **+0.2661** | > 0 | **43× reasoning** |
| `wrong_traj_llr` — another event's trajectory | **+1.0680** | >> 0 | 170× reasoning |

The three nulls landing at *exactly* 0.0 rules out position/shift misalignment anywhere in
the scoring — the history-token null especially, where causal attention makes zero a
structural invariant (those tokens precede the cot span, so dependence is impossible). And
misleading vision costs 0.27 nats through the same code path that scores reasoning at
−0.006, so the probe is not insensitive: reasoning is simply not something it responds to.

Reference `log p(a*)` per future token is **−1.788**, not the −7.3 previously quoted — that
figure was inflated by the high-entropy history tokens, so the old "LLR is 3% of baseline
NLL" calibration was wrong in both numerator and denominator.

## Result

`phase0_llr_per_token.py --blank-mode splice`, full OOD reasoning split (train+val), 8 shards,
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

Alpamayo 2 Super's `+0.037` was produced by the same `loss_future_traj` scoring and the
same span-inclusive blanking, so it is **retracted pending re-measurement** on the same
corrected basis. See `~/repos/alpamayo2/examples/llr/README.md`.

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
