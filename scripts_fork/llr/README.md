# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.

# Reasoning–action consistency (LLR) diagnostics

Phase 0 of [`langforce_readme.md`](langforce_readme.md) (the spec, alongside this README): does
Alpamayo's Chain-of-Causation reasoning actually carry information the trajectory prediction
uses, or is it a post-hoc rationalization riding a vision/ego shortcut? Everything here is
**measurement only** — no training. Run against the PAI-AV OOD reasoning split
(`reasoning/ood_reasoning.parquet`), which carries real gold CoC annotations.

### Start here

| if you want… | read |
|---|---|
| the findings, as a page with figures | **[Does Alpamayo use its reasoning -- Claude artifact](https://claude.ai/artifact/DA6NETdxGBUtbQSa6hGYyz))** — open in a browser, self-contained |
| the findings, in prose with every number | [`results/phase0_llr_action_direction.md`](results/phase0_llr_action_direction.md) |
| what to run, and what each arm controls for | the [Scripts](#scripts) table below |
| where the parquets are | [Run artifacts](#run-artifacts--where-the-numbers-actually-come-from) — all on EFS, world-readable |
| the traps that cost the most time | [Traps](#traps-worth-reading-before-touching-any-of-this) — read trap 6 before running anything |

**One-line summary of the discrete-token head:** the reasoning's *content* does nothing —
another event's prose scores as well as the prose written for this scene, across three
independent designs. Something being in the reasoning slot is worth a small amount, but the
filler sweep shows most of that is the cost of an off-distribution empty control rather than a
benefit of reasoning.

The Alpamayo-2-Super counterpart lives in `~/repos/alpamayo2/examples/llr/`. **The
flow-matching (deployed) action head is a separate line of work in this same directory** —
see [`FM_HEAD_STATUS.md`](FM_HEAD_STATUS.md), [`fm_llr_method.md`](fm_llr_method.md), and the
section at the end of this README. Its numbers are **not** comparable to the ones here: the two
heads are measured with different estimators in different units.

## What the measurement is (read this before touching anything)

**Two forward passes and a subtraction.** One pass with the gold CoC prose, one with
`coc_text=""`; difference the trajectory-token log probs.

```
llr_act = log p(a* | v, ell_gold) - log p(a* | v, ell_empty)
```

There is nothing clever in the denominator. `coc_text=""` re-tokenizes normally and is the same
sequence the already-validated `--no_coc` eval arm produces. **If you find yourself reaching for
token surgery, attention masks, or padding to build the denominator, stop** — that is the path
that produced the retracted `+0.227`.

**Vocabulary — one name per thing.** Four names for the denominator ("splice", "empty",
"no_coc", "blank") were used interchangeably across these scripts and docs, which is a large part
of why this kept getting re-litigated. Canonically, and in the code:

| term | what it is | valid `p(a*\|v)`? |
|---|---|---|
| **`empty`** | `coc_text=""`. Prose gone, `<\|cot_start\|>`/`<\|cot_end\|>` kept. **The denominator.** | **yes** |
| **`pad`** | Prose overwritten with pad tokens, markers preserved, length held fixed. Occupancy probe only. | no |
| **`donor`** | Other events' real prose, averaged over `--n-donors` draws (default 5). Content control. | yes |

> ⚠️ **`empty` means something structurally different in the nav experiment.** Here it is
> `coc_text=""` and the `<|cot_start|>`/`<|cot_end|>` markers **remain**. In
> [`results/phase0_llr_nav.md`](results/phase0_llr_nav.md) it is `nav_text=None` and the route
> section is absent **entirely, markers included**. Both are the correct in-distribution
> `p(a*|v)` for their case, and the difference is forced by the code: `construct_route` returns
> `[]` on `None`, whereas the CoC span is marker-inclusive and overwriting its markers is what
> produced the retracted `+0.227`. A marker-retaining nav variant was implemented and then
> **removed**, precisely because training never emits an empty route pair — so for nav there is
> only one denominator and no flag to choose it.

`gold − empty` and `gold − donor` answer **different questions**, and both are legitimate:

- **`gold − empty`** is the PMI, `log p(a*|v,ℓ) − log p(a*|v)`. "Does conditioning on the
  reasoning raise the action's likelihood?" The empty sequence is the model's genuine
  representation of *no reasoning*, so its being shorter is intrinsic, not a confound.
- **`gold − donor`** holds the slot occupied and fluent and varies only *content*. "Is the model
  reading **this scene's** reasoning?" This is the one to quote when the claim is about content.

They differ because an occupied cot span buys likelihood even when its content is worthless —
`pad` tokens recover ~72% of the effect on 1.5's largest-LLR events (n=4, suggestive only).
Donors are deliberately **not** truncated to gold's token length: truncating mid-sentence
degrades fluency, which confounds in the same direction as the effect, whereas untruncated donor
lengths simply vary about gold's in expectation. Averaging over donors is not optional — per-event
σ ≈ 0.05 against an effect ≈ 0.01, so a single donor draw injects donor-choice noise straight into
the estimate.

"Blank" is retired as a term: it describes what the deleted buggy mode did, and the vagueness of
the word is what let the bug through review. The CLI flag is `--denominator {empty,pad,donor}`; it was
previously `--blank-mode {splice,interior,span}`.

**The length asymmetry is not a problem.** The `empty` denominator is shorter than the numerator
(15 tokens on 1.5, 22 on A2S — prose leaving, markers staying). Both passes teacher-force the
same 128 ground-truth trajectory tokens, so their log probs are directly comparable and no
correction is needed. It has exactly one consequence, which is an implementation rule and not a
caveat on the result:

> **Pair trajectory tokens by rank within each pass, never by absolute sequence position.**

Each pass derives its own mask; future-value tokens are then zipped by rank. Reusing the
numerator's positional indices against the denominator's logits would difference gold token *k*
against empty token *k+15* — different targets — for a plausible, meaningless number.

## Headline result

**Two channels, opposite verdicts.** The *reasoning* span's effect is content-free, scales with
token count, and is not demonstrably about the trajectory being predicted — it behaves like a
generic likelihood shift. The *navigation* command's effect is small but directional,
channel-specific, and survives its falsification test. Details below and in
[`results/phase0_llr_nav.md`](results/phase0_llr_nav.md).

| | reasoning (CoC) | navigation |
|---|---|---|
| effect (robust) | +0.0029 (1.5) / +0.0171 (A2S) | +0.0018 |
| content / directional? | **no** — donor ≈ 0 (p=0.99 / 0.58) | **yes** — swap kills it (paired p=0.023) |
| scales with span length? | **across scenes only** — +0.0012–0.0016 /token, 7–9σ, but does not reproduce within a scene, and the filler curve is non-monotonic (see below) | **no** — 5–9× below that rate |
| channel-specific? | no | **yes** — turns → curvature only (p=4.3e−03) |
| target-specific? | **not detectably** — wrong-target +0.0017 (p=0.014), paired diff n.s. | not tested |


**The reasoning's *content* does nothing; the slot's *presence* is worth ~0.1% of the baseline.**
Full OOD reasoning split (train+val, 2,071/2,077 events, 265k tokens), both denominators over an
identical event set:

| | `gold − empty` (PMI) | `gold − donor` (content) |
|---|---|---|
| mean | +0.0010 (0.88σ) | −0.0037 (−3.92σ) |
| median / 10% trimmed | +0.0021 / **+0.0029** | +0.0009 / **−0.0004** |
| fraction > 0 | 54.4% (sign p=**6.3e−5**) | 52.4% (p=0.028) |
| Wilcoxon | p=**6.2e−5** | p=**0.99** |

**Do not quote either mean unqualified — the distribution is left-skewed (−0.87/−1.37) and the
mean misleads in both directions.** `gold − donor`'s −3.9σ is a tail artifact (median positive,
Wilcoxon p=0.99): content is **zero**. `gold − empty`'s "0.9σ null" is understated by the same
skew: by rank tests the center is robustly **positive** at +0.002–0.003. So another event's prose
does as well as the correct prose, while having *something* there beats having nothing — the
measurable effect is **occupancy, not content**, exactly as the pad probe predicted.

Flat across the 6.4 s horizon (±0.007 per 0.8 s bin) and across both control channels
(accel −0.0001, curvature +0.0021).

### Three later designs, and what each one changed

The full-split `empty`/`donor` runs above compare gold prose against a few random donors per
event. Three designs since then control things that comparison could not. **Two of them revised
conclusions stated earlier in this file.**

| design | script | what it controls | result |
|---|---|---|---|
| within-scene donor sweep | `phase0_llr_donor_sweep.py` | scene difficulty (scene + target fixed, 100 donors) | per-scene length slopes **disagree in sign** (−0.028 to +0.024, 6/9 significant); pooled +0.0025, R²=1.2%. The across-scene slope does **not** reproduce within a scene. |
| filler dose-response | `phase0_llr_filler_dose.py` | content, **by construction** (filler cannot say more) | **inverted U**, peaking at n≈24. One arbitrary token is worth +0.0041 — about **twice** the whole gold-vs-empty effect. Filler at 16–32 tokens ≥ gold; matched-length gold-vs-filler is a coin flip (p=0.48 / 0.14). |
| token-clean directive flips | `phase0_llr_flip_coc.py` | tokenization (equal token count, one differing position) | on tail events the **false** directive wins (−0.039, Wilcoxon p=0.021) — but the sweep shows donors move those same tails **further** (+0.108 on min, −0.066 on max), so this is regression to the mean, not directive-reading. |

**Three consequences for how the numbers above should be quoted.**

1. **The length story is bounded, not general.** Real CoCs are 14.9 ± 4.2 tokens, which is exactly
   the rising limb of the filler curve. The across-scene +0.00122 is real *over the range real
   reasoning occupies* and does not generalise. A straight line through the full filler curve gives
   **−0.00039/token at t=−16.5** — a confident *negative* slope that is pure misspecification. No
   single slope describes this relationship, including the +0.00122 quoted above.
2. **The presence term is largely the empty control's cost.** `log p(cot_end | cot_start)` on an
   empty block is **−14.41 nats (≈5×10⁻⁷)** versus −0.04 after real prose. One filler token recovers
   twice the gold effect. Treat `gold − empty` as "gold vs something alien", not "gold vs nothing".
3. **Tail events cannot carry a qualitative argument.** Anything selected on extreme LLR regresses
   under *any* perturbation. Both flip results (positive and negative tails) are explained by this.

The content null itself **survives all three** and is the most robust thing in this file: donor at
full split (p=0.99 / 0.58) and matched-length gold-vs-filler (p=0.48 / 0.14) agree on zero, and the
within-scene sweep adds nothing against it.

**Alpamayo 2 Super is 18× larger — but also occupancy, not content.** Its presence term is
**+0.0179 nats, 15.4σ**, symmetric (skew +0.04) and robust to every estimator (median +0.0152,
trimmed +0.0171, sign p=3.0e−70). But its **donor arm at full split (n=2,070) is zero**: trimmed
+0.0003, Wilcoxon p=0.58, ~2% of the total. So the two models differ in *magnitude, not in kind* —
neither trajectory head uses what its reasoning says.

This **refutes** the earlier ~63%-content figure for A2S (n=4, and a 6-event smoke agreed at ~72%
before the full split said ~2%). It also reinterprets A2S's channel asymmetry: on the donor arm
accel/curvature are −0.0043/−0.0010, so the accel bias is a property of the presence term —
plausibly because acceleration is the harder prediction and gains more from extra prefix compute
— not evidence that longitudinal reasoning content is being read.
Its acceleration channel carries ~3× its curvature channel (+0.0265 vs +0.0093).

Whether that effect is *content* or mere *presence* is **not established**: the wrong-prose arm
suggests ~63% content, but it is **n=4** and has never been run at scale on A2S. Treat it as a
hint, not a result. A2S never had either defect that invalidated 1.5's number; see
`~/repos/alpamayo2/examples/llr/README.md`. (Different camera profile and recipe, so each model is
measured against its own baseline.)

**The natural yardstick is the model's own discrimination.** `log p(true traj) - log p(foreign
traj)` under fixed conditioning is **+0.7748 nats/token** (1.5, n=2,062, trimmed) -- what the model gains
from knowing which trajectory is right. Against that: the cameras are worth **34%**, the
navigation command **0.23%**, and the reasoning's *content* **0.05%**. Full table in the results
doc, "Everything on one scale".

That null is only meaningful because the same code path registers other conditioning changes:

| | nats | vs. reasoning |
|---|---|---|
| 3 null controls | **exactly 0.0** | noise floor |
| **remove reasoning** (`p(a*\|v)`) | **+0.001** | — |
| **wrong** reasoning prose | −0.008 | ~same as removing it |
| blank vision | +0.102 | ~100× |
| **wrong** vision | +0.266 | **240×** |
| wrong trajectory target | **+0.7748** | ~700× |

The wrong-trajectory row is the **full-split separation** (n=2,062, 10% trimmed; raw +0.8094),
not the n=10 ladder. An earlier +1.068 from the 10-event ladder has been retired: that estimate
carried std 0.71 with one event at −0.21, because ~92% of this split is "Continue straight" and an
ungated donor is often a near-duplicate of the true future. Every other row in this table is still
n=10 and should be read as indicative.

A2S's own ladder (n=24): wrong vision +0.2785, wrong trajectory +0.9173, so its reasoning is worth
~6% of its cameras against ~0.4% for 1.5.

> ### ⚠️ The previously published `+0.227 nats` is RETRACTED
>
> It was an artifact of our own ablation. Two causes, both in
> [`results/phase0_llr_action_direction.md`](results/phase0_llr_action_direction.md) in full:
>
> 1. **`loss_future_traj` does not score the future trajectory.** Its span is 178 tokens, not
>    128: plus 48 *history* trajectory tokens (same token-id block, so the mask's id-range test
>    catches them) and the 2 `traj_future_start`/`traj_future_end` delimiters.
> 2. **The ablation deleted the reasoning block's markers.** `get_label_mask`'s cot span is
>    inclusive of `<|cot_start|>`/`<|cot_end|>` (`get_label_mask.py:45`, `start : end + 1`), so
>    pad-blanking it left `<|traj_future_start|>` following an `<|endoftext|>`. The delimiters
>    went from `log p` of exactly 0.0 to −18/−28 nats, contributing a near-constant **+0.27
>    offset** — larger than the whole reported effect — while every bit of the apparent
>    *spread* came from the 128 real tokens being scored against a corrupted prefix.
>
> Alpamayo 2 Super's `+0.037` came from the same scoring path but was **not** affected by either
> defect; re-measured on the corrected basis it is +0.0179 (from an inflated +0.0395 old basis).

## Run artifacts — where the numbers actually come from

Quote a number only if you can point at the parquet behind it.

| run | artifact | status |
|---|---|---|
| 1.5 `empty`, full split (original) | `/mnt/efs/users/rod/results/llr_splice_a15/pt.shard*-of-8.parquet` + `merged.parquet` | complete, 2,071 events |
| **A2S `empty`, full split (corrected)** | `/mnt/efs/users/rod/results/llr_splice_a2s/pt.shard*-of-8.parquet` | complete, 2,071 events — **this is the +0.0179** |
| A2S, full split (superseded) | — | the retracted +0.037 run. **Data deleted 2026-09-18**; its numbers survive in the results-doc appendix only. |
| 1.5, full split (superseded) | — | the retracted +0.227 run (`phase0_llr_action_direction/`, mean +0.2274, 4,142 events). **Data deleted 2026-09-18.** |
| 1.5 `empty`, full split (re-run) | `/mnt/efs/users/rod/results/llr_phase0_a15/empty.shard{0,1}-of-2.parquet` | complete; reproduces the original exactly |
| **1.5 `donor` K=2, full split** | `/mnt/efs/users/rod/results/llr_phase0_a15/donor_k2.shard{0,1}-of-2.parquet` | complete, 2,071 events — the content result |
| **A2S `donor` K=2, full split** | `/mnt/efs/users/rod/results/llr_phase0_a2s/donor_k2.shard*-of-8.parquet` | complete, 2,070 events (7 failed of 2,077) — **A2S content = zero** |
| **nav, full split** | `/mnt/efs/users/rod/results/llr_nav_a15/nav_gold.shard*-of-6.parquet`, `nav_cotempty.shard*-of-2.parquet` | complete, 2,071 events — +0.0018 trimmed, Wilcoxon p=6e−16 |
| **within-scene donor sweep** | `/mnt/efs/users/rod/results/llr_artifact/llr_donor_sweep_all.parquet` | complete, 9 scenes × 100 donors |
| **filler dose-response** | `/mnt/efs/users/rod/results/llr_filler_a15/filler.shard*-of-7.parquet` | complete, 70 scenes × 14 doses × 2 filler kinds |
| **token-clean flips, negative tail** | `/mnt/efs/users/rod/results/llr_artifact/llr_flip_rlvenv.parquet` | complete, 12 events, all reproducing full-split `llr_gold`. **Must be run under `a1x_rl_b300`** — see trap below |
| target-swap (wrong trajectory) | `/mnt/efs/users/rod/results/llr_target_a15/` | complete, 2,062 events; underpowered (paired p=0.11) |
| separation / gap-donor | `/mnt/efs/users/rod/results/llr_gap_donor_a15/` | complete, 2,062 events. Effect is 0.9% of the separation — **not quoted**, retained only as data |
| nav smoke, n=12 | `/mnt/efs/users/rod/results/llr_nav_smoke/` | smoke only; superseded once the above lands |
| 1.5 controls, n=10 | scratch only | artifact not retained; numbers survive in the results doc |
| A2S donor/pad, n=4 | output of `phase0_llr_verify_setup_a2s.py` (in the sibling **alpamayo2** repo, `examples/llr/`) | **refuted** by the full-split donor run above (it said ~63% content; the truth is ~2%). Kept as the record of how an n=4 probe misled. |

**Everything above is on EFS** (`/mnt/efs/users/rod/results/`, world-readable), so it is
shareable and survives instance replacement. Two runs originally written to local NVMe
(`/opt/dlami/nvme`) were copied to EFS on 2026-09-24 as `llr_splice_a15` and `llr_splice_a2s`;
the NVMe originals are ephemeral and should not be cited. The two superseded runs' data has been **deleted** so their numbers cannot be quoted by
accident; if you need to sanity-check which run a stray parquet is, the row count tells you:
2,031 or 4,142 events = superseded, 2,071 = corrected.

## ⚠️ Run these from a recipe directory, not from here

These live at the repo root because the work **spans two recipes** and belongs to
neither — the `phase0_*` scripts import `alpamayo1_5_sft.models.sft_base_model`
(`TrainableReasoningVLA`), the `extract_llr_viz_predictions*` scripts import
`alpamayo1_x_rl.models.reasoning_vla` (`RLWrapperReasoningVLA`), and the two
plotting scripts import neither.

But they still have to be **invoked with a recipe's venv, from that recipe's
directory** — both the venv path and their `sys.path[:0] = ["../../../src", "../.."]`
bootstrap are resolved relative to the working directory:

```bash
cd recipes/alpamayo1_x_rl          # CWD matters -- do not run from scripts_fork/llr/
CUDA_VISIBLE_DEVICES=0 a1x_rl_b300/bin/python \
  ../../scripts_fork/llr/phase0_llr_per_token.py --denominator empty --out <path>
```

Always use the `a1x_rl_b300` venv for this directory, whatever the box —
see the root `CLAUDE.md`, which tells you to check the hardware first. Each
script's own docstring carries its exact sharded invocation.

## Scripts

**Start here:**

| script | what it is |
|---|---|
| `phase0_llr_per_token.py` | **The measurement.** Defaults are correct — `--denominator empty`. Resolves LLR per trajectory token, so the horizon profile and the accel-vs-curvature split come out of one run; the only code beyond the two passes is the rank pairing described above. `--denominator pad` is the occupancy probe. **The `span` mode that produced the retracted +0.227 has been deleted**; see the results doc for what it did. |
| `phase0_llr_donor_sweep.py` | Within-scene sweep: one scene, one target, ~100 donor CoCs. Removes the scene-difficulty confound from the length regression, and provides the regression-to-the-mean control that explains both flip results. |
| `phase0_llr_filler_dose.py` | Filler dose-response: n tokens of meaningless filler in the reasoning block, swept over n, two filler kinds (a repeated word, in-distribution; the pad token, not). The only design where content is zero **by construction**. `--denominator pad` in the main script is the single point n=n_gold of this curve. |
| `phase0_llr_sanity_controls.py` | **Run this before believing any null.** Three controls that must be 0 (identical re-score, fresh re-tokenize, and the history tokens — where causality makes 0 a structural invariant, so it is the sharpest alignment check available) and three that must be large (blank vision, wrong vision, wrong trajectory). The last group is what distinguishes "reasoning doesn't matter" from "the probe is dead". |
| `phase0_llr_nav_per_token.py` | **The same measurement for NAVIGATION-COMMAND conditioning**: `llr_nav = log p(a*\|v,nav) − log p(a*\|v)`, per trajectory token. The command is an oracle synthesized from the GT future (`oracle_nav.derive_oracle_nav`) or from the gold CoC prose, since PAI-AV ships no route annotation. Note the denominator inverts the CoC rule: `construct_route` returns `[]` for `nav_text=None`, so the in-distribution no-nav sequence has **no route markers at all** — nothing is overwritten, so no marker artifact is possible. `--cot {gold,empty}` controls whether the gold reasoning stays in context (it often states the maneuver, which makes nav look redundant). Results: [`results/phase0_llr_nav.md`](results/phase0_llr_nav.md). |
| `phase0_llr_nav_sanity_controls.py` | The nav control ladder. Four nulls that must be exactly 0 — including `null_template`, which checks the `r1` template (no `route` component) against `r1_5` with nav absent and requires byte-identical `input_ids`, tying the nav path to the validated CoC path. Positive controls (wrong vision, wrong trajectory) plus the nav content arms: counterfactual `swap_direction` nav, and "Continue straight" as a wrong-but-valid command on turn rows. |
| `phase0_llr_wrong_coc.py` | Content control: gold CoC vs. **another event's real prose**, truncated to the same token length. **Superseded by `phase0_llr_per_token.py --denominator donor`**, which averages over `--n-donors` draws, does not truncate, and resolves per token. Kept because it is the script the A2S ~63%-content figure came from — a figure that is **n=4 and not established**. |

**Superseded, kept as the record:**

| script | why it's here |
|---|---|
| `phase0_llr_action_direction.py` | Produced the retracted `+0.227`. Scores via `loss_future_traj` (role-mixed span) with marker-inclusive pad-blanking. Its docstring's rationale for blanking-over-splicing is the error described above. |
| `phase0_llr_diagnostic.py` | §4.1 language-direction form. Its `[v, a*, cot]` reordering puts the trajectory in a sequence position the model was never trained on. |
| `phase0_llr_ordering_control.py` | The control that **proved** §4.1 broken: substituting a *wrong* trajectory into the reordered slot moved the score only 0.19 nats out of a 3.3–3.5 nat drop, i.e. ~95% positional artifact. |
| `extract_llr_viz_predictions.py` | §4.3 behavioral ablation whose "no reasoning" arm dropped `cot` from `components_prompt` *structurally*. Produced implausible trajectories. |

**Visualization:**

| script | what it is |
|---|---|
| `extract_llr_viz_v2.py` | Camera frames + GT trajectory for selected events; BEV plots with a ±8 m minimum lateral span so straight paths stay readable. |
| `extract_llr_viz_predictions_v2.py` | Behavioral ablation via the validated `coc_text=""` (empty) denominator. |
| `extract_llr_viz_velocity.py` | Speed-vs-time plots, so accel/decel is visible. |
| `render_gt_only_viz.py` | Re-renders the BEV/velocity panels showing **only** gold CoC + GT trajectory, dropping self-generated overlays (which mixed conditioning regimes with the scoring). |
| `extract_llr_viz_v3.py` | Selects showcase events on the **corrected** per-event value. v2 bucketed by the retracted metric, so its selection is not comparable. |

**Tail probes — all n=4, illustrative only:**

| script | what it is | status |
|---|---|---|
| `phase0_llr_flip_coc.py` | One-word directive inversion, asserting `llr_gold` reproduces first. `--cases-json` loads **pre-verified token-clean** cases: the built-in list was written at the string level and two entries silently changed 2–3 tokens (`deceleration` is 3 tokens, `acceleration` 1). | **Do not read these as directive-sensitivity.** 12 token-clean negative-tail flips give −0.039 (p=0.021), but the donor sweep moves the same tails further (+0.108), so it is regression to the mean. Only 33 of the 120 most-negative events admit a clean inversion at all. |
| `phase0_llr_occlusion.py` | Knocks out parts of the CoC (token / role / all) in pad and delete modes, to localize which span carries the effect. | **Known bug:** prefix re-tokenization maps `"Steer "` and `"Steer left"` to equal token counts, so `direction_word` selects a zero-length span — the pad rows are invalid (signature: `n_occluded_tokens=0`, attribution exactly `+0.0000`); delete rows are valid. |
| `phase0_llr_register_control.py` | Gold vs. filler vs. donor vs. nonsense prose, length-matched, testing whether distributional typicality explains the effect. | **It does not** — r=+0.218, and nonsense at −14.44 nats/token still retains 70%. Hypothesis tested and rejected. |

## On disk but not documented above

Two scripts exist in this directory and are deliberately not part of the story above. They work,
their data is kept, and neither is referenced by any result:

| script | what it is | why it is parked |
|---|---|---|
| `phase0_llr_coc_grid.py` | Crossed N x N scene x CoC grid (42x42 run: `/mnt/efs/users/rod/results/llr_coc_grid_a15/`). Fits `y = mu + alpha[s] + beta[c] + eps` and tests the diagonal. | The content result it gives is sound and **exactly invariant** to the denominator (diagonal contrast identical to 4e-16 whether or not the empty baseline is subtracted). But its headline framing — a variance split of scene 70% / CoC 3.1% / interaction 26.9% — is an artefact of that denominator: on raw `logp` the same data gives 99.7 / 0.0 / 0.3, because whatever you subtract sets the size of the scene term. The baseline-free quantities are the two sds (CoC identity 0.0069, interaction 0.0201). Pulled 2026-09-22 pending a reporting form that does not depend on the baseline. |
| `phase0_llr_flow_matching.py` | Scores the Stage-2 **flow-matching action head** — the deployed one — with the denoising loss under gold / cluster-matched / word-shuffled / empty conditioning, using common random numbers. | Results exist but are not written up here yet. |

## Traps worth reading before touching any of this

**1. `loss_future_traj` is misnamed, and its composition is model-specific.** On Alpamayo 1.5 it
spans 178 tokens in three roles — 48 history (LLR ≡ 0 by causality, pure dilution), 2 delimiters
(saturated, formatting-only), 128 real. On Alpamayo 2 Super it is 130: history and future tokens
live in *disjoint* id blocks there, so no history token is caught, but the 2 delimiters move by up
to 3.95 nats and can invert the sign of the mask-mean on their own. Read the composition off the
model you are using; do not port the counts. Always split by role; never average them. Per-token mean over the full mask reconciles exactly against
`-out.loss_future_traj`, which is the check that catches mask/shift drift.

**2. The "cot span" includes its markers.** Anything that overwrites
`get_label_mask(..., ["cot"])` wholesale destroys `<|cot_end|>` and injects ~23 nats. Either use
`empty` (re-tokenize, nothing overwritten) or, if you must hold length fixed, overwrite the prose
*interior* only and never the two markers.

**3. `empty` vs. `pad`, for scoring.** An earlier version of this README said the opposite of
what follows, so read carefully. **`empty` is correct for scoring.** It changes sequence length,
but that is in-distribution: component *order* is preserved and the model sees variable-length
CoC every training step. §4.1's positional artifact came from **reordering components**, not from
a length change — don't over-generalize it into a fixed-length requirement, which is what
motivated the marker-destroying overwrite in the first place.

Note the sign of this trap **flips for nav conditioning**: there the route section's markers
leave with the command (`construct_route` returns `[]` on `None`), and it is the
*marker-retaining* variant that is off-distribution. See
[`results/phase0_llr_nav.md`](results/phase0_llr_nav.md).

**4. Scoring vs. generation are different ablations.** Scoring conditions on **gold** CoC and
teacher-forces the **GT** trajectory; the behavioral ablation uses the model's **self-generated**
CoC and a **sampled** trajectory. Don't read a correlation between `llr` and trajectory
divergence across that gap — they are different conditioning regimes. (Empirically there is no
clean correlation either way.)

**5. Trajectory token layout.** `tokens_per_future_traj = 128` is `(64 waypoints, 2)` from
`UnicycleAccelCurvatureActionSpace` flattened row-major with `dt = 0.1`, i.e.
`stack([accel, kappa], -1)`. So token `k` → `t = (k // 2 + 1) × 0.1 s` and
**even `k` = acceleration, odd `k` = curvature**. It is *interleaved*, not one token per
timestep — and the 48 history tokens must be excluded by **position** before ranking, since
their ids are indistinguishable from future ones.

**6. The two `_b300` venvs give different numbers, and neither is more correct.** The same script,
the same checkpoint, byte-identical inputs, scored under `a1_5_sft_b300` vs `a1x_rl_b300`:

| event | `a1_5_sft_b300` | `a1x_rl_b300` |
|---|---|---|
| 920b5869 `llr_gold` | −0.2969 | **−0.3066** |
| 515ffffa `llr_gold` | −0.1832 | **−0.1710** |

~0.01 nats per event — **25× the content term** and 8× the per-token length effect. On one flip
case it reverses the sign of `gold − flip` (−0.0051 vs +0.0060).

*Cause*, diagnosed not guessed: torch 2.8.0+cu128 vs 2.11.0+cu130 use different CUDA reduction
kernels for the fp32 last-dim `mean` inside `Qwen2RMSNorm`, ~96 sites in the LM. The vision tower
is **bit-identical** (reduction width 1280, not 4096); divergence begins at decoder layer 0 and
compounds to mean |Δlogit| ≈ 0.04. Attention backend (this path is sdpa/eager, **not**
flash_attention_2 despite the config default), flash-attn, cuBLAS, TF32, video decode,
tokenization and the source trees were all checked and are bit-identical.

Against an fp64 CPU reference the two reductions are statistically indistinguishable (mean abs
err 3.84e−8 vs 3.95e−8), so **±0.01 nats is an environment-level noise floor**, not a bug with a
right answer. *Within* one venv the measurement is bit-exact — verified across repeated scoring,
dataset reloads, separate processes, int-vs-float `t0_us`, and an independent re-run of a whole
sweep scene (max |Δ| = 0).

**What to do:** run everything with `recipes/alpamayo1_x_rl/a1x_rl_b300/bin/python`, including
one-off probes — every number in this file came from there. When a `--tol` reproduction check
fails by ~0.01, suspect the venv before hunting for a data bug. Note this cuts against the
natural reading of the repo `CLAUDE.md` table, which pairs the 1.5 recipe with `a1_5_sft_b300`:
for this work, *consistency* of environment matters more than which recipe a script belongs to.
Effects below ~0.01 nats are only meaningful as within-environment paired comparisons over many
events — which the content term (0.0004) and the length slope (0.0012) both are.

## Where this leaves Phase 1

A ~0 nat Phase-0 result makes the §5 regularizer *more* interesting, not less: `L_total =
L_SFT − β·L_LLR` would be **creating** a reasoning→action dependence rather than reinforcing an
existing one, and the corrected `llr_act` is the quantity it should move. §2.1's
gradient-trivial trap concerns the training objective, not this measurement.

> ### ✅ The standing caveat has been addressed — see [`FM_HEAD_STATUS.md`](FM_HEAD_STATUS.md)
>
> Everything in *this* file measures Stage 1's discrete-token pathway. The **deployed** head is
> Stage 2's flow-matching action expert, which exposes no trajectory-token logits. That gap is
> now closed, and **the deployed head gives the same answer** (full split, n=2,071):
>
> | | token head | flow-matching head |
> |---|---|---|
> | CoC **content** | ~0 (donor p=0.99) | **0.27–0.46% of separation** (trimmed / mean); `donor` ≈ `donor_cluster` ≈ `shuffled`, paired p=0.19–0.52 |
> | cameras | 34% of separation | 9.0% trimmed / 19.8% mean |
> | nav **direction** | turns → curvature, p=4.3e−3 | *not pursued* — surrogate-only, never validated against the exact likelihood |
> | null controls | exactly 0 | exactly 0 |
>
> Same verdict, same dissociation: reasoning **content** behaves like occupancy (and touches only
> acceleration), while the navigation **direction** is channel-specific and physically sensible
> (touches only curvature). The estimator is *not* a likelihood ratio on the ODE — the deployed
> sampler is a 10-step Euler map, so the continuous-time density is a distribution the model never
> samples from. It is the flow-matching loss gap reweighted by `t/(1-t)`, which is the diffusion
> ELBO in nats; derivation and the abandoned ODE route are in
> [`fm_llr_method.md`](fm_llr_method.md); the abandoned ODE route is in
> [`ode_likelihood_explained.md`](ode_likelihood_explained.md).
>
> **Update 2026-09-24 — the flow-matching figures above are from the ELBO surrogate, whose
> absolute levels are now known to be inflated ~6–8×.** An exact likelihood of the deployed
> 10-step Euler sampler (no variational slack) gives `wrongtraj` +54.9 not +418, `blankvision`
> +13.2 not +82.6, and **`donor` −0.30 ± 0.39, p=0.24** — i.e. the flow-matching content term is
> a *true zero*, matching the token head rather than merely being small. Read the ratios above,
> not the levels. See [`FM_HEAD_STATUS.md`](FM_HEAD_STATUS.md).
