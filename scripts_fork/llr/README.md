# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.

# Reasoning–action consistency (LLR) diagnostics

Phase 0 of [`langforce_readme.md`](langforce_readme.md) (the spec, alongside this README): does
Alpamayo's Chain-of-Causation reasoning actually carry information the trajectory prediction
uses, or is it a post-hoc rationalization riding a vision/ego shortcut? Everything here is
**measurement only** — no training. Run against the PAI-AV OOD reasoning split
(`reasoning/ood_reasoning.parquet`), which carries real gold CoC annotations.

Results: [`results/phase0_llr_action_direction.md`](results/phase0_llr_action_direction.md).
The Alpamayo-2-Super counterpart lives in `~/repos/alpamayo2/examples/llr/`.

## Headline result

**No measurable dependence.** Over the full OOD reasoning split (train+val, 2,071/2,077 events,
265k scored tokens), removing the reasoning moves `log p(a*)` by **+0.0010 nats** per future
trajectory token — **0.9σ from zero**, with 54.4% of events positive (a coin flip) and 0.05% of
the −1.932 baseline NLL. Flat across the 6.4 s horizon (±0.007 per 0.8 s bin) and across both
control channels (accel −0.0001, curvature +0.0021).

That null is only meaningful because the same code path registers other conditioning changes:

| | nats | vs. reasoning |
|---|---|---|
| 3 null controls | **exactly 0.0** | noise floor |
| **remove reasoning** (`p(a*\|v)`) | **+0.001** | — |
| **wrong** reasoning prose | −0.008 | ~same as removing it |
| blank vision | +0.102 | ~100× |
| **wrong** vision | +0.266 | **240×** |
| wrong trajectory target | +1.068 | ~1000× |

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
> Alpamayo 2 Super's `+0.037` used the same scoring and is retracted pending re-measurement.

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
  ../../scripts_fork/llr/phase0_llr_per_token.py --blank-mode splice --out <path>
```

Use the `_b300` venv on this box (sm_103) — see the root `CLAUDE.md`. Each
script's own docstring carries its exact sharded invocation.

## Scripts

**Start here:**

| script | what it is |
|---|---|
| `phase0_llr_per_token.py` | **The measurement.** `--blank-mode splice` (default) is the corrected one: the denominator re-tokenizes with an empty CoC, giving a true `p(a*\|v)`. Resolves LLR per trajectory token, so the horizon profile and the accel-vs-curvature split come out of the same run. `--blank-mode interior`/`span` reproduce the fixed-length and broken ablations for comparison. |
| `phase0_llr_sanity_controls.py` | **Run this before believing any null.** Three controls that must be 0 (identical re-score, fresh re-tokenize, and the history tokens — where causality makes 0 a structural invariant, so it is the sharpest alignment check available) and three that must be large (blank vision, wrong vision, wrong trajectory). The last group is what distinguishes "reasoning doesn't matter" from "the probe is dead". |
| `phase0_llr_wrong_coc.py` | Content control: gold CoC vs. **another event's real prose**, truncated to the same token length — in-distribution *and* position-preserved, so it separates "the content is wrong" from "there is no text here". Largely moot now that the total effect is ~0 (there is nothing left to decompose), but it is the right tool if a future run shows a non-zero effect. |

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
| `extract_llr_viz_predictions_v2.py` | Behavioral ablation via the validated `coc_text=""` splice. |
| `extract_llr_viz_velocity.py` | Speed-vs-time plots, so accel/decel is visible. |
| `render_gt_only_viz.py` | Re-renders the BEV/velocity panels showing **only** gold CoC + GT trajectory, dropping self-generated overlays (which mixed conditioning regimes with the scoring). |

## Traps worth reading before touching any of this

**1. `loss_future_traj` is misnamed.** 178 tokens in three roles — 48 history (LLR ≡ 0 by
causality, pure dilution), 2 delimiters (saturated, formatting-only), 128 real. Always split by
role; never average them. Per-token mean over the full mask reconciles exactly against
`-out.loss_future_traj`, which is the check that catches mask/shift drift.

**2. The "cot span" includes its markers.** Anything that overwrites
`get_label_mask(..., ["cot"])` wholesale destroys `<|cot_end|>` and injects ~23 nats. Blank the
*interior* only, or splice.

**3. Splice vs. blank, for scoring.** The earlier version of this README said the opposite of
what follows, so read carefully. **Splicing is correct for scoring.** It changes sequence length,
but that is in-distribution: component *order* is preserved and the model sees variable-length
CoC every training step. §4.1's positional artifact came from **reordering components**, not from
a length change — don't over-generalize it into a fixed-length requirement, which is what
motivated the marker-destroying pad-blank in the first place.

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

## Where this leaves Phase 1

A ~0 nat Phase-0 result makes the §5 regularizer *more* interesting, not less: `L_total =
L_SFT − β·L_LLR` would be **creating** a reasoning→action dependence rather than reinforcing an
existing one, and the corrected `llr_act` is the quantity it should move. §2.1's
gradient-trivial trap concerns the training objective, not this measurement.

Standing caveat: all of this reaches Stage 1's discrete-token pathway. The *deployed* head is
Stage 2's diffusion/flow-matching expert, which exposes no trajectory-token logits — reaching it
needs §8's consistency term.
