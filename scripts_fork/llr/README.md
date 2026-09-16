# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.

# Reasoning–action consistency (LLR) diagnostics

Phase 0 of `langforce_readme.md` (repo root): does Alpamayo's Chain-of-Causation
reasoning actually carry information the trajectory prediction uses, or is it a
post-hoc rationalization riding a vision/ego shortcut? Everything here is
**measurement only** — no training. Run against the PAI-AV OOD reasoning split
(`reasoning/ood_reasoning.parquet`), which carries real gold CoC annotations.

Results: [`results/phase0_llr_action_direction.md`](results/phase0_llr_action_direction.md).
The Alpamayo-2-Super counterpart lives in `~/repos/alpamayo2/examples/llr/`.

## ⚠️ Run these from a recipe directory, not from here

These live at the repo root because the work **spans two recipes** and belongs to
neither — the `phase0_*` scripts import `alpamayo1_5_sft.models.sft_base_model`
(`TrainableReasoningVLA`), the `extract_llr_viz_predictions*` scripts import
`alpamayo1_x_rl.models.reasoning_vla` (`RLWrapperReasoningVLA`), and the two
plotting scripts import neither.

But they still have to be **invoked with a recipe's venv, from that recipe's
directory** — both the venv path and their `sys.path[:0] = ["../../../src",
"../.."]` bootstrap are resolved relative to the working directory:

```bash
cd recipes/alpamayo1_x_rl          # CWD matters -- do not run from scripts_fork/llr/
CUDA_VISIBLE_DEVICES=0 a1x_rl_b300/bin/python \
  ../../scripts_fork/llr/phase0_llr_action_direction.py --split both --out <path>
```

Use the `_b300` venv on this box (sm_103) — see the root `CLAUDE.md`. Each
script's own docstring carries its exact sharded invocation.

## Scripts, in the order the work actually happened

| script | what it is |
|---|---|
| `phase0_llr_diagnostic.py` | **Superseded — kept as the record.** §4.1 language-direction form. Its `[v, a*, cot]` reordering puts the trajectory in a sequence position the model was never trained on. |
| `phase0_llr_ordering_control.py` | The control that **proved** §4.1 broken: substituting a *wrong* trajectory in the same reordered slot moved the score only 0.19 nats out of a 3.3–3.5 nat drop, i.e. ~95% positional-novelty artifact, not content signal. |
| `phase0_llr_action_direction.py` | **The real measurement.** §4.2 action-direction form, both passes in natural trained order `[v, cot, traj_future]`. This produced the headline numbers. |
| `extract_llr_viz_v2.py` | Pulls camera frames + GT trajectory for selected events, renders BEV plots (min ±8 m lateral span so straight paths stay readable). |
| `extract_llr_viz_predictions.py` | **Superseded — kept as the record.** §4.3 behavioral ablation whose "no reasoning" arm dropped `cot` from `components_prompt` structurally, putting the model outside its trained sequence shape. Produced implausible trajectories (one event accelerating to 14 m/s where ground truth braked to 0.36). |
| `extract_llr_viz_predictions_v2.py` | **The corrected ablation.** Uses the validated `coc_text=""` splice (see below). Divergences collapsed vs. v1 — e.g. 10.4 m → 1.0 m on the worst case. |
| `extract_llr_viz_velocity.py` | Speed-vs-time plots for the same four series as the BEV plots, so accel/decel is visible. |

## The one trap worth reading before touching any of this

Two ablations here look like "remove the reasoning" but are **not
interchangeable**:

- **Measurement** (`phase0_llr_action_direction.py`) blanks the CoC span with pad
  tokens **at identical length and position**. Numerator and denominator must
  differ *only* in content — otherwise `a*` shifts position between passes and you
  reintroduce exactly the artifact that broke §4.1.
- **Generation** (`extract_llr_viz_predictions_v2.py`) splices the span down to
  `[cot_end, traj_future_start]` (`coc_text=""`), the mechanism
  `~/repos/alpamayo1.5/eval_pai_av_val.py --no_coc` already validated. Correct
  here because the model just continues from the given prefix and already sees
  variable-length CoC in training — but wrong for scoring, for the reason above.

Full rationale: `results/phase0_llr_action_direction.md`, section
"Why blanking, not `coc_text=\"\"`".

## Headline result

| | Alpamayo 1.5 | Alpamayo 2 Super |
|---|---|---|
| mean `llr_act` | **+0.227 nats** | **+0.037 nats** |
| median | +0.248 | +0.036 |
| range | −0.586 → +0.751 | −0.325 → +0.532 |
| n (ok / total) | 2,071 / 2,077 | 2,031 / 2,077 |

Reasoning is load-bearing on both (every event cluster's mean is positive) but
modest against a ~7.3–7.6 nat baseline per-token NLL, and **~6× weaker on
Alpamayo 2 Super**. Separately: on the corrected behavioral ablation, `llr_act`
magnitude shows **no clean correlation** with how much the generated trajectory
actually changes — a single dramatic example is not evidence until the ablation
mechanism itself has been checked.
