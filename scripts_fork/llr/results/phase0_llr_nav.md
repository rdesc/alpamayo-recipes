# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.

# Phase 0 — navigation-command LLR (`llr_nav`) on Alpamayo 1.5

**Status: SMOKE ONLY (n=12). Nothing in the "Results" section below is established at a sample
size worth quoting as a finding.** The control ladder is the part that is established.

**FULL SPLIT COMPLETE** (2026-09-18). Headline below; the smoke numbers further down are
superseded and kept only as a record of how badly n=12 misled (it reported right turns at
+0.0197 and straight at -0.0021; at n=2,071 those reverse).

### Result: small, real, directional, and NOT a length effect

`llr_nav` = **+0.0018** nats/token (10% trimmed; raw mean +0.0032 overstates it, skew +1.33),
Wilcoxon **p=6e-16**, 56.4% of events positive, n=2,071.

| by command | n | trimmed | Wilcoxon |
|---|---|---|---|
| straight | 1884 | +0.0018 | 1.1e-14 |
| left | 84 | +0.0020 | 0.015 |
| right | 103 | +0.0011 | 0.24 (n.s.) |

**The channel dissociation is the finding.** On turn events the effect is entirely lateral:

| turns (n=187) | trimmed | Wilcoxon |
|---|---|---|
| curvature | **+0.0030** | **7.3e-04** |
| acceleration | -0.0004 | 0.52 |
| paired (curv - accel), within event | **+0.0036** | **4.3e-03** |

Straight commands behave oppositely -- their effect sits in acceleration (+0.0042). Turns'
curvature effect is 1.9x straight's, so it is not a generic consequence of adding tokens.

**Swap control (the falsification test): PASSED.** Left<->right swapped on the same 187 turn
events: curvature drops to +0.0012 and becomes non-significant (p=0.19), paired gold-swap
+0.0025 (p=0.023). Swapping retains only 41% and what remains is indistinguishable from zero.

**Not occupancy.** At the CoC span's measured rate of +0.0012 nats/token, an 8-token turn command
should buy +0.0098; observed is +0.0011 to +0.0020, 5-9x less. Route tokens are not worth what
reasoning tokens are worth.

**Not redundant with the reasoning.** `--cot empty` gives +0.0019 trimmed; paired difference
against `--cot gold` is -0.0006, p=0.43.

**Checks passed:** per-shard consistency; train (p=1.1e-14) and val (p=0.013) both positive;
jackknife range [+0.0031,+0.0039] and bootstrap 95% CI [+0.0009,+0.0059] on the turn-curvature
effect; left and right both positive (no sign bug); no event with LLR identically 0; route spans
decode correctly per command; markers/reconciliation/token-count/paired-target asserts active.

**Caveats.** Commands are oracles derived from the GT future (a ceiling, not deployable). Turns
are 9% of the split, so 187 events carry the key result. Null controls are still only n=12.

Original launch note: `--cot gold` on 6 shards and `--cot empty`
on 2, `--nav-from traj`, output `/mnt/efs/users/rod/results/llr_nav_a15/`. When it lands, apply
the **robust-estimator treatment** used on the CoC runs — median, 10% trimmed mean, sign test,
Wilcoxon, skew — and not the mean alone. On all three CoC runs so far the mean was misleading:
it understated 1.5's presence term and manufactured a spurious ~3-4σ *negative* content effect on
both 1.5 and A2S. The nav smoke's per-event skew is unknown, so assume nothing.

## What this measures

The sibling of the Chain-of-Causation measurement in
[`phase0_llr_action_direction.md`](phase0_llr_action_direction.md), with the navigation command in
place of the reasoning prose:

```
llr_nav = log p(a* | v, nav) − log p(a* | v)
```

per future trajectory token, in nats. Positive = the navigation command **helps**. It is a
conditional-PMI diagnostic: does route conditioning carry information the trajectory prediction
actually uses?

Two forward passes and a subtraction. Numerator: the route section present. Denominator:
`nav_text=None`, re-tokenized. Scripts:
[`phase0_llr_nav_per_token.py`](../phase0_llr_nav_per_token.py),
[`phase0_llr_nav_sanity_controls.py`](../phase0_llr_nav_sanity_controls.py).

## Setup, verified rather than assumed

| thing | value | how it was verified |
|---|---|---|
| model | `/mnt/efs/users/rod/ckpts/Alpamayo-1.5-10B-rl-training` | same checkpoint as the CoC run, so the numbers sit on one scale |
| prompt path | `chat_template_version="r1_5"`, `components_order = [image, traj_history, route, prompt, cot, traj_future]` | `r1` has no `route` case at all (`alpamayo/chat_template/r1_5.py:51`); this order matches `alpamayo1_5.helper.create_message`'s `hist_traj_placeholder + route_section + prompt_text` |
| nav injection | `<\|route_start\|>{nav_text}<\|route_end\|>` | `alpamayo/chat_template/components.py:construct_route` |
| route span decoded per event | asserted `<\|route_start\|>` first, `<\|route_end\|>` last | `get_label_mask(..., ["route"])` is marker-inclusive; the script asserts the layout every event and records the decoded span in the parquet |
| scored tokens | the **128 future** tokens only | `loss_future_traj`'s mask is 178 tokens (48 history + 2 delimiters + 128 future); reconciled against `-out.loss_future_traj` to <1e-3 every event |
| route markers trained? | **yes** | embedding norms: `route_start` 0.809, `route_end` 0.845 against a 0.806 vocab mean |
| `<\|route_pad\|>` trained? | **no** | norm 0.105, the initialization value (same as the unused `meta_action_start`). This is *why* the `pad` denominator was removed: it injected an embedding the model has never conditioned on. |

### A correction to the working premise about the denominator

The CoC rule was "the empty denominator must **retain** both markers", because
`get_label_mask`'s span is marker-inclusive and overwriting `<|cot_end|>` injects ~23 nats — the
defect behind the retracted `+0.227`.

**That rule does not transfer to nav, and following it here would have been the wrong choice.**
`construct_route` returns `[]` when `nav_text is None`, so the in-distribution no-nav sequence has
**no route markers at all**. That is correct precisely because it is produced by *re-tokenizing*,
not by overwriting: nothing is destroyed, and no token ends up predicting off a corrupted marker.
It is also exactly what the validated no-nav eval arm produces (`eval_pai_av_val_nav.py:80`,
`cond = None`).

The decisive evidence that this denominator is in distribution is the `null_template` control:
the `r1_5` template with nav absent produces **byte-identical `input_ids`** to the plain `r1`
template, on every event. `r1` has no `route` case at all, so the no-nav sequence is literally
the sequence this model family is trained on — not merely a plausible variant of it.

> Earlier revisions of this section also cited training's `nav_dropout_prob` as evidence. That
> does not apply here: `nav_dropout_prob` exists only in `alpamayo/data/navsim.py` and
> `alpamayo/data/truckdrive.py`, not in `alpamayo/data/pai_nav.py`, so it says nothing about the
> PAI-AV path this checkpoint was trained on. The `null_template` argument above is both correct
> and stronger, and does not depend on it.

Here the *marker-retaining* variant is the off-distribution one — training never emits an empty
`<|route_start|><|route_end|>` pair — which is the reverse of the CoC case. It was implemented,
measured, and then **removed along with `pad`**: since `nav_text=None` is the only valid
`p(a*|v)` for nav, there is no denominator choice to make and the scripts carry no
`--denominator` flag at all. The scripts still decode and assert the route span every event
rather than assuming its layout.

### Vocabulary

| term | what it is | valid `p(a*\|v)`? |
|---|---|---|
| **`empty`** | `nav_text=None`. Route section absent entirely. **The denominator.** | **yes** |
| ~~`markers`~~ | `nav_text=""`, an empty `<\|route_start\|><\|route_end\|>` pair. **REMOVED** — training never emits it, so it is not a valid `p(a*\|v)`. | no |
| ~~`pad`~~ | Command overwritten with `<\|route_pad\|>`. **REMOVED** — that token is untrained in this checkpoint (norm 0.105 vs a 0.81 vocab mean), so it injects an initialization-value embedding and read +0.0207, 3× the real effect. An artifact, not an occupancy result. | no |
| `donor` | A real, well-formed, **wrong** command in the same slot: `swap_direction` (left↔right) or "Continue straight" on a turn row. Present, fluent, same length — so what it buys is **presence**; content is the residual `empty_llr - swap_llr`. | yes |

Trajectory tokens are paired **by rank within each pass**, never by absolute position: the `empty`
denominator is shorter, and each pass re-derives its own mask. The script additionally asserts the
paired target token ids are equal between passes.

## Where the navigation command comes from

PAI-AV's OOD reasoning split ships **no route annotation** (`ood_reasoning.parquet` has only
`split`, `event_cluster`, and an `events` JSON with `event_start_timestamp` + `coc`). The command
is therefore synthesized by `/home/rod/repos/alpamayo1.5/oracle_nav.py`, loaded by path:

- `--nav-from traj` (default): `derive_oracle_nav(gt_xy)` from the GT ego-future geometry →
  `"Turn left in 12m"` / `"Turn right in 8m"` / `"Continue straight"`. Thresholds: 25° net
  end-of-horizon heading change, 10° onset, 1 m/s moving. A lane change nets ~0° and is reported
  as straight, by design.
- `--nav-from coc`: `derive_nav_from_coc(gold_coc)` from the gold prose → direction only.

Both are **oracles** (they use the answer), so this measures a recoverable ceiling, not a
deployable number. Confirmed: `"Continue straight"` is a real command string, not an empty one —
the eval merely suppresses it by default — so straight rows are usable and are reported as their
own arm.

Measured command mix on the first 25 events of the split (n=25, `--turn-deg 25`):
**23 straight / 1 left / 1 right**. Turns are ~8% of events, and each classification costs a clip
load (~17 s), which is why the smoke runs use `--balance-commands` and why a full run should stay
unweighted and report the breakdown as it falls.

## The CoC confound (`--cot`)

`--cot gold` (the default, for comparability with the CoC run) keeps the gold reasoning prose in
context in **both** passes. But the gold CoC frequently states the maneuver, so the route
information may already be present and nav would be measured as redundant. `--cot empty` runs both
passes with `cot_text=""`, making the nav command the only linguistic route cue. Both are reported;
they answer different questions and neither is "the" answer.

## Smoke artifacts

`/mnt/efs/users/rod/results/llr_nav_smoke/` — `nav_sanity_smoke.parquet` (control ladder),
`nav_pt_gold.parquet` / `nav_pt_empty.parquet` (per-token, the two `--cot` arms), plus the three
run logs. The 12 events were: left `01052304`, `08dfff23`, `0ae32163`, `0af40d4b`; right
`00c18025`, `037150fe`, `056535bc`, `066d71ee`; straight `000548db` (×2), `000ba013`, `002d7967`.
(These runs predate the `<out>.events.json` selection dump, so re-runs on the same events need
that list; later runs write it automatically.)

## Control ladder — n = 12 events (4 left / 4 right / 4 straight)

`phase0_llr_nav_sanity_controls.py --limit 12 --balance-commands --cot gold`, 12/12 events ok,
7.2 min. Every number below is `log p(a*|reference) − log p(a*|alternative)` per future
trajectory token, so positive = the correct command is better.

| control | mean | std | n | expected | verdict |
|---|---|---|---|---|---|
| `null_identical` — same command, scored twice | **0.0000** | 0.0000 | 12 | == 0 | **PASS** |
| `null_retokenize` — fresh tokenize, same inputs | **0.0000** | 0.0000 | 12 | == 0 | **PASS** |
| `history_llr` — the 48 history trajectory tokens | **0.0000** | 0.0000 | 12 | == 0 | **PASS** |
| `null_template` — `r1` vs `r1_5`-without-nav | **0.0000** | 0.0000 | 12 | == 0 | **PASS** |
| **`empty_llr` — remove the nav command** | **+0.0073** | 0.0177 | 12 | — | the result |
| `markers_llr` — vs. empty marker pair | +0.0061 | 0.0157 | 12 | ? | ~same as removing it |
| `pad_llr` — vs. `<\|route_pad\|>` | +0.0207 | 0.0276 | 12 | ? | inflated; that token is untrained |
| `swap_llr` — vs. the **wrong-direction** command | +0.0031 | 0.0081 | 12 | ? | ~40% of `empty_llr` |
| `straight_llr` — turn rows, vs. "Continue straight" | +0.0095 | 0.0207 | 8 | ? | ~same as removing it |
| `wrong_vision_llr` — another scene's frames | **+0.2360** | 0.0444 | 12 | > 0 | **PASS** |
| `wrong_traj_llr` — another event's trajectory | **+0.6809** | 0.7110 | 12 | >> 0 | PASS (see note) |

**All four nulls are exactly 0.0 on all 12 events.** The `history_llr` null is the sharp one —
those tokens precede the route section, so causal attention makes 0 a structural invariant, and a
one-token shift anywhere would break it. `null_template` additionally confirmed **byte-identical
`input_ids`** between the `r1` template (no `route` component) and `r1_5` with nav absent on all
12 events, which ties this measurement to the already-validated CoC scoring path. The full
178-token mask mean reconciled against `-loss_future_traj` to <1e-3 on every pass of every event.

**On `wrong_traj_llr`'s "SUSPECT" flag:** the script's threshold is `> 1.0`, calibrated on the
CoC run's own 10-event ladder (a figure since retired as unstable; the full-split separation is
+0.7748, which is itself below this threshold). At n=12 the mean is +0.681 with std 0.711 and per-event values spanning
−0.208 → +2.521 (one event negative). That is a small-sample spread, not a dead probe:
`wrong_vision_llr` is tight (+0.236 ± 0.044, and the CoC run's figure on the same checkpoint was
+0.266), and it is the decisive positive control here. Do not read the +0.681 as a considered
estimate of the wrong-trajectory scale.

## Results — SMOKE ONLY, n = 12 events / 1,536 tokens

`phase0_llr_nav_per_token.py --limit 12 --balance-commands`, both `--cot` arms on the **same** 12
events (selection is deterministic), `--nav-from traj`. The denominator is always
`nav_text=None`; there is no flag.

| | `--cot gold` | `--cot empty` |
|---|---|---|
| mean `llr_nav` | **+0.0073** | +0.0043 |
| per-event std | 0.0177 | 0.0127 |
| σ from zero | **1.43** | 1.18 |
| fraction > 0 | 58.3% | 66.7% |
| reference `log p(a*\|v,nav)` | −1.8916 | −1.8931 |
| n | 12 events / 1,536 tokens | 12 events / 1,536 tokens |

**Neither arm is distinguishable from zero at n=12.** Removing the gold CoC did *not* increase
nav's measured contribution, so at this sample size the "the gold CoC already states the
maneuver, which is why nav looks redundant" hypothesis gets **no support** — if anything the
number is smaller without it.

### By command (n = 4 each — far too small to quote as a finding)

`--cot gold`, per-event means:

| command | mean | std | n |
|---|---|---|---|
| left | +0.0044 | 0.0064 | 4 |
| right | +0.0197 | 0.0276 | 4 |
| straight | −0.0021 | 0.0036 | 4 |
| **turns (left+right)** | **+0.0120** (1.68σ, 75% positive) | 0.0203 | **8** |

The right-turn mean is dominated by a **single event** (`00c18025`, +0.0539); the other three
right rows are +0.0294, +0.0030, −0.0076. **No claim about left-vs-right is supportable at n=4.**
The direction of the overall pattern — turns above straight, straight at ~0 — is consistent with
the hypothesis that nav conditioning matters mainly on turns, but 1.68σ on n=8 is a hint, not a
result.

### Presence vs. content — the most informative arm

`swap_llr` is what a **wrong** command buys relative to no command at all. A wrong command is
present, fluent and the same length, so whatever it buys is **presence**, not content. Content is
therefore the *residual* — the part only the correct command buys:

    presence = swap_llr                     content = empty_llr - swap_llr

| subset | `empty_llr` | presence (`swap_llr`) | content (residual) | **content share** |
|---|---|---|---|---|
| all (n=12) | +0.0073 | +0.0031 (43%) | +0.0042 | **57%** |
| turns (n=8) | +0.0120 | +0.0047 (39%) | +0.0073 | **61%** |
| left (n=4) | +0.0044 | +0.0003 (6%) | +0.0041 | **94%** |
| right (n=4) | +0.0197 | +0.0091 (46%) | +0.0106 | **54%** |

> ⚠️ **An earlier version of this table had presence and content transposed**, labelling
> `swap_llr` as the content term and so reporting content shares of 43 / 39 / 6 / 46% — the
> presence shares. The conclusion drawn from it ("mostly presence, unlike A2S's CoC") was
> therefore backwards. Corrected above.

The `straight` arm does **not** agree with the swap arm. On turn rows, replacing the correct turn
with "Continue straight" costs +0.0095 of the +0.0120 — it keeps 79%, so it says presence
dominates, while swap (keeps 39%) says content dominates. Two in-distribution wrong-command arms,
opposite conclusions.

**Read: at n=12 this decomposition carries no signal, and no presence-vs-content claim should be
made from it.** Two tells, both fatal:

- The `straight` arm keeps **3% on left turns but 96% on right turns** — a 30× disagreement
  between two subsets of 4 that should behave alike.
- Over all 12 events the `straight` arm keeps **130%** of the effect, i.e. a *negative* content
  share, which is not a physically meaningful quantity.

These are ratios of means of order 0.01 whose per-event std is of the same order, so the ratio is
unstable by construction. Settling presence-vs-content needs the full split with a swap numerator
over the same events; until then this section records arm values, not a finding.

### Channel split

| | accel | curvature |
|---|---|---|
| all (n=12, 768 tokens each) | +0.0065 | +0.0081 |
| left (n=4) | +0.0026 | +0.0061 |
| right (n=4) | +0.0165 | +0.0229 |
| straight (n=4) | +0.0004 | −0.0046 |

Curvature runs ~1.4× accel on turn rows. That is the direction the prior predicts (nav is a
lateral instruction) and it differs from the CoC result, where A2S's effect concentrated in
accel. But the gap is well inside the per-event noise at n=12 and **should not be reported as an
established asymmetry.**

### Horizon profile (turns only, `--cot gold`, n=8)

| t (s) | accel | curvature |
|---|---|---|
| 0.0–1.6 | −0.0015 | +0.0020 |
| 1.6–3.2 | +0.0246 | +0.0377 |
| 3.2–4.8 | +0.0110 | +0.0157 |
| 4.8–6.4 | +0.0045 | +0.0034 |

The effect is concentrated in the 1.6–3.2 s window rather than being flat, which is where the
turn is executed for these oracle distances (3–32 m). Suggestive and mechanistically plausible,
but n=8 and driven partly by the one large event.

## On the CoC scale

| quantity | value | source |
|---|---|---|
| CoC `llr_act`, full split | +0.0010 (0.9σ, n=2,071) | `phase0_llr_action_direction.md` |
| **nav `llr_nav`, smoke** | **+0.0073 (1.4σ, n=12)** | this run |
| nav `llr_nav`, turns only, smoke | +0.0120 (1.7σ, n=8) | this run |
| baseline `log p(a*)` per future token | −1.892 (nav run) vs −1.932 (CoC run) | both |
| wrong vision | +0.236 (n=12) vs +0.266 (CoC) | both |
| wrong trajectory | +0.681 (n=12, std 0.71) vs +0.7748 (CoC, full split) | both |

So `llr_nav` is nominally ~7× the CoC's full-split number and ~3% of the wrong-vision control.
**But the CoC figure is an n=2,071 measurement and this is n=12.** The two are not comparable as
evidence, only as magnitudes, and a +0.007 mean with per-event std 0.018 at n=12 is exactly the
kind of number that regresses toward zero with more data.

## What is and isn't established

**Established (n=12 unless noted):**

- The probe is correctly aligned: four independent nulls at exactly 0.0, including the structural
  history-token null and byte-identical `input_ids` against the validated CoC template path; and
  the full-mask reconciliation against `-loss_future_traj` holds on every pass.
- The probe is live: another scene's frames cost +0.236 ± 0.044 nats through the identical code
  path.
- The nav plumbing is what we think it is: the route span decodes as
  `<|route_start|>Turn left in 3m<|route_end|>` (4–9 tokens), markers asserted first/last every
  event.
- `route_start`/`route_end` are trained in this checkpoint; `<|route_pad|>` is **not** (norm
  0.105 vs 0.806 vocab mean), so the `pad` arm's larger reading (+0.0207) was an artifact of
  injecting an untrained embedding. That arm has since been **removed** from the scripts, along
  with `markers`.
- `"Continue straight"` is a real command string, so straight rows are usable.
- Command mix is ~8% turns (n=25 probe: 23 straight / 1 left / 1 right), so a full run will be
  straight-dominated and the turn subset will be ~160 of ~2,077 events.

**NOT established:**

- Whether `llr_nav` is non-zero at all. 1.43σ on n=12 (1.68σ on the n=8 turn subset) is not
  evidence. **Do not quote +0.0073, +0.0120, or any of the per-command numbers as findings.**
- Any left-vs-right difference (n=4 per cell, one event dominating the right mean).
- The curvature > accel asymmetry (inside the noise at this n).
- The horizon concentration at 1.6–3.2 s (same caveat, plus the one dominant event).
- The presence-vs-content split — and it is the *weakest* thing in the smoke run, not the
  sturdiest. The two in-distribution wrong-command arms **disagree**: `swap` keeps 43% (content
  dominates) while `straight` keeps 79% (presence dominates). Worse, `straight` keeps 3% on left
  turns against 96% on right turns, and 130% over all 12 — a negative content share. These are
  ratios of ~0.01 means whose per-event std is the same order, so they are unstable by
  construction and support no claim in either direction.

**What a full run would settle:** at 2,077 events the standard error on the mean drops ~13×, so
a +0.007 effect would land at ~18σ if real. Cost: ~2 forward passes plus one clip load per event,
≈47 s/event, ≈27 GPU-hours, ≈3.5 h across 8 shards. Recommended shape: unweighted full split with
`--cot gold` (comparable to the CoC run), plus a `--nav-swap` numerator run over the same events
via `--events-json` to nail the presence-vs-content split at scale.

