# Reasoning–Action Consistency Regularization for Alpamayo (LangForce-style LLR)

**Purpose of this doc.** Hand this to a Claude Code agent working *inside the Alpamayo/StarVLA training codebase*. It specifies (a) a pre-training diagnostic to run **before** touching training, and (b) how to add a Log-Likelihood-Ratio (LLR) regularizer that forces the driving trajectory to depend on the model's Chain-of-Causation (CoC) reasoning, rather than riding a vision/ego shortcut.

**Golden rule for the agent:** several steps depend on Alpamayo-internal details (token layout, which head is trained, whether ego status is a maskable input). Wherever this doc says **[CONFIRM]**, locate the ground truth in the codebase (grep, read configs, read the forward pass) before writing code. Do not assume.

---

## 0. TL;DR — what you are building

1. **Phase 0 (no training):** measure the LLR on the *current trained checkpoint*. This tells you whether reasoning already informs the action (nothing to fix) or is decorative (a real target exists). **Do this first. It may end the project early, which is a good outcome, not a failure.**
2. **Phase 1 (training):** add one auxiliary loss term, `L_total = L_SFT − β·L_LLR`, where `L_LLR` is computed on the **language side** via a reordered teacher-forced pass. Two extra forward passes per step; no architecture change.

---

## 1. Background (concise)

**The shortcut.** In goal-driven / smooth-continuation driving data, the action `a` is nearly predictable from a cheap signal `S` (raw vision `v`, and especially **ego status** `e` — velocity/accel/yaw/history). When `S` already predicts `a`, the model never needs the reasoning `ℓ`, so the generated CoC becomes a post-hoc rationalization the trajectory ignores. This is the driving analogue of LangForce's "vision shortcut."

**The fix (information-theoretic).** We want the action to carry information about the reasoning **beyond** the shortcut, i.e. maximize the conditional pointwise mutual information

```
PMI(a ; ℓ | v)  =  log p(a | v, ℓ) − log p(a | v)      (action direction)
              =  log p(ℓ | v, a) − log p(ℓ | v)      (language direction)   ← identical by symmetry
```

Both directions are the same quantity (PMI is symmetric). Which one you use differs between **measurement** and **training** (see §2).

**Why this model.** Alpamayo-R1 = Cosmos-Reason VLM backbone + trajectory decoder, generates CoC reasoning then a trajectory. It **[CONFIRM]** can emit the trajectory as discrete autoregressive tokens from the base LM (in addition to / instead of the diffusion decoder). The discrete-token path is what makes the language-modeling likelihoods below tractable.

---

## 2. The objective — and a trap to avoid

### 2.1 The trap (do NOT implement this)

The "obvious" action-direction term

```
L_LLR_wrong = log p(a* | v, ℓ) − sg(log p(a* | v))
```

is **gradient-trivial**. The numerator `log p(a*|v,ℓ)` is exactly the SFT trajectory-token loss (negated), and the stop-grad (`sg`) removes the denominator from the gradient entirely. Net effect on the gradient = up-weighting SFT. It does **not** break the shortcut. Do not build this.

### 2.2 The correct term (language side, following LangForce)

```
L_LLR = mean_ℓ[ log p(ℓ | v, a*) ]  −  sg( mean_ℓ[ log p(ℓ | v) ] )
```

- **Numerator** `log p(ℓ | v, a*)`: score the reasoning tokens `ℓ` in a **reordered** teacher-forced pass `[v, a*, ℓ]` — i.e. reasoning conditioned on vision AND the ground-truth trajectory tokens `a*`. This is **not** the SFT reasoning loss (SFT scores `ℓ` given `v` only, in generation order `[v, ℓ, a]`), so its gradient does real work: it forces the shared backbone to make the reasoning genuinely predictable from the action, hence the action dependent on the reasoning.
- **Denominator** `log p(ℓ | v)`: score `ℓ` in a normal pass `[v, ℓ]` (reasoning from vision only). **Stop-gradient.** Its role is (i) to prevent gaming by degrading language ability, and (ii) as a monitoring baseline so you can watch the *true ratio*, not just the numerator (see §6).

**Training loss** (minimized; SFT already contains the usual trajectory + reasoning losses):

```
L_total = L_SFT  −  β · L_LLR
```

Maximizing `L_LLR` ⇔ subtracting `β·L_LLR` from the minimization objective. Start `β` small (LangForce used 0.1) and expect to need a warmup ramp (§5.4).

> Note: because the discrete trajectory tokens `a*` serve directly as the action's representation in the sequence, you do **not** need LangForce's "latent action query" tokens. Feed `a*` in directly.

---

## 3. Confirm-first checklist (do before writing any training code)

Grep/read the codebase and answer each. Write the answers at the top of your working notes.

- **[CONFIRM-1] Discrete trajectory tokens exist and are trained.** Is there an AR discrete-trajectory-token head with its own vocab/tokenizer, trained with a next-token loss? If the trajectory is *only* produced by the diffusion decoder and the AR tokens are an untrained afterthought, the LM-likelihoods will be miscalibrated — flag this and stop to discuss (you may need the diffusion-head variant in §8).
- **[CONFIRM-2] Token layout & attention mask.** What is the exact input sequence order (`[vision, reasoning, trajectory]`?) and the attention mask? You need to (a) build a reordered `[v, a*, ℓ]` pass, and (b) mask the reasoning span so trajectory/reasoning tokens can be selectively hidden. Locate where the attention mask is assembled.
- **[CONFIRM-3] Reasoning (ℓ) generation loss / LM head.** Confirm the LM head scores CoC reasoning tokens and you can extract per-token log-probs for an arbitrary token span.
- **[CONFIRM-4] Ego status as an input.** Is ego status (velocity/accel/history) a distinct, maskable input (tokens or a conditioning vector)? Needed only for the ego-status extension (§8); if absent, that extension may not apply.
- **[CONFIRM-5] Vision-prefix KV reuse.** Can you prefill and reuse the vision-token KV across the multiple forward passes? This keeps overhead marginal (see §5.3).
- **[CONFIRM-6] Dataset fields.** Confirm each training example has: multi-view images `v`, ground-truth CoC reasoning `ℓ`, and ground-truth trajectory (both continuous and its discrete tokenization `a*`).

---

## 4. Phase 0 — Pre-training diagnostics (RUN THIS FIRST)

**Goal:** answer "is the LLR already high?" on the current trained checkpoint, with **zero training**. This is just the training forward passes run in eval mode, averaged over a held-out split (e.g. the PhysicalAI-AV OOD val set). No gradients.

### 4.1 Language-direction LLR (the quantity training will optimize)

For each example, compute:

```
llr_lang = mean_ℓ[ log p(ℓ | v, a*) ]  −  mean_ℓ[ log p(ℓ | v) ]
```

- Pass A (numerator): forward `[v, a*, ℓ]`, teacher-forced, collect log-probs of the `ℓ` tokens.
- Pass B (denominator): forward `[v, ℓ]`, teacher-forced, collect log-probs of the `ℓ` tokens.
- Report mean and distribution of `llr_lang` (in nats/token) over the split.

### 4.2 Action-direction LLR (cross-check; interpretable in trajectory nats)

```
llr_act = mean_a[ log p(a* | v, ℓ) ]  −  mean_a[ log p(a* | v) ]
```

- Pass C: forward full context `[v, ℓ, a*]`, score `a*` tokens.
- Pass D: forward reasoning-masked `[v, (ℓ masked), a*]`, score `a*` tokens.
- (Symmetry means `llr_act` and `llr_lang` estimate the same PMI; large numerical disagreement signals a tokenization/calibration problem worth investigating.)

### 4.3 Behavioral ablation (does masking reasoning change the trajectory?)

At inference, generate the trajectory with reasoning present vs. with the reasoning span masked from the trajectory's attention. Measure the change in the **decoded trajectory** (e.g. mean displacement / your existing `V_between/V_total`-style spread and any planning metric). This is the closed-loop-flavored counterpart of §4.1.

### 4.4 Decision gate

| Phase-0 result | Interpretation | Action |
|---|---|---|
| `llr` clearly **> 0**, masking reasoning **hurts** trajectory | Reasoning already used on this head | Little to fix here; check the *deployed* (diffusion) head instead (§8), or reconsider the premise |
| `llr ≈ 0`, masking reasoning **barely changes** trajectory | Shortcut confirmed; reasoning decorative | Proceed to Phase 1. Record the baseline `llr` — it sets a realistic target and informs `β`/warmup |
| `llr` differs a lot between §4.1 and §4.2 | Likely miscalibrated AR head ([CONFIRM-1]) | Resolve before training |

**Record the achievable magnitude.** If the gap between "reasoning present" and "reasoning masked" is tiny even for the *ground-truth* action, that ceiling limits how much any regularizer (SFT or RL) can achieve — and a near-zero ceiling is itself a publishable finding ("reasoning is redundant given perception"). Do not skip writing this number down.

---

## 5. Phase 1 — Implement the regularizer

### 5.1 The two forward passes (per training step)

Per batch, in addition to the standard SFT forward pass:

1. **Numerator pass** `[v, a*, ℓ]`, teacher-forced, **gradients on**, extract `logp_num = Σ log p(ℓ_t | v, a*, ℓ_<t)` (sum or mean over the reasoning span).
2. **Denominator pass** `[v, ℓ]`, teacher-forced, **detached**, extract `logp_den = Σ log p(ℓ_t | v, ℓ_<t)`.

### 5.2 The loss

```python
llr        = logp_num - logp_den.detach()      # per-example
loss_llr   = llr.mean()
loss_total = loss_sft - beta * loss_llr        # minimize
```

Reduction: mean over reasoning tokens per example, then mean over the batch. Keep the SFT loss exactly as-is.

### 5.3 Efficiency

The vision prefix is identical across the SFT pass, the numerator pass, and the denominator pass. **[CONFIRM-5]** Prefill the vision-token KV once and reuse it; only the reasoning/trajectory suffix differs. This turns "3× forward" into roughly "1× vision + 3× short suffix," which is the trick that keeps overhead marginal.

### 5.4 Hyperparameters & schedule

- `beta`: start at 0.1; sweep {0.03, 0.1, 0.3}. If the baseline `llr` from Phase 0 was near zero, the early gradient is weak — **ramp `beta` with a linear warmup** over the first ~10–20% of steps rather than applying it from step 0.
- Intervene **during SFT** (reasoning still plastic), not on a frozen-reasoning checkpoint where the backbone has already decided reasoning is ignorable.
- Keep a control run (`beta = 0`) for every experiment so any change is attributable to the term.

### 5.5 Pseudocode (framework-agnostic)

```python
for batch in loader:
    v, ell, a_star = batch.vision, batch.reasoning_tokens, batch.traj_tokens

    # --- standard SFT (unchanged) ---
    out_sft  = model(seq=[v, ell, a_star])           # natural order
    loss_sft = sft_loss(out_sft, targets=[ell, a_star])

    # --- LLR numerator: reasoning given vision + ground-truth action ---
    out_num  = model(seq=[v, a_star, ell])           # REORDERED
    logp_num = token_logprobs(out_num, span=ell)     # gradients flow

    # --- LLR denominator: reasoning given vision only (detached) ---
    with torch.no_grad():
        out_den  = model(seq=[v, ell])
        logp_den = token_logprobs(out_den, span=ell)

    llr        = logp_num.mean() - logp_den.mean()   # den already no-grad
    loss_total = loss_sft - beta * llr

    loss_total.backward(); opt.step(); opt.zero_grad()

    log({"loss_sft": loss_sft, "llr": llr,
         "logp_num": logp_num.mean(), "logp_den": logp_den.mean(), "beta": beta})
```

> Reuse the vision-prefix KV across `out_sft`, `out_num`, `out_den` per §5.3 rather than three full passes.

---

## 6. Training-time monitoring / sanity checks

- **Log the ratio, not just the numerator.** Track `llr = logp_num − logp_den` over training. The failure mode: `logp_num` climbs but `logp_den` climbs equally (the model just got more fluent at reasoning generally, not more faithful). If `llr` is flat while `logp_num` rises, the term is not doing its job.
- **Verify the stop-grad.** Assert no gradient reaches the denominator pass (e.g. its params' grad contribution is zero when `loss_sft` is ablated). A leaked gradient here silently degrades reasoning-from-vision.
- **Watch vision-only driving competence.** Periodically evaluate trajectory quality with reasoning masked. The point is to *raise* reasoning-dependence, not to *destroy* the vision/ego prior (safety-critical). If masked-reasoning driving degrades sharply, reduce `beta`.
- **Re-run Phase 0 §4.1 on checkpoints.** The whole thesis is "did the LLR go up?" — measure it every N steps on the fixed val split, not just training-batch `llr`.
- **Guard against trajectory degradation on easy scenes.** On unambiguous straight-cruise cases, reasoning genuinely adds ~0 info; the term should be near-zero there and not inject spurious swerving. Spot-check a few easy scenes' trajectories vs. the `beta=0` control.

---

## 7. Honest risks & expected outcomes

- **Vanishing signal.** If Phase 0 shows `llr ≈ 0` because reasoning carries almost no action-relevant information, the regularizer has little to grab and may barely move the metric. This is the single biggest risk. A clean null ("enforcing reasoning-action MI does not raise it, because the information isn't there") is a legitimate, publishable result given the existing faithfulness-measurement literature — treat it as a real outcome, not a failed run.
- **AR-vs-deployed-head transfer.** This term shapes the AR/language pathway and the shared backbone. If the *deployed* head is the diffusion decoder, confirm the gain transfers (re-measure the diffusion head's reasoning-mask behavioral gap, §4.3). If it doesn't transfer, add the diffusion-head consistency term in §8.
- **Reordered-pass realism.** The `[v, a*, ℓ]` ordering is a training-only artifact (the model normally generates `ℓ` before `a`). That's fine for a teacher-forced auxiliary loss, but do not use that ordering at inference.

---

## 8. Optional extensions (after the core term works)

- **Diffusion-head consistency (if the deployed head is the DiT).** Add a flow-matching-loss *gap*: `L_consist = sg(L_FM(traj | v, reasoning-masked)) − L_FM(traj | v, reasoning-present)`, maximized, with the reasoning KV masked at every layer of the action expert. Forces the *deployed* head to consume the reasoning. Complementary to the language-side LLR (one shapes the representation, the other polices the consumer).
- **Nested ego → perception → reasoning (chain rule).** Add a second layer targeting the ego-status shortcut, using the MI chain rule so the layers don't double-count:
  - `L1` (perception beyond ego): `log p(a|e,v) − sg(log p(a|e))`
  - `L2` (reasoning beyond ego+perception): `log p(a|e,v,ℓ) − sg(log p(a|e,v))`
  - Each higher layer conditions on the union of all cheaper signals. Requires ego status to be a maskable input ([CONFIRM-4]). Note the same action-direction caveat from §2.1 applies — realize each via the language/likelihood side where the numerator is not the SFT target, or via the diffusion-gap variant.
- **RL / C-GRPO form.** The expected LLR is a KL: `E[log π(a|v,ℓ) − log π(a|v)] = KL(π(·|v,ℓ) ‖ π(·|v))`. Rather than a reward bonus (gameable), impose it as a **constraint** `E[KL] ≥ δ` in a constrained-GRPO setup: the dual variable auto-tunes the weight, and the constraint *floor* removes the incentive to over-diverge (the main reward-hacking failure). Set `δ` from the Phase-0 measured achievable KL — an infeasible `δ` destabilizes the dual. Score `log π(a|v)` against a **frozen SFT reference**, not the live policy.

---

## 9. Prior art to differentiate from (so novelty is defensible)

- **Faithfulness measurement in this exact setting already exists** ("Is VLA Reasoning Faithful?", arXiv 2605.17268 — Alpamayo-R1 on PhysicalAI-AV). Differentiate by *fixing*, not just probing.
- **RL with a reasoning-alignment reward in driving exists** (CoPhy / "Distill to Think, Foresee to Act", 2605.21139 — GRPO + cosine-similarity "cognitive" reward). Differentiate by principled *conditional-MI / constrained* formulation that breaks a shortcut, vs. representation-similarity.
- **"Reasoning is optional" counter-evidence exists** (NoRD, 2602.21172). Motivate why enforced reasoning-dependence buys long-tail/safety value that a reasoning-free policy does not. NoRD also independently reports GRPO difficulty-bias on weak policies (relevant to the RL extension — budget for a Dr.-GRPO-style fix).

---

*Formulation note for the implementing agent: the action-direction LLR with a stop-gradded denominator collapses to SFT (§2.1) — the language-side term (§2.2) is the one with a non-trivial, shortcut-breaking gradient. If in doubt, verify empirically: a `beta`-only run whose gradient equals the SFT gradient (up to scale) means you built the wrong term.*
