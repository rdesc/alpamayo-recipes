# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.

# Measuring `log p(a* | v, coc)` on the flow-matching action head

How the deployed Stage-2 action expert -- which emits **no logits** -- is made to yield a
log-likelihood ratio in **nats**, comparable to the discrete token head's.

Companion docs: [`README.md`](README.md) (token-head study),
[`FM_HEAD_STATUS.md`](FM_HEAD_STATUS.md) (results and running log),
[`ode_likelihood_explained.md`](ode_likelihood_explained.md) (the exact-ODE route, built and
abandoned).

**Everything in sections 3-5 is established published material, not a method we invented.** The
citation trail is given inline. What is ours is the experimental design (section 7), the control
ladder (section 8) and the engineering.

---

## Contents

1. [Why a new measurement was needed](#1-why-a-new-measurement-was-needed)
2. [Where language enters the action expert](#2-where-language-enters-the-action-expert)
3. [The training objective, verbatim](#3-the-training-objective-verbatim)
4. [Log-SNR, and why a weighted MSE is a log-likelihood](#4-log-snr-and-why-a-weighted-mse-is-a-log-likelihood)
5. [The weight for this interpolant](#5-the-weight-for-this-interpolant)
6. [Sampling uniformly in lambda](#6-sampling-uniformly-in-lambda)
7. [The estimator as implemented](#7-the-estimator-as-implemented)
8. [The control ladder](#8-the-control-ladder)
9. [What this licenses, and what it does not](#9-what-this-licenses-and-what-it-does-not)

---

## 1. Why a new measurement was needed

The **Stage-1 discrete head** emits 128 trajectory tokens (64 waypoints x 2 channels: accel,
curvature), each a softmax over 3000 bins. The ground-truth trajectory's log-probability is a
lookup:

```
log p(a* | v, coc) = sum_{k=1..128} log softmax( z_k(v, coc) )[ b_k(a*) ]
```

One teacher-forced pass, read the logits at the GT bin indices, sum. Every number in
`README.md` is a difference of two such quantities.

The **Stage-2 action expert** -- the head that actually drives -- is a flow-matching model. It
has no softmax, no bins, and **no logits**. Its whole output is a velocity field

```
v_theta( x, t, context )   in   R^(64 x 2)
```

There is nothing to index into. `log p(a*)` is not something the network computes; it is a
property of the ODE the network induces. So the central question of the project had a clean
answer on a head that is not deployed and no answer at all on the head that is.

Two routes out:

- **(a) Compute the exact density.** A deterministic ODE has a tractable likelihood via the
  instantaneous change of variables. We built this. It works and it self-tests. It was
  **abandoned** -- see [`ode_likelihood_explained.md`](ode_likelihood_explained.md) section 5;
  the decisive reason is that the deployed sampler is a 10-step Euler map, so the
  continuous-time density is a distribution the model never samples from (>16 nats apart).
- **(b) Use the training objective itself.** The flow-matching MSE, with the correct weight
  applied, is exactly a variational bound on `log p(a*)` in nats. We were already computing the
  MSE and simply were not weighting it. **This is what we use**, and it is the subject of this
  document.

---

## 2. Where language enters the action expert

This is what makes the token head's LLR design transfer unchanged.

The expert is a **decoder-only** transformer built from a copy of the VLM's text config, with
**fresh weights** and `embed_tokens` deleted:

```python
expert_config = copy.deepcopy(self.vlm.config.text_config)   # same arch
self.expert   = AutoModel.from_config(expert_config)          # different weights
del self.expert.embed_tokens                                  # cannot embed a token at all
```

It has no cross-attention *module*. Conditioning arrives as the VLM's KV cache, handed to the
expert as if it were the expert's own prefix:

```python
prompt_cache = vlm_outputs.past_key_values      # K/V computed by VLM weights
expert_out   = self.expert(
    inputs_embeds   = future_token_embeds,      # the 128 noisy-action tokens, that is ALL
    past_key_values = prompt_cache,             # <-- the entire conditioning channel
    attention_mask  = attention_mask,           # masks everything after <traj_future_start>
)
```

So at expert layer `i`, with prefix length `P`:

```
Q = X_act W_Q^(exp,i)                       [128 x 2048]   -- action tokens only, no prefix rows

     [ H^(vlm,i) W_K^(vlm,i) ]   P x 1024    <- from the cache, VLM weights
K =  [                       ]
     [ X_act    W_K^(exp,i) ]   128 x 1024   <- expert weights

(same for V)
```

Verified dimensions -- this is why the concatenation is even legal:

| | layers | hidden | Q heads | KV heads | head_dim | K/V width |
|---|---|---|---|---|---|---|
| VLM text tower (`Cosmos-Reason2-8B`) | 36 | 4096 | 32 | 8 | 128 | 1024 |
| expert (`expert_cfg` override) | 36 *(inherited)* | 2048 | 16 | 8 *(inherited)* | 128 | 1024 |

`expert_cfg` overrides `hidden_size` and `num_attention_heads` but leaves `head_dim` pinned at
128 and `num_key_value_heads` inherited. The expert is a *narrower* tower that can still consume
the wide VLM's cache because the per-head K/V geometry matches exactly.

**Gradients.** In Stage-2 training `cotrain_vlm=false` (pinned by
`tests/test_recipe_static_contracts.py:103`), and there are three independent brakes: the VLM
params get `requires_grad=False`, the whole VLM forward runs under `torch.no_grad()`, and
`stop_grad_from_vlm` explicitly detaches the cache. So `W_K^(vlm)`, `W_V^(vlm)` receive **zero
gradient**; the expert's own projections are fully trained. At *scoring* time nothing is trained
at all -- the scorer runs entirely under `no_grad`.

**The consequence that matters.** There is exactly **one** channel from language into the action
head, and it is the cropped KV cache. So when we swap gold CoC -> donor CoC:

- the VLM re-runs, producing different prefix K/V
- the expert weights and its 128 input embeddings are **bit-identical** (same gold `x`, same
  `t`, same `eps`)
- therefore 100% of any change in `v_theta` is attributable to the prose swap

Same intervention as the token head, same isolation, different readout.

---

## 3. The training objective, verbatim

From `alpamayo_r1/diffusion/flow_matching.py` (byte-identical across every venv we use):

```python
def construct_training_data(self, x):
    t = self.beta_dist.sample((batch_size,))                      # Beta(1.5, 1.0)
    t = self.beta_scale_constant - t * self.beta_scale_constant   # 0.999
    noise   = torch.randn_like(x)
    noisy_x = t * x + (1 - t) * noise

def compute_loss_from_pred(self, training_data, pred):
    target = (training_data["x"] - training_data["noise"]).to(dtype=pred.dtype)
    return torch.nn.functional.mse_loss(target, pred)
```

Three things to read off it.

**(a) The interpolant, and its orientation.**

```
x_t = t*x + (1-t)*eps ,      eps ~ N(0, I)
      ^^^              ^^^^^^^
      alpha_t = t      sigma_t = 1-t
```

`t=0` is pure noise, `t=1` is clean data -- the rectified-flow convention, **opposite** to the
usual diffusion one. Every sign below depends on this. Here `x` is the ground-truth action `a*`,
a `[64, 2]` tensor.

Note this is the linear / optimal-transport interpolant (`alpha + sigma = 1`), not
variance-preserving (`alpha^2 + sigma^2 = 1`): `Var[x_t]` dips to about half at `t=0.5`.

**(b) The target is a constant.** `target = x - eps` is the velocity of the straight line from
`eps` to `x`, which does not depend on `t`. The model's job: given a point on the segment and
told where on it you are, name the direction of travel.

**(c) The loss is unweighted.** A plain mean over all `B x 64 x 2` entries, with no `t`-dependent
factor. Section 4 says the ELBO *requires* one; its absence is exactly why the raw MSE gap is not
in nats.

**Where training samples `t`.** With `t = 0.999*(1 - Beta(1.5,1))` the density is
`p(t) ~ (1 - t/0.999)^(1/2)` -- concentrated at the **noisy** end:

```
P(t > 0.9)   = 3.1%
P(t > 0.99)  = 0.085%
P(t > 0.999) = 0        exactly
```

Hold onto this; section 5 shows the ELBO weight is largest precisely where that support vanishes.

---

## 4. Log-SNR, and why a weighted MSE is a log-likelihood

### SNR

`x_t` mixes signal and noise. SNR is the ratio of their **powers** (hence the squares, since
`eps` has unit variance per component):

```
SNR(t) = alpha_t^2 / sigma_t^2 = t^2 / (1-t)^2
lambda = log SNR                = 2 * log( t / (1-t) )
```

| t | x_t is... | amplitude ratio | SNR | lambda |
|---|---|---|---|---|
| 0.1 | `0.1x + 0.9eps` | 1 : 9 | 0.012 | -4.39 |
| 0.5 | `0.5x + 0.5eps` | 1 : 1 | 1 | **0** |
| 0.9 | `0.9x + 0.1eps` | 9 : 1 | 81 | +4.39 |
| 0.99 | `0.99x + 0.01eps` | 99 : 1 | 9801 | +9.19 |

Same quantity as decibels, different base (`dB = 10*log10(SNR)`; `lambda = ln(SNR)`).

### Why lambda is the right axis

`t` is an arbitrary parameterization. `lambda` is intrinsic -- a property of the corrupted
distribution, not of how you index it. That is what lets different schedules be compared, and it
is why Kingma & Gao can tabulate every published objective in one frame.

The deeper reason: **equal intervals of lambda cost equal nats.** Shannon's Gaussian channel
capacity is `C = 0.5*log(1 + SNR)` nats per dimension, so with `SNR = e^lambda`:

```
dC/dlambda = 0.5 * e^lambda / (1 + e^lambda)  ->  0.5   for lambda >> 0
```

Every unit of log-SNR buys exactly **half a nat per dimension**, at any SNR. `dlambda` is
therefore the natural information measure.

### The unified form

Kingma & Gao (arXiv 2303.00848) Eq. 4 writes every diffusion objective as

```
L_w(x) = (1/2) * integral w(lambda) * E_eps || eps_hat(x_lambda; lambda) - eps ||^2  dlambda
```

Papers differ **only** in `w(lambda)`. The headline result:

```
w(lambda) = 1   <==>   L_w(x) = -ELBO(x) + const
```

The leading `1/2` is the Gaussian-channel `1/2` above: read `E||eps_hat - eps||^2` as the
*fraction* of the available half-nat the model fails to capture (it is ~1 for a model that
predicts nothing, ~0 for a perfect denoiser). The whole integral is "nats left on the table,
summed over noise levels".

Citation trail:

| claim | source |
|---|---|
| log-SNR parameterization; continuous-time diffusion ELBO | Kingma et al., VDM (2107.00630) |
| likelihood weighting of the score-matching loss | Song et al. (2101.09258) |
| unified `w(lambda)` form; `w=1` is the ELBO; Table 1 of objectives | Kingma & Gao (2303.00848) |
| eps- and v-parameterizations are affinely equivalent | Salimans & Ho (2202.00512) |
| sampling `t` uniformly in logit/log-SNR space | Esser et al., SD3 (2403.03206) |
| comparing ELBOs across conditionings, with shared `(t, eps)` | Li et al., Diffusion Classifier (2303.16203) |

### The bound, and why the constant cancels

```
-log p(a* | c)  <=  L_{w=1}(a* | c)  +  C ,        C = L_prior + L_recon
```

`C` does **not** depend on `c`: the prior is `N(0, I)` whatever prose conditioned the model, and
the reconstruction term is a function of `a*` and the schedule. So in a difference it cancels:

```
LLR = log p(a*|gold) - log p(a*|ctrl)  ~=  L_{w=1}(a*|ctrl) - L_{w=1}(a*|gold)
```

Note the order flip -- `L` is a negative log-likelihood, so **control minus gold** is positive
when gold helps. That matches the token head's sign convention.

The `~=` is doing real work. See section 9.

---

## 5. The weight for this interpolant

Two conversions: our loss is on **velocity**, not `eps`; our variable is **t**, not `lambda`.

### (a) Velocity error <-> noise error

Eliminate `x` between the two definitions:

```
x_t = t*x + (1-t)*eps  and  v = x - eps  =>  x = v + eps
x_t = t(v + eps) + (1-t)eps = t*v + eps
=>  eps = x_t - t*v
```

So at fixed `x_t`, `eps` is affine in `v` with slope `-t`:

```
eps_hat - eps = -t * (v_hat - v)        =>    ||eps_hat - eps||^2 = t^2 * ||v_hat - v||^2
```

Sanity: near `t=0` the eps-error is crushed to nothing -- correct, because `x_t ~= eps` there, so
`eps` can be read straight off the input however badly the velocity is predicted.

### (b) The Jacobian

```
lambda    = 2*[ log t - log(1-t) ]
dlambda/dt = 2*[ 1/t + 1/(1-t) ] = 2 / ( t(1-t) )
=>  dlambda = 2/(t(1-t)) dt      and      dt = (t(1-t)/2) dlambda
```

### (c) Combine

```
L_{w=1} = (1/2) * integral ||eps_hat - eps||^2 dlambda
        = (1/2) * integral t^2 ||v_hat - v||^2 * 2/(t(1-t)) dt

=>  L_{w=1} = integral_0^1  [ t/(1-t) ] * E|| v_theta(x_t,t) - (x - eps) ||^2  dt
```

**The ELBO weight on the velocity MSE is `t/(1-t)`, leading constant exactly 1.** The `1/2` and
the `2` cancel -- a small piece of luck specific to this interpolant. This single line is what
the whole surrogate rests on.

### What flow matching optimizes, in ELBO terms

Running it backwards, training minimizes `integral p_train(t) ||v_hat - v||^2 dt`, so its implied
weight is

```
w(lambda) = p_train(t) * (1-t)/t
```

For textbook uniform-`t` flow matching (`p_train = 1`) that is `w = (1-t)/t = e^(-lambda/2)` --
the OT / rectified-flow row of Kingma & Gao's Table 1. So:

> **Flow matching is an ELBO with an exponentially decaying weight in log-SNR.** It deliberately
> de-emphasizes high SNR -- which is good for sample quality and bad for likelihood.

Our model is doubly suppressed, since `p_train(t)` also decays to zero at the top.

### The collision

```
 t        lambda   ELBO weight t/(1-t)   training mass above t
 0.5       0.00            1                    35%
 0.9      +4.39            9                    3.1%
 0.99     +9.19           99                    0.085%
 0.999   +13.8          999                     0        <-- weight 999, zero training data
 ->1      ->inf         ->inf                   0
```

Two distinct failures, and it matters that they are separate:

1. **Estimator variance -- fixable.** A uniform-`t` grid has unbounded per-draw weight; with 32
   strata the top one sits at `t ~= 0.984`, weight ~62, and the estimate is hostage to one
   sample. Section 6 fixes this exactly.
2. **Extrapolation -- not fixable.** `v_theta` at `t = 0.9999` is untrained. Nats collected there
   are a readout of extrapolation, not of the model's density. This is why every table carries a
   `lambda_max`, why `analyze_fm_nats.py` sets `LAM_QUOTE = 6.0`, and why the truncation sweep is
   reported alongside every figure.

---

## 6. Sampling uniformly in lambda

Do not sample `t` and then apply an unbounded weight. Sample in the measure the ELBO is written
in, and let the Jacobian do the work. With a midpoint grid of `N` points on `[lam_lo, lam_hi]`:

```python
lam_lo = 2*log( t_min/(1-t_min) );  lam_hi = 2*log( t_max/(1-t_max) )
lam    = lam_lo + ((arange(N) + 0.5)/N) * (lam_hi - lam_lo)
t_all  = sigmoid(lam / 2)                          # inverse of lambda = 2*logit(t)
```

and then, since `L = (1/2) * integral t^2 ||v_hat-v||^2 dlambda`, each draw's contribution is

```
w_j = ( (lam_hi - lam_lo) / N ) * t_j^2 / 2                <-- BOUNDED, since t <= 1
```

That is `per_draw_cell()` in `analyze_fm_nats.py`, verbatim.

On the headline grid (`N=32`, `lambda in [-13.81, +9.19]`, `dlambda = 0.719`):

| | uniform-t | uniform-lambda (used) |
|---|---|---|
| per-draw weight | `t/(1-t)` -- unbounded | `dlambda * t^2/2` <= **0.359** |
| top draw's share of total weight | dominated by one sample | 9.7% |
| top 4 draws | -- | 38.1% |

Two further properties worth having:

- **It is deterministic quadrature, not random sampling.** The `t` grid contributes a *bias*
  (discretization), not a variance. Only `eps` is random.
- **Truncation is free.** Each draw owns its own lambda-cell, so summing a subset of draws IS the
  integral truncated to a lower `lambda_max`. The whole truncation sweep is post-hoc, no GPU.

The weight is applied **downstream** in `analyze_fm_nats.py`, never in the scorer, so any
existing run can be re-weighted without re-running anything.

> `--t-sampler beta` is rejected by the analyzer with a hard error: sampling from `pi(t)` is
> *itself* equivalent to weighting by `(t/(1-t))*pi(t)`, so applying the weight again
> double-counts it.

---

## 7. The estimator as implemented

Per event, per condition `c`, with `x` the **gold** action:

```
L(c) = (1/N) sum_j || v_theta( t_j*x + (1-t_j)*eps_j , t_j , KV(v,c) ) - (x - eps_j) ||^2
```

and the reported quantity is the weighted difference

```
LLR(c) = sum_j  w_j * ( ||v(...,c)_j - u_j||^2 - ||v(...,gold)_j - u_j||^2 )      nats
```

Four deliberate differences from training:

| | training | scorer |
|---|---|---|
| `x` | a sampled batch | always the **gold** `a*` for this event |
| `t, eps` | fresh random per step | **fixed array**, reused byte-for-byte across arms |
| `t` law | `Beta(1.5,1)` | uniform-lambda midpoint grid |
| reduction | scalar `mse_loss` | kept at `[64, 2]` |

### Common random numbers (CRN)

The per-draw loss has sd ~0.27 against condition gaps of ~0.02-0.12, so a naive difference of two
independently-sampled means is hopeless. Reusing the identical `(t_j, eps_j)` across arms and
differencing **per draw** before averaging cancels the shared term and buys a 2-9x SE reduction.
This is the Diffusion Classifier trick (2303.16203 sec 3.3).

It also makes `null` (gold re-scored against itself) come out at **exactly** 0.000000 -- a
determinism check, though *not* a noise floor; see the seed-7 section of `FM_HEAD_STATUS.md`.

### Keeping the residual unreduced

`sq = ((target - pred) ** 2).float()` is stored at `[N, 64, 2]`, not collapsed to a scalar.
Collapsing destroys the accel-vs-curvature split and the horizon profile, and they cannot be
recovered afterwards -- and the nav result lives entirely in that split. Index convention:
`resid` is flattened to 128 with `EVEN = accel`, `ODD = curvature`, waypoint `w` at `(w+1)*0.1 s`.

### Validation

`--verify` runs the fast path against the real `model.forward` with `model.diffusion`
monkeypatched to consume the identical draws. Per-draw level agreement is `max_abs ~ 1.5e-3`
(bf16 kernel selection differs with batch shape). Those errors do **not** cancel in the contrast
-- gold's 1.5e-3 and the control's 1.7e-3 *add* to 2.6e-3 -- so `--verify` asserts on the
**signed bias of the contrast** (+9.8e-5 +/- 1.5e-4), not on the level.

---

## 8. The control ladder

A near-zero content term is meaningless without a scale. Every arm is scored against the same
gold pass on the same CRN draws.

| arm | what changes | what it is for |
|---|---|---|
| `null` | nothing | must be **exactly** 0; determinism check |
| `wrongtraj` | the **target** `a*` -> another event's action | the yardstick: how many nats is a genuinely wrong trajectory worth? All content figures are quoted as a share of this |
| `blankvision` | camera frames zeroed | a large, unambiguous positive control |
| `random` | CoC -> another clip's prose | the headline content term |
| `cluster` | CoC -> another clip's prose, **same `event_cluster`** | holds topic fixed; isolates scene-specificity from genre |
| `shuffled` | this event's **own** words, permuted | same length, same bag of words, syntax destroyed. If the head reads the sentence this should hurt less than unrelated prose; if it reads keywords, not at all |
| `navnone` | route section absent entirely | nav *presence* |
| `navswap` | left <-> right inverted | nav *direction* -- the falsification test |

Nav requires `--nav`, which switches the component order to include `route` and the chat template
to `r1_5` (`r1` has no route case at all). Note `construct_route` returns `[]` for
`nav_text=None`, so the markers leave entirely -- **inverted** relative to the CoC case, where
the markers stay and only the prose leaves.

---

## 9. What this licenses, and what it does not

Write the bound with its gap named:

```
L(c) = -log p(a*|c) + Delta(c) ,      Delta(c) >= 0

LLR_hat = L(ctrl) - L(gold)
        = [ log p(a*|gold) - log p(a*|ctrl) ]  +  [ Delta(ctrl) - Delta(gold) ]
          \___________ what we want __________/     \______ the bias ________/
```

**The bias is the *difference* of slacks, not the slack.** `Delta` itself can be enormous and cost
nothing. Only asymmetry hurts.

And here is the uncomfortable part: `Delta(c)` measures how far `v_theta(., ., c)` is from being a
consistent velocity field -- i.e. **how well the model denoises**, which is exactly the quantity
being differenced. If gold conditioning denoises better than donor conditioning, then
`Delta(gold) < Delta(ctrl)` and the bias is **positive** -- inflating the effect in the direction
we are looking for. This is the leading candidate explanation for the 4.2x cross-head separation
mismatch (FM +3.27 vs token head +0.775).

**Why the headline survives it anyway:**

1. **The headline is a null.** A bias that inflates effects makes a null *more* credible. Content
   arms come in at 0.27-0.46% of separation.
2. **The content arms are compared to each other.** `random` / `cluster` / `shuffled` are all
   degraded conditioning in similar ways, so their slacks should be comparable -- which is why
   the paired tests between them are the cleanest thing in the study.
3. **The nav result is a channel dissociation.** Slack asymmetry is a scalar property of overall
   denoising quality; there is no route by which it produces zero on accel and large on
   curvature. An artefact would smear.

**Where it does bite:** every absolute nats level; the cross-head comparison; any ratio whose
numerator and denominator could be inflated differently.

### Three densities, never to be confused

| | what it is | status |
|---|---|---|
| `p_ODE` | continuous-time probability-flow density | computed, then **abandoned** -- >16 nats from the deployed map |
| `p_ELBO-bounded` | what this surrogate bounds (an SDE model, per Albergo et al. 2303.08797 Remark 2.11) | **what we use** |
| `p_10-step` | the actual deployed Euler map | **the validation arm** -- see below |

### The test that would settle it

The deployed sampler is a deterministic 10-step Euler map:

```
x_0 ~ N(0, I);   for k = 0..9:  x_{k+1} = x_k + dt * v_theta(x_k, t_k),   dt = 1/10, t_k = k/10
```

A deterministic diffeomorphism has an **exact** log-density by change of variables at each step
-- no bound, no slack, no Monte Carlo:

```
log p(y) = log N(x_0; 0, I) - sum_k log |det( I + dt * J_v(x_k, t_k) )| ,   x_0 = T^{-1}(y)
```

At `D = 128` the full Jacobian is 128 forward-mode JVPs per step, 10 steps -- affordable, and
crucially there is **no discretization error to converge away**, because the 10-step map *is* the
object of study. The inverse is obtained by fixed-point iteration on each step (contracts while
`dt * Lip(v) < 1`), and the round-trip residual `||T(x_0) - y||` is the only approximation in the
whole estimator.

If this reproduces the surrogate's ratios, the slack asymmetry is inert. If separation collapses
toward ~0.8 nats, we have found the inflation. Status in `FM_HEAD_STATUS.md`.
