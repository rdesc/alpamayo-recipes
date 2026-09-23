# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.

# An exact likelihood for the deployed action head

How to get a real `log p(a* | v, ell)` in **nats** out of the Stage-2 flow-matching action
expert -- the head that actually drives, and the one every other measurement in this directory
cannot reach.

Companion to [`README.md`](README.md) (the token-head study) and
[`phase0_llr_flow_matching.py`](phase0_llr_flow_matching.py) (the MSE surrogate). Read this
before touching the ODE path.

---

## 1. Why a new measurement was needed

Alpamayo has two ways of producing a trajectory, and the whole LLR study so far measured the
one that is not deployed.

**The discrete-token head (Stage 1).** The trajectory is emitted as 128 tokens -- 64 waypoints
x (acceleration, curvature). Being a language-model head, it answers the question directly:
teacher-force the ground-truth tokens, read the cross-entropy, and that *is*
`log p(a* | context)` in nats. Run it twice -- reasoning present, reasoning absent -- and
subtract. That subtraction is the entire study: the +0.0029 presence term, the zero content
term, the +0.775 separation yardstick.

**The flow-matching action expert (Stage 2).** The head that actually drives. It emits no
tokens and exposes **no logits**, so `log p(a*)` is not available to be read off, and the
subtraction the study is built on does not exist.

Hence the standing caveat at the bottom of `README.md`, and item #1 of the priority list in
`langforce_readme.md`: *"Highest priority: it decides whether Phase 1 is worth doing at all."*

This matters substantively, not just for completeness. The finding so far is a null -- the
reasoning's *content* does nothing. A null on a head nobody ships is a far weaker claim than a
null on the one that drives.

Two ways to recover a score. Both now exist:

| | what it gives | status |
|---|---|---|
| **MSE surrogate** (`phase0_llr_flow_matching.py`) | squared-error gap, **not** nats | works; cheap |
| **ODE likelihood** (this document) | a real log-density, **in nats** | prototype works |

---

## 2. What a flow-matching model is

The token head generates by sampling symbols. The action expert generates by **moving a point
through space**.

A trajectory is one point in 128-dimensional space (64 waypoints x 2 channels). The model learns
a **velocity field** `v_theta(x, t, c)`: given position `x`, time `t`, and conditioning `c`, which
way to move.

```
t = 0  ------------------------------------>  t = 1
pure noise                                    real trajectory
x ~ N(0, I)                                   driving behaviour
```

### Training

Stand on the straight line between a real trajectory `x` and some noise `eps`
(`flow_matching.py:140-173`):

```python
noisy_x = t * x + (1 - t) * noise
target  = x - noise
return torch.nn.functional.mse_loss(target, pred)
```

The target is the **time-derivative of the path**:

```
x_t      = t*x + (1 - t)*eps
dx_t/dt  = x - eps
```

Since the path is straight, that velocity is **constant along it** -- the same arrow at every
`t`. The model must produce it from a corrupted position, never seeing `x` or `eps` separately.

Written as an objective:

```
L(c) = E_{x, eps, t}  || v_theta( t*x + (1-t)*eps , t , c )  -  (x - eps) ||^2
```

### Inputs, and where the conditioning enters

```python
noisy_x       = t * x + (1 - t) * eps            # (B, 64, 2)   WHERE you are
action_embeds = model.action_in_proj(noisy_x, t) # (B, 128, H)  position + time -> tokens
eo = model.expert(
    inputs_embeds   = action_embeds,
    past_key_values = chunk_cache,               # <- ALL conditioning lives here
)
pred = model.action_out_proj(...).view(B, 64, 2)
```

| input | route |
|---|---|
| `noisy_x` -- current position | embedded into action tokens |
| `t` -- current time | embedded jointly with `noisy_x` (the Fourier `freqs` buffers) |
| `c` -- scene, prompt, **CoC reasoning** | **only** as the VLM's cropped KV cache |

That third row is what the whole action-head arm rests on. The expert has **no cross-attention
and receives no text embeddings**; everything linguistic reaches it through the cached prefix,
cropped at `<traj_future_start>+1`. This is why swapping CoC prose swaps the conditioning in the
same sense as on the token head.

Shapes: 128 dims in -> 128 dims out. That squareness gives the Jacobian a well-defined trace,
which Section 4 needs.

### Two useful identities

Given `noisy_x`, `t` and a predicted velocity `v`, both endpoints are recoverable:

```
x   = noisy_x + (1 - t)*v
eps = noisy_x - t*v
```

So **predicting the velocity is equivalent to predicting the clean trajectory**. The MSE
surrogate is therefore asking: does the right reasoning improve the model's guess at the true
trajectory, seen through noise?

### Why this defines a distribution

Many `(x, eps)` pairs give the same `noisy_x`. The model cannot recover which, so the best it can
do is `E[x - eps | noisy_x, t, c]`. That irreducible ambiguity is what spreads probability mass
-- and it is strongly `t`-dependent:

- near `t = 1`: `noisy_x ~= x`, nearly determined, **easy**, little room for conditioning to help
- near `t = 0`: `noisy_x ~= eps`, the model is guessing, **hard** -- and this is where reasoning
  content *could* matter

Consequences: most of the per-draw variance (sd ~0.27) is `t`, not `eps`; and a single scalar
averaged over `t` blends a regime where nothing can matter with one where everything could. The
`Delta`-vs-`t` check is not optional.

### The `t` range

```python
self.beta_scale_constant = 0.999                       # flow_matching.py:58
t = self.beta_scale_constant - t * self.beta_scale_constant
```

With `b ~ Beta(1.5, 1)`, training draws `t = 0.999*(1 - b)`, so `t` in **[0, 0.999]** -- capped
just below 1, where `noisy_x = x` exactly and the input carries no noise. The `Beta(1.5, 1)` shape
concentrates mass at **small `t`**: training deliberately oversamples the hard, noisy regime.

Our `--t-sampler stratified` replaces this with a flat grid `((j+0.5)/N)`, uniform on `(0,1)`.
Lower variance, but a **different weighting of the schedule**. This is why `--t-sampler beta`
exists, and why the empty-vs-gold contrast **flipped sign between N=16 and N=32**: different `N`
means a different grid means a genuinely different quantity. *Hold `N` fixed across anything you
compare.*

### Generation

```python
x = torch.randn(batch_size, *self.x_dims)               # start at noise, t=0
time_steps = torch.linspace(0.0, 1.0, inference_step + 1)
for i in range(inference_step):
    v = step_fn(x=x, t=t_start)
    x = x + dt * v
```

Two properties of this make a likelihood possible:

1. **Deterministic.** All randomness is in the starting point. Same `x(0)`, same trajectory.
2. **Invertible.** Run the arrows backward from a real trajectory and you land on the specific
   noise point that would have produced it.

A deterministic, invertible map with a known distribution at one end gives an exact density at
the other.

---

## 3. How a density falls out of an invertible map

**Probability mass is conserved; volume is not.** Density is mass per unit volume, so knowing how
much the map stretches space tells you how much the density must drop.

In one dimension, with `z ~ N(0,1)` and `x = 2z`, the same mass is spread over twice the space:

```
p_x(x) = p_z(z) * |dz/dx|
```

In `D` dimensions the stretch factor is a Jacobian determinant:

```
log p_1(x) = log p_0(z) + log |det dz/dx|
```

A 128x128 determinant would be unpleasant. But the map is not one jump -- it is an ODE built from
infinitesimal steps, and in that limit the determinant collapses to the **instantaneous change of
variables**:

```
d log p_t(x(t)) / dt  =  - div v_theta(x(t), t, c)
```

The log-determinant becomes an integral of the **divergence** -- the *trace* of the Jacobian, a
sum of 128 diagonal entries, not a determinant.

> **Divergence, physically:** the rate at which the flow expands volume at a point. A small blob
> of dye carried by the field spreads if divergence is positive (density thins), concentrates if
> negative (density rises).

Integrating from noise to data:

```
log p(a* | v, ell)  =  log p_0(x(0))  -  integral_0^1  div v_theta(x(t), t, c) dt
                       \___________/     \_________________________________/
                       known Gaussian     volume change along the path
```

Which is literally the benchmark's inner loop:

```python
div_total = div_total - div.double() * dt.double()     # d log p/dt = -div v
x = x + dt * v                                         # step backward: dt < 0
...
lp0 = -0.5 * (x.double() ** 2).sum() - 0.5 * D * np.log(2 * np.pi)   # log N(x(0); 0, I)
return (lp0 + div_total).item()
```

Two lines of accounting: where the noise point landed, and how much the flow stretched space.

**Nothing here is a bound, a surrogate, or a sampled expectation.** Given the velocity field,
`log p(a*)` is determined. The only error is *numerical* -- the discretization of the integral --
which is a knob we control and can check by convergence. Contrast the MSE, which is proportional
to an NLL bound only under a `t`-weighting the code does not apply.

### What a log-density is, and is not

`P(X = x) = 0` for a continuous variable. `p(x)` is **not** a probability -- it is probability
*per unit volume*:

```
p(x) = lim_{vol -> 0}  P(X in box around x) / vol(box)
```

a *rate*, which may exceed 1. So `log p(a*) = -315.66` does **not** claim `a*` has probability
`e^-315.66`. Magnitudes are large because `D = 128` and density carries units of `1/volume`. For
scale, a standard normal in 128 dimensions has log-density about **-181.6** at a typical point.

**The consequence that matters.** Because density has units of `1/volume`, its absolute value is
**coordinate-dependent** -- rescale the action space and every `log p` shifts by `log|det J|` of
the rescaling. An absolute log-density is not a quotable quantity. But

```
LLR = log p(a* | v, ell) - log p(a* | v)
```

uses the same point, the same coordinates, the same volume element -- **the units cancel
exactly**. The LLR is coordinate-free even though neither term is. This is the continuous-case
form of the rule already in `README.md`: *quote the contrast, never the level.*

**Comparing to the token head.** The token head yields a probability over **bins**
(`num_bins: 3000` over `dims_min/max = +/-10`), not a density. The two relate by roughly

```
log P(bin) ~= log p(a*) + 128 * log(bin width)
```

a fixed offset of about **-641 nats** here. So the two heads' *levels* are different kinds of
object and must not share a table -- but that offset is constant, so it **cancels in the LLR**,
and the *differences* legitimately can. This is a better answer than the share-of-separation
workaround.

It also gives a free correctness check: convert the ODE density to a bin probability and see
whether it lands near the token head's measured `log p`. Order-of-magnitude agreement is evidence
the ODE is right; wild disagreement means a bug. **Not yet run.**

---

## 4. Computing the divergence

`div v` is the **trace** of the Jacobian -- 128 diagonal entries, not a determinant.

```
div v = sum_i d v_i / d x_i  =  trace(J),    J = dv/dx   (128 x 128)
```

**Hutchinson's estimator** (`E[u^T J u] = trace(J)` for random `u`) is the standard choice and
costs one Jacobian-vector product. We rejected it: it is unbiased but noisy, the noise enters at
every integration step and accumulates, and the effects here are ~1e-3 nats. *(In hindsight this
was over-caution -- with the same probes reused across conditions the error is largely
common-mode and cancels in a paired difference. Nobody in the literature does that, so it was
untested either way.)*

**Exact, via 128 Jacobian-vector products.** Forward-mode AD gives `J @ u` for any direction `u`;
feed it basis vector `e_j`, take the `j`-th entry of the result, and that is `J_jj`. Batch rows
are independent, so 128 copies of `x` with 128 different tangents give the whole diagonal in one
(chunked) call. Reverse mode via `vjp` gives the same numbers -- verified to `|d| = 0` on three
analytic fields -- and is both faster and compatible with the fast attention kernel.

The price is 128 network evaluations per integration step. Only the small action expert repeats;
the 10B VLM runs once per condition and its cropped KV cache is reused.

**Three engineering traps, all of which cost real time:**

| trap | symptom | fix |
|---|---|---|
| `expand()` gives stride-0 views | forward AD refuses to `make_dual` on aliased memory | `.contiguous()` on the primal only |
| `use_cache=True` | 79 GB OOM: the expert concatenates its keys onto the batch-expanded image-heavy prefix, once per row | `use_cache=False`; each step re-attends to the same fixed prefix |
| no rule for fused SDPA | `NotImplementedError` under forward AD | force `SDPBackend.MATH`, or use reverse mode and keep the fast kernel |

Plus one that did **not** crash: the running divergence sum retained an autograd graph across
every chunk and every step (78 GB) until it was detached.

---

## 5. Why this path was abandoned

Three findings, in increasing order of how decisive they are.

**(a) Euler is hopeless here, and better solvers only half-fix it.** On the analytic test
`v = a*t*x`, reaching `|err| < 1e-4` costs **2,560** divergence evaluations with Euler and **20**
with RK4 or Heun -- a 128x difference, with measured convergence orders 1.0 / 1.9 / 3.9. On the
model, RK4 duly converged the *level* to ~24-25 nats with as few as 8 evaluations. But the
gold-vs-donor **contrast** scattered between -0.30 and +3.55 with sign flips and no trend.

**(b) The contrast is noise-limited, not solver-limited.** Two runs at an identical config differ
by **0.029 nats** -- roughly 30x the effects of interest. Everything ran under `bf16` autocast
(~8 mantissa bits); accumulating a 128-term trace over dozens of steps toward a 1e-3 target is
below that floor, and unlike the surrogate (where common random numbers make rounding
common-mode) the ODE states diverge between conditions, so nothing cancels.

**(c) The decisive one: it is the wrong distribution.** The deployed sampler is
`num_inference_steps = 10`, `int_method = euler` -- verified in the checkpoint config, absent
from every YAML so the class default applies, with `inference_guidance_weight = 1.0` (no CFG).
**The deployed policy IS a 10-step map with h = 0.1.** The continuous-time ODE density is the
`n -> infinity` limit: a distribution the car never samples from, and our own numbers put the two
more than 16 nats apart. No solver fixes that.

If an exact likelihood is ever wanted, the right object is the **discrete 10-step map**:
`log p(a*) = log N(x_0; 0, I) - sum_{i=0}^{9} log|det(I + h * dv/dx(x_i, t_i))|`, which has
**zero** discretization error because the 10-step map is the model. It costs about what our
n=10 ODE run did. That is the sensible validation arm; the continuous limit is not.

---

## 6. What we actually use: the reweighted denoising gap

The flow-matching velocity MSE is the diffusion ELBO's eps-MSE under a known weighting, so
dividing the weighting out recovers a genuine likelihood -- at the cost of the *cheap* measurement
we already had, with no ODE at all.

Kingma & Gao (arXiv 2303.00848) Eq. 4 writes the family as
`L_w = 1/2 * int w(lambda) E||eps_hat - eps||^2 dlambda`, with the **ELBO at `w(lambda) = 1`** and
the **flow-matching OT path at `w(lambda) = e^{-lambda/2} = (1-t)/t`** (their Table 1). For our
interpolant, `lambda = 2 log(t/(1-t))`, `dlambda/dt = 2/(t(1-t))`, and
`eps_hat - eps = -t (v_hat - v)`. Substituting, the 1/2 and the 2 cancel and the weight on the
**velocity** MSE is simply

```
LLR(c1 vs c2) = E_{t,eps}[ (t/(1-t)) * ( ||v(x_t,t,c2) - u||^2 - ||v(x_t,t,c1) - u||^2 ) ]   nats
```

with `u = a* - eps` and the SAME `(t, eps)` in both arms. Every additive constant cancels at
fixed `a*`. **Verified numerically** against a Gaussian where the optimal velocity field is
closed-form and the bound is tight: recovered the true log-density difference to ~2e-3 nats
(Monte-Carlo limited).

**Sample uniformly in `lambda`, not in `t`.** The weight `t/(1-t)` is 255 at the top stratum of a
128-point uniform-`t` grid, so uniform-`t` puts almost no samples where nearly all the weight
lives. Under a uniform-`lambda` grid, `dt = (t(1-t)/2) dlambda` turns the per-draw weight into a
**bounded** `(lam_hi - lam_lo) * t^2 / 2`. Measured: 7x more accurate at equal budget.

**And truncate inside training support.** The training sampler is `Beta(1.5,1)` scaled by 0.999,
which puts ~3.1% of its mass above `t = 0.9`, ~0.085% above `t = 0.99`, and exactly zero above
`t = 0.999` -- while the ELBO weight there is 9, 99 and 999. *The weight is largest exactly where
the model was barely trained*, and the integral of the weight diverges, so any absolute nats
figure is a function of where you truncate. **Report the LLR as a curve in `lambda_max`, with
evidence of a plateau, never as a single number.** Do not combine `--t-sampler beta` with the
weight: sampling from `pi(t)` is itself equivalent to weighting by `(t/(1-t)) pi(t)`, so applying
both double-counts.

Precedent, which is the point -- this design is deliberately not novel: Flow Policy Optimization
(arXiv 2507.21053) uses exactly this ratio for action distributions with shared `(eps, t)`;
Diffusion Classifier (arXiv 2303.16203, sec 3.3) originates the shared-noise trick; Kong et al.
(arXiv 2310.07972) give the pointwise LLR identity and a lower-variance magnitude-only variant.

**Caveats to state in any writeup:** a difference of two ELBO bounds is not a bound on the
difference (the slack differs per condition; Song et al. 2101.09258 Table 2 puts the
ELBO-vs-exact gap at 0.10-0.13 bpd on CIFAR-10); the bound is on the SDE model, not the ODE one
(Albergo et al. 2303.08797, Remark 2.11); and Cross & Ragni (arXiv 2409.06364) find conditional
diffusion likelihood largely agnostic to its conditioning, i.e. it can measure nuisance
structure -- which is what the length-matched `shuffled` arm exists to rule out.

---

## Status

See [`FM_HEAD_STATUS.md`](FM_HEAD_STATUS.md) for the running log and current numbers.
