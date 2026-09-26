# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Exact log-density of a DETERMINISTIC K-step Euler flow map. No variational slack.

The deployed sampler (alpamayo_r1/diffusion/flow_matching.py::_euler) is

    x_0 ~ N(0, I_D);   for k = 0..K-1:  x_{k+1} = x_k + dt * v(x_k, t_k)
    dt = 1/K,  t_k = k/K,  output = x_K

That is a composition of K diffeomorphisms, so its pushforward density is EXACT by the
change-of-variables formula -- no ELBO, no bound, no Monte Carlo:

    log p(y) = log N(x_0; 0, I) - sum_k log|det( I + dt * J_v(x_k, t_k) )|,   x_0 = T^{-1}(y)

Everything here is solver-exact by construction: the 10-step map IS the object of study, so
unlike the continuous-time ODE there is no discretisation error to converge away.
"""
from __future__ import annotations

import math
import torch


def jac_transpose(vfield, x, t, D, tchunk=16):
    """Full Jacobian of ``vfield`` at (x, t), returned TRANSPOSED: M[i, :] = dv/dx[:, i].

    Built with D forward-mode JVPs, batched in chunks. ``vfield`` must act batch-elementwise,
    so a batch of identical x with tangent e_i in row i yields column i of J in row i.
    logdet is transpose-invariant, so the caller can use M directly.
    """
    shp = x.shape[1:]
    eye = torch.eye(D, device=x.device, dtype=x.dtype)
    rows = []
    # no_grad is REQUIRED, not just an optimisation: forward-mode AD needs no reverse graph, but
    # without this every JVP also builds one and memory grows across the D/tchunk chunks until it
    # OOMs regardless of tchunk. (Same class of bug as the ODE divergence-sum leak.)
    with torch.no_grad():
        for s in range(0, D, tchunk):
            e = min(s + tchunk, D)
            xb = x.expand(e - s, *shp).contiguous()  # fwd-AD rejects stride-0 expand views
            tan = eye[s:e].reshape(e - s, *shp)
            _, jv = torch.func.jvp(lambda z: vfield(z, t), (xb,), (tan,))
            rows.append(jv.reshape(e - s, D).double().detach())
    return torch.cat(rows, 0)


def invert_map(y, vfield, K, tol=None, maxit=200):
    """Solve T(x_0) = y by walking the Euler recursion backwards.

    Each step x_{k+1} = x_k + dt*v(x_k,t_k) is implicit in x_k and is solved by a DAMPED fixed
    point x <- (1-w)x + w(x_{k+1} - dt*v(x,t_k)), which converges when w*dt*Lip(v) < 1. Two
    details matter, both learned from failures concentrated on off-manifold (wrongtraj) targets,
    where |x_0| ~ 10.8 against ~7.2 for gold and the field is stiffer:

      * the initial guess is one explicit back-step, not x_{k+1} itself, which squares the
        starting error;
      * w is halved whenever the residual grows, so a step that would oscillate still contracts.

    Returns x_0, the worst per-step residual, and the total iteration count. Note the per-step
    residual UNDERSTATES the final error, because errors compound backwards through the K steps
    -- always judge accuracy by the caller's round-trip, never by this.
    """
    if tol is None:
        tol = 1e-12 if y.dtype == torch.float64 else 1e-7
    dt = 1.0 / K
    x_next = y
    worst, iters = 0.0, 0
    for k in range(K - 1, -1, -1):
        t_k = k / K
        x = x_next - dt * vfield(x_next, t_k)      # one explicit back-step as the initial guess
        w, prev = 1.0, None
        d = float("inf")
        for _ in range(maxit):
            iters += 1
            g = x_next - dt * vfield(x, t_k)
            xn = x + w * (g - x)
            d = (xn - x).abs().max().item()
            if prev is not None and d > prev:      # diverging at this damping -> back off
                w = max(w * 0.5, 0.05)
            prev = d
            x = xn
            if d < tol:
                break
        worst = max(worst, d)
        x_next = x
    return x_next, worst, iters


def forward_replay(x0, vfield, K):
    """Run the deployed Euler map forward, returning [x_0, ..., x_K]."""
    dt = 1.0 / K
    xs = [x0]
    x = x0
    for k in range(K):
        x = x + dt * vfield(x, k / K)
        xs.append(x)
    return xs


def log_n01(z, D):
    return float(-0.5 * (z.double() ** 2).sum().item() - 0.5 * D * math.log(2.0 * math.pi))


def logp_discrete_map(y, vfield, K, D, tchunk=16, tol=None, max_outer=8, rt_tol=1e-5):
    """Exact log-density at ``y`` of the K-step Euler pushforward of N(0, I_D).

    Inversion is a fixed point followed by **damped Newton**. The fixed point alone is not enough:
    its per-step residual compounds backwards through the K steps (a measured per-step 5e-2 gave a
    round-trip of 9.6e-1). Since the log-determinant already needs every step Jacobian, the full
    map Jacobian ``G = A_{K-1}...A_0``, ``A_k = I + dt*J_v(x_k,t_k)``, comes for free, so a Newton
    step ``x0 += G^{-1}(y - T(x0))`` costs only a 128x128 solve.

    An UNGUARDED Newton step diverges badly on a minority of rows -- it drove round-trips to 1e16
    and, at the absurd x0 it lands on, made cond(G) look like 1e20 and produced spurious
    det(A_k) < 0. Those pathologies were an artefact of the diverged iterate, NOT a property of
    the map: on converged rows the spectrum of G is FULL RANK with cond ~ 2e2..5e3. So every step
    is now **line-searched on a cheap forward replay** (no Jacobians) and accepted only if it
    reduces the round-trip; the best iterate ever seen is what gets returned.

    ``diag['roundtrip']`` is ||T(x0) - y||_inf for the RETURNED x0 -- the density is exact at
    T(x0), so this is the only approximation, and rows must be filtered on it.
    ``diag['neg_sign']`` counts steps with det(A_k) < 0; nonzero means the map is locally
    orientation-reversing there and a single-preimage change of variables is invalid.
    """
    dt = 1.0 / K
    I = torch.eye(D, dtype=torch.float64, device=y.device)

    def replay_rt(x):
        with torch.no_grad():
            xs = forward_replay(x, vfield, K)
        return xs, (xs[-1] - y).abs().max().item()

    def jacs(xs):
        Ms = [jac_transpose(vfield, xs[k], k / K, D, tchunk) for k in range(K)]
        GT = I.clone()
        for M in Ms:
            GT = GT @ (I + dt * M)
        return Ms, GT

    with torch.no_grad():
        x0, fp_resid, iters = invert_map(y, vfield, K, tol=tol)
    xs, rt = replay_rt(x0)
    best = (x0, xs, rt)
    n_newton, n_reject = 0, 0

    for _ in range(max_outer):
        if rt < rt_tol:
            break
        Ms, GT = jacs(xs)
        try:
            z = torch.linalg.solve(GT.transpose(0, 1), (y - xs[-1]).reshape(-1).double())
        except Exception:
            break
        if not torch.isfinite(z).all():
            break
        improved = False
        for alpha in (1.0, 0.5, 0.25, 0.125):                 # backtracking line search
            xt = x0 + alpha * z.reshape(x0.shape).to(x0.dtype)
            xst, rtt = replay_rt(xt)
            if rtt < rt:
                x0, xs, rt, improved = xt, xst, rtt, True
                n_newton += 1
                if rt < best[2]:
                    best = (x0, xs, rt)
                break
            n_reject += 1
        if not improved:
            break

    x0, xs, rt = best
    Ms, GT = jacs(xs)                                          # Jacobians AT the returned iterate

    logdet, neg_sign = 0.0, 0
    for M in Ms:
        sign, ld = torch.linalg.slogdet(I + dt * M)
        if sign.item() <= 0:
            neg_sign += 1
        logdet += float(ld.item())

    sv = torch.linalg.svdvals(GT).cpu().numpy()
    lp0 = log_n01(x0.reshape(-1), D)
    return lp0 - logdet, {
        "svals": sv,
        "roundtrip": rt, "fp_resid": fp_resid, "fp_iters": iters,
        "newton": n_newton, "nrej": n_reject,
        "logdet": logdet, "logp0": lp0, "x0_norm": float(x0.double().norm().item()),
        "neg_sign": neg_sign, "cond_G": float(sv[0] / max(sv[-1], 1e-300)),
        "sv_max": float(sv[0]), "sv_min": float(sv[-1]),
        "rank_1em6": int((sv > sv[0] * 1e-6).sum()),
    }


# --------------------------------------------------------------------------------------
def selftest():
    """Analytic check: a linear field v(x, t) = a*x makes the K-step map a pure scaling.

        x_{k+1} = (1 + dt*a) x_k   =>   T(x_0) = c * x_0,  c = (1 + a/K)^K

    The pushforward of N(0, I) under scaling by c is N(0, c^2 I), whose density is known in
    closed form. The two must agree to float64.
    """
    torch.manual_seed(0)
    D, K = 8, 10
    for a in (0.7, -0.4, 2.0):
        def vf(x, t, a=a):
            return a * x
        y = torch.randn(1, D, dtype=torch.float64)
        lp, diag = logp_discrete_map(y, vf, K, D, tchunk=3)
        c = (1 + a / K) ** K
        exact = float(-0.5 * D * math.log(2 * math.pi * c * c) - (y.double() ** 2).sum().item() / (2 * c * c))
        print(f"  a={a:+.2f}  c={c:.6f}  logp={lp:+.9f}  exact={exact:+.9f}  "
              f"err={abs(lp-exact):.3e}  roundtrip={diag['roundtrip']:.2e}")
        assert abs(lp - exact) < 1e-9, "LINEAR-FIELD SELFTEST FAILED"

    # time-dependent field: v(x,t) = a*t*x  =>  c = prod_k (1 + dt*a*t_k)
    for a in (1.5, -0.8):
        def vf(x, t, a=a):
            tt = float(t) if not torch.is_tensor(t) else float(t)
            return a * tt * x
        y = torch.randn(1, D, dtype=torch.float64)
        lp, diag = logp_discrete_map(y, vf, K, D, tchunk=3)
        c = 1.0
        for k in range(K):
            c *= (1 + (1.0 / K) * a * (k / K))
        exact = float(-0.5 * D * math.log(2 * math.pi * c * c) - (y.double() ** 2).sum().item() / (2 * c * c))
        print(f"  a*t, a={a:+.2f}  c={c:.6f}  logp={lp:+.9f}  exact={exact:+.9f}  err={abs(lp-exact):.3e}")
        assert abs(lp - exact) < 1e-9, "TIME-DEPENDENT SELFTEST FAILED"

    # affine field with a shift: v(x,t) = a*x + b  =>  still a diffeo, logdet unchanged by b
    b = torch.randn(1, D, dtype=torch.float64)
    a = 0.5
    def vfb(x, t):
        return a * x + b
    y = torch.randn(1, D, dtype=torch.float64)
    lp, _ = logp_discrete_map(y, vfb, K, D, tchunk=3)
    c = (1 + a / K) ** K
    # T(x0) = c*x0 + s  with s = sum_k (1+a/K)^(K-1-k) * (b/K)
    s = torch.zeros_like(b)
    for k in range(K):
        s = s * (1 + a / K) + b / K
    x0 = (y - s) / c
    exact = float(-0.5 * D * math.log(2 * math.pi) - 0.5 * (x0 ** 2).sum().item() - D * math.log(c))
    print(f"  affine a={a:+.2f}  logp={lp:+.9f}  exact={exact:+.9f}  err={abs(lp-exact):.3e}")
    assert abs(lp - exact) < 1e-9, "AFFINE SELFTEST FAILED"
    print("  ALL SELFTESTS PASSED")


if __name__ == "__main__":
    selftest()
