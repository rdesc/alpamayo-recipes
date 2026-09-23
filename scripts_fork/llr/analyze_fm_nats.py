# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Turn phase0_llr_flow_matching.py's per-draw output into a log-likelihood ratio in NATS.

The scorer stores the raw per-draw squared error. The ELBO weighting is applied HERE, so any
existing run can be re-weighted without touching a GPU. See THE NATS ESTIMATOR in
phase0_llr_flow_matching.py for the derivation; in brief, for the linear interpolant the
velocity MSE is the ELBO's eps-MSE under w(lambda) = (1-t)/t, so dividing that out gives

    LLR = E[ (t/(1-t)) * ( ||v(c_ctrl) - u||^2 - ||v(gold) - u||^2 ) ]        nats

and under a uniform-lambda grid the change of variables makes the per-draw weight

    w_j = (lam_hi - lam_lo) * t_j^2 / 2                                       (bounded)

POSITIVE = the gold reasoning makes the true trajectory MORE likely. This matches the sign of
the token head's llr, and unlike the raw MSE gap it is in the same units, so the two heads'
CONTRASTS may be compared directly (the levels may not -- see the doc).

Usage:
    python analyze_fm_nats.py /path/to/pilot.shard*.parquet
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd


def per_draw_cell(df: pd.DataFrame, n_draws: int) -> np.ndarray:
    """Each draw's CELL contribution to the weighted integral (sum these, do not average).

    Under a uniform-lambda grid of n_draws points spanning [lam_lo, lam_hi], draw j represents a
    cell of width (lam_hi-lam_lo)/n_draws, and the change of variables dt = (t(1-t)/2) dlambda
    turns the ELBO weight t/(1-t) into a BOUNDED t^2/2. So

        contribution_j = ((lam_hi - lam_lo) / n_draws) * (t_j^2 / 2) * delta_j

    Summing a subset of draws therefore gives the integral truncated to that lambda range, which
    is what makes the lambda_max sweep below a pure post-hoc operation.
    """
    tt = df["t"].to_numpy(dtype=np.float64)
    sampler = df["t_sampler"].iloc[0] if "t_sampler" in df.columns else "stratified"
    if sampler == "lambda":
        lam_lo = float(df["lam_lo"].iloc[0])
        lam_hi = float(df["lam_hi"].iloc[0])
        return ((lam_hi - lam_lo) / n_draws) * tt**2 / 2.0
    if sampler == "beta":
        raise SystemExit(
            "--t-sampler beta is already an importance distribution: sampling from pi(t) is "
            "itself equivalent to weighting by (t/(1-t))*pi(t), so applying the weight again "
            "double-counts it. Re-run with --t-sampler lambda, or divide pi out explicitly.")
    # uniform-in-t grid: cell width 1/n_draws, weight t/(1-t) -- unbounded at the top stratum.
    return (1.0 / n_draws) * tt / (1.0 - tt)


def lam_of(t: np.ndarray) -> np.ndarray:
    return 2.0 * np.log(t / (1.0 - t))


LAM_QUOTE = 6.0   # quote inside training support; see the truncation warning below


def main(paths: list[str]) -> None:
    files: list[str] = []
    for p in paths:
        files.extend(sorted(glob.glob(p)))
    if not files:
        raise SystemExit(f"no parquet matched {paths}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    sampler = df["t_sampler"].iloc[0] if "t_sampler" in df.columns else "stratified"
    print(f"{len(files)} shard(s), {len(df):,} rows, "
          f"{df[['clip_id','event_idx']].drop_duplicates().shape[0]:,} events, sampler={sampler}")
    if sampler != "lambda":
        print("WARNING: uniform-t sampling puts almost no draws where the ELBO weight lives; "
              "treat the nats figure as indicative only.")
    n_draws = int(df["draw"].max()) + 1
    df["w"] = per_draw_cell(df, n_draws)
    df["lam"] = lam_of(df["t"].to_numpy(dtype=np.float64))

    gold = (df[df.cond == "gold"]
            .set_index(["clip_id", "event_idx", "draw"])[["loss", "w", "lam"]]
            .rename(columns={"loss": "loss_gold"}))
    out = []
    for cond in sorted(set(df.cond) - {"gold"}):
        arm = (df[df.cond == cond]
               .groupby(["clip_id", "event_idx", "draw"])["loss"].mean().rename("loss_ctrl"))
        j = gold.join(arm, how="inner").dropna()
        if j.empty:
            continue
        # weight and difference PER DRAW (CRN pairing), SUM over draws within an event
        j["contrib"] = j["w"] * (j["loss_ctrl"] - j["loss_gold"])
        per_event = j.groupby(["clip_id", "event_idx"])["contrib"].sum()
        raw = (j.assign(r=j["loss_ctrl"] - j["loss_gold"])
                 .groupby(["clip_id", "event_idx"])["r"].mean())
        v = per_event.to_numpy()
        n = len(v)
        from scipy import stats
        wp = stats.wilcoxon(v).pvalue if n > 10 else float("nan")
        out.append(dict(arm=cond, n=n, nats=v.mean(), se=v.std(ddof=1) / np.sqrt(n),
                        median=float(np.median(v)), trimmed=float(stats.trim_mean(v, 0.1)),
                        frac_pos=float((v > 0).mean()), wilcoxon_p=wp,
                        raw_mse_gap=float(raw.mean())))
    r = pd.DataFrame(out)
    print("\n=== LLR vs gold, in NATS (positive = gold reasoning helps) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(r.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))

    # ---------------------------------------------------------------------------------
    # THE LOAD-BEARING DIAGNOSTIC: LLR as a function of where we truncate.
    #
    # The ELBO weight t/(1-t) is largest exactly where the model was barely trained -- the
    # training sampler is Beta(1.5,1) scaled by 0.999, which puts only ~3% of its mass above
    # t=0.9 and ~0.08% above t=0.99, while the weight there is 9 and 99. The integral of the
    # weight diverges, so ANY absolute nats figure is a function of the truncation. The
    # DIFFERENCE between two conditionings should decay at both ends and be truncation-
    # insensitive -- if it is not, the number is an artefact of the untrained tail and must
    # not be quoted. Report the curve, not a point.
    # ---------------------------------------------------------------------------------
    print("\n=== LLR vs truncation lambda_max (t_max in brackets) ===")
    print("    quote a number only where this has plateaued, and only inside training support")
    print("    training mass:  t>0.9 ~ 3.1%   t>0.99 ~ 0.085%   t>0.999 = 0")
    cuts = [2.0, 4.0, 6.0, 9.2, 11.0, 13.9]
    hdr = "arm".ljust(14) + "".join(f"  lam<={c:<5.1f}[t<={1/(1+np.exp(-c/2)):.4f}]" for c in cuts)
    print(hdr)
    for cond in sorted(set(df.cond) - {"gold"}):
        arm = (df[df.cond == cond]
               .groupby(["clip_id", "event_idx", "draw"])["loss"].mean().rename("loss_ctrl"))
        j = gold.join(arm, how="inner").dropna()
        if j.empty:
            continue
        j["contrib"] = j["w"] * (j["loss_ctrl"] - j["loss_gold"])
        cells = []
        for c in cuts:
            sub = j[j["lam"] <= c]
            v = sub.groupby(["clip_id", "event_idx"])["contrib"].sum()
            cells.append(f"{v.mean():+21.5f}" if len(v) else f"{'--':>21}")
        print(cond.ljust(14) + "".join(cells))

    # ---------------------------------------------------------------------------------
    # LENGTH CONFOUND TEST.
    #
    # Swapping CoC prose changes its token count, which changes the cropped prefix length, which
    # shifts the action tokens' position ids (rope_deltas + cache length). That is a nuisance
    # channel with a signature -- an effect concentrated in the earliest waypoints, which is what
    # the pilot showed. Two independent checks:
    #   (a) regress the per-event LLR on the signed prefix-length difference. A real content
    #       effect should have slope ~0 and a nonzero intercept.
    #   (b) `shuffled` is length-matched to gold by construction (permuting words preserves the
    #       token count), so its effect CANNOT be length. If shuffled ~ donor, length is not the
    #       driver; if shuffled ~ 0 while donor is positive, it is.
    # ---------------------------------------------------------------------------------
    if "prefix_len" in df.columns and (df["prefix_len"] > 0).any():
        print("\n=== length confound (lambda<=%.1f) ===" % LAM_QUOTE)
        sub = df[df["lam"] <= LAM_QUOTE]
        pl = (sub.groupby(["clip_id", "event_idx", "cond", "donor_idx"])["prefix_len"]
              .first().reset_index())
        plg = (pl[pl.cond == "gold"].set_index(["clip_id", "event_idx"])["prefix_len"]
               .rename("plen_gold"))
        gsub = (sub[sub.cond == "gold"]
                .set_index(["clip_id", "event_idx", "draw"])[["loss", "w"]]
                .rename(columns={"loss": "lg"}))
        from scipy import stats
        print(f"{'arm':<14}{'d_prefix (mean|abs|)':>22}{'slope':>12}{'p':>10}{'intercept':>12}")
        for cond in sorted(set(sub.cond) - {"gold"}):
            a = (sub[sub.cond == cond]
                 .groupby(["clip_id", "event_idx", "draw"])["loss"].mean().rename("lc"))
            j = gsub.join(a, how="inner").dropna()
            if j.empty:
                continue
            per_event = (j.assign(c=j["w"] * (j["lc"] - j["lg"]))
                         .groupby(level=[0, 1])["c"].sum())
            dl = (pl[pl.cond == cond].groupby(["clip_id", "event_idx"])["prefix_len"].mean()
                  - plg)
            m = pd.concat([per_event.rename("llr"), dl.rename("dlen")], axis=1).dropna()
            if len(m) < 10 or m["dlen"].std() == 0:
                print(f"{cond:<14}{m['dlen'].abs().mean():>22.2f}"
                      f"{'(length-matched)':>34}")
                continue
            lr = stats.linregress(m["dlen"], m["llr"])
            print(f"{cond:<14}{m['dlen'].abs().mean():>22.2f}{lr.slope:>+12.6f}"
                  f"{lr.pvalue:>10.2e}{lr.intercept:>+12.5f}")
        print("    slope ~ 0 with a nonzero intercept => the effect is NOT length.")
        print("    `shuffled` is NEARLY length-matched: permuting words usually but not always")
        print("    preserves the token count (measured mean |d_prefix| ~ 0.4, vs ~3.9 for donor),")
        print("    so its row is the cleanest control -- but it is not exactly 0.")

    # ---------------------------------------------------------------------------------
    # CHANNEL AND HORIZON DECOMPOSITION, from the stored (64, 2) residual.
    #
    # `resid` is flattened row-major, so index = waypoint*2 + channel: EVEN = acceleration,
    # ODD = curvature, waypoint w at t = (w+1)*0.1 s. Same layout as the token head's 128
    # trajectory tokens, which is what makes the two studies' splits comparable.
    #
    # Caveat: the velocity field couples dimensions, so this is a decomposition of the LOSS,
    # not a set of independent per-dimension likelihoods. Read it the way the token head's
    # per-token LLR is read -- as attribution, not as separable evidence.
    # ---------------------------------------------------------------------------------
    if "resid" in df.columns:
        print("\n=== channel split (nats, lambda<=%.1f) ===" % LAM_QUOTE)
        sub = df[df["lam"] <= LAM_QUOTE]
        R = np.stack(sub["resid"].to_numpy())                      # (rows, 128)
        acc = R[:, 0::2].mean(axis=1)
        cur = R[:, 1::2].mean(axis=1)
        s = sub[["clip_id", "event_idx", "draw", "cond", "w"]].copy()
        s["acc"], s["cur"] = acc, cur
        g2 = (s[s.cond == "gold"].set_index(["clip_id", "event_idx", "draw"])
              [["acc", "cur", "w"]].rename(columns={"acc": "acc_g", "cur": "cur_g"}))
        print(f"{'arm':<14}{'accel':>12}{'curvature':>12}{'ratio a/c':>12}")
        for cond in sorted(set(sub.cond) - {"gold"}):
            a2 = (s[s.cond == cond].groupby(["clip_id", "event_idx", "draw"])[["acc", "cur"]]
                  .mean().rename(columns={"acc": "acc_c", "cur": "cur_c"}))
            j2 = g2.join(a2, how="inner").dropna()
            if j2.empty:
                continue
            na = j2.groupby(level=[0, 1]).apply(
                lambda d: (d["w"] * (d["acc_c"] - d["acc_g"])).sum(), include_groups=False).mean()
            nc = j2.groupby(level=[0, 1]).apply(
                lambda d: (d["w"] * (d["cur_c"] - d["cur_g"])).sum(), include_groups=False).mean()
            rat = na / nc if abs(nc) > 1e-12 else float("nan")
            print(f"{cond:<14}{na:>+12.5f}{nc:>+12.5f}{rat:>12.2f}")

        print("\n=== horizon profile (nats per 0.8 s bin, lambda<=%.1f) ===" % LAM_QUOTE)
        print("    bins AVERAGE to the arm's total above; reasoning should help MORE at long")
        print("    horizon, where vision and ego motion run out. The token head is flat (+/-0.007).")
        W = R.reshape(len(R), 64, 2).mean(axis=2)                  # mean over channels
        hb = np.stack([W[:, i * 8:(i + 1) * 8].mean(axis=1) for i in range(8)], axis=1)
        hcols = [f"h{i}" for i in range(8)]
        sh = sub[["clip_id", "event_idx", "draw", "cond", "w"]].copy()
        for i, c in enumerate(hcols):
            sh[c] = hb[:, i]
        gh = (sh[sh.cond == "gold"].set_index(["clip_id", "event_idx", "draw"])[hcols + ["w"]])
        print(f"{'arm':<14}" + "".join(f"{(i+1)*0.8:>9.1f}s" for i in range(8)))
        for cond in sorted(set(sub.cond) - {"gold"}):
            # average donors within (event, draw) FIRST -- otherwise every donor is counted
            # separately and the profile comes out n_donors times too large.
            ah = (sh[sh.cond == cond]
                  .groupby(["clip_id", "event_idx", "draw"])[hcols].mean())
            jh = gh.join(ah, how="inner", lsuffix="_g", rsuffix="_c").dropna()
            if jh.empty:
                continue
            nev = jh.reset_index()[["clip_id", "event_idx"]].drop_duplicates().shape[0]
            wv = jh["w"].to_numpy()
            prof = np.array([(wv * (jh[f"{c}_c"].to_numpy() - jh[f"{c}_g"].to_numpy())).sum()
                             for c in hcols]) / max(nev, 1)
            print(f"{cond:<14}" + "".join(f"{x:>+10.5f}" for x in prof)
                  + f"   (mean {prof.mean():+.5f})")

    # where in the schedule the effect actually lives
    print("\n=== weighted contribution by lambda bin (per arm) ===")
    print("    a real effect should have COMPACT SUPPORT: ~0 at both ends. A monotone ramp into")
    print("    the top bin means the untrained tail is driving it.")
    edges = np.array([-14, -9, -6, -3, 0, 3, 6, 9, 14.0])
    for cond in sorted(set(df.cond) - {"gold"}):
        arm = (df[df.cond == cond]
               .groupby(["clip_id", "event_idx", "draw"])["loss"].mean().rename("loss_ctrl"))
        j = gold.join(arm, how="inner").dropna().reset_index()
        if j.empty:
            continue
        j["contrib"] = j["w"] * (j["loss_ctrl"] - j["loss_gold"])
        b = j.groupby(pd.cut(j["lam"], edges, include_lowest=True), observed=False)["contrib"].sum()
        nev = j[["clip_id", "event_idx"]].drop_duplicates().shape[0]
        print(f"  {cond:<13}" + " ".join(f"{x/max(nev,1):+9.5f}" for x in b.to_numpy()))
    print("   bins:        " + " ".join(f"{lo:+4.0f}..{hi:<4.0f}"
                                        for lo, hi in zip(edges[:-1], edges[1:])))


if __name__ == "__main__":
    main(sys.argv[1:] or ["/mnt/efs/users/rod/results/llr_fm_nats_pilot/pilot.shard*.parquet"])
