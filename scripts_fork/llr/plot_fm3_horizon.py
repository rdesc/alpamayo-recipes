# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Three edits over the horizon, flow-matching head -- the companion to the token head's
full3_horizon.png, on the SAME 901 events selected by the SAME rule.

Rows are the two control channels. The scorer keeps the unreduced (64,2) residual, which is the
same layout as the token head's 128 trajectory tokens (64 waypoints x 2 channels) -- BOTH heads
resolve acceleration and curvature, so this is a like-for-like decomposition, not a capability
unique to this head. The first waypoint is t=0.1s, which is where the Stop effect turns out to
live.

Solid = all events. Dashed = top presence decile. Same palette as the token-head figure, so the
two read as one study; validated for CVD (normal dE 18.9, deuteranopia 9.0, protanopia 8.4).
"""
import glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

TH  = "/mnt/efs/users/rod/results/llr_fm_flip_th"
LAD = "/mnt/efs/users/rod/results/llr_fm_ladder"
PRE = "/mnt/efs/users/rod/results/llr_fm_presence"

def rd(pat):
    x = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(pat))], ignore_index=True)
    N = int(x.draw.max()) + 1
    lo, hi = float(x.lam_lo.iloc[0]), float(x.lam_hi.iloc[0])
    x["w"] = ((hi - lo) / N) * x.t ** 2 / 2
    return x

def percell(x, cond):
    """Per-event weighted nats contrast vs gold, resolved to all 128 cells."""
    g = x[x.cond == "gold"].copy()
    G = np.stack(g["resid"].to_numpy())
    gk = g[["clip_id", "event_idx", "draw"]].reset_index(drop=True)
    gi = {(c, int(e), int(d)): i for i, (c, e, d) in enumerate(gk.itertuples(index=False))}
    s = x[x.cond == cond].copy()
    S = np.stack(s["resid"].to_numpy())
    rows = s[["clip_id", "event_idx", "draw", "w"]].reset_index(drop=True)
    idx = np.array([gi.get((c, int(e), int(d)), -1) for c, e, d in
                    zip(rows.clip_id, rows.event_idx, rows.draw)])
    ok = idx >= 0
    D = (S[ok] - G[idx[ok]]) * rows["w"].to_numpy()[ok, None]
    key = pd.MultiIndex.from_arrays([rows.clip_id[ok], rows.event_idx[ok]])
    # donors: average over donor_idx first, then sum the draw-cells
    df = pd.DataFrame(D, index=key)
    nd = s.groupby(["clip_id", "event_idx"])["donor_idx"].nunique().clip(lower=1)
    out = df.groupby(level=[0, 1]).sum()
    return out.div(nd.reindex(out.index).to_numpy()[:, None])

TH_D, LAD_D, PRE_D = rd(f"{TH}/*.parquet"), rd(f"{LAD}/*.parquet"), rd(f"{PRE}/*.parquet")
KIND = TH_D[TH_D.cond == "flipdir"].groupby(["clip_id", "event_idx"])["flip_kind"].first()
pres = percell(PRE_D, "empty").sum(axis=1)
pct = pres.rank(pct=True) * 100

ARMS = [("flip", percell(TH_D, "flipdir"), "FLIP — invert one directive word\n(Stop↔Proceed, left↔right)"),
        ("donor", percell(TH_D, "donor"),  "DONOR — replace the whole CoC\nwith another event's prose"),
        ("shuffled", percell(LAD_D, "shuffled"), "SHUFFLED — keep the words,\npermute their order")]

INK, LONG, LAT_C = "#16171c", "#1d5c54", "#8f3228"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#9aa7b4", "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": "#64707c", "ytick.color": "#64707c",
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
T = (np.arange(64) + 1) * 0.1

def curve(ax, M, ch, col, ls, lw, lab):
    A = M.to_numpy().reshape(len(M), 64, 2)[:, :, ch]
    m, se = A.mean(0), A.std(0, ddof=1) / np.sqrt(len(A))
    ax.plot(T, m, color=col, ls=ls, lw=lw, label=lab)
    ax.fill_between(T, m - se, m + se, color=col, alpha=0.13, lw=0)

# sharey="row": the three edits must be comparable WITHIN a channel, or the panels
# cannot be read against each other. Across channels the scales genuinely differ.
fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6), sharex=True, sharey="row")
for r, (ch, chname) in enumerate([(0, "acceleration"), (1, "curvature")]):
    for c, (name, M, title) in enumerate(ARMS):
        ax = axes[r, c]
        for k, col, lab in [("directive-longitudinal", LONG, "longitudinal"),
                            ("directive-lateral", LAT_C, "lateral")]:
            ev = KIND[KIND == k].index.intersection(M.index)
            if len(ev) < 10: continue
            curve(ax, M.loc[ev], ch, col, "-", 2.4, f"{lab} · all (n={len(ev)})")
            tv = pct.reindex(ev).dropna()
            top = tv[tv >= 90].index
            if len(top) >= 40:
                curve(ax, M.loc[top], ch, col, "--", 1.5, f"{lab} · top decile (n={len(top)})")
        ax.axhline(0, color="#4e4c47", lw=.9)
        ax.set_xlim(0, 6.4); ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
        if r == 1: ax.set_xlabel("horizon time (s)")
        if r == 0: ax.set_title(title, fontsize=9.5, loc="left")
        if c == 0: ax.set_ylabel(f"gold − edited   (nats / cell)\n{chname.upper()}")
axes[0, 0].legend(fontsize=7.4, frameon=False, loc="upper right")
fig.suptitle("Trajectory log-likelihood under gold vs. edited Chain-of-Causation — flow-matching head",
             fontsize=12.5, x=0.005, ha="left", y=0.985)
fig.text(0.005, 0.955,
         "Alpamayo 1.5, DEPLOYED action expert. Same 901 events and same flip rule as the "
         "discrete-token figure. Positive = the gold CoC makes the true trajectory more likely "
         "than the edited one.\nRows split the two control channels, as the token-head figure "
         "can also do. Bands ±1 SE over events. Reweighted-ELBO surrogate: compare shapes and "
         "ratios, not levels.", fontsize=8.6, ha="left", va="top", color="#4a4d57")
fig.tight_layout(rect=[0, 0, 1, 0.88])
for ext in ("png", "pdf"):
    fig.savefig(f"{TH}/fm3_horizon.{ext}", bbox_inches="tight")
print("saved", f"{TH}/fm3_horizon.png")
