# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.

"""Analysis for Phase 2, Experiment A (``phase2_sample_displacement.py``).

Nine readouts, in the order they should be read. The first is a gate: if `null` is not exactly
zero the rest of the file means nothing.

 1. determinism      `null` displacement must be exactly 0 (same conditioning, same x_0).
 2. headline         displacement per arm in metres, against the `blankvision` yardstick.
 3. arm separation   paired Wilcoxon donor/cluster/shuffled. Phase 1 could not tell these three
                     apart, and that indistinguishability IS its occupancy result. If the
                     sampler can tell them apart, the behavioural question has a different
                     answer from the likelihood question.
 4. scale-free       displacement / the sampler's OWN spread, per event. "Does swapping the
                     prose move the car further than re-rolling the dice?"
 5. accuracy         does the swap make the prediction WORSE against ground truth? Displacement
                     alone only says the rollouts differ; this says whether the difference is a
                     degradation, which is the part a LangForce-style regulariser would need.
 6. reliability      split-half over draws, Spearman-Brown. Required before ANY correlation in 7.
 7. cross-phase      per-event displacement vs the Phase-1 per-event likelihood gap. Same scenes?
 8. horizon          displacement vs waypoint. A real behavioural divergence compounds with
                     horizon; Phase 1's content arms instead spiked at 0.8 s, which never made
                     physical sense.
 9. channel          longitudinal vs lateral, in the gold rollout's own frame. Phase 1: content
                     moved acceleration only, curvature exactly zero. Does that survive?

Usage:
    python analyze_phase2_displacement.py --glob '/mnt/efs/.../llr_fm_sampdisp/samp.shard*.parquet'
"""

import argparse
import glob as globmod

import numpy as np
import pandas as pd
from scipy import stats

ARM_ORDER = ["null", "donor", "cluster", "shuffled", "blankvision"]

ap = argparse.ArgumentParser()
ap.add_argument("--glob", default="/mnt/efs/users/rod/results/llr_fm_sampdisp/samp.shard*.parquet")
ap.add_argument("--phase1-events",
                default="/mnt/efs/users/rod/results/llr_fm_ladder/per_event_fm.parquet",
                help="Phase-1 per-event surrogate LLR per arm, for readout 7. Skipped if absent.")
ap.add_argument("--trim", type=float, default=0.10)
ap.add_argument("--split", default=None, help="Restrict to one dataset split (train|val).")
args = ap.parse_args()

paths = sorted(globmod.glob(args.glob))
if not paths:
    raise SystemExit(f"no shards matched {args.glob}")
df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
if args.split:
    df = df[df["split"] == args.split].copy()
df["key"] = df["clip_id"].astype(str) + "|" + df["event_idx"].astype(str)

# Keep only events where every arm present in the file survived, so every comparison below is
# paired on exactly the same events. Mixing arm-specific event sets is how a selection effect
# gets mistaken for an arm effect.
arms = [a for a in ARM_ORDER if a in set(df["cond"])]
n_per = df[df["cond"].isin(arms + ["gold"])].groupby("key")["cond"].nunique()
complete = set(n_per[n_per == len(arms) + 1].index)
dropped = df["key"].nunique() - len(complete)
df = df[df["key"].isin(complete)].copy()
S = int(df["n_draws"].iloc[0])
print(f"shards={len(paths)}  events={len(complete)} (dropped {dropped} incomplete)  "
      f"draws/event={S}  arms={arms}")

piv = df.pivot_table(index="key", columns="cond", values="ade_mean")
gtp = df.pivot_table(index="key", columns="cond", values="ade_gt_mean")
gtmin = df.pivot_table(index="key", columns="cond", values="ade_gt_min")
spread = df[df["cond"] == "gold"].set_index("key")["spread_gold"]


def trimmed(x):
    return float(stats.trim_mean(np.asarray(x, float), args.trim))


# ---- 1. determinism -----------------------------------------------------------------------
if "null" in arms:
    m = float(piv["null"].abs().max())
    print(f"\n[1] DETERMINISM  max |null displacement| = {m:.3e} m  "
          f"({'PASS -- exactly 0' if m == 0.0 else 'FAIL: see notes, do not read on'})")

# ---- 2. headline --------------------------------------------------------------------------
yard = float(piv["blankvision"].mean()) if "blankvision" in arms else float("nan")
sp = float(spread.mean())
print(f"\n[2] DISPLACEMENT from the gold rollout, matched initial noise (metres)")
print(f"{'arm':<13}{'mean':>9}{'trim10':>9}{'median':>9}{'%blankvis':>11}{'%spread':>9}"
      f"{'frac>0':>9}")
for a in arms:
    v = piv[a].to_numpy(float)
    print(f"{a:<13}{v.mean():>9.4f}{trimmed(v):>9.4f}{np.median(v):>9.4f}"
          f"{100 * v.mean() / yard:>11.2f}{100 * v.mean() / sp:>9.2f}{(v > 0).mean():>9.3f}")
print(f"{'(spread)':<13}{sp:>9.4f}{trimmed(spread):>9.4f}{np.median(spread):>9.4f}"
      f"{100 * sp / yard:>11.2f}{'--':>9}{'--':>9}"
      "   <- mean pairwise ADE among the gold rollouts themselves")
gg = gtp["gold"].to_numpy(float)
print(f"{'(gold vs GT)':<13}{gg.mean():>9.4f}{trimmed(gg):>9.4f}{np.median(gg):>9.4f}"
      f"{'--':>11}{'--':>9}{'--':>9}   <- the model's own accuracy scale")

# ---- 3. do the content arms come apart? ----------------------------------------------------
content = [a for a in ("donor", "cluster", "shuffled") if a in arms]
print(f"\n[3] ARM SEPARATION -- paired Wilcoxon on per-event displacement")
for i, a in enumerate(content):
    for b in content[i + 1:]:
        d = piv[a].to_numpy(float) - piv[b].to_numpy(float)
        w = stats.wilcoxon(d)
        print(f"    {a:>9} vs {b:<11} median diff {np.median(d):+.4f} m   "
              f"mean {d.mean():+.4f}   p={w.pvalue:.3g}   {a}>{b} in {(d > 0).mean():.1%}")
print("    (Phase 1 likelihood, same three arms: p=0.20 / 0.52 / 0.19 -- indistinguishable)")

# ---- 4. scale-free ------------------------------------------------------------------------
print(f"\n[4] DISPLACEMENT / SAMPLER SPREAD, per event (ratio; 1.0 = the prose swap moves the "
      f"car as far as a fresh noise draw)")
print(f"{'arm':<13}{'mean':>9}{'median':>9}{'p90':>9}{'frac>1':>9}")
for a in arms:
    r = (piv[a] / spread).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    print(f"{a:<13}{r.mean():>9.4f}{np.median(r):>9.4f}{np.percentile(r, 90):>9.4f}"
          f"{(r > 1).mean():>9.3f}")

# ---- 5. accuracy --------------------------------------------------------------------------
print(f"\n[5] ACCURACY vs ground truth -- paired against the gold rollout (metres; + = the swap "
      f"makes it WORSE)")
print(f"{'arm':<13}{'d_ADE':>10}{'p':>11}{'d_minADE':>11}{'p':>11}{'worse':>8}")
for a in arms:
    d = gtp[a].to_numpy(float) - gtp["gold"].to_numpy(float)
    dm = gtmin[a].to_numpy(float) - gtmin["gold"].to_numpy(float)
    pa = stats.wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
    pm = stats.wilcoxon(dm).pvalue if np.any(dm != 0) else float("nan")
    print(f"{a:<13}{d.mean():>+10.4f}{pa:>11.3g}{dm.mean():>+11.4f}{pm:>11.3g}"
          f"{(d > 0).mean():>8.3f}")

# ---- 6. per-event reliability --------------------------------------------------------------
print(f"\n[6] PER-EVENT RELIABILITY of the displacement (split-half over the {S} draws, "
      f"Spearman-Brown)")
ade = {a: np.stack(df[df["cond"] == a].set_index("key").loc[sorted(complete), "ade"].to_numpy())
       for a in arms}
for a in arms:
    A = ade[a]
    h1, h2 = A[:, 0::2].mean(1), A[:, 1::2].mean(1)
    if np.allclose(A, 0):
        print(f"    {a:<13} n/a (identically zero)")
        continue
    r = float(np.corrcoef(h1, h2)[0, 1])
    sb = f"{2 * r / (1 + r):+.3f}" if r > -0.999 else "n/a"
    print(f"    {a:<13} half-half r={r:+.3f}   Spearman-Brown {sb}")

# ---- 7. cross-phase ------------------------------------------------------------------------
print(f"\n[7] CROSS-PHASE: does the car move on the same events where the likelihood moved?")
print(f"    `blankvision` is the POSITIVE CONTROL. Both phases agree the cameras matter, so if"
      f"\n    this test has any power at all it must show up there. A null on the content arms"
      f"\n    means nothing without it.")
try:
    p1 = pd.read_parquet(args.phase1_events)
    p1.index = (p1.index.get_level_values("clip_id").astype(str) + "|"
                + p1.index.get_level_values("event_idx").astype(str))
    ren = {"donor": "donor", "donor_cluster": "cluster",
           "shuffled": "shuffled", "blankvision": "blankvision"}
    print(f"\n{'arm':<13}{'n':>6}{'Pearson r':>12}{'p':>11}{'Spearman':>11}{'p':>11}")
    for p1col, a in ren.items():
        if a not in arms or p1col not in p1.columns:
            continue
        j = pd.DataFrame({"disp": piv[a], "llr": p1[p1col]}).dropna()
        r = stats.pearsonr(j["disp"], j["llr"])
        rho = stats.spearmanr(j["disp"], j["llr"])
        print(f"{a:<13}{len(j):>6}{r.statistic:>+12.4f}{r.pvalue:>11.3g}"
              f"{rho.statistic:>+11.4f}{rho.pvalue:>11.3g}")
    print("    Phase-1 per-event reliability (surrogate, content term) was +0.872; the"
          "\n    attenuation ceiling for each arm is sqrt(0.872 * rel_disp) from readout 6.")
except Exception as e:  # noqa: BLE001 -- readout 7 is optional, the rest must still print
    print(f"    skipped ({type(e).__name__}: {e})")

# ---- 8 & 9. horizon and channel -------------------------------------------------------------
print(f"\n[8] HORIZON -- mean displacement (m) by waypoint, 0.1 s apart")
print(f"\n[9] CHANNEL -- same displacement split in the gold rollout's own frame")
if "traj_xy" in df.columns:
    keys = sorted(complete)
    T = {a: np.stack(df[df["cond"] == a].set_index("key").loc[keys, "traj_xy"].to_numpy()
                     ).reshape(len(keys), S, 64, 2) for a in arms + ["gold"]}
    G = T["gold"]
    # unit tangent of the gold rollout, central difference, per (event, draw, waypoint)
    tan = np.gradient(G, axis=2)
    tan /= np.maximum(np.linalg.norm(tan, axis=-1, keepdims=True), 1e-9)
    nrm = np.stack([-tan[..., 1], tan[..., 0]], axis=-1)
    idx = [0, 7, 15, 23, 31, 47, 63]
    print(f"{'arm':<13}" + "".join(f"{(i + 1) * 0.1:>8.1f}s" for i in idx))
    for a in arms:
        d = np.linalg.norm(T[a] - G, axis=-1).mean(axis=1)  # (events, 64)
        print(f"{a:<13}" + "".join(f"{d[:, i].mean():>9.4f}" for i in idx))
    print()
    print(f"{'arm':<13}{'|longitudinal|':>16}{'|lateral|':>12}{'lat/lon':>10}")
    for a in arms:
        dv = T[a] - G
        lon = np.abs((dv * tan).sum(-1)).mean(axis=(1, 2))
        lat = np.abs((dv * nrm).sum(-1)).mean(axis=(1, 2))
        print(f"{a:<13}{lon.mean():>16.4f}{lat.mean():>12.4f}"
              f"{lat.mean() / max(lon.mean(), 1e-9):>10.3f}")
    print("    (Phase 1 likelihood: content moved acceleration only, curvature exactly 0 --"
          " i.e. it predicts lat/lon well BELOW the vision arm's)")
else:
    print("    skipped: run with --save-traj to record rollouts")

# ---- 10. the length confound -----------------------------------------------------------------
# The alternative explanation for readout 3. `shuffled` preserves gold's token count almost
# exactly (Phase 1 measured mean |d_prefix| = 0.38 tokens, against 3.89 for `donor`), so if
# displacement is really driven by how much the PREFIX LENGTH changed -- an occupancy/compute
# effect -- then donor > shuffled follows with no content in it at all. Phase 1 killed this
# hypothesis on the likelihood (slope +0.0008 against an intercept that was the whole effect);
# it has to be killed again here, on behaviour, because it is the same confound.
print(f"\n[10] LENGTH CONFOUND -- is displacement explained by how much the prefix length moved?")
plen = df.pivot_table(index="key", columns="cond", values="prefix_len")
print(f"{'arm':<13}{'mean|dlen|':>12}{'slope':>10}{'p':>11}{'intercept':>11}{'mean disp':>11}"
      f"{'r':>8}")
for a in arms:
    if a == "null":
        continue
    dl = (plen[a] - plen["gold"]).abs().to_numpy(float)
    v = piv[a].to_numpy(float)
    if np.allclose(dl, dl[0]):
        print(f"{a:<13}{dl.mean():>12.2f}   (no variation in |dlen| -- regression undefined)")
        continue
    lr = stats.linregress(dl, v)
    print(f"{a:<13}{dl.mean():>12.2f}{lr.slope:>10.5f}{lr.pvalue:>11.3g}"
          f"{lr.intercept:>11.4f}{v.mean():>11.4f}{lr.rvalue:>8.3f}")
print("    Read the INTERCEPT against `mean disp`: if length were the driver the intercept")
print("    would collapse toward 0 and the slope would carry the effect.")
# The sharpest version: restrict to events where donor happens to be length-matched to gold as
# closely as shuffled is, and re-run the donor-vs-shuffled test there.
if {"donor", "shuffled"} <= set(arms):
    dl_d = (plen["donor"] - plen["gold"]).abs()
    dl_s = (plen["shuffled"] - plen["gold"]).abs()
    tight = dl_d <= max(float(dl_s.quantile(0.9)), 1.0)
    if int(tight.sum()) > 20:
        d = (piv["donor"] - piv["shuffled"])[tight].to_numpy(float)
        w = stats.wilcoxon(d)
        print(f"\n    Length-matched subset: {int(tight.sum())} events where |dlen(donor)| is "
              f"within\n    the 90th pct of |dlen(shuffled)| ({float(dl_s.quantile(0.9)):.0f} "
              f"tokens). donor - shuffled there:\n    mean {d.mean():+.4f} m, median "
              f"{np.median(d):+.4f}, p={w.pvalue:.3g}, donor>shuffled in {(d > 0).mean():.1%}")
