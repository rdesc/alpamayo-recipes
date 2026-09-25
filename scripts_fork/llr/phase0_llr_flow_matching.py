# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Phase-0 LLR on the CONTINUOUS action head (Stage-2 flow-matching expert).

Everything else in this directory measures the DISCRETE trajectory-token head: 128 traj tokens,
teacher-forced, ``log p(a*|v,coc) - log p(a*|v,control)`` in nats. The flow-matching expert emits
no token logits, so that likelihood is not available. This script is the analogue.

WHY THE LLR DESIGN TRANSFERS UNCHANGED
--------------------------------------
The action expert has NO cross-attention module and receives NO text embeddings. Its only input
embeddings are the noisy-action tokens (``sft_alpamayo_r1.py:198-204``). All language -- images,
prompt, and the CoC span -- reaches it solely as the VLM's KV cache, taken at
``sft_alpamayo_r1.py:206`` and cropped at ``<traj_future_start>+1`` at ``:207``, then handed to
the expert as ``past_key_values`` at ``:222``. So swapping the CoC prose swaps the conditioning
exactly the way ``phase0_llr_per_token.py`` does on the token head. Same checkpoint, same events,
same split: ``Alpamayo-1.5-10B-rl-training`` carries the full expert (396 ``expert.*`` +
``action_in_proj.*`` + ``action_out_proj.*``); ``RLWrapperReasoningVLA`` simply never loads them.

THE ESTIMATOR
-------------
Per event, per condition c, with x the GOLD action (64 waypoints x (accel, curvature))::

    L(c) = (1/N) sum_j || v_theta( t_j*x + (1-t_j)*eps_j , t_j , KV(v, c) ) - (x - eps_j) ||^2

which is exactly the training objective (``flow_matching.py:140-173``) evaluated at the gold
action. The reported terms mirror the token-head vocabulary::

    content  = L(donor) - L(gold)     <- THE HEADLINE. Isolates what the prose SAYS.
    presence = L(empty) - L(gold)     <- diagnostic only, see below.

THE ARMS (``--arms``), and what each one controls for
------------------------------------------------------
Every arm is scored against the SAME gold pass on the SAME CRN draws, so all of them are
directly comparable row-for-row.

    random    Donors drawn from the whole split (other clips only). The headline content term.
              Confound it leaves open: a random donor is usually from a DIFFERENT KIND of scene,
              so "content" here could be nothing more than topic mismatch -- a highway CoC read
              over a yard scene.
    cluster   Donors drawn from the SAME ``event_cluster`` (still other clips). Holds topic
              roughly fixed and varies only which specific scene the prose describes. If the gap
              SHRINKS versus ``random``, a chunk of the headline was topic mismatch; if it does
              NOT shrink, the expert is tracking this scene's reasoning rather than its genre.
    shuffled  This event's OWN gold prose with its words permuted (seeded). Same length, same
              bag of words, destroyed syntax. The sharp one. If Delta(shuffled) ~ Delta(random),
              the expert needs word ORDER / structure, so it is reading the sentence. If
              shuffled scores like gold (Delta ~ 0), the expert is only picking up keywords, and
              the "content" effect is lexical overlap -- which is exactly what an artefact
              concentrated at the noisy end of the t schedule would also look like.
    empty     ``coc_text=""``. Diagnostic only -- see below. Measured null on the 40-event pilot
              (median +0.002, sign 23+/17-), so it is no longer in the default arm set.

THE NATS ESTIMATOR -- how the MSE gap becomes a log-likelihood ratio
--------------------------------------------------------------------
The raw gap below is a squared-error difference, NOT nats. But the flow-matching velocity MSE
is the diffusion ELBO's eps-MSE under a known weighting, so dividing that weighting out
recovers a genuine likelihood. For the linear interpolant used here
(``x_t = t*a + (1-t)*eps``, target ``v = a - eps``), with lambda = 2*log(t/(1-t)) the log-SNR:

    Kingma & Gao (arXiv 2303.00848) Eq. 4:  L_w = 1/2 * int w(lambda) E||eps_hat - eps||^2 dlambda
    ELBO  <=>  w(lambda) = 1;  flow-matching OT path  <=>  w(lambda) = e^{-lambda/2} = (1-t)/t

Substituting dlambda/dt = 2/(t(1-t)) and eps_hat - eps = -t*(v_hat - v), the 1/2 and the 2
cancel exactly and the weight on the VELOCITY MSE is simply ``t/(1-t)``:

    LLR(c1 vs c2) = E_{t,eps}[ (t/(1-t)) * ( ||v(x_t,t,c2) - u||^2 - ||v(x_t,t,c1) - u||^2 ) ]

in NATS, with u = a - eps and the SAME (t, eps) draws in both arms. Every additive constant --
prior term, endpoints, D/2*log(2*pi*e) -- cancels at fixed a*. Verified numerically against an
analytic Gaussian where the bound is tight (agreement to ~2e-3 nats, Monte-Carlo limited).

SAMPLE t UNIFORMLY IN LAMBDA, NOT IN t (``--t-sampler lambda``). The weight t/(1-t) reaches 255
at the top stratum of a 128-point uniform-t grid, so uniform-t puts almost no samples where
nearly all the weight lives. Under a uniform-lambda grid the change of variables
dt = (t(1-t)/2) dlambda turns the per-draw weight into

    w_j = (lam_hi - lam_lo) * t_j^2 / 2          <- BOUNDED, since t <= 1

which is why that sampler is not cosmetic. ``lam_lo``/``lam_hi`` are written into every output
row so the weight is reconstructible downstream; the weighting itself is applied in ANALYSIS,
not here, so existing runs can be re-weighted without re-running the model.

Precedent (this design is not novel, which is the point): Flow Policy Optimization
(arXiv 2507.21053) uses exactly this ratio for action distributions with shared (eps, t);
Diffusion Classifier (arXiv 2303.16203 sec 3.3) is the origin of the shared-noise trick;
Kong et al. (arXiv 2310.07972) give the pointwise LLR identity.

SIGN CONVENTION -- IT IS INVERTED RELATIVE TO EVERY OTHER SCRIPT HERE.
What this file writes out is a LOSS, not a log-likelihood. **Positive = the gold CoC fits the
gold action BETTER**, which agrees in sign with ``phase0_llr_per_token.py``'s ``llr``.

The RAW gap in these parquets is a squared-error difference and is NOT nats. The ELBO weighting
that converts it is applied downstream in ``analyze_fm_nats.py`` (see THE NATS ESTIMATOR below),
deliberately, so that any existing run can be re-weighted without a GPU. Quote nats only from
that script's output, never from ``loss`` directly.

Once weighted, the two heads' CONTRASTS are in the same units (nats per cell vs nats per token,
128 of each) and may be compared. Their LEVELS may not: this is a difference of two variational
bounds, and the slack need not cancel -- see ``fm_llr_method.md`` section 9. An exact likelihood
of the deployed 10-step Euler map is the validation arm that bounds that gap; the continuous-time
probability-flow ODE was built and abandoned (``ode_likelihood_explained.md`` section 5).

VARIANCE REDUCTION -- this is the whole game
--------------------------------------------
The per-draw loss has sd ~0.27 against condition gaps of ~0.02-0.12, so a naive difference of two
independently-sampled means is hopeless. The analogue of the token head's "PAIR BY RANK WITHIN
EACH PASS" invariant is COMMON RANDOM NUMBERS:

    1. ONE array of (t_j, eps_j) is drawn per event and reused BYTE-FOR-BYTE across gold, empty
       and every donor. Difference PER DRAW, then average. Never average-then-subtract.
    2. The target x is the same gold action in all conditions, so (x - eps_j) is identical too.
    3. t is STRATIFIED on a ((j+0.5)/N) grid rather than drawn from the training Beta(1.5,1)
       sampler, removing timestep variance outright. (--t-sampler beta reproduces the training
       distribution if you want it; it is noisier.)
    4. Donors are averaged over --n-donors draws per event, same rationale as the token head:
       donor choice, not Monte-Carlo noise, dominates a single-donor estimate.

Measured on one event at N=128: CRN cuts the standard error 2-9x, i.e. 5-80x fewer passes.
Roughly 20-45 draws per event clear 3 sigma on that event's content term.

WHY THE EMPTY ARM IS DIAGNOSTIC ONLY
-------------------------------------
Identical reasoning to ``results/phase0_llr_action_direction.md`` and the notes in
``phase0_llr_per_token.py``: ``coc_text=""`` is shorter than the gold sequence and puts the model
in a configuration training rarely produces, so ``L(empty) - L(gold)`` confounds content with slot
occupancy and with length. On the very first event measured here it comes out NEGATIVE (empty
fits the gold action BETTER than gold prose does), which is the same off-distribution pathology
the token-head study documented. Quote ``content``; report ``presence`` only as a control.

THE FAST PATH (and why it is exact)
------------------------------------
``TrainableAlpamayoR1.forward`` runs one full VLM pass per call and computes the VLM's full-vocab
LM-head cross-entropy, which this measurement never uses (and which OOMs above micro-batch 4).
But conditioning enters only through the cropped KV cache, so for a fixed condition that cache is
IDENTICAL for every noise draw. This script therefore:

    one VLM pass (batch 1, inner ``vlm.model`` -- no LM head, no CE)
      -> crop the cache at <traj_future_start>+1  (same index, same assertion as :186-194)
      -> snapshot the cropped per-layer (k, v)
      -> for each chunk of draws, rebuild a DynamicCache whose k/v are batch-EXPANDED VIEWS
         (expand, not repeat: no copy, no extra memory)
      -> one expert forward for the whole chunk

position_ids are rebuilt with the same formula as ``_process_position_ids_qwen2_5_vl``
(``:112-128``), using ``rope_deltas`` plus the CROPPED cache length -- the original reads
``get_seq_length()`` after the crop because ``kv_cache is vlm_outputs.past_key_values``. A fresh
cache is built per chunk because the expert appends its own keys with ``use_cache=True``; reusing
the object would leave the batch dimension materialised.

``--verify`` checks the fast path against the slow one (``model.forward`` with the diffusion
module patched to consume the same fixed draws) on one event and prints the max abs deviation.
Run it after any change to the model code. A fast-but-different scorer is worse than a slow one.

Same venv/CWD rules as the rest of this directory (always ``a1x_rl_b300``, whatever the box --
run from a recipe directory; see ``README.md`` and the root ``CLAUDE.md``).

Usage::

    cd recipes/alpamayo1_x_rl

    # exactness check -- do this first
    CUDA_VISIBLE_DEVICES=0 a1x_rl_b300/bin/python ../../scripts_fork/llr/phase0_llr_flow_matching.py \\
      --verify --n-draws 16 --out /tmp/fm_verify.parquet

    # pilot
    CUDA_VISIBLE_DEVICES=0 a1x_rl_b300/bin/python ../../scripts_fork/llr/phase0_llr_flow_matching.py \\
      --limit 40 --n-draws 64 --n-donors 5 --out /tmp/fm_pilot.parquet

    # control arms: random vs topic-matched vs word-shuffled, all on the same gold draws
    CUDA_VISIBLE_DEVICES=0 a1x_rl_b300/bin/python ../../scripts_fork/llr/phase0_llr_flow_matching.py \\
      --arms random,cluster,shuffled --limit 40 --n-draws 32 --n-donors 5 --out /tmp/fm_arms.parquet

    # the effect is strongly t-dependent, so also check it under the TRAINING t distribution
    CUDA_VISIBLE_DEVICES=0 a1x_rl_b300/bin/python ../../scripts_fork/llr/phase0_llr_flow_matching.py \\
      --arms random --t-sampler beta --limit 20 --n-draws 32 --out /tmp/fm_beta.parquet

Output: one row per (event, condition, draw) -- clip_id, event_idx, cond, donor_idx, draw, t,
loss, resid. Everything downstream (content term, robust battery, Delta-vs-t curve) is computed
from that, so the analysis can be redone without re-running the model.

``loss`` is the scalar mean squared residual, as before. ``resid`` is that residual BEFORE the
reduction: 128 float32 = (64 waypoints, 2 channels) flattened row-major, so index = w*2 + c with
EVEN = acceleration and ODD = curvature, matching the token head's trajectory-token layout
(README trap 5). ``loss == resid.mean()`` exactly, by construction.

Keep the resolution. The horizon profile and the accel-vs-curvature split are load-bearing in the
token-head study -- the navigation result's claim to being real conditioning is precisely that it
is channel-specific -- and a scalar-only run cannot reproduce either, at any sample size, without
re-running the model. When differencing conditions, difference PER DRAW AND PER ENTRY and only
then average: collapsing to a scalar first and re-expanding discards the common-random-number
pairing that the whole estimator is built on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch

DEFAULT_MODEL_PATH = "/mnt/efs/users/rod/ckpts/Alpamayo-1.5-10B-rl-training"
DEFAULT_PARQUET = "/opt/dlami/nvme/rod/datasets/pai_av_dataset/reasoning/ood_reasoning.parquet"
MIN_T0_US = 1_700_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--split", default="both", choices=["train", "val", "both"])
    p.add_argument("--n-draws", type=int, default=64, help="CRN (t, eps) draws, shared by all conditions.")
    p.add_argument("--chunk", type=int, default=32, help="Draws per expert forward.")
    p.add_argument("--n-donors", type=int, default=5)
    p.add_argument("--donor-seed", type=int, default=0)
    p.add_argument(
        "--donor-traj-min-ade", type=float, default=5.0,
        help="Minimum ADE (metres over the 6.4 s horizon) between this event's GT future and the "
             "donor trajectory used by `wrongtraj`. MATCHES the token head's gate of the same "
             "name (phase0_llr_sanity_controls.py), and must stay matched: the separation is the "
             "denominator both heads are normalised by, so an ungated donor here silently makes "
             "the two heads' ratios incomparable. Without the gate the donor is often a "
             "NEAR-DUPLICATE -- ~92%% of this split is 'Continue straight' -- the 'wrong' target "
             "is not wrong, and the yardstick collapses. The token head retired an ungated "
             "+1.068 for exactly this reason (std 0.71, one event at -0.21); the ungated "
             "flow-matching arm shows the same signature (per-event gaps +3.3 to +229). "
             "Set 0 to disable.")
    p.add_argument("--donor-traj-max-tries", type=int, default=32,
                   help="Random picks from the pool before falling back to the most distant one.")
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument(
        "--t-sampler", default="stratified", choices=["stratified", "beta", "lambda"],
        help="'stratified' is the ((j+0.5)/N) grid (variance reduction); 'beta' reproduces the "
             "training sampler (Beta(1.5,1) scaled) and is noisier; 'lambda' is a stratified "
             "grid uniform in log-SNR, which is the one to use for the NATS estimator -- see "
             "THE NATS ESTIMATOR in the module docstring.",
    )
    p.add_argument("--lam-t-min", type=float, default=1e-3,
                   help="'lambda' sampler: truncation at the noise end (t_min).")
    p.add_argument("--lam-t-max", type=float, default=1 - 1e-3,
                   help="'lambda' sampler: truncation at the data end (t_max).")
    p.add_argument(
        "--arms", default="random,empty",
        help="Comma list of control arms to score against gold: random | cluster | shuffled | "
             "empty | null | wrongtraj | blankvision | flipdir | navnone | navswap. See THE ARMS "
             "in the module docstring for what each controls for. Default keeps the original "
             "behaviour (random donors + the empty diagnostic).",
    )
    p.add_argument(
        "--nav", action="store_true",
        help="NAVIGATION-COMMAND conditioning instead of (well, alongside) CoC. Switches the "
             "component order to include `route` and the chat template to r1_5, and synthesises "
             "an oracle command from the GT future. Enables the navnone/navswap arms. Nav is the "
             "ONE channel with a directional, channel-specific effect on the token head "
             "(+0.0018, turns->curvature p=7.3e-4), so it is the likeliest thing to transfer.")
    p.add_argument("--nav-from", default="traj", choices=["traj", "coc"],
                   help="'traj' derives the command from GT geometry; 'coc' from the gold prose.")
    p.add_argument("--no-empty", action="store_true",
                   help="Back-compat alias: drop 'empty' from --arms.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--verify", action="store_true", help="Fast-vs-slow exactness check, then exit.")
    p.add_argument("--out", required=True)
    return p.parse_args()


def _swap_direction(nav_text: str) -> str:
    """left <-> right, via a placeholder so the two substitutions cannot chase each other.

    The nav falsification test: if the effect is directional, inverting the command must destroy
    it. On the token head this dropped +0.0018 to +0.0012 (n.s.).
    """
    ph = "\x00SWAP\x00"
    out = re.sub(r"\bleft\b", ph, nav_text, flags=re.IGNORECASE)
    out = re.sub(r"\bright\b", "left", out, flags=re.IGNORECASE)
    return out.replace(ph, "right")


_LAT_VERBS = r"\b(?:steer|turn|merge|move|shift|bear|veer|swerve|change)\b"


def _flip_directive(text: str) -> tuple[str, str, int] | None:
    """Flip the DIRECTIVE in a CoC sentence, changing exactly one word. Returns (flipped, kind,
    word position), or None if the sentence carries no flippable directive.

    The sharpest possible content probe: one word, perfectly length-matched (callers MUST still
    verify the token count -- see below), and the word that most directly names the manoeuvre.
    If the head reads the reasoning for content at all, it should read this.

    Faithful to the 12 hand-verified cases in flip_cases_clean.json, which matters in one
    non-obvious way: only the FIRST, directive occurrence is flipped. Case n2 is
    "Steer left and merge into the left lane ..." -> "Steer right and merge into the left lane",
    leaving the incidental second mention alone. A global left<->right swap (what
    ``_swap_direction`` does, correctly, for the nav ROUTE field) is a different and weaker
    manipulation: it rewrites scene description as well as command.

    Two kinds, matching the verified set:
      directive-lateral       left <-> right, only when a steering verb precedes it
      directive-longitudinal  leading "Stop" -> "Proceed"
    """
    words = text.split()
    for i, w in enumerate(words[:6]):
        core = re.sub(r"[^A-Za-z]", "", w).lower()
        if core in ("left", "right"):
            # Search the WHOLE prefix, not a fixed window: case n6 is "Lane change to the left",
            # where the verb sits three words back. `i < 6` already confines this to a directive
            # position, so a wider lookback cannot reach an incidental mention.
            if not re.search(_LAT_VERBS, " ".join(words[:i]), flags=re.IGNORECASE):
                continue
            repl = "right" if core == "left" else "left"
            if w[:1].isupper():
                repl = repl.capitalize()
            out = list(words)
            out[i] = re.sub(core, repl, w, flags=re.IGNORECASE)
            return " ".join(out), "directive-lateral", i
    if words and re.sub(r"[^A-Za-z]", "", words[0]).lower() == "stop":
        out = list(words)
        out[0] = "Proceed" if words[0][:1].isupper() else "proceed"
        return " ".join(out), "directive-longitudinal", 0
    return None


def load_events(parquet_path: str, split: str) -> list[dict]:
    """Flatten ood_reasoning.parquet into one row per annotated event.

    Same loader as phase0_llr_per_token.py / phase0_llr_action_direction.py -- see those files
    for provenance. Kept byte-compatible so event indices line up across the two heads.
    """
    df = pd.read_parquet(parquet_path)
    if split != "both":
        df = df[df["split"] == split].copy()
    df["events"] = df["events"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    df = df.dropna(subset=["events"])
    rows = []
    for clip_id, row in df.iterrows():
        for ei, ev in enumerate(row["events"]):
            rows.append(
                {
                    "clip_id": clip_id,
                    "event_idx": ei,
                    "t0_us": max(int(ev["event_start_timestamp"]), MIN_T0_US),
                    "gold_coc": ev["coc"],
                    "event_cluster": row["event_cluster"],
                    "split": row["split"],
                }
            )
    return rows


def main() -> None:  # noqa: C901 -- one linear measurement, splitting it hides the invariants
    args = parse_args()
    torch.manual_seed(42)

    sys.path[:0] = ["../../../src", "../.."]
    import physical_ai_av
    from alpamayo_r1.helper import to_device
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo.processor.qwen_processor import (
        build_processor,
        collate_fn_from_model_config,
        get_preprocess_data_fn_from_model_config,
    )
    from transformers.cache_utils import DynamicCache

    from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1

    print(f"[fm-llr] loading {args.model_path} ...", flush=True)
    model = TrainableAlpamayoR1.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    ).cuda().eval()
    # NB the "newly initialized" warning for action_space.{accel,curvature}_{mean,std} and the
    # Fourier `freqs` is benign and was verified: those are register_buffers built from
    # action_space_cfg / torch.logspace, not shard weights. Values live in config.json.

    # `route` sits between traj_history and prompt; "r1" has no route case at all, so nav
    # conditioning requires the r1_5 template. Everything else r1_5 delegates to r1, so the
    # non-nav path is byte-identical to before.
    CHAT_TEMPLATE = "r1_5" if args.nav else "r1"
    COMPONENTS = (["image", "traj_history", "route", "prompt", "cot", "traj_future"] if args.nav
                  else ["image", "traj_history", "prompt", "cot", "traj_future"])
    oracle_nav = None
    if args.nav:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "oracle_nav", "/home/rod/repos/alpamayo1.5/oracle_nav.py")
        oracle_nav = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oracle_nav)

    preprocess_fn = get_preprocess_data_fn_from_model_config(
        components_order=COMPONENTS,
        components_prompt=["cot", "traj_future"],
        label_components=["cot"],
        generation_mode=False,
        include_camera_ids=True,
        include_frame_nums=True,
        model_config=model.config,
        chat_template_version=CHAT_TEMPLATE,
    )
    build_processor(
        vlm_name_or_path=model.config.vlm_name_or_path,
        traj_vocab_size=model.config.traj_vocab_size,
        min_pixels=model.config.min_pixels,
        max_pixels=model.config.max_pixels,
        include_camera_ids=True,
        include_frame_nums=True,
        chat_template_version=CHAT_TEMPLATE,
    )

    future_start_token_id = model.config.traj_token_ids["future_start"]
    adims = tuple(model.action_space.get_action_space_dims())
    n_action_tokens = model.config.tokens_per_future_traj
    print(f"[fm-llr] action dims={adims} expert tokens={n_action_tokens}", flush=True)

    # ---------------- common random numbers ----------------
    N = args.n_draws
    g = torch.Generator().manual_seed(args.noise_seed)
    eps_all = torch.randn((N, *adims), generator=g)
    lam_lo = lam_hi = float("nan")
    if args.t_sampler == "stratified":
        t_all = ((torch.arange(N, dtype=torch.float64) + 0.5) / N).float()
    elif args.t_sampler == "lambda":
        # Uniform grid in log-SNR lambda = 2*log(t/(1-t)), i.e. t = sigmoid(lambda/2).
        # Mandatory for the nats estimator: the ELBO weight t/(1-t) reaches 255 at the top
        # stratum of a uniform-t grid, so uniform-t puts almost no samples where nearly all
        # of the weight lives. Under this sampler the per-draw weight becomes
        # (lam_hi-lam_lo)*t^2/2, which is BOUNDED -- see the docstring.
        lo, hi = args.lam_t_min, args.lam_t_max
        lam_lo = 2.0 * math.log(lo / (1 - lo))
        lam_hi = 2.0 * math.log(hi / (1 - hi))
        lam = lam_lo + ((torch.arange(N, dtype=torch.float64) + 0.5) / N) * (lam_hi - lam_lo)
        t_all = torch.sigmoid(lam / 2).float()
    else:
        torch.manual_seed(args.noise_seed)
        b = model.diffusion.beta_dist.sample((N,)).cpu().float()
        t_all = model.diffusion.beta_scale_constant - b * model.diffusion.beta_scale_constant
    print(f"[fm-llr] {N} CRN draws, t-sampler={args.t_sampler}, "
          f"t in [{t_all.min():.5f}, {t_all.max():.5f}]"
          + (f", lambda in [{lam_lo:.2f}, {lam_hi:.2f}]" if args.t_sampler == "lambda" else ""),
          flush=True)

    # ---------------- events / donor pool ----------------
    events = load_events(args.parquet, args.split)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if args.no_empty:
        arms = [a for a in arms if a != "empty"]
    for a in arms:
        if a in ("navnone", "navswap") and not args.nav:
            raise SystemExit(f"arm {a!r} requires --nav")
        if a not in ("random", "cluster", "shuffled", "empty", "flipdir",
                     "null", "wrongtraj", "blankvision", "navnone", "navswap"):
            raise SystemExit(
                f"unknown arm {a!r}; choose from random/cluster/shuffled/empty/flipdir "
                f"(content arms) or null/wrongtraj/blankvision (the control ladder)")
    print(f"[fm-llr] arms: {arms}", flush=True)

    # Donor pool carries event_cluster so the 'cluster' arm can hold topic fixed.
    donor_pool = [
        (e["clip_id"], e["gold_coc"], e["event_cluster"])
        for e in events
        if isinstance(e.get("gold_coc"), str) and e["gold_coc"].strip()
    ]
    if len(donor_pool) < args.n_donors + 1:
        raise RuntimeError(f"donor pool {len(donor_pool)} too small for --n-donors {args.n_donors}")

    def _rng(tag: str, clip_id: str, event_idx: int) -> np.random.Generator:
        """Deterministic per (arm, event). hashlib, NOT hash(): CPython salts str hashing per
        process (PYTHONHASHSEED), so hash() would give different draws in every shard and
        re-run. The tag keeps the arms from sharing a stream."""
        key = f"{args.donor_seed}|{tag}|{clip_id}|{int(event_idx)}".encode()
        return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))

    def draw_donors(clip_id: str, event_idx: int, cluster=None) -> list[str]:
        """Donor prose from OTHER clips. With `cluster` set, restricted to that event_cluster
        so topic is held roughly fixed and only the specific scene varies."""
        tag = "random" if cluster is None else "cluster"
        rng = _rng(tag, clip_id, event_idx)
        cands = [
            i for i, (cid, _, cl) in enumerate(donor_pool)
            if cid != clip_id and (cluster is None or cl == cluster)
        ]
        if not cands:
            return []
        picks = rng.choice(len(cands), size=min(args.n_donors, len(cands)), replace=False)
        return [donor_pool[cands[int(i)]][1] for i in picks]

    def draw_shuffled(clip_id: str, event_idx: int, gold: str) -> list[str]:
        """This event's OWN prose with words permuted: length, token count and bag-of-words all
        preserved, syntax destroyed. --n-donors independent permutations, averaged."""
        words = gold.split()
        if len(words) < 3:
            return []
        rng = _rng("shuffled", clip_id, event_idx)
        out = []
        for _ in range(args.n_donors):
            w = list(words)
            for _try in range(10):  # a permutation equal to the original is not a control
                rng.shuffle(w)
                if w != words:
                    break
            out.append(" ".join(w))
        return out

    sel = list(events)
    if args.num_shards > 1:
        sel = sel[args.shard_idx :: args.num_shards]
    if args.limit:
        sel = sel[: args.limit]
    if args.verify:
        sel = sel[:1]
    print(f"[fm-llr] shard {args.shard_idx}/{args.num_shards}: {len(sel)} events", flush=True)

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    # ---------------- the two scorers ----------------
    def tokenize(data: dict, coc_text: str, n_copies: int = 1, blank_vision: bool = False,
                 nav_text: str | None = None) -> dict:
        sample = {
            "image_frames": (torch.zeros_like(data["image_frames"]) if blank_vision
                             else data["image_frames"]),
            "camera_indices": data["camera_indices"],
            "relative_timestamps": data["relative_timestamps"],
            "ego_history_xyz": data["ego_history_xyz"].squeeze(0),
            "ego_history_rot": data["ego_history_rot"].squeeze(0),
            "cot": coc_text,
        }
        if args.nav:
            # construct_route returns [] for nav_text=None, so the no-nav sequence has NO route
            # markers at all. That is the in-distribution p(a*|v) here -- the reverse of the CoC
            # case, where the markers stay and only the prose leaves.
            sample["nav_text"] = nav_text
        sample["tokenized_data"] = preprocess_fn(sample)
        batch = collate_fn_from_model_config(
            [dict(sample) for _ in range(n_copies)],
            model_config=model.config,
            include_camera_ids=True,
            include_frame_nums=True,
            chat_template_version=CHAT_TEMPLATE,
        )
        return batch["tokenized_data"]

    def gold_action(data: dict) -> torch.Tensor:
        """The teacher-forced target x, identical across every condition (CRN item 2)."""
        a = model.action_space.traj_to_action(
            traj_history_xyz=data["ego_history_xyz"].cuda(),
            traj_history_rot=data["ego_history_rot"].cuda(),
            traj_future_xyz=data["ego_future_xyz"].cuda(),
            traj_future_rot=data["ego_future_rot"].cuda(),
        )
        return a.reshape(-1, *adims)  # [1, 64, 2]

    def score_fast(data: dict, coc_text: str, x_override: torch.Tensor | None = None,
                   blank_vision: bool = False, nav_text: str | None = None) -> np.ndarray:
        """ONE VLM pass, then all N draws through the expert on a batch-expanded KV cache.

        ``x_override`` scores a DIFFERENT action under this conditioning (the wrong-trajectory
        control -- the analogue of the token head's +0.775 separation, and the yardstick that
        makes a near-zero content term interpretable rather than merely unresolved).
        ``blank_vision`` zeroes the camera frames (large positive control).
        """
        td = to_device({"td": dict(tokenize(data, coc_text, 1, blank_vision, nav_text))},
                       "cuda")["td"]
        input_ids = td.pop("input_ids")
        input_ids = model.fuse_traj_tokens(
            input_ids,
            {
                "ego_history_xyz": data["ego_history_xyz"].cuda(),
                "ego_history_rot": data["ego_history_rot"].cuda(),
                "ego_future_xyz": data["ego_future_xyz"].cuda(),
                "ego_future_rot": data["ego_future_rot"].cuda(),
            },
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            # inner .model, not the ForConditionalGeneration wrapper: skips lm_head + the
            # full-vocab cross-entropy this measurement never uses (and that OOMs at bs>4).
            vlm_out = model.vlm.model(input_ids=input_ids, use_cache=True, **td)

        fs = (input_ids == future_start_token_id).nonzero(as_tuple=False)
        assert fs.shape[0] == 1, f"expected exactly one <traj_future_start>, got {fs.shape[0]}"
        crop_at = int(fs[0, 1].item()) + 1
        # Record the conditioning length. Swapping CoC prose changes the token count, which
        # changes the cropped prefix length, which shifts the action tokens' position ids via
        # rope_deltas + cache length. That is a length confound with a signature -- an effect
        # concentrated in the EARLIEST waypoints -- and it must be testable downstream, so the
        # per-condition prefix length is written into every row.
        score_fast.last_prefix_len = crop_at

        cache = vlm_out.past_key_values
        cache.crop(crop_at)
        prefix_len = cache.get_seq_length()
        snap = [(lyr.keys, lyr.values) for lyr in cache.layers]  # batch-1 tensors
        delta = int(vlm_out.rope_deltas.flatten()[0].item()) + int(prefix_len)

        x1 = gold_action(data) if x_override is None else x_override  # [1, 64, 2]
        # Per-draw squared residual at FULL resolution: [N, 64, 2]. Collapsing this to a
        # scalar here would throw away the two decompositions the token-head study leans on
        # (horizon profile, accel-vs-curvature), and they cannot be recovered afterwards.
        out = np.empty((N, *adims), dtype=np.float64)
        for s in range(0, N, args.chunk):
            idx = list(range(s, min(s + args.chunk, N)))
            B = len(idx)
            # expand, not repeat: a view, so the prefix KV costs nothing extra per draw.
            chunk_cache = DynamicCache(
                ddp_cache_data=[
                    (k.expand(B, *k.shape[1:]), v.expand(B, *v.shape[1:])) for k, v in snap
                ]
            )
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                x = x1.expand(B, *x1.shape[1:])
                t = t_all[idx].to(x.device, x.dtype).view(B, *([1] * (x.dim() - 1)))
                eps = eps_all[idx].to(x.device, x.dtype)
                noisy_x = t * x + (1 - t) * eps
                action_embeds = model.action_in_proj(noisy_x, t)
                # same formula as _process_position_ids_qwen2_5_vl (:112-128); the original
                # reads get_seq_length() AFTER the crop because the cache object is shared.
                pos = torch.arange(action_embeds.shape[1], device=action_embeds.device)
                position_ids = pos.view(1, 1, -1).expand(3, B, -1).clone() + delta
                fwd = {}
                if model.config.expert_non_causal_attention:
                    fwd["is_causal"] = False
                eo = model.expert(
                    inputs_embeds=action_embeds,
                    position_ids=position_ids,
                    past_key_values=chunk_cache,
                    attention_mask=None,
                    use_cache=True,
                    **fwd,
                )
                pred = model.action_out_proj(eo.last_hidden_state[:, -action_embeds.shape[1] :])
                pred = pred.view(-1, *adims)
                target = (x - eps).to(pred.dtype)
                # Squared residual for THIS chunk of draws, unreduced: [B, 64, 2]. Same
                # quantity the caller calls `resid`; `out` is just the chunks assembled.
                sq = ((target - pred) ** 2).float()
            out[idx] = sq.detach().cpu().numpy()
            del chunk_cache
        del cache, snap
        return out

    def score_slow(data: dict, coc_text: str, micro_bs: int = 4) -> np.ndarray:
        """Reference path: the real ``model.forward``, with the diffusion module patched to
        consume the SAME fixed draws. Used only by --verify."""
        state: dict = {"idx": None, "sq": None}
        orig_ctd = model.diffusion.construct_training_data
        orig_loss = model.diffusion.compute_loss_from_pred

        def ctd(x):
            i = state["idx"]
            t = t_all[i].to(x.device, x.dtype).view(-1, *([1] * (x.dim() - 1)))
            noise = eps_all[i].to(x.device, x.dtype)
            return {"x": x, "noisy_x": t * x + (1 - t) * noise, "timesteps": t,
                    "noise": noise, "is_drop_guidance": None}

        def clfp(training_data, pred):
            target = (training_data["x"] - training_data["noise"]).to(pred.dtype)
            state["sq"] = ((target - pred) ** 2).float().detach().cpu()  # [B, 64, 2]
            # model.forward still needs a scalar; the stored tensor keeps full resolution.
            return state["sq"].mean()

        model.diffusion.construct_training_data = ctd
        model.diffusion.compute_loss_from_pred = clfp
        try:
            out = np.empty((N, *adims), dtype=np.float64)
            for s in range(0, N, micro_bs):
                idx = list(range(s, min(s + micro_bs, N)))
                B = len(idx)
                td = to_device({"td": dict(tokenize(data, coc_text, B))}, "cuda")["td"]
                rep = lambda k: data[k].expand(B, *data[k].shape[1:]).contiguous().cuda()  # noqa: E731
                state["idx"] = idx
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model(
                        tokenized_data=td,
                        ego_history_xyz=rep("ego_history_xyz"),
                        ego_history_rot=rep("ego_history_rot"),
                        ego_future_xyz=rep("ego_future_xyz"),
                        ego_future_rot=rep("ego_future_rot"),
                        labels_mask=None,
                    )
                out[idx] = state["sq"].numpy()
            return out
        finally:
            model.diffusion.construct_training_data = orig_ctd
            model.diffusion.compute_loss_from_pred = orig_loss

    # ---------------- verify ----------------
    if args.verify:
        ev = sel[0]
        data = load_physical_aiavdataset(ev["clip_id"], t0_us=ev["t0_us"], avdi=avdi)
        print(f"[verify] event {ev['clip_id']} idx={ev['event_idx']}", flush=True)
        keep: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, txt in (("gold", ev["gold_coc"]), ("empty", "")):
            t0 = time.time()
            f = score_fast(data, txt)
            tf = time.time() - t0
            t0 = time.time()
            s = score_slow(data, txt)
            ts = time.time() - t0
            keep[name] = (f, s)
            fm, sm = f.mean(axis=(1, 2)), s.mean(axis=(1, 2))  # per-draw scalars
            d = np.abs(fm - sm)
            rel = d / np.maximum(np.abs(sm), 1e-12)
            print(
                f"[verify] {name:6s} LEVEL   max_abs={d.max():.3e} mean_abs={d.mean():.3e} "
                f"max_rel={rel.max():.3e}  fast={tf:.1f}s slow={ts:.1f}s speedup={ts/tf:.1f}x",
                flush=True,
            )
            # bf16 has ~3 decimal digits; a matching path differs only by reduction order.
            print(f"[verify] {name:6s} VERDICT: "
                  f"{'MATCH' if rel.max() < 5e-3 else 'MISMATCH -- DO NOT USE THE FAST PATH'}",
                  flush=True)

        # THE CHECK THAT MATTERS. Nothing downstream quotes a level; every reported number is a
        # per-draw CONTRAST between two conditions. A level deviation that is common-mode
        # cancels there and is harmless; one that is not, is not. Only this can tell them apart.
        (fg, sg), (fe, se) = keep["gold"], keep["empty"]
        c_fast = (fe - fg).mean(axis=(1, 2))
        c_slow = (se - sg).mean(axis=(1, 2))
        err = c_fast - c_slow                      # SIGNED, per draw
        bias = err.mean()                          # survives averaging over draws
        se = err.std(ddof=1) / np.sqrt(N)          # shrinks as 1/sqrt(draws * events)
        print(f"[verify] CONTRAST (empty - gold), per draw, n={N}")
        print(f"[verify]   mean contrast        = {c_slow.mean():+.6f}  (slow path)")
        print(f"[verify]   max |fast - slow|    = {np.abs(err).max():.3e}")
        print(f"[verify]   BIAS (mean signed)   = {bias:+.3e}  +/- {se:.1e}")
        # Do NOT assume the two levels' fast-vs-slow errors are common-mode: measured on the
        # pilot event they ADD rather than cancel (level errors 1.5e-3 / 1.7e-3, contrast error
        # 2.6e-3), i.e. they are independent per condition, not a shared offset. What actually
        # matters is therefore the BIAS: the random part averages down over draws and events,
        # a systematic part does not and sets a floor under every number this script reports.
        TARGET = 2e-3  # a content term this size is the smallest thing we would want to claim
        ok = abs(bias) < 0.2 * TARGET
        print(f"[verify]   vs target effect {TARGET:.0e}: bias is "
              f"{100 * abs(bias) / TARGET:.1f}% of it")
        print(f"[verify] CONTRAST VERDICT: "
              f"{'MATCH -- bias is small against the smallest effect worth claiming' if ok else 'CAUTION -- fast-path bias is NOT negligible against the target effect; widen n-draws or use the slow path for the headline'}",
              flush=True)
        return

    # ---------------- pilot / full run ----------------
    rows: list[dict] = []
    # (action, future_xy) so the wrongtraj donor can be ADE-gated against the current event.
    action_pool: list[tuple[torch.Tensor, np.ndarray]] = []

    def _future_xy(d: dict) -> np.ndarray:
        return np.asarray(d["ego_future_xyz"].detach().cpu()).reshape(-1, 3)[:, :2]

    def _ade(a: np.ndarray, b: np.ndarray) -> float:
        n = min(len(a), len(b))
        return float(np.linalg.norm(a[:n] - b[:n], axis=1).mean())

    def tok_of(txt: str) -> list:
        return model.tokenizer(txt, add_special_tokens=False)["input_ids"]

    n_flip_drop = [0]   # flip candidates rejected because the one-word edit changed token count

    t_start = time.time()
    for i, ev in enumerate(sel):
        try:
            data = load_physical_aiavdataset(ev["clip_id"], t0_us=ev["t0_us"], avdi=avdi)
            # Rolling pool of previously-seen gold actions. Using these as wrong-trajectory
            # targets costs nothing -- no extra dataset loads, no extra VLM passes.
            if "wrongtraj" in arms:
                if len(action_pool) >= 64:
                    action_pool.pop(0)
            nav_gold = None
            if args.nav:
                nav_gold = (oracle_nav.derive_nav_from_coc(ev["gold_coc"])
                            if args.nav_from == "coc" else
                            oracle_nav.derive_oracle_nav(
                                data["ego_future_xyz"].detach().cpu().numpy()[0, 0, :, :2]))
            base = {"nav": nav_gold} if args.nav else {}
            conds: list[tuple[str, int, str, dict]] = [("gold", -1, ev["gold_coc"], dict(base))]
            if "navnone" in arms:
                # nav_text=None -> route section absent entirely. THE nav denominator.
                conds.append(("navnone", -1, ev["gold_coc"], {"nav": None}))
            if "navswap" in arms and nav_gold:
                sw = _swap_direction(nav_gold)
                if sw != nav_gold:   # only meaningful on rows that name a direction
                    conds.append(("navswap", -1, ev["gold_coc"], {"nav": sw}))
            if "empty" in arms:
                conds.append(("empty", -1, "", dict(base)))
            # ---- the control ladder ----
            if "null" in arms:
                # Identical conditioning, identical target, re-scored. MUST come out exactly 0.
                # Not decoration: it is the only check that this path is deterministic, and the
                # ODE arm failed precisely here (0.036 nat run-to-run spread).
                conds.append(("null", -1, ev["gold_coc"], dict(base)))
            if "blankvision" in arms:
                conds.append(("blankvision", -1, ev["gold_coc"], {**base, "blank_vision": True}))
            if "flipdir" in arms:
                fl = _flip_directive(ev["gold_coc"])
                if fl is not None:
                    ftxt, fkind, fpos = fl
                    # The point of this arm is a length-matched ONE-WORD edit. If left/right (or
                    # Stop/Proceed) happen to tokenize to different lengths the prefix shifts and
                    # the length confound is back in through the front door. Drop those rows
                    # rather than explain them afterwards.
                    if len(tok_of(ev["gold_coc"])) == len(tok_of(ftxt)):
                        conds.append(("flipdir", -1, ftxt,
                                      {**base, "flip_kind": fkind, "flip_pos": fpos}))
                    else:
                        n_flip_drop[0] += 1
            if "wrongtraj" in arms and action_pool:
                rng = _rng("wrongtraj", ev["clip_id"], ev["event_idx"])
                mine = _future_xy(data)
                for di in range(min(args.n_donors, len(action_pool))):
                    # Draw until the donor is genuinely a different manoeuvre; fall back to the
                    # most distant candidate seen. `donor_ade` is recorded so the gate is
                    # auditable after the fact rather than merely asserted.
                    best_k, best_a = None, -1.0
                    for _try in range(max(1, args.donor_traj_max_tries)):
                        k = int(rng.integers(len(action_pool)))
                        a = _ade(mine, action_pool[k][1])
                        if a > best_a:
                            best_k, best_a = k, a
                        if a >= args.donor_traj_min_ade:
                            break
                    conds.append(("wrongtraj", di, ev["gold_coc"],
                                  {**base, "x": action_pool[best_k][0], "ade": best_a}))
            if "random" in arms:
                for di, d in enumerate(draw_donors(ev["clip_id"], ev["event_idx"])):
                    conds.append(("donor", di, d, dict(base)))
            if "cluster" in arms:
                for di, d in enumerate(
                    draw_donors(ev["clip_id"], ev["event_idx"], cluster=ev["event_cluster"])
                ):
                    conds.append(("donor_cluster", di, d, dict(base)))
            if "shuffled" in arms:
                for di, d in enumerate(
                    draw_shuffled(ev["clip_id"], ev["event_idx"], ev["gold_coc"])
                ):
                    conds.append(("shuffled", di, d, dict(base)))
            for name, di, txt, opts in conds:
                resid = score_fast(data, txt, x_override=opts.get("x"),
                                   blank_vision=opts.get("blank_vision", False),
                                   nav_text=opts.get("nav"))  # [N, 64, 2]
                losses = resid.mean(axis=(1, 2))  # the scalar every earlier run reported
                for j in range(N):
                    rows.append(
                        {
                            "clip_id": ev["clip_id"], "event_idx": ev["event_idx"],
                            "split": ev["split"], "event_cluster": ev["event_cluster"],
                            "cond": name, "donor_idx": di, "draw": j,
                            "donor_ade": float(opts.get("ade", np.nan)),
                            "flip_kind": opts.get("flip_kind", ""),
                            "flip_pos": int(opts.get("flip_pos", -1)),
                            "t": float(t_all[j]), "loss": float(losses[j]),
                            "prefix_len": int(getattr(score_fast, "last_prefix_len", -1)),
                            "n_tok_coc": len(txt.split()),
                            "t_sampler": args.t_sampler,
                            "lam_lo": lam_lo, "lam_hi": lam_hi,
                            # Flattened row-major, so index = waypoint * 2 + channel:
                            # EVEN = acceleration, ODD = curvature; waypoint w is t=(w+1)*0.1 s.
                            # Same layout as the token head's 128 traj tokens (README trap 5),
                            # which is what makes the two studies' decompositions comparable.
                            "resid": resid[j].reshape(-1).astype(np.float32),
                        }
                    )
        except Exception as e:  # one bad clip must not kill a shard
            print(f"[fm-llr] SKIP {ev['clip_id']}#{ev['event_idx']}: {type(e).__name__}: {e}",
                  flush=True)
            continue
        if "wrongtraj" in arms:
            action_pool.append((gold_action(data).detach().clone(), _future_xy(data)))
        if (i + 1) % 5 == 0 or i + 1 == len(sel):
            el = time.time() - t_start
            print(f"[fm-llr] {i+1}/{len(sel)} events  {el:.0f}s  "
                  f"{el/(i+1):.1f}s/event", flush=True)

    if "flipdir" in arms:
        print(f"[fm-llr] flipdir: {n_flip_drop[0]} candidates dropped for unequal token count",
              flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out = args.out
    if args.num_shards > 1:
        out = out.replace(".parquet", f".shard{args.shard_idx}.parquet")
    df.to_parquet(out)
    print(f"[fm-llr] wrote {out}  rows={len(df)}  events={df[['clip_id','event_idx']].drop_duplicates().shape[0]}",
          flush=True)


if __name__ == "__main__":
    main()
