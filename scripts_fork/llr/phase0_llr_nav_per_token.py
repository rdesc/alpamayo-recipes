# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Phase-0 LLR for NAVIGATION-COMMAND conditioning, resolved PER TRAJECTORY TOKEN.

Sibling of ``phase0_llr_per_token.py``, which asks the same question of the Chain-of-Causation
prose. Here the conditioning under test is the navigation command (the ``<|route_start|>...
<|route_end|>`` section), so the quantity is::

    llr_nav[k] = log p(a*_k | a*_<k, v, nav) - log p(a*_k | a*_<k, v)

Positive ``llr_nav`` = the navigation command HELPS. This is a conditional PMI diagnostic: does
route conditioning carry information the trajectory prediction actually uses?

THE METHOD, IN FULL
-------------------
Two forward passes and a subtraction. One with the navigation command present, one without it.
Nothing else. If you find yourself reaching for token surgery, attention masks or pad tokens to
build the denominator, stop -- that is the path that produced the retracted +0.227 on the CoC
version (see ``results/phase0_llr_action_direction.md``).

VOCABULARY -- one name per thing
--------------------------------
    empty    -- THE denominator. ``nav_text=None``, so ``construct_route`` returns [] and the
                route section is absent entirely. A genuine p(a*|v): it is exactly what the
                training-time ``nav_dropout_prob`` path produces and exactly what the validated
                no-nav eval arm produces (``eval_pai_av_val_nav.py:80``, ``cond = None``).
    (REMOVED) markers -- ``nav_text=""``, i.e. ``<|route_start|><|route_end|>``. NOT a
                valid p(a*|v): training never emits an empty route section (dropout drops the
                whole component, it does not empty it). Occupancy probe only.
    pad      -- route content overwritten with ``<|route_pad|>``, markers preserved, length held
                fixed. Also not a valid p(a*|v); a second occupancy probe.
    donor    -- another kind of real command in the same slot. Here the natural donors are the
                left<->right swap (``swap_direction``) and, on turn rows, "Continue straight".
                Those live in ``phase0_llr_nav_sanity_controls.py``.

"splice" and "blank" are retired as terms.

A PREMISE CORRECTION, RECORDED HERE ON PURPOSE
----------------------------------------------
On the CoC version the rule was "the empty denominator must RETAIN both markers", because
``get_label_mask``'s span is marker-inclusive and destroying ``<|cot_end|>`` injects ~23 nats.
That rule is about *editing an existing span*, and it does not transfer to nav:

``alpamayo/chat_template/components.py:construct_route`` returns ``[]`` when ``nav_text is
None``. So the in-distribution no-nav sequence has **no route markers at all** -- and that is
correct, because it is produced by re-tokenizing, not by overwriting anything. Nothing is
destroyed and no token follows a corrupted marker. The marker-retaining variant (``markers``,
``nav_text=""``) is the *off-distribution* one here, which is the reverse of the CoC case. Both
are provided; ``empty`` is the default and the one to quote. The scripts decode and assert the
route span's layout rather than assuming it either way.

THE ONE INVARIANT
-----------------
The ``empty`` denominator is SHORTER than the numerator (the whole route section leaves). That is
not a problem and needs no correction -- both passes teacher-force the same 128 ground-truth
trajectory tokens, so their log probs are directly comparable. It has exactly one consequence:

    PAIR TRAJECTORY TOKENS BY RANK WITHIN EACH PASS, NEVER BY ABSOLUTE SEQUENCE POSITION.

Each pass re-derives its own mask from its own ids; future-value tokens are then zipped by rank.

WHAT IS SCORED
--------------
``loss_future_traj`` is misnamed. On Alpamayo 1.5 its ``traj_mask`` spans 178 tokens: 48 *history*
trajectory tokens (history and future share one token-id block starting at
``traj_token_start_idx``, so the mask's id-range test scoops them up), the 2
``traj_future_start``/``traj_future_end`` delimiters, and only then the 128 actual future tokens.
This script scores the **128 future tokens only**. "Is this a future token" is a question about
POSITION (after the ``traj_future_start`` delimiter), not about token id. The per-token mean over
the full 178-token mask is reconciled against ``-out.loss_future_traj`` every event, which is what
catches mask/shift drift.

Token layout: ``tokens_per_future_traj = 128`` is ``(64 waypoints, 2)`` from
``UnicycleAccelCurvatureActionSpace`` flattened row-major with ``dt = 0.1``, i.e.
``stack([accel, kappa], -1)``. So token ``k`` -> ``t = (k // 2 + 1) * 0.1 s`` and even ``k`` =
acceleration, odd ``k`` = curvature. Nav is a *lateral* instruction, so if it is load-bearing the
effect should concentrate in **curvature**.

NAV COMMAND SOURCE
------------------
The released PAI-AV OOD reasoning set ships no route annotation, so the command is synthesized by
``/home/rod/repos/alpamayo1.5/oracle_nav.py`` (loaded by path; it is standalone):

  --nav-from traj  (default)  ``derive_oracle_nav(gt_xy)`` from the GT ego-future geometry ->
                              "Turn left in 12m" | "Turn right in 8m" | "Continue straight"
  --nav-from coc              ``derive_nav_from_coc(gold_coc)`` from the gold prose ->
                              "Turn left" | "Turn right" | "Continue straight"

Both are ORACLES (they use the answer), so this measures a recoverable ceiling, not a deployable
number. ``"Continue straight"`` is a real command, not an empty string -- the eval merely
suppresses it by default -- so straight rows ARE usable and are reported as their own arm.

THE COT CONFOUND -- read before interpreting
--------------------------------------------
``--cot gold`` (default, for comparability with the CoC run) conditions BOTH passes on the gold
CoC prose. But the gold CoC often states the maneuver ("the ego vehicle turns left"), so the route
information may already be in the context and nav would be measured as redundant. ``--cot empty``
removes that confound by running both passes with ``cot_text=""``, making nav the only linguistic
route cue. Run both; they answer different questions and the smoke run reports both.

Same venv/CWD rules as the rest of this directory (``_b300`` venv on this sm_103 box, invoked from
a recipe directory -- see ``README.md``).

Usage::

    cd recipes/alpamayo1_5_sft
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \\
      a1_5_sft_b300/bin/python ../../scripts_fork/llr/phase0_llr_nav_per_token.py \\
      --limit 12 --out /tmp/llr_nav_smoke.parquet

    # full split, sharded
    for i in 0 1 2 3 4 5 6 7; do
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$i \\
        a1_5_sft_b300/bin/python ../../scripts_fork/llr/phase0_llr_nav_per_token.py \\
        --num-shards 8 --shard-idx $i --out /opt/dlami/nvme/rod/results/llr_nav/pt.parquet &
    done; wait
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

DEFAULT_MODEL_PATH = "/mnt/efs/users/rod/ckpts/Alpamayo-1.5-10B-rl-training"
DEFAULT_PARQUET = "/opt/dlami/nvme/rod/datasets/pai_av_dataset/reasoning/ood_reasoning.parquet"
ORACLE_NAV_PATH = "/home/rod/repos/alpamayo1.5/oracle_nav.py"
MIN_T0_US = 1_700_000
DT_S = 0.1  # action_space_cfg.dt -- one accel/curvature PAIR per 0.1 s

# The nav-conditioned component order. `route` sits between traj_history and prompt, which is the
# same placement `alpamayo1_5.helper.create_message` uses at inference
# (`user_text = hist_traj_placeholder + route_section + prompt_text`). `route` is not in
# `construct_user_prompt`'s template dict, so its presence does not change the prompt sentence --
# dropping nav therefore changes the route section and nothing else.
COMPONENTS_ORDER = ["image", "traj_history", "route", "prompt", "cot", "traj_future"]
CHAT_TEMPLATE = "r1_5"  # "r1" has no `route` case at all; r1_5 delegates everything else to r1


def load_oracle_nav():
    """Load ~/repos/alpamayo1.5/oracle_nav.py by path.

    It is standalone (argparse/re/numpy only), so importing it by path avoids dragging in
    alpamayo1_5's model package, which would collide with the recipe's alpamayo_r1.
    """
    spec = importlib.util.spec_from_file_location("oracle_nav", ORACLE_NAV_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load oracle_nav from {ORACLE_NAV_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def swap_direction(nav_text: str) -> str:
    """Flip left<->right in a navigation instruction. "Continue straight" is unchanged.

    Byte-for-byte the algorithm of ``alpamayo1_5.nav_utils.swap_direction``
    (``/home/rod/repos/alpamayo1.5/src/alpamayo1_5/nav_utils.py:199``). It is re-stated here
    rather than imported because ``nav_utils`` imports ``alpamayo1_5.helper`` and
    ``alpamayo1_5.models.base_model``, which collide with the recipe's ``alpamayo_r1`` package in
    this venv. Keep the two in sync; the placeholder trick avoids double-swapping.
    """
    PH = "___PH___"
    result = re.sub(r"\bleft\b", PH, nav_text, flags=re.IGNORECASE)
    result = re.sub(
        r"\bright\b",
        lambda m: "left" if m.group()[0].islower() else "Left",
        result,
        flags=re.IGNORECASE,
    )
    return result.replace(PH, "right")


def nav_class(nav_text: str | None) -> str:
    """left / right / straight / none from a command string."""
    if nav_text is None:
        return "none"
    low = nav_text.lower()
    if "left" in low:
        return "left"
    if "right" in low:
        return "right"
    return "straight"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--split", default="both", choices=["train", "val", "both"])
    p.add_argument(
        "--nav-from",
        default="traj",
        choices=["traj", "coc"],
        help=(
            "Oracle source for the navigation command. 'traj' derives it from the GT ego-future "
            "geometry (derive_oracle_nav, gives a distance: 'Turn left in 12m'); 'coc' derives the "
            "direction from the gold CoC prose (derive_nav_from_coc, direction only)."
        ),
    )
    p.add_argument("--turn-deg", type=float, default=25.0, help="oracle_nav.TURN_YAW, in degrees.")
    p.add_argument(
        "--cot",
        default="gold",
        choices=["gold", "empty"],
        help=(
            "What the CoC span holds in BOTH passes. 'gold' matches the CoC LLR run and keeps the "
            "reasoning prose in context -- but the gold CoC often states the maneuver, so nav can "
            "look redundant. 'empty' makes the nav command the only linguistic route cue. Neither "
            "is wrong; they answer different questions. See THE COT CONFOUND in the docstring."
        ),
    )
    # NOTE: there is deliberately NO --denominator flag. `nav_text=None` (route section absent)
    # is the only in-distribution p(a*|v) for nav, so it is the only denominator. Two variants
    # were implemented and removed:
    #   markers -- nav_text="", an empty <|route_start|><|route_end|> pair. Training NEVER emits
    #              that, so it is off-distribution. (This is the reverse of the CoC case, where
    #              retaining the markers is mandatory because that span is marker-inclusive.)
    #   pad     -- command overwritten with <|route_pad|>, which is UNTRAINED in this checkpoint
    #              (embedding norm 0.105 vs a 0.81 vocab mean; route_start/route_end are
    #              0.81/0.84). It injects an initialization-value embedding the model has never
    #              conditioned on, and duly read +0.0207 -- 3x the real effect -- which is an
    #              artifact, not an occupancy result.
    # Presence-vs-content is measured with the swap / straight arms in the sanity-controls
    # script, which stay in distribution because they are real commands.
    p.add_argument(
        "--nav-swap",
        action="store_true",
        help="Numerator uses the left<->right swapped command (counterfactual nav). Straight rows "
        "are unchanged by the swap, so they become a built-in null for this arm.",
    )
    p.add_argument("--first-event-only", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--events-json",
        default=None,
        help="Restrict to the events in this manifest (list of dicts with clip_id/event_idx, or a "
        "dict with an 'events'/'items' key). Use it to run several arms (--cot gold vs empty, "
        "--nav-swap) over the SAME events -- classifying events by command costs a clip load "
        "each, so --balance-commands is best done once and its selection reused. Every run writes "
        "its selection to <out>.events.json for exactly that purpose.",
    )
    p.add_argument(
        "--balance-commands",
        action="store_true",
        help="With --limit, pick events so left/right/straight are each ~limit/3 of the run "
        "instead of taking the first N. 'Continue straight' dominates the split, so the first N "
        "are nearly all straight -- use this for smoke runs, not for the full-split run (which "
        "should stay unweighted and report the command breakdown as it falls).",
    )
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--out", required=True, help="Output .parquet: one row per (event, token).")
    return p.parse_args()


def load_events(parquet_path: str, split: str, all_events: bool) -> list[dict]:
    """Flatten ood_reasoning.parquet into one row per annotated event.

    Identical to phase0_llr_per_token.py's loader (see that file for provenance).
    """
    df = pd.read_parquet(parquet_path)
    if split != "both":
        df = df[df["split"] == split].copy()
    df["events"] = df["events"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    df = df.dropna(subset=["events"])
    rows = []
    for clip_id, row in df.iterrows():
        events = row["events"]
        idxs = range(len(events)) if all_events else [0]
        for ei in idxs:
            ev = events[ei]
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


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    oracle_nav = load_oracle_nav()
    oracle_nav.TURN_YAW = np.deg2rad(args.turn_deg)

    sys.path[:0] = ["../../../src", "../.."]
    import physical_ai_av
    from alpamayo_r1.helper import to_device
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo.processor.qwen_processor import (
        collate_fn_from_model_config,
        get_preprocess_data_fn_from_model_config,
        build_processor,
    )
    from alpamayo.utils.get_label_mask import get_label_mask

    from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA

    print(f"[nav] loading model {args.model_path} ...", flush=True)
    model = TrainableReasoningVLA.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).cuda().eval()

    preprocess_fn = get_preprocess_data_fn_from_model_config(
        components_order=COMPONENTS_ORDER,
        components_prompt=["cot", "traj_future"],
        label_components=["cot"],
        generation_mode=False,
        include_camera_ids=True,
        include_frame_nums=True,
        model_config=model.config,
        chat_template_version=CHAT_TEMPLATE,
    )
    processor = build_processor(
        vlm_name_or_path=model.config.vlm_name_or_path,
        traj_vocab_size=model.config.traj_vocab_size,
        min_pixels=model.config.min_pixels,
        max_pixels=model.config.max_pixels,
        include_camera_ids=True,
        include_frame_nums=True,
        chat_template_version=CHAT_TEMPLATE,
    )
    tokenizer = processor.tokenizer

    traj_lo = model.future_token_start_idx
    traj_hi = traj_lo + model.config.traj_vocab_size
    sid = model.special_token_ids
    route_start_id = tokenizer.convert_tokens_to_ids("<|route_start|>")
    route_end_id = tokenizer.convert_tokens_to_ids("<|route_end|>")
    print(
        f"[nav] traj value-token id range [{traj_lo}, {traj_hi}); "
        f"tokens_per_future_traj={model.config.tokens_per_future_traj}; dt={DT_S}s; "
        f"route_start={route_start_id} route_end={route_end_id}",
        flush=True,
    )

    events = load_events(args.parquet, args.split, all_events=not args.first_event_only)
    if args.num_shards > 1:
        events = events[args.shard_idx :: args.num_shards]
    if args.events_json:
        with open(args.events_json) as f:
            man = json.load(f)
        if isinstance(man, dict):
            man = man.get("events", man.get("items", []))
        want = {(m["clip_id"], int(m["event_idx"])) for m in man}
        events = [e for e in events if (e["clip_id"], int(e["event_idx"])) in want]
        print(f"[nav] --events-json: matched {len(events)}/{len(want)} requested events", flush=True)
    if args.limit and not args.balance_commands:
        events = events[: args.limit]
    print(f"[nav] shard {args.shard_idx}/{args.num_shards}: {len(events)} events", flush=True)

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    # The nav command is derived from the GT future, so classifying an event means loading its
    # clip. Cache that, so a balanced smoke run does not load anything twice.
    _cache: dict[tuple, dict] = {}

    def _load(ev: dict) -> dict:
        key = (ev["clip_id"], ev["t0_us"])
        if key in _cache:
            return _cache.pop(key)  # consume: camera frames are large, never hold the whole split
        d = load_physical_aiavdataset(ev["clip_id"], t0_us=ev["t0_us"], avdi=avdi)
        if args.balance_commands:
            _cache[key] = d
        return d

    def _derive_nav(ev: dict, d: dict) -> str:
        if args.nav_from == "coc":
            return oracle_nav.derive_nav_from_coc(ev["gold_coc"])
        return oracle_nav.derive_oracle_nav(d["ego_future_xyz"].detach().cpu().numpy()[0, 0, :, :2])

    if args.balance_commands and args.limit:
        want = max(1, args.limit // 3)
        picked: dict[str, list] = {"left": [], "right": [], "straight": []}
        n_probed = 0
        for ev in events:
            if all(len(v) >= want for v in picked.values()):
                break
            try:
                d = _load(ev)
            except Exception:
                continue
            n_probed += 1
            c = nav_class(_derive_nav(ev, d))
            if c in picked and len(picked[c]) < want:
                picked[c].append(ev)
            else:
                _cache.pop((ev["clip_id"], ev["t0_us"]), None)  # don't hoard rejected clips
        events = [e for lst in picked.values() for e in lst]
        print(
            f"[nav] balanced selection over {n_probed} probed events: "
            + ", ".join(f"{k}={len(v)}" for k, v in picked.items()),
            flush=True,
        )

    # Record the selection so other arms can reuse it via --events-json instead of paying the
    # clip-load cost of classifying events all over again.
    sel_path = f"{args.out}.events.json"
    os.makedirs(os.path.dirname(sel_path) or ".", exist_ok=True)
    with open(sel_path, "w") as f:
        json.dump([{"clip_id": e["clip_id"], "event_idx": int(e["event_idx"])} for e in events], f)
    print(f"[nav] selection written to {sel_path}", flush=True)

    def _per_token_logp(td: dict, data: dict, fused: torch.Tensor, traj_mask: torch.Tensor):
        """log p for each token in the traj span, in sequence order."""
        model_inputs = to_device({"tokenized_data": dict(td)}, "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                tokenized_data=model_inputs["tokenized_data"],
                ego_history_xyz=data["ego_history_xyz"].cuda(),
                ego_history_rot=data["ego_history_rot"].cuda(),
                ego_future_xyz=data["ego_future_xyz"].cuda(),
                ego_future_rot=data["ego_future_rot"].cuda(),
                labels_mask=None,
            )
        # Replicate sft_base_model._compute_next_token_loss's shift/gather exactly, but with
        # reduction="none" so each token keeps its own value. Mean of this == loss_future_traj.
        m = traj_mask[:, 1:]
        shift_labels = fused[:, 1:][m].contiguous()
        shift_logits = out.logits[..., :-1, :][m].contiguous().float()
        logp = -F.cross_entropy(shift_logits, shift_labels, reduction="none")
        # Sequence position each scored token predicts (the +1 undoes the left-shift), so history
        # and future trajectory tokens can be separated by position.
        seq_pos = m[0].nonzero().flatten() + 1
        return (
            logp.cpu().numpy(),
            shift_labels.cpu().numpy(),
            seq_pos.cpu().numpy(),
            -out.loss_future_traj.item(),
        )

    def _tokenize(data: dict, cot_text: str, nav_text: str | None) -> dict:
        """Tokenize+collate one event at a given (cot, nav) pair.

        ``nav_text=None`` makes ``construct_route`` return [], so the route section -- markers
        included -- is absent: the in-distribution no-nav sequence. ``nav_text=""`` keeps the
        markers with no content.
        """
        sample = {
            "image_frames": data["image_frames"],
            "camera_indices": data["camera_indices"],
            "relative_timestamps": data["relative_timestamps"],
            "ego_history_xyz": data["ego_history_xyz"].squeeze(0),
            "ego_history_rot": data["ego_history_rot"].squeeze(0),
            "cot": cot_text,
            "nav_text": nav_text,
        }
        sample["tokenized_data"] = preprocess_fn(sample)
        batch = collate_fn_from_model_config(
            [sample], model_config=model.config,
            include_camera_ids=True, include_frame_nums=True, chat_template_version=CHAT_TEMPLATE,
        )
        return batch["tokenized_data"]

    def _prep_span(input_ids: torch.Tensor, data: dict):
        """Fuse trajectory tokens and build the model's own traj_mask plus the future delimiter
        position. Re-derived PER PASS: the 'empty' denominator is shorter, so its mask cannot be
        shared with the numerator's. See THE ONE INVARIANT in the module docstring."""
        traj_data = {
            "ego_history_xyz": data["ego_history_xyz"].cuda(),
            "ego_history_rot": data["ego_history_rot"].cuda(),
            "ego_future_xyz": data["ego_future_xyz"].cuda(),
            "ego_future_rot": data["ego_future_rot"].cuda(),
        }
        fused = model.fuse_traj_tokens(input_ids.clone().cuda(), traj_data)
        # Replicates the model's own traj_mask (sft_base_model.py:660-667) EXACTLY, including its
        # quirk: history-trajectory tokens come from the same id block as future ones, so the
        # id-range test catches all 48 tokens_per_history_traj as well as the 128 future tokens.
        traj_mask = (
            ((fused >= traj_lo) & (fused < traj_hi))
            | (fused == sid["traj_future_start"])
            | (fused == sid["traj_future_end"])
        )
        fsp = (fused[0] == sid["traj_future_start"]).nonzero().flatten()
        if fsp.numel() == 0:
            raise RuntimeError("traj_future_start token not found in fused sequence")
        return fused, traj_mask, int(fsp[-1].item())

    def _future_sel(labels, seq_pos, fut_start_pos):
        return (labels >= traj_lo) & (labels < traj_hi) & (seq_pos > fut_start_pos)

    rows: list[dict] = []
    t_start = time.time()
    n_ok = 0
    for i, ev in enumerate(events):
        clip_id, t0_us = ev["clip_id"], ev["t0_us"]
        try:
            data = _load(ev)

            # --- the navigation command (oracle) ---
            nav_text = _derive_nav(ev, data)
            nav_used = swap_direction(nav_text) if args.nav_swap else nav_text
            cmd = nav_class(nav_text)

            cot_text = ev["gold_coc"] if args.cot == "gold" else ""

            # --- numerator: p(a* | v, nav) ---
            td = _tokenize(data, cot_text, nav_used)
            input_ids_real = td["input_ids"]

            # Verify the route span's layout by decoding rather than assuming it. get_label_mask
            # returns a span INCLUSIVE of both markers (get_label_mask.py:45), so the first and
            # last set positions must be <|route_start|> / <|route_end|>.
            route_mask = get_label_mask(input_ids_real, tokenizer, ["route"])
            n_route_tokens = int(route_mask.sum().item())
            if n_route_tokens == 0:
                raise RuntimeError("route span not found in tokenized sequence")
            route_pos = route_mask[0].nonzero().flatten()
            assert input_ids_real[0, route_pos[0]].item() == route_start_id, (
                "first token of the route span is not <|route_start|> -- span layout changed"
            )
            assert input_ids_real[0, route_pos[-1]].item() == route_end_id, (
                "last token of the route span is not <|route_end|> -- span layout changed"
            )
            route_decoded = tokenizer.decode(input_ids_real[0, route_pos].tolist())

            fused, traj_mask, fut_start_pos = _prep_span(input_ids_real, data)
            td_num = dict(td)
            td_num["input_ids"] = input_ids_real.clone()
            lp_num, labels_seq, seq_pos, scalar_num = _per_token_logp(td_num, data, fused, traj_mask)

            # Reconcile against the model's own mean over the full 178-token mask -- catches any
            # mask/shift drift immediately.
            recon_err = abs(float(lp_num.mean()) - scalar_num)
            if recon_err > 1e-3:
                raise RuntimeError(
                    f"per-token mean {lp_num.mean():.6f} != loss_future_traj {scalar_num:.6f} "
                    f"(err {recon_err:.2e}) -- span mask or shift is wrong"
                )

            # --- denominator ---
            # nav_text=None drops the whole route section -- the sequence the validated no-nav
            # eval arm produces, and (per null_template) byte-identical to the plain r1
            # template, which has no route case at all. Length changes, so this pass needs its
            # own fuse/mask and pairing must be by RANK.
            td_den = _tokenize(data, cot_text, None)
            fused_den, traj_mask_den, fut_start_den = _prep_span(td_den["input_ids"], data)
            lp_den_all, labels_den, seq_pos_den, scalar_den = _per_token_logp(
                td_den, data, fused_den, traj_mask_den
            )
            sel_den = _future_sel(labels_den, seq_pos_den, fut_start_den)
            sel_num = _future_sel(labels_seq, seq_pos, fut_start_pos)
            if int(sel_num.sum()) != int(sel_den.sum()):
                raise RuntimeError(
                    f"future-token count differs between passes ({int(sel_num.sum())} vs "
                    f"{int(sel_den.sum())}) -- per-token pairing would be misaligned"
                )
            # Both passes teacher-force the SAME ground-truth trajectory, so the k-th future
            # token is the same target in both and pairing by rank is exact.
            if not np.array_equal(labels_seq[sel_num], labels_den[sel_den]):
                raise RuntimeError("paired future targets differ between passes")
            num_vals = lp_num[sel_num]
            den_vals = lp_den_all[sel_den]
            # Map rank -> (horizon time, control channel). Ranking is over FUTURE value tokens
            # only; the 48 history tokens share the future tokens' id block and precede the route
            # span (so their LLR is structurally 0 by causal attention), and including them would
            # shift every future token's timestep.
            n_fut = len(num_vals)
            if n_fut != int(model.config.tokens_per_future_traj):
                raise RuntimeError(
                    f"scored {n_fut} future tokens, expected {model.config.tokens_per_future_traj}"
                )
            for traj_idx in range(n_fut):
                rows.append(
                    {
                        "clip_id": clip_id,
                        "event_idx": ev["event_idx"],
                        "event_cluster": ev["event_cluster"],
                        "split": ev["split"],
                        "nav_text": nav_text,
                        "nav_used": nav_used,
                        "nav_command": cmd,
                        "nav_from": args.nav_from,
                        "n_route_tokens": n_route_tokens,
                        "route_decoded": route_decoded,
                        "cot_arm": args.cot,
                        "denominator": "empty",
                        "nav_swap": bool(args.nav_swap),
                        "traj_idx": traj_idx,
                        "channel": "accel" if traj_idx % 2 == 0 else "curvature",
                        "t_s": (traj_idx // 2 + 1) * DT_S,
                        "logp_nav": float(num_vals[traj_idx]),
                        "logp_den": float(den_vals[traj_idx]),
                        "llr_nav": float(num_vals[traj_idx] - den_vals[traj_idx]),
                    }
                )
            n_ok += 1
            ev_llr = float(np.mean(num_vals - den_vals))
            print(
                f"[nav] [{i + 1}/{len(events)}] {clip_id[:8]} cmd={cmd:8s} "
                f"nav={nav_text!r:24s} llr_nav={ev_llr:+.4f} logp={float(num_vals.mean()):+.4f} "
                f"({(time.time() - t_start) / 60:.1f} min)",
                flush=True,
            )
        except Exception as e:
            torch.cuda.empty_cache()
            print(f"[nav] [{i + 1}/{len(events)}] {clip_id} FAILED: {type(e).__name__}: {e}", flush=True)

    df = pd.DataFrame(rows)
    out_path = args.out
    if args.num_shards > 1:
        root, _, ext = args.out.rpartition(".")
        out_path = f"{root}.shard{args.shard_idx}-of-{args.num_shards}.{ext}"
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(
        f"\n[nav] done in {(time.time() - t_start) / 60:.1f} min. "
        f"{n_ok}/{len(events)} events ok, {len(df)} token rows. Wrote {out_path}",
        flush=True,
    )
    if not len(df):
        return

    ev_mean = df.groupby(["clip_id", "event_idx"])["llr_nav"].mean()
    n_ev = len(ev_mean)
    sem = ev_mean.std() / max(np.sqrt(n_ev), 1e-9)
    print(
        f"\n[nav] llr_nav per future trajectory token, n={n_ev} events / {len(df)} tokens\n"
        f"        mean {ev_mean.mean():+.4f}  per-event std {ev_mean.std():.4f}  "
        f"sigma-from-zero {ev_mean.mean() / sem if sem else float('nan'):.2f}  "
        f"frac>0 {(ev_mean > 0).mean() * 100:.1f}%\n"
        f"        reference log p(a*|v,nav) = {df['logp_nav'].mean():+.4f} nats/token",
        flush=True,
    )
    print("\n[nav] by command (per-event means):", flush=True)
    per_ev = df.groupby(["clip_id", "event_idx", "nav_command"])["llr_nav"].mean().reset_index()
    print(per_ev.groupby("nav_command")["llr_nav"].agg(["mean", "std", "count"]), flush=True)
    print("\n[nav] by channel (token-level):", flush=True)
    print(df.groupby("channel")["llr_nav"].agg(["mean", "std", "count"]), flush=True)
    print("\n[nav] horizon profile (token-level mean):", flush=True)
    bins = pd.cut(df["t_s"], [0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6, 6.4])
    print(df.groupby([bins, "channel"], observed=True)["llr_nav"].mean().unstack(), flush=True)


if __name__ == "__main__":
    main()
