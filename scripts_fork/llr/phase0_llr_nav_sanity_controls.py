# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Sanity controls for the navigation-conditioning LLR -- is "nav buys ~X nats" a FINDING, or a
broken probe that would report the same number no matter what?

Sibling of ``phase0_llr_sanity_controls.py`` (the CoC control ladder). Same discipline, same
scoring path as ``phase0_llr_nav_per_token.py``: a null is only believable if the instrument
demonstrably responds to things it SHOULD respond to and stays silent on things it should not.

Sign convention throughout: every ``*_llr`` column is ``log p(a* | reference) - log p(a* |
alternative)``, so POSITIVE means the reference (the correct navigation command) is better.

Controls that MUST come out at exactly 0.0 (if any is non-zero the pipeline is misaligned):

  null_identical    re-tokenize with the SAME command and score twice. Any deviation from 0.0
                    means the scoring path is non-deterministic.
  null_retokenize   a second fresh tokenize/collate of identical inputs -- same sequence, so also
                    exactly 0.0. Catches state leaking between passes (``forward`` pops
                    "input_ids" from its dict, which is a real hazard here).
  history_llr       LLR on the 48 history-trajectory tokens. In the nav-conditioned component
                    order the route section sits AFTER traj_history, so causal attention makes any
                    dependence impossible: exactly 0.0 is a structural invariant, and it is the
                    sharpest alignment check available -- a one-token position/shift error
                    anywhere breaks it immediately.
  null_template     the r1 chat template (which has no ``route`` component at all) against the
                    r1_5 template with ``nav_text=None``. r1_5 only ADDS route/question/answer
                    cases and delegates everything else to r1, so with nav absent the two must
                    produce identical input_ids and identical log probs. This is what ties the
                    nav denominator to the already-validated CoC scoring path, and makes the
                    reference log p(a*) directly comparable to the CoC run's -1.932.

Controls that MUST come out LARGE (if any is ~0, the probe is dead and any null is meaningless):

  wrong_vision_llr  another scene's camera frames. Misleading rather than blank vision: blank
                    pixels are merely uninformative, another scene's frames assert something
                    false. On the CoC run this registered +0.266 through this same code path.
  wrong_traj_llr    teacher-force another event's ground-truth trajectory. +1.068 on the CoC run.

Two occupancy arms were implemented and REMOVED, because neither is a valid p(a*|v) for nav:
``markers`` (``nav_text=""``, an empty ``<|route_start|><|route_end|>`` pair -- training never
emits that) and ``pad`` (``<|route_pad|>``, which is UNTRAINED in this checkpoint: embedding norm
0.105 against a 0.81 vocab mean, so it injected an initialization-value embedding and read
+0.0207, 3x the real effect). Presence-vs-content is measured with the swap/straight arms, which
stay in distribution because they are real commands.

Nav-specific arms (the actual question, not pass/fail):

  empty_llr         the measurement: correct command vs. no route section at all. THE llr_nav.
  swap_llr          counterfactual: the left<->right swapped command. Being a real, well-formed,
                    in-distribution command that is WRONG for the scene, what it buys over no
                    command is PRESENCE; content is the residual ``empty_llr - swap_llr``
                    ``swap_direction`` returns "Continue straight" unchanged, so straight rows are
                    a built-in null for this arm and are reported separately.
  straight_llr      turn rows only: the correct turn command vs. "Continue straight" -- another
                    wrong-but-valid command, and the one the deployed eval actually suppresses.

Read the output as a ladder. wrong_vision/wrong_traj set the scale at which this instrument
registers a real conditioning change; the nulls set the noise floor; the nav arms are then
interpretable relative to both.

Same venv/CWD rules as the rest of this directory. Usage::

    cd recipes/alpamayo1_5_sft
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \\
      a1_5_sft_b300/bin/python ../../scripts_fork/llr/phase0_llr_nav_sanity_controls.py \\
      --limit 12 --out /tmp/llr_nav_sanity.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from phase0_llr_nav_per_token import (  # noqa: E402  (sibling module, same directory)
    CHAT_TEMPLATE,
    COMPONENTS_ORDER,
    DEFAULT_MODEL_PATH,
    DEFAULT_PARQUET,
    load_events,
    load_oracle_nav,
    nav_class,
    swap_direction,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--split", default="both", choices=["train", "val", "both"])
    p.add_argument("--nav-from", default="traj", choices=["traj", "coc"])
    p.add_argument("--turn-deg", type=float, default=25.0)
    p.add_argument(
        "--cot",
        default="gold",
        choices=["gold", "empty"],
        help="What the CoC span holds in every pass. See THE COT CONFOUND in "
        "phase0_llr_nav_per_token.py.",
    )
    p.add_argument("--limit", type=int, default=12)
    p.add_argument(
        "--events-json",
        default=None,
        help="Restrict to the events in this manifest (as written by "
        "phase0_llr_nav_per_token.py's <out>.events.json), so the control ladder runs on exactly "
        "the events the measurement ran on. Classifying an event by command costs a clip load, so "
        "reuse beats re-deriving.",
    )
    p.add_argument(
        "--balance-commands",
        action="store_true",
        help="Pick events so left/right/straight are represented rather than taking the first N "
        "(straight dominates the split, so the first N are nearly all straight).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

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

    print(f"[nav-sanity] loading model {args.model_path} ...", flush=True)
    model = TrainableReasoningVLA.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).cuda().eval()

    def _make_preprocess(order, template):
        return get_preprocess_data_fn_from_model_config(
            components_order=order,
            components_prompt=["cot", "traj_future"],
            label_components=["cot"],
            generation_mode=False,
            include_camera_ids=True,
            include_frame_nums=True,
            model_config=model.config,
            chat_template_version=template,
        )

    preprocess_fn = _make_preprocess(COMPONENTS_ORDER, CHAT_TEMPLATE)
    # The exact config phase0_llr_per_token.py (the CoC run) uses -- for null_template.
    preprocess_fn_r1 = _make_preprocess(
        ["image", "traj_history", "prompt", "cot", "traj_future"], "r1"
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

    events = load_events(args.parquet, args.split, all_events=True)
    if args.events_json:
        with open(args.events_json) as f:
            man = json.load(f)
        if isinstance(man, dict):
            man = man.get("events", man.get("items", []))
        want = {(m["clip_id"], int(m["event_idx"])) for m in man}
        events = [e for e in events if (e["clip_id"], int(e["event_idx"])) in want]
        print(f"[nav-sanity] --events-json: matched {len(events)}/{len(want)}", flush=True)
    print(f"[nav-sanity] {len(events)} candidate events", flush=True)

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    def _tokenize(
        data: dict,
        cot_text: str,
        nav_text: str | None,
        vision_from: dict | None = None,
        use_r1: bool = False,
    ) -> dict:
        sample = {
            "image_frames": data["image_frames"] if vision_from is None else vision_from["image_frames"],
            "camera_indices": data["camera_indices"],
            "relative_timestamps": data["relative_timestamps"],
            "ego_history_xyz": data["ego_history_xyz"].squeeze(0),
            "ego_history_rot": data["ego_history_rot"].squeeze(0),
            "cot": cot_text,
            "nav_text": nav_text,
        }
        template = "r1" if use_r1 else CHAT_TEMPLATE
        sample["tokenized_data"] = (preprocess_fn_r1 if use_r1 else preprocess_fn)(sample)
        batch = collate_fn_from_model_config(
            [sample], model_config=model.config,
            include_camera_ids=True, include_frame_nums=True, chat_template_version=template,
        )
        return batch["tokenized_data"]

    def _score(td: dict, traj_src: dict) -> dict:
        """Mean log p over future / history / delimiter token roles.

        No ego-history ablation here, for the reason documented in
        phase0_llr_sanity_controls.py: the action space derives its t0 velocity from ego history,
        so swapping it re-encodes the FUTURE TARGET tokens too -- that scores a different target
        rather than ablating context. wrong_vision is the clean context ablation, since images
        never reach the trajectory tokenizer.
        """
        input_ids = td["input_ids"]
        traj_data = {
            "ego_history_xyz": traj_src["ego_history_xyz"].cuda(),
            "ego_history_rot": traj_src["ego_history_rot"].cuda(),
            "ego_future_xyz": traj_src["ego_future_xyz"].cuda(),
            "ego_future_rot": traj_src["ego_future_rot"].cuda(),
        }
        fused = model.fuse_traj_tokens(input_ids.clone().cuda(), traj_data)
        traj_mask = (
            ((fused >= traj_lo) & (fused < traj_hi))
            | (fused == sid["traj_future_start"])
            | (fused == sid["traj_future_end"])
        )
        fsp = int((fused[0] == sid["traj_future_start"]).nonzero().flatten()[-1].item())

        model_inputs = to_device({"tokenized_data": dict(td)}, "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                tokenized_data=model_inputs["tokenized_data"],
                ego_history_xyz=traj_src["ego_history_xyz"].cuda(),
                ego_history_rot=traj_src["ego_history_rot"].cuda(),
                ego_future_xyz=traj_src["ego_future_xyz"].cuda(),
                ego_future_rot=traj_src["ego_future_rot"].cuda(),
                labels_mask=None,
            )
        m = traj_mask[:, 1:]
        shift_labels = fused[:, 1:][m].contiguous()
        shift_logits = out.logits[..., :-1, :][m].contiguous().float()
        logp = -F.cross_entropy(shift_logits, shift_labels, reduction="none")
        seq_pos = m[0].nonzero().flatten() + 1
        is_val = (shift_labels >= traj_lo) & (shift_labels < traj_hi)
        is_fut = is_val & (seq_pos > fsp)
        is_hist = is_val & ~is_fut
        # Reconcile the full-mask mean against the model's own scalar -- catches mask/shift drift.
        recon_err = abs(float(logp.mean().item()) + float(out.loss_future_traj.item()))
        if recon_err > 1e-3:
            raise RuntimeError(
                f"per-token mean != -loss_future_traj (err {recon_err:.2e}) -- mask or shift wrong"
            )
        return {
            "future": float(logp[is_fut].mean().item()),
            "hist": float(logp[is_hist].mean().item()) if bool(is_hist.any()) else float("nan"),
            "delim": float(logp[~is_val].mean().item()),
            "n_future": int(is_fut.sum().item()),
            "n_hist": int(is_hist.sum().item()),
            "recon_err": recon_err,
            "input_ids": td["input_ids"],
        }

    def _derive_nav(ev: dict, data: dict) -> str:
        if args.nav_from == "coc":
            return oracle_nav.derive_nav_from_coc(ev["gold_coc"])
        gt_xy = data["ego_future_xyz"].detach().cpu().numpy()[0, 0, :, :2]
        return oracle_nav.derive_oracle_nav(gt_xy)

    # --- event selection -------------------------------------------------------------------
    # Straight dominates the split, so taking the first N would give an all-straight smoke run.
    # --balance-commands probes events until each command class has ~limit/3 members. The nav
    # command needs the GT future, which means loading the clip, so cache what we load.
    loaded: dict[tuple, dict] = {}

    def _load(ev):
        key = (ev["clip_id"], ev["t0_us"])
        if key not in loaded:
            loaded[key] = load_physical_aiavdataset(ev["clip_id"], t0_us=ev["t0_us"], avdi=avdi)
        return loaded[key]

    if args.balance_commands:
        want = max(1, args.limit // 3)
        picked: dict[str, list] = {"left": [], "right": [], "straight": []}
        for ev in events:
            if all(len(v) >= want for v in picked.values()):
                break
            try:
                d = _load(ev)
            except Exception:
                continue
            c = nav_class(_derive_nav(ev, d))
            if c in picked and len(picked[c]) < want:
                picked[c].append(ev)
            else:
                loaded.pop((ev["clip_id"], ev["t0_us"]), None)  # camera frames are large
        use = [e for lst in picked.values() for e in lst]
        print(
            f"[nav-sanity] balanced selection: "
            + ", ".join(f"{k}={len(v)}" for k, v in picked.items()),
            flush=True,
        )
    else:
        use = events[: args.limit]

    rows: list[dict] = []
    t0 = time.time()
    for i, ev in enumerate(use):
        try:
            data = _load(ev)
            cot = ev["gold_coc"] if args.cot == "gold" else ""
            nav = _derive_nav(ev, data)
            cmd = nav_class(nav)

            # --- reference: p(a* | v, nav) ---
            td_nav = _tokenize(data, cot, nav)
            ref = _score(td_nav, data)

            route_mask = get_label_mask(td_nav["input_ids"], tokenizer, ["route"])
            rpos = route_mask[0].nonzero().flatten()
            n_route = int(route_mask.sum().item())
            assert n_route > 0, "route span not found"
            assert td_nav["input_ids"][0, rpos[0]].item() == route_start_id
            assert td_nav["input_ids"][0, rpos[-1]].item() == route_end_id
            route_decoded = tokenizer.decode(td_nav["input_ids"][0, rpos].tolist())

            res = {
                "nav_text": nav,
                "nav_command": cmd,
                "n_route_tokens": n_route,
                "route_decoded": route_decoded,
                "ref_future": ref["future"],
                "ref_delim": ref["delim"],
                "hist_logp": ref["hist"],
                "n_hist": ref["n_hist"],
                "n_future": ref["n_future"],
                "recon_err": ref["recon_err"],
            }

            # ---------- nulls: must be exactly 0 ----------
            res["null_identical"] = _score(td_nav, data)["future"] - ref["future"]
            res["null_retokenize"] = _score(_tokenize(data, cot, nav), data)["future"] - ref["future"]

            td_none = _tokenize(data, cot, None)  # the 'empty' denominator
            none_scored = _score(td_none, data)
            # History tokens precede the route section, so this is structurally 0.
            res["history_llr"] = ref["hist"] - none_scored["hist"]

            # r1 template (no route component) vs r1_5 with nav absent: must be the same sequence.
            td_r1 = _tokenize(data, cot, None, use_r1=True)
            ids_match = bool(
                td_r1["input_ids"].shape == td_none["input_ids"].shape
                and torch.equal(td_r1["input_ids"], td_none["input_ids"])
            )
            res["null_template_ids_match"] = ids_match
            res["null_template"] = _score(td_r1, data)["future"] - none_scored["future"]

            # ---------- the measurement under test ----------
            res["empty_llr"] = ref["future"] - none_scored["future"]

            # ---------- occupancy probes ----------

            # ---------- nav content controls ----------
            nav_sw = swap_direction(nav)
            res["nav_swapped"] = nav_sw
            # On straight rows swap_direction is a no-op, so this arm is a null there by
            # construction -- reported separately in the summary, never averaged in.
            res["swap_llr"] = ref["future"] - _score(_tokenize(data, cot, nav_sw), data)["future"]
            if cmd in ("left", "right"):
                res["straight_llr"] = (
                    ref["future"] - _score(_tokenize(data, cot, "Continue straight"), data)["future"]
                )
            else:
                res["straight_llr"] = float("nan")

            # ---------- positive controls: must be large ----------
            rng = np.random.default_rng(args.seed + i)
            j = int(rng.choice([k for k in range(len(events)) if events[k]["clip_id"] != ev["clip_id"]]))
            other = load_physical_aiavdataset(events[j]["clip_id"], t0_us=events[j]["t0_us"], avdi=avdi)
            traj_swap = dict(data)
            traj_swap["ego_future_xyz"] = other["ego_future_xyz"]
            traj_swap["ego_future_rot"] = other["ego_future_rot"]
            res["wrong_traj_llr"] = ref["future"] - _score(td_nav, traj_swap)["future"]
            res["wrong_vision_llr"] = (
                ref["future"] - _score(_tokenize(data, cot, nav, vision_from=other), data)["future"]
            )

            rows.append({"clip_id": ev["clip_id"], "event_idx": ev["event_idx"], "split": ev["split"], **res})
            print(
                f"[nav-sanity] [{i + 1}/{len(use)}] {ev['clip_id'][:8]} {cmd:8s} "
                f"empty={res['empty_llr']:+.4f} swap={res['swap_llr']:+.4f} "
                f"straight={res['straight_llr']:+.4f} "
                f"wrong_vis={res['wrong_vision_llr']:+.4f} wrong_traj={res['wrong_traj_llr']:+.4f} "
                f"nulls=({res['null_identical']:+.1e},{res['history_llr']:+.1e},"
                f"{res['null_template']:+.1e}) ({(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )
        except Exception as e:
            torch.cuda.empty_cache()
            print(
                f"[nav-sanity] [{i + 1}/{len(use)}] {ev['clip_id'][:8]} FAILED: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    parent = os.path.dirname(args.out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\n[nav-sanity] {len(df)} events in {(time.time() - t0) / 60:.1f} min -> {args.out}\n")
    if not len(df):
        return

    print("=" * 84)
    print(f"{'control':20s} {'mean':>10s} {'std':>10s} {'n':>4s} {'expected':>12s}  verdict")
    print("=" * 84)
    checks = [
        ("null_identical", "zero", "== 0"),
        ("null_retokenize", "zero", "== 0"),
        ("history_llr", "zero", "== 0"),
        ("null_template", "zero", "== 0"),
        ("empty_llr", None, "the result"),
        ("swap_llr", None, "?"),
        ("straight_llr", None, "?"),
        ("wrong_vision_llr", "large", "> 0"),
        ("wrong_traj_llr", "big", ">> 0"),
    ]
    for col, kind, exp in checks:
        v = df[col].dropna()
        if not len(v):
            print(f"{col:20s} {'n/a':>10s}")
            continue
        if kind == "zero":
            ok = "PASS" if v.abs().max() < 1e-6 else f"FAIL (max |dev| {v.abs().max():.2e})"
        elif kind == "big":
            ok = "PASS" if v.mean() > 1.0 else f"SUSPECT -- probe may be insensitive ({v.mean():+.3f})"
        elif kind == "large":
            ok = "PASS" if v.mean() > 0.05 else f"SUSPECT ({v.mean():+.3f})"
        else:
            ok = ""
        print(f"{col:20s} {v.mean():+10.4f} {v.std():10.4f} {len(v):4d} {exp:>12s}  {ok}")
    print("=" * 84)
    if not df["null_template_ids_match"].all():
        print(
            "WARNING: r1 and r1_5-without-nav did NOT produce identical input_ids on "
            f"{(~df['null_template_ids_match']).sum()}/{len(df)} events -- the nav denominator is "
            "not the same sequence as the validated CoC path. Investigate before trusting numbers."
        )
    else:
        print("r1 vs r1_5-without-nav: input_ids identical on all events (plumbing verified).")

    print("\nBy command:")
    print(
        df.groupby("nav_command")[["empty_llr", "swap_llr", "straight_llr", "ref_future"]].agg(
            ["mean", "count"]
        )
    )
    print(
        "\nNOTE: swap_llr on straight rows is 0 BY CONSTRUCTION (swap_direction is a no-op on "
        '"Continue straight") -- read it as a null there, not as an effect.'
    )
    print(
        f"\nScale check: wrong VISION costs {df['wrong_vision_llr'].mean():.3f} nats and swapping "
        f"the TRAJECTORY costs {df['wrong_traj_llr'].mean():.3f}, while removing the NAV command "
        f"costs {df['empty_llr'].mean():.4f}. Reference log p(a*) per future token = "
        f"{df['ref_future'].mean():.3f} (CoC run's comparable figure: -1.932).",
        flush=True,
    )


if __name__ == "__main__":
    main()
