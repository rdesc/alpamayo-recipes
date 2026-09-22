# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Sanity controls for the corrected LLR measurement -- is "reasoning buys ~0 nats" a FINDING, or
a broken probe that would report ~0 no matter what?

The corrected measurement (``phase0_llr_per_token.py --denominator empty``) says removing the
Chain-of-Causation moves log p(a*) by about -0.007 nats. A null result is only meaningful if the
instrument demonstrably responds to things it SHOULD respond to, and stays silent on things it
should not. This script runs both kinds of control on the same events and the same scoring path.

Controls that MUST come out at ~0 (if any is non-zero, the pipeline is misaligned):

  null_identical   re-tokenize with the SAME gold CoC and score twice. Any deviation from exactly
                   0.0 means the scoring path is non-deterministic.
  null_retokenize  numerator scored from a fresh tokenize/collate of the identical inputs -- lands
                   on the same sequence, so also exactly 0.0. Catches state leaking between passes
                   (the `forward` that pops "input_ids" from its dict is a real hazard here).
  history_tokens   LLR on the 48 history-trajectory tokens. These PRECEDE the cot span, so causal
                   attention makes any dependence impossible: exactly 0.0 is a structural
                   invariant. It is the sharpest alignment check available -- a one-token
                   position/shift error anywhere would break it immediately.

Controls that MUST come out LARGE (if any is ~0, the probe is dead and the headline null is
meaningless):

  no_vision        drop the camera frames. Alpamayo's trajectory prediction is overwhelmingly
                   vision+ego driven, so log p(a*) must fall hard. THE decisive positive control:
                   it proves the metric can detect a conditioning change at all, through the exact
                   same code path that reports ~0 for reasoning.
  wrong_traj       teacher-force ANOTHER event's ground-truth trajectory. The model should find a
                   trajectory inconsistent with this scene much less likely, so log p must fall.
                   Confirms we are scoring a scene-conditioned quantity, not echoing the tokenizer.
  wrong_coc        another event's real CoC prose, truncated to the same token length (in
                   distribution, position-preserved). Magnitude is the open question, not a
                   pass/fail -- it separates "content wrong" from "text absent".
  presence_on_donortraj
                   the presence effect measured on a WRONG target: log p(a'|v,gold) -
                   log p(a'|v,empty) for a donor trajectory a'. Asks whether a filled cot span
                   raises the likelihood of ANY trajectory (generic shift) or specifically of the
                   correct one (prediction improvement). Compare against empty_llr: a ratio near
                   1 means the presence term is a generic likelihood shift and carries no
                   information about prediction quality; near 0 means it is target-specific.
                   Distinct from wrong_traj, which varies the TARGET at fixed conditioning --
                   this varies the CONDITIONING at a fixed wrong target.

Read the output as a ladder: no_vision and wrong_traj set the scale at which this instrument
registers a real change in conditioning; the nulls set the noise floor; the CoC ablations are then
interpretable relative to both.

Same venv/CWD rules as the rest of this directory (``a1x_rl_b300``, run from a recipe dir).

Usage::

    CUDA_VISIBLE_DEVICES=0 a1x_rl_b300/bin/python ../../scripts_fork/llr/phase0_llr_sanity_controls.py \\
      --limit 6 --out /tmp/llr_sanity.parquet
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

DEFAULT_MODEL_PATH = "/mnt/efs/users/rod/ckpts/Alpamayo-1.5-10B-rl-training"
DEFAULT_PARQUET = "/opt/dlami/nvme/rod/datasets/pai_av_dataset/reasoning/ood_reasoning.parquet"
MIN_T0_US = 1_700_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--split", default="both", choices=["train", "val", "both"])
    p.add_argument("--limit", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--donor-traj-min-ade", type=float, default=5.0,
        help="Minimum ADE (metres over the 6.4 s horizon) between this event's GT future and the "
             "donor trajectory used by wrong_traj / presence_on_donortraj. Without this gate the "
             "donor is often a NEAR-DUPLICATE: ~92%% of this split is 'Continue straight', so a "
             "random donor frequently describes almost the same motion, the 'wrong' target is not "
             "wrong, and both arms are vacuous. A near-duplicate donor is the likely explanation "
             "for wrong_traj_llr's instability in earlier runs (std 0.71, one event at -0.21).",
    )
    p.add_argument(
        "--donor-traj-max-tries", type=int, default=5,
        help="Candidate donors to load before giving up on the ADE gate. On failure the "
             "highest-ADE candidate seen is used and donor_traj_gate_ok is recorded False, so "
             "those events can be excluded at analysis time rather than silently included.",
    )
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_events(parquet_path: str, split: str) -> list[dict]:
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


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

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

    print(f"[sanity] loading model {args.model_path} ...", flush=True)
    model = TrainableReasoningVLA.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).cuda().eval()

    preprocess_fn = get_preprocess_data_fn_from_model_config(
        components_order=["image", "traj_history", "prompt", "cot", "traj_future"],
        components_prompt=["cot", "traj_future"],
        label_components=["cot"],
        generation_mode=False,
        include_camera_ids=True,
        include_frame_nums=True,
        model_config=model.config,
        chat_template_version="r1",
    )
    processor = build_processor(
        vlm_name_or_path=model.config.vlm_name_or_path,
        traj_vocab_size=model.config.traj_vocab_size,
        min_pixels=model.config.min_pixels,
        max_pixels=model.config.max_pixels,
        include_camera_ids=True,
        include_frame_nums=True,
        chat_template_version="r1",
    )
    tokenizer = processor.tokenizer
    traj_lo = model.future_token_start_idx
    traj_hi = traj_lo + model.config.traj_vocab_size
    sid = model.special_token_ids

    events = load_events(args.parquet, args.split)
    donor_tok = [
        np.asarray(tokenizer(e["gold_coc"], add_special_tokens=False)["input_ids"], dtype=np.int64)
        for e in events
    ]
    use = events[: args.limit]
    print(f"[sanity] {len(use)} events; donor pool {len(events)}", flush=True)

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    def _tokenize(
        data: dict,
        cot_text: str,
        with_vision: bool = True,
        vision_from: dict | None = None,
        ego_from: dict | None = None,
    ) -> dict:
        sample = {
            "image_frames": data["image_frames"],
            "camera_indices": data["camera_indices"],
            "relative_timestamps": data["relative_timestamps"],
            "ego_history_xyz": data["ego_history_xyz"].squeeze(0),
            "ego_history_rot": data["ego_history_rot"].squeeze(0),
            "cot": cot_text,
        }
        if not with_vision:
            # Zero the pixels rather than removing the image component: that keeps the sequence
            # shape, the image placeholder tokens and every position identical, so the ONLY thing
            # that changes is the visual content -- the same discipline as content-blanking, and
            # here it is legitimate because we WANT a large response, not a clean null.
            sample["image_frames"] = torch.zeros_like(data["image_frames"])
        if vision_from is not None:
            # Actively MISLEADING vision rather than blank vision: a different scene's frames.
            # Blank pixels are merely uninformative (the model can fall back on ego history and
            # lose little), whereas another scene's frames assert something false. This is the
            # vision-side analogue of the wrong_coc control, and the fairer test of whether the
            # probe registers a change in visual conditioning.
            sample["image_frames"] = vision_from["image_frames"]
        if ego_from is not None:
            # Another event's ego history. Expected to be the LARGEST effect of all: the ego-only
            # MLP baseline already beats the fine-tuned model on K=1 ADE, so trajectory prediction
            # here is thought to be dominated by ego history rather than vision. If THIS is also
            # ~0 the probe is genuinely insensitive to context and the reasoning null means
            # nothing; if it is large, then "vision is worth 0.07 nats" is a fact about the model,
            # not an artifact of the instrument.
            sample["ego_history_xyz"] = ego_from["ego_history_xyz"].squeeze(0)
            sample["ego_history_rot"] = ego_from["ego_history_rot"].squeeze(0)
        sample["tokenized_data"] = preprocess_fn(sample)
        batch = collate_fn_from_model_config(
            [sample], model_config=model.config,
            include_camera_ids=True, include_frame_nums=True, chat_template_version="r1",
        )
        return batch["tokenized_data"]

    def _score(td: dict, traj_src: dict) -> dict:
        """Mean log p over future / history / delimiter token roles.

        NOTE on why there is no ego-history ablation here: the obvious one is confounded twice
        over. The action space derives its t0 velocity from ego history
        (`estimate_t0_states`), so handing `fuse_traj_tokens` a different history re-encodes the
        FUTURE TARGET tokens too -- that scores a different target sequence rather than ablating
        the context. Patching only the 48 history token ids in the fused tensor does not fix it
        either, because `forward` pops `input_ids` and re-fuses from its own `ego_history_*`
        arguments, so a locally edited tensor changes the labels while the model still sees the
        original history. `wrong_vision` is the clean context ablation: images never reach the
        trajectory tokenizer, so the targets are provably identical across the two passes.
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
        # Full-sequence next-token logprob, so non-trajectory tokens (e.g. the cot delimiters)
        # can be scored too. Used by the empty-CoC in-distribution check below.
        _all_lp = -F.cross_entropy(
            out.logits[0, :-1, :].float(), fused[0, 1:], reduction="none"
        )
        _score.last_all_logp = _all_lp.detach().cpu().numpy()
        _score.last_ids = fused[0].detach().cpu().numpy()

        m = traj_mask[:, 1:]
        shift_labels = fused[:, 1:][m].contiguous()
        shift_logits = out.logits[..., :-1, :][m].contiguous().float()
        logp = -F.cross_entropy(shift_logits, shift_labels, reduction="none")
        seq_pos = m[0].nonzero().flatten() + 1
        is_val = (shift_labels >= traj_lo) & (shift_labels < traj_hi)
        is_fut = is_val & (seq_pos > fsp)
        is_hist = is_val & ~is_fut
        return {
            "future": float(logp[is_fut].mean().item()),
            "hist": float(logp[is_hist].mean().item()) if bool(is_hist.any()) else float("nan"),
            "delim": float(logp[~is_val].mean().item()),
            "n_future": int(is_fut.sum().item()),
        }

    rows: list[dict] = []
    t0 = time.time()
    for i, ev in enumerate(use):
        try:
            data = load_physical_aiavdataset(ev["clip_id"], t0_us=ev["t0_us"], avdi=avdi)
            gold = ev["gold_coc"]

            # --- reference: p(a* | v, ell_gold) ---
            td_gold = _tokenize(data, gold)
            ref = _score(td_gold, data)

            n_cot = int(get_label_mask(td_gold["input_ids"], tokenizer, ["cot"]).sum().item())

            res = {"ref_future": ref["future"], "ref_delim": ref["delim"], "hist_logp": ref["hist"]}

            # ---------- nulls: must be 0 ----------
            res["null_identical"] = _score(_tokenize(data, gold), data)["future"] - ref["future"]
            res["null_retokenize"] = _score(_tokenize(data, gold), data)["future"] - ref["future"]
            # history tokens cannot depend on the cot span (they precede it) -> exact 0
            td_spl = _tokenize(data, "")
            res["history_llr"] = _score(td_spl, data)["hist"] - ref["hist"]

            # ---------- the measurement under test ----------
            res["empty_llr"] = ref["future"] - _score(td_spl, data)["future"]

            # ---------- IS AN EMPTY CoC IN DISTRIBUTION FOR THIS CHECKPOINT? ----------
            # If it is, the model should not be surprised when <|cot_start|> is followed
            # IMMEDIATELY by <|cot_end|>. If it is OOD, that close carries very low probability --
            # and an OOD denominator depresses log p for EVERY target, which would explain the
            # whole "presence term" as an artefact rather than a compute effect.
            #   cotend_lp_empty : log p(cot_end | ..., cot_start)      in the EMPTY sequence
            #   cotend_lp_gold  : log p(cot_end | ..., last prose tok) in the GOLD sequence
            #   firstprose_lp   : log p(first prose token | cot_start) in the GOLD sequence
            def _lp_of(token_id, after_id, ids, lp):
                pos = np.where(ids == token_id)[0]
                if len(pos) == 0:
                    return float("nan")
                k = int(pos[-1])
                if after_id is not None and (k == 0 or ids[k - 1] != after_id):
                    return float("nan")
                return float(lp[k - 1]) if k > 0 else float("nan")

            _score(td_spl, data)  # refresh cached arrays for the EMPTY sequence
            ids_e, lp_e = _score.last_ids, _score.last_all_logp
            res["cotend_lp_empty"] = _lp_of(sid["cot_end"], sid["cot_start"], ids_e, lp_e)
            _score(td_gold, data)  # refresh for the GOLD sequence
            ids_g, lp_g = _score.last_ids, _score.last_all_logp
            res["cotend_lp_gold"] = _lp_of(sid["cot_end"], None, ids_g, lp_g)
            _cs = np.where(ids_g == sid["cot_start"])[0]
            res["firstprose_lp_gold"] = float(lp_g[int(_cs[-1])]) if len(_cs) else float("nan")

            # ---------- positive controls: must be large ----------
            res["no_vision_llr"] = ref["future"] - _score(_tokenize(data, gold, with_vision=False), data)["future"]

            rng = np.random.default_rng(args.seed + i)
            cand = [k for k in range(len(events)) if events[k]["clip_id"] != ev["clip_id"]]

            def _ade(a, b):
                """Mean L2 over the horizon between two ego-future paths, in metres."""
                pa = a.detach().float().cpu().numpy().reshape(-1, a.shape[-1])[:, :2]
                pb = b.detach().float().cpu().numpy().reshape(-1, b.shape[-1])[:, :2]
                m = min(len(pa), len(pb))
                return float(np.linalg.norm(pa[:m] - pb[:m], axis=-1).mean())

            # Gate the donor on ADE: a random donor is often a near-duplicate (see
            # --donor-traj-min-ade), which makes the 'wrong' target not actually wrong.
            best = None
            for _try in range(max(1, args.donor_traj_max_tries)):
                jj = int(rng.choice(cand))
                oth = load_physical_aiavdataset(events[jj]["clip_id"], t0_us=events[jj]["t0_us"], avdi=avdi)
                a = _ade(data["ego_future_xyz"], oth["ego_future_xyz"])
                if best is None or a > best[0]:
                    best = (a, jj, oth)
                if a >= args.donor_traj_min_ade:
                    break
            donor_ade, j, other = best
            res["donor_traj_ade"] = donor_ade
            res["donor_traj_gate_ok"] = bool(donor_ade >= args.donor_traj_min_ade)
            # Swap ONLY the future trajectory being teacher-forced; context stays this scene's.
            traj_swap = dict(data)
            traj_swap["ego_future_xyz"] = other["ego_future_xyz"]
            traj_swap["ego_future_rot"] = other["ego_future_rot"]
            res["wrong_traj_llr"] = ref["future"] - _score(td_gold, traj_swap)["future"]

            # ---------- IS THE PRESENCE TERM TARGET-SPECIFIC? ----------
            # The presence term (empty_llr) says a filled cot span raises log p(a*_GT). Two very
            # different mechanisms give the same number:
            #   (a) a GENERIC likelihood shift -- the span raises log p of ANY trajectory, so it
            #       says nothing about prediction quality; or
            #   (b) a PREDICTION improvement -- the extra compute specifically favours the
            #       CORRECT trajectory.
            # Distinguish them by measuring the presence effect on a WRONG target: hold the donor
            # trajectory fixed and vary only the conditioning. This is NOT wrong_traj_llr, which
            # varies the target under fixed conditioning.
            #   ratio ~1  -> generic shift (a). The presence term is not about this trajectory.
            #   ratio ~0  -> target-specific (b).
            res["presence_on_donortraj"] = (
                _score(td_gold, traj_swap)["future"] - _score(td_spl, traj_swap)["future"]
            )

            # Misleading rather than blank vision -- the vision-side analogue of wrong_coc.
            res["wrong_vision_llr"] = (
                ref["future"] - _score(_tokenize(data, gold, vision_from=other), data)["future"]
            )

            # ---------- content control ----------
            dt = donor_tok[j]
            if len(dt) >= n_cot - 2:
                cot_mask = get_label_mask(td_gold["input_ids"], tokenizer, ["cot"]).clone()
                sp = cot_mask[0].nonzero().flatten()
                cot_mask[0, sp[0]] = False
                cot_mask[0, sp[-1]] = False  # keep the markers -- only swap the prose
                n_int = int(cot_mask.sum().item())
                ids_w = td_gold["input_ids"].clone()
                ids_w[cot_mask] = torch.from_numpy(dt[:n_int]).to(ids_w.dtype)
                td_w = dict(td_gold)
                td_w["input_ids"] = ids_w
                res["wrong_coc_llr"] = ref["future"] - _score(td_w, data)["future"]
            else:
                res["wrong_coc_llr"] = float("nan")

            rows.append({"clip_id": ev["clip_id"], "event_idx": ev["event_idx"], **res})
            print(
                f"[sanity] [{i + 1}/{len(use)}] {ev['clip_id'][:8]} "
                f"empty={res['empty_llr']:+.4f} no_vision={res['no_vision_llr']:+.4f} "
                f"wrong_traj={res['wrong_traj_llr']:+.4f} wrong_coc={res['wrong_coc_llr']:+.4f} "
                f"nulls=({res['null_identical']:+.1e},{res['history_llr']:+.1e}) "
                f"({(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )
        except Exception as e:
            torch.cuda.empty_cache()
            print(f"[sanity] [{i + 1}/{len(use)}] {ev['clip_id'][:8]} FAILED: {type(e).__name__}: {e}", flush=True)

    df = pd.DataFrame(rows)
    parent = os.path.dirname(args.out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print(f"\n[sanity] {len(df)} events in {(time.time() - t0) / 60:.1f} min -> {args.out}\n")
    if not len(df):
        return
    print("=" * 78)
    print(f"{'control':18s} {'mean':>10s} {'std':>10s} {'expected':>12s}  verdict")
    print("=" * 78)
    checks = [
        ("null_identical", 0.0, "== 0"),
        ("null_retokenize", 0.0, "== 0"),
        ("history_llr", 0.0, "== 0"),
        ("empty_llr", None, "the result"),
        ("firstprose_lp_gold", None, "gold CoC: first prose token"),
        ("cotend_lp_gold", None, "gold CoC: the trained close"),
        ("cotend_lp_empty", None, "empty CoC: how surprising is an immediate close?"),
        ("presence_on_donortraj", None, "presence on a WRONG target -- ~= empty_llr means generic"),
        ("wrong_coc_llr", None, "?"),
        ("no_vision_llr", None, "> 0"),
        ("wrong_vision_llr", None, "> 0"),
        ("wrong_traj_llr", None, ">> 0"),
    ]
    for col, want, exp in checks:
        v = df[col]
        if want == 0.0:
            ok = "PASS" if v.abs().max() < 1e-6 else f"FAIL (max |dev| {v.abs().max():.2e})"
        elif exp == ">> 0":
            ok = "PASS" if v.mean() > 1.0 else f"SUSPECT -- probe may be insensitive ({v.mean():+.3f})"
        else:
            ok = ""
        print(f"{col:18s} {v.mean():+10.4f} {v.std():10.4f} {exp:>12s}  {ok}")
    print("=" * 78)
    print(
        f"\nScale check: removing VISION costs {df['no_vision_llr'].mean():.3f} nats and swapping the\n"
        f"trajectory costs {df['wrong_traj_llr'].mean():.3f} nats, while removing REASONING costs\n"
        f"{df['empty_llr'].mean():.4f}. Reference log p(a*) per future token = {df['ref_future'].mean():.3f}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
