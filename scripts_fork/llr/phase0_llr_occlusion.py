# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Which PART of the Chain-of-Causation is responsible for the LLR? Occlusion attribution.

We know the aggregate LLR (`log p(a*|v,ell) - log p(a*|v)`) and we know that inverting the
directive ("steer left" -> "steer right") barely changes it. This localizes the effect instead of
testing one hypothesis at a time: knock out one piece of the CoC at a time and see which knockout
destroys the LLR.

    llr_full          = log p(a* | v, ell)        - log p(a* | v)
    llr_occluded[s]   = log p(a* | v, ell \\ s)    - log p(a* | v)
    attribution[s]    = llr_full - llr_occluded[s]

A large positive `attribution[s]` means span `s` is carrying the effect: remove it and the LLR
collapses. Near-zero means the model was not using that span at all. The denominator
`log p(a*|v)` is the same empty-CoC pass for every row, so attributions are directly comparable
within an event.

THREE GRANULARITIES, because they answer different questions:

  token   leave-one-token-out across the whole cot span. The fine-grained map -- "does the word
          `left` matter more than the word `cones`".
  role    hand-labelled span knockouts separating the DIRECTIVE (the manoeuvre verb and its
          direction: "Steer left", "Stop") from the SCENE description (the objects and their
          arrangement: "the pedestrian crossing the road at the crosswalk", "construction zone
          guided by traffic cones"). This is the decisive one for the open question: does the
          model read WHAT TO DO, or merely WHAT IS THERE? Substituting another event's prose
          changes both at once, which is why that control could not separate them.
  all     the whole prose knocked out -- reproduces `llr_full` as a consistency check (its
          attribution must equal llr_full, since the occluded pass IS the empty-CoC pass).

TWO OCCLUSION MODES, run for every span, because neither is clean alone:

  pad     overwrite the span's tokens with the pad token, holding length and every downstream
          position fixed. Positionally exact, but pad tokens mid-prose are a sequence the model
          never saw in training. (Harmless here in a way it was NOT harmless for the retracted
          headline measurement: that one overwrote the <|cot_start|>/<|cot_end|> MARKERS, which
          injected ~23 nats. This only ever touches prose strictly between the markers -- with
          markers intact the delimiters moved by 4e-6 on Alpamayo 1.5.)
  delete  splice the span out and re-tokenize, so the remaining prose is genuine fluent text.
          In-distribution, but shortens the sequence and so shifts downstream positions slightly.

If the two modes agree on which spans matter, the attribution is trustworthy. Where they disagree,
the pad-token novelty is doing the work and the row should not be believed. Both are reported
side by side rather than averaged.

Run from a recipe directory with the _b300 venv (see README.md)::

    CUDA_VISIBLE_DEVICES=0 a1x_rl_b300/bin/python ../../scripts_fork/llr/phase0_llr_occlusion.py \\
      --out /tmp/llr_occlusion.parquet
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
MANIFEST = (
    "/tmp/claude-1001/-mnt-efs-users-rod-repos-alpamayo-recipes/"
    "2313ddf8-517c-4f41-8959-3dd623c291a8/scratchpad/llr_viz/alpamayo1_5_corrected/manifest.json"
)

# The four highest-LLR Alpamayo-1.5 events. `roles` gives hand-labelled substrings to knock out;
# each must appear verbatim in `gold` (asserted at runtime, and the selected token span is decoded
# back and printed so a mis-selection is visible rather than silent).
CASES = [
    {
        "tag": "max1",
        "clip_prefix": "50fe6c70",
        "t0_us": 1_700_000.0,
        "gold": "Go straight following temporary traffic cones when navigating through the construction zone.",
        "roles": {
            "directive": "Go straight",
            "scene": "temporary traffic cones",
            "scene_context": "the construction zone",
        },
    },
    {
        "tag": "max2",
        "clip_prefix": "13f0af4c",
        "t0_us": 14_648_839.0,
        "gold": "Steer left to pass the pedestrian crossing the road at the crosswalk.",
        "roles": {
            "directive": "Steer left",
            "direction_word": "left",
            "scene": "the pedestrian crossing the road",
            "scene_context": "at the crosswalk",
        },
    },
    {
        "tag": "max3",
        "clip_prefix": "93f00fcf",
        "t0_us": 1_700_000.0,
        "gold": "Stop for the construction zone with workers and traffic cones.",
        "roles": {
            "directive": "Stop",
            "scene": "the construction zone",
            "scene_context": "with workers and traffic cones",
        },
    },
    {
        "tag": "max4",
        "clip_prefix": "1bd51369",
        "t0_us": 9_883_757.0,
        "gold": "Shift left to adjust lane position for the construction zone guided by traffic cones.",
        "roles": {
            "directive": "Shift left",
            "direction_word": "left",
            "scene": "the construction zone guided by traffic cones",
            "scene_context": "to adjust lane position",
        },
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--manifest", default=MANIFEST)
    p.add_argument("--skip-tokens", action="store_true", help="Role/all knockouts only (faster).")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    sys.path[:0] = ["../../../src", "../.."]
    import physical_ai_av
    from alpamayo_r1.helper import to_device
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo.processor.qwen_processor import (
        collate_fn_from_model_config,
        get_preprocess_data_fn_from_model_config,
        build_processor,
    )

    from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA

    with open(args.manifest) as f:
        man = json.load(f)
    by_prefix = {e["clip_id"][:8]: e for e in man}

    print(f"[occl] loading model {args.model_path} ...", flush=True)
    model = TrainableReasoningVLA.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    ).cuda().eval()

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
    blank_id = tokenizer.pad_token_id
    if blank_id is None:
        blank_id = tokenizer.eos_token_id
    traj_lo = model.future_token_start_idx
    traj_hi = traj_lo + model.config.traj_vocab_size
    sid = model.special_token_ids

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    def tokenize(data: dict, cot_text: str) -> dict:
        sample = {
            "image_frames": data["image_frames"],
            "camera_indices": data["camera_indices"],
            "relative_timestamps": data["relative_timestamps"],
            "ego_history_xyz": data["ego_history_xyz"].squeeze(0),
            "ego_history_rot": data["ego_history_rot"].squeeze(0),
            "cot": cot_text,
        }
        sample["tokenized_data"] = preprocess_fn(sample)
        batch = collate_fn_from_model_config(
            [sample], model_config=model.config,
            include_camera_ids=True, include_frame_nums=True, chat_template_version="r1",
        )
        return batch["tokenized_data"]

    def cot_bounds(input_ids: torch.Tensor) -> tuple[int, int]:
        """(first, last_exclusive) positions of the prose strictly between the markers."""
        s = (input_ids[0] == sid["cot_start"]).nonzero().flatten()
        e = (input_ids[0] == sid["cot_end"]).nonzero().flatten()
        if s.numel() == 0 or e.numel() == 0:
            raise RuntimeError("cot markers not found")
        return int(s[0]) + 1, int(e[0])

    def score_future(td: dict, data: dict) -> tuple[float, int]:
        input_ids = td["input_ids"]
        traj_data = {
            "ego_history_xyz": data["ego_history_xyz"].cuda(),
            "ego_history_rot": data["ego_history_rot"].cuda(),
            "ego_future_xyz": data["ego_future_xyz"].cuda(),
            "ego_future_rot": data["ego_future_rot"].cuda(),
        }
        fused = model.fuse_traj_tokens(input_ids.clone().cuda(), traj_data)
        traj_mask = (
            ((fused >= traj_lo) & (fused < traj_hi))
            | (fused == sid["traj_future_start"])
            | (fused == sid["traj_future_end"])
        )
        fsp = int((fused[0] == sid["traj_future_start"]).nonzero().flatten()[-1].item())
        mi = to_device({"tokenized_data": dict(td)}, "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                tokenized_data=mi["tokenized_data"],
                ego_history_xyz=traj_data["ego_history_xyz"],
                ego_history_rot=traj_data["ego_history_rot"],
                ego_future_xyz=traj_data["ego_future_xyz"],
                ego_future_rot=traj_data["ego_future_rot"],
                labels_mask=None,
            )
        m = traj_mask[:, 1:]
        sl = fused[:, 1:][m].contiguous()
        lg = out.logits[..., :-1, :][m].contiguous().float()
        logp = -F.cross_entropy(lg, sl, reduction="none")
        seq_pos = m[0].nonzero().flatten() + 1
        is_fut = (sl >= traj_lo) & (sl < traj_hi) & (seq_pos > fsp)
        return float(logp[is_fut].mean().item()), int(is_fut.sum().item())

    def span_token_range(gold: str, phrase: str, lo: int, ids: torch.Tensor) -> tuple[int, int]:
        """Map a character substring of `gold` to token positions inside the cot span.

        Uses prefix re-tokenization: the token count of gold[:start] gives the offset. Verified by
        decoding the resulting range and comparing against the phrase, so a bad mapping surfaces.
        """
        ci = gold.index(phrase)
        n_before = len(tokenizer(gold[:ci], add_special_tokens=False)["input_ids"])
        n_through = len(tokenizer(gold[: ci + len(phrase)], add_special_tokens=False)["input_ids"])
        return lo + n_before, lo + n_through

    rows: list[dict] = []
    t0 = time.time()
    for c in CASES:
        ent = by_prefix.get(c["clip_prefix"])
        if ent is None:
            print(f"[occl] {c['tag']}: {c['clip_prefix']} not in manifest -- skipped", flush=True)
            continue
        gold = c["gold"]
        for phrase in c["roles"].values():
            assert phrase in gold, f"{c['tag']}: role phrase {phrase!r} not in gold CoC"

        data = load_physical_aiavdataset(ent["clip_id"], t0_us=c["t0_us"], avdi=avdi)

        td_gold = tokenize(data, gold)
        ids_gold = td_gold["input_ids"]
        lo, hi = cot_bounds(ids_gold)
        n_prose = hi - lo
        lp_full, n_f = score_future(td_gold, data)
        lp_none, n_n = score_future(tokenize(data, ""), data)
        if n_f != n_n:
            raise RuntimeError(f"{c['tag']}: future-token count differs between passes")
        llr_full = lp_full - lp_none

        tok_strs = [tokenizer.convert_ids_to_tokens(int(t)) for t in ids_gold[0, lo:hi]]
        print(
            f"\n[occl] === {c['tag']} {c['clip_prefix']}  llr_full={llr_full:+.4f}  "
            f"prose={n_prose} tokens ===\n[occl] {gold}\n[occl] tokens: {tok_strs}",
            flush=True,
        )

        def emit(kind, label, lp, mode, n_occ):
            llr = lp - lp_none
            rows.append(
                {
                    "tag": c["tag"], "clip_id": ent["clip_id"], "clip8": c["clip_prefix"],
                    "gold_coc": gold, "occl_kind": kind, "occl_label": label, "mode": mode,
                    "n_occluded_tokens": n_occ, "n_prose_tokens": n_prose,
                    "logp": lp, "logp_none": lp_none,
                    "llr_full": llr_full, "llr_occluded": llr,
                    "attribution": llr_full - llr,
                    "frac_of_effect": (llr_full - llr) / llr_full if llr_full else float("nan"),
                }
            )
            return llr

        # ---------- role knockouts ----------
        for role, phrase in c["roles"].items():
            a, b = span_token_range(gold, phrase, lo, ids_gold)
            decoded = tokenizer.decode(ids_gold[0, a:b]).strip()
            # pad mode
            ids_p = ids_gold.clone()
            ids_p[0, a:b] = blank_id
            td_p = dict(td_gold); td_p["input_ids"] = ids_p
            lp_p, _ = score_future(td_p, data)
            llr_p = emit("role", role, lp_p, "pad", b - a)
            # delete mode
            deleted = (gold[: gold.index(phrase)] + gold[gold.index(phrase) + len(phrase):])
            deleted = " ".join(deleted.split())
            lp_d, _ = score_future(tokenize(data, deleted), data)
            llr_d = emit("role", role, lp_d, "delete", b - a)
            print(
                f"[occl]   role {role:14s} [{decoded!r:52s}] "
                f"pad: llr {llr_p:+.4f} attr {llr_full-llr_p:+.4f} | "
                f"del: llr {llr_d:+.4f} attr {llr_full-llr_d:+.4f}",
                flush=True,
            )

        # ---------- whole-prose knockout (consistency: attribution must equal llr_full) ----------
        ids_a = ids_gold.clone()
        ids_a[0, lo:hi] = blank_id
        td_a = dict(td_gold); td_a["input_ids"] = ids_a
        lp_a, _ = score_future(td_a, data)
        emit("all", "whole prose", lp_a, "pad", n_prose)
        emit("all", "whole prose", lp_none, "delete", n_prose)
        print(
            f"[occl]   ALL prose      pad: llr {lp_a-lp_none:+.4f} | "
            f"del: llr +0.0000 attr {llr_full:+.4f} (must equal llr_full {llr_full:+.4f})",
            flush=True,
        )

        # ---------- leave-one-token-out ----------
        if not args.skip_tokens:
            for k in range(n_prose):
                ids_k = ids_gold.clone()
                ids_k[0, lo + k] = blank_id
                td_k = dict(td_gold); td_k["input_ids"] = ids_k
                lp_k, _ = score_future(td_k, data)
                emit("token", tok_strs[k], lp_k, "pad", 1)
            sub = [r for r in rows if r["tag"] == c["tag"] and r["occl_kind"] == "token"]
            top = sorted(sub, key=lambda r: -abs(r["attribution"]))[:5]
            print(
                "[occl]   top tokens by |attribution|: "
                + ", ".join(f"{r['occl_label']}={r['attribution']:+.4f}" for r in top),
                flush=True,
            )

    df = pd.DataFrame(rows)
    parent = os.path.dirname(args.out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\n[occl] {len(df)} rows in {(time.time()-t0)/60:.1f} min -> {args.out}", flush=True)

    if len(df):
        print("\n" + "=" * 96)
        print("ROLE ATTRIBUTION -- fraction of each event's LLR destroyed by knocking out that span")
        print("=" * 96)
        roles = df[df.occl_kind == "role"]
        piv = roles.pivot_table(index="occl_label", columns="mode", values="frac_of_effect", aggfunc="mean")
        print((100 * piv).round(1).to_string())
        print("\n(100% = knocking this span out destroys the whole effect; ~0% = the model was not using it)")
        agree = roles.pivot_table(index=["tag", "occl_label"], columns="mode", values="attribution")
        if {"pad", "delete"}.issubset(agree.columns):
            d = (agree["pad"] - agree["delete"]).abs()
            print(f"\npad-vs-delete agreement: mean |diff| {d.mean():.4f} nats, max {d.max():.4f}")
            print("  (small => the attribution is not an artifact of pad-token novelty)")


if __name__ == "__main__":
    main()
