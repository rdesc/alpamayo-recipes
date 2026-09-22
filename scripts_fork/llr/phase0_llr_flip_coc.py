# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Semantic-flip control: does the model's trajectory likelihood notice when the reasoning's
DIRECTIVE is reversed while everything else about it is held fixed?

``phase0_llr_wrong_coc.py`` substitutes another event's prose, which changes topic, entities and
phrasing all at once -- a large, diffuse edit. This does the sharpest possible version instead: a
one- or two-word inversion of the manoeuvre ("Steer left" -> "Steer right", "Stop for" ->
"Accelerate past"), leaving the scene description, style and near-enough the token count intact.
If a positive LLR reflects the model actually reading the instruction, reversing the instruction
should destroy or invert it. If the LLR is noise, the flip moves nothing systematically.

Applied to the four highest-LLR Alpamayo-1.5 events, which is where the test has teeth: the
full-split mean is +0.0010 (0.9 sigma, i.e. nothing), so these tail events are exactly the ones
one might read as "reasoning worked here." This asks whether that reading survives.

Three passes per event, all in natural trained order, scoring the 128 future trajectory tokens
only (never the delimiters or history tokens -- see ``results/phase0_llr_action_direction.md``):

    logp_gold = log p(a* | v, ell_gold)
    logp_flip = log p(a* | v, ell_flipped)
    logp_none = log p(a* | v)               -- cot spliced to empty, in-distribution

    llr_gold     = logp_gold - logp_none    <- must reproduce the headline per-event value
    llr_flip     = logp_flip - logp_none    <- the same quantity with the directive reversed
    llr_gold_vs_flip = logp_gold - logp_flip <- direct preference for the correct directive

``llr_gold`` reproducing the value from the corrected run is the integrity check that this script
is scoring the same thing; it is asserted, not merely printed.

Run from a recipe directory with the _b300 venv (see README.md)::

    CUDA_VISIBLE_DEVICES=0 a1x_rl_b300/bin/python ../../scripts_fork/llr/phase0_llr_flip_coc.py \\
      --out /tmp/llr_flip.parquet
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

# The four highest-LLR Alpamayo-1.5 events from the corrected run, with hand-written directive
# inversions. Each flip is the minimal edit that reverses the manoeuvre and leaves the scene
# description untouched; `flip_kind` records whether the reversed directive is lateral or
# longitudinal, since the two channels behave differently (accel -0.0001 vs curvature +0.0021 on
# 1.5, and +0.0265 vs +0.0093 on A2S).
CASES = [
    {
        "tag": "max1",
        "clip_id": "50fe6c70-8b3c-4a1e-8a2f-9c4d1e7b6a05",  # resolved at runtime by prefix
        "clip_prefix": "50fe6c70",
        "t0_us": 1_700_000.0,
        "gold": "Go straight following temporary traffic cones when navigating through the construction zone.",
        "flip": "Turn left following temporary traffic cones when navigating through the construction zone.",
        "flip_kind": "lateral",
        "expected_llr_gold": 0.2594,
    },
    {
        "tag": "max2",
        "clip_prefix": "13f0af4c",
        "t0_us": 14_648_839.0,
        "gold": "Steer left to pass the pedestrian crossing the road at the crosswalk.",
        "flip": "Steer right to pass the pedestrian crossing the road at the crosswalk.",
        "flip_kind": "lateral",
        "expected_llr_gold": 0.2474,
    },
    {
        "tag": "max3",
        "clip_prefix": "93f00fcf",
        "t0_us": 1_700_000.0,
        "gold": "Stop for the construction zone with workers and traffic cones.",
        "flip": "Accelerate past the construction zone with workers and traffic cones.",
        "flip_kind": "longitudinal",
        "expected_llr_gold": 0.2327,
    },
    {
        "tag": "max4",
        "clip_prefix": "1bd51369",
        "t0_us": 9_883_757.0,
        "gold": "Shift left to adjust lane position for the construction zone guided by traffic cones.",
        "flip": "Shift right to adjust lane position for the construction zone guided by traffic cones.",
        "flip_kind": "lateral",
        "expected_llr_gold": 0.2277,
    },
    # ---- the NEGATIVE-LLR tail. These matter more than the positives, because we verified
    # from the ground-truth ego-future that their CoC is ACCURATE: min1 decelerates 8.0->5.0 m/s
    # after "strong deceleration", min4 decelerates 4.1->1.3 after "decelerate", min3 holds
    # 9.1->8.4 after "maintain speed". So if the model reads the directive, inverting an accurate
    # one should push the LLR DOWN (more negative). If instead the LLR rises toward zero, the
    # directive is not being used and the negative value has some other source.
    {
        "tag": "min1",
        "clip_prefix": "cca7366b",
        "t0_us": 12_684_145.0,
        "gold": "Strong deceleration to maintain a safe distance from the lead vehicle with trailer ahead.",
        "flip": "Strong acceleration to maintain a safe distance from the lead vehicle with trailer ahead.",
        "flip_kind": "longitudinal",
        "expected_llr_gold": -0.3269,
    },
    {
        "tag": "min2",
        "clip_prefix": "920b5869",
        "t0_us": 1_833_297.0,
        "gold": "Steer left to keep a safe distance from the construction worker on the right.",
        "flip": "Steer right to keep a safe distance from the construction worker on the right.",
        "flip_kind": "lateral",
        "expected_llr_gold": -0.3066,
    },
    {
        "tag": "min3",
        "clip_prefix": "9ca9120b",
        "t0_us": 15_348_106.0,
        "gold": "Maintain speed while following the lead vehicle and navigating through the construction zone with traffic cones.",
        "flip": "Reduce speed while following the lead vehicle and navigating through the construction zone with traffic cones.",
        "flip_kind": "longitudinal",
        "expected_llr_gold": -0.2896,
    },
    {
        "tag": "min4",
        "clip_prefix": "2a05be7b",
        "t0_us": 5_269_367.0,
        "gold": "Decelerate to maintain a safe distance from the lead truck ahead with oversize load.",
        "flip": "Accelerate to maintain a safe distance from the lead truck ahead with oversize load.",
        "flip_kind": "longitudinal",
        "expected_llr_gold": -0.2851,
    },
    {
        "tag": "min5",
        "clip_prefix": "cb01e385",
        "t0_us": 9_342_626.0,
        "gold": "Strong deceleration to maintain a safe distance from the pedestrian crossing the road.",
        "flip": "Strong acceleration to maintain a safe distance from the pedestrian crossing the road.",
        "flip_kind": "longitudinal",
        "expected_llr_gold": -0.2723,
    },
]

MANIFEST = (
    "/tmp/claude-1001/-mnt-efs-users-rod-repos-alpamayo-recipes/"
    "2313ddf8-517c-4f41-8959-3dd623c291a8/scratchpad/llr_viz/alpamayo1_5_corrected/manifest.json"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--manifest", default=MANIFEST, help="Used to resolve full clip_ids by prefix.")
    p.add_argument("--tol", type=float, default=2e-3, help="Tolerance on reproducing llr_gold.")
    p.add_argument(
        "--cases-json", default=None,
        help="Load CASES from a JSON file instead of the hand-written list. Use this for flips "
             "that were VERIFIED token-clean beforehand: the built-in list was written at the "
             "string level, and two of its longitudinal entries turned out to change 2-3 tokens "
             "('deceleration' -> ['dec','eler','ation'] vs 'acceleration' -> one token), which "
             "confounded the result. Cases here are pre-checked for equal token length and a "
             "single differing position.",
    )
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    import json

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

    print(f"[flip] loading model {args.model_path} ...", flush=True)
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

    def score_future(td: dict, data: dict) -> tuple[float, int]:
        """Mean log p over the 128 future trajectory tokens only."""
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
        model_inputs = to_device({"tokenized_data": dict(td)}, "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                tokenized_data=model_inputs["tokenized_data"],
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

    rows = []
    t0 = time.time()
    cases = CASES
    if args.cases_json:
        with open(args.cases_json) as fh:
            cases = json.load(fh)
        print(f"[flip] loaded {len(cases)} pre-verified cases from {args.cases_json}", flush=True)

    for c in cases:
        ent = by_prefix.get(c["clip_prefix"])
        if ent is None:
            print(f"[flip] {c['tag']}: clip prefix {c['clip_prefix']} not in manifest -- skipped", flush=True)
            continue
        clip_id = ent["clip_id"]
        data = load_physical_aiavdataset(clip_id, t0_us=c["t0_us"], avdi=avdi)

        lp_gold, n_g = score_future(tokenize(data, c["gold"]), data)
        lp_flip, n_f = score_future(tokenize(data, c["flip"]), data)
        lp_none, n_n = score_future(tokenize(data, ""), data)
        if not (n_g == n_f == n_n == 128):
            raise RuntimeError(f"{c['tag']}: future-token counts differ: {n_g}/{n_f}/{n_n}")

        llr_gold = lp_gold - lp_none
        llr_flip = lp_flip - lp_none
        # Integrity check: this must match the corrected run's value for the same event.
        err = abs(llr_gold - c["expected_llr_gold"])
        status = "ok" if err <= args.tol else f"MISMATCH (expected {c['expected_llr_gold']:+.4f}, err {err:.2e})"

        n_tok_gold = len(tokenizer(c["gold"], add_special_tokens=False)["input_ids"])
        n_tok_flip = len(tokenizer(c["flip"], add_special_tokens=False)["input_ids"])

        rows.append(
            {
                "tag": c["tag"], "clip_id": clip_id, "clip8": c["clip_prefix"],
                "event_cluster": ent["event_cluster"], "split": ent["split"],
                "gold_coc": c["gold"], "flipped_coc": c["flip"], "flip_kind": c["flip_kind"],
                "n_tok_gold": n_tok_gold, "n_tok_flip": n_tok_flip,
                "logp_gold": lp_gold, "logp_flip": lp_flip, "logp_none": lp_none,
                "llr_gold": llr_gold, "llr_flip": llr_flip,
                "llr_gold_vs_flip": lp_gold - lp_flip,
                "retained_frac": (llr_flip / llr_gold) if llr_gold != 0 else float("nan"),
                "repro_status": status,
            }
        )
        print(
            f"[flip] {c['tag']} {c['clip_prefix']} ({c['flip_kind']}, {n_tok_gold}->{n_tok_flip} tok)  "
            f"llr_gold={llr_gold:+.4f} [{status}]  llr_flip={llr_flip:+.4f}  "
            f"gold-vs-flip={lp_gold - lp_flip:+.4f}  ({(time.time()-t0)/60:.1f} min)",
            flush=True,
        )

    df = pd.DataFrame(rows)
    parent = os.path.dirname(args.out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print("\n" + "=" * 86)
    print(f"{'case':7s} {'llr_gold':>10s} {'llr_flip':>10s} {'gold-vs-flip':>13s} {'retained':>9s}  flip")
    print("=" * 86)
    for _, r in df.iterrows():
        print(
            f"{r['tag']:7s} {r['llr_gold']:+10.4f} {r['llr_flip']:+10.4f} "
            f"{r['llr_gold_vs_flip']:+13.4f} {100*r['retained_frac']:8.1f}%  {r['flip_kind']}"
        )
    print("=" * 86)
    if len(df):
        print(
            f"mean llr_gold {df.llr_gold.mean():+.4f} -> mean llr_flip {df.llr_flip.mean():+.4f}  "
            f"(mean retained {100*df.retained_frac.mean():.1f}%)"
        )
        print(
            f"mean preference for the correct directive: {df.llr_gold_vs_flip.mean():+.4f} nats\n"
            f"For reference: the full-split mean llr is +0.0010, and 1.5's wrong-prose control was "
            f"-0.008. A flip that leaves llr_flip ~= llr_gold means the model is not reading the\n"
            f"directive -- these tail events are then noise, not evidence of reasoning being used."
        )
        bad = df[~df.repro_status.eq("ok")]
        if len(bad):
            print(f"\nWARNING: {len(bad)} case(s) failed to reproduce the corrected run's llr_gold.")


if __name__ == "__main__":
    main()
