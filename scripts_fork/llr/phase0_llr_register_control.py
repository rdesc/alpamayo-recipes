# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Is the LLR about WHAT the reasoning says, or merely that plausible driving prose is present?

Three prior results set this up, all on the four highest-LLR Alpamayo-1.5 events:

  * inverting the directive ("steer left" -> "steer right") retains ~105% of the LLR, and the
    model slightly prefers the reversed instruction -- so the DIRECTION is not being used;
  * replacing the prose with nonsense ("Bananas telephone quantum sandwich violin.") destroys the
    whole LLR in 3 of 4 events -- so the span IS read, and read strongly;
  * knocking out the directive span destroys ~50% of the effect, but deleting just the word
    "left" costs 0.105 nats where FLIPPING it to "right" costs nothing.

Together those say the trajectory head responds to the span's presence and register rather than
its instruction. This script tests that directly, with the one control the others were missing:

    filler = fluent, on-register driving prose of MATCHED TOKEN LENGTH that is irrelevant to this
             particular scene -- no directive that applies, none of the scene's objects.

    llr_gold   = log p(a* | v, ell_gold)   - log p(a* | v)
    llr_filler = log p(a* | v, ell_filler) - log p(a* | v)

  * If llr_filler ~= llr_gold, the LLR is a register/length effect and carries no information
    about the scene at all -- the apparent "reasoning helps" is just "some plausible driving
    sentence occupies the span".
  * If llr_filler ~= 0 while llr_gold is large, the content genuinely matters (the model needs
    prose about THIS scene) even though the directive's direction does not.

That distinction decides how to read every positive LLR in this work, including Alpamayo 2
Super's non-zero full-split result, so it is worth its own script rather than a footnote.

Also re-run here for a like-for-like comparison on identical events, since prior numbers came
from different scripts: nonsense prose, and a length-matched substitution of ANOTHER event's real
gold CoC (real prose, real driving register, wrong scene -- the closest thing to filler that is
not hand-written).

Length matching is reported, not assumed: the token count of every variant is printed next to
gold's. A variant more than ~2 tokens off gold is flagged, because sequence length is itself a
candidate explanation (the empty-CoC denominator is always the shortest sequence).

Run from a recipe directory with the _b300 venv (see README.md)::

    CUDA_VISIBLE_DEVICES=0 a1x_rl_b300/bin/python \\
      ../../scripts_fork/llr/phase0_llr_register_control.py --out /tmp/llr_register.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd
import torch
import torch.nn.functional as F

DEFAULT_MODEL_PATH = "/mnt/efs/users/rod/ckpts/Alpamayo-1.5-10B-rl-training"
MANIFEST = (
    "/tmp/claude-1001/-mnt-efs-users-rod-repos-alpamayo-recipes/"
    "2313ddf8-517c-4f41-8959-3dd623c291a8/scratchpad/llr_viz/alpamayo1_5_corrected/manifest.json"
)
NONSENSE = "Bananas telephone quantum sandwich violin."

# `filler` sentences are hand-written to be (a) fluent, (b) unmistakably driving-register,
# (c) free of any object or manoeuvre belonging to the event's actual scene, and (d) close to
# gold's token count. `donor` is another event's real gold CoC, chosen from a different scenario
# cluster -- real prose rather than hand-written, as a second opinion on the same question.
CASES = [
    {
        "tag": "max1",
        "clip_prefix": "50fe6c70",
        "t0_us": 1_700_000.0,
        "gold": "Go straight following temporary traffic cones when navigating through the construction zone.",
        "filler": "Maintain a steady course along the marked lane while continuing past the parked delivery vehicles.",
        "donor": "Stop to yield to the upcoming vehicle at the roundabout.",
    },
    {
        "tag": "max2",
        "clip_prefix": "13f0af4c",
        "t0_us": 14_648_839.0,
        "gold": "Steer left to pass the pedestrian crossing the road at the crosswalk.",
        "filler": "Maintain a steady course along the marked lane behind the slower moving vehicle.",
        "donor": "Adapt speed for the road curvature while following the temporary lane ahead.",
    },
    {
        "tag": "max3",
        "clip_prefix": "93f00fcf",
        "t0_us": 1_700_000.0,
        "gold": "Stop for the construction zone with workers and traffic cones.",
        "filler": "Continue along the marked lane beside the parked delivery vehicles.",
        "donor": "Steer left to pass the pedestrian crossing at the crosswalk.",
    },
    {
        "tag": "max4",
        "clip_prefix": "1bd51369",
        "t0_us": 9_883_757.0,
        "gold": "Shift left to adjust lane position for the construction zone guided by traffic cones.",
        "filler": "Continue along the marked lane while maintaining a steady distance behind the slower vehicle.",
        "donor": "Stop to yield to the upcoming vehicle at the roundabout ahead.",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--manifest", default=MANIFEST)
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

    print(f"[reg] loading model {args.model_path} ...", flush=True)
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

    def tokenize(data, cot_text):
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

    def score_future(td, data):
        traj_data = {
            "ego_history_xyz": data["ego_history_xyz"].cuda(),
            "ego_history_rot": data["ego_history_rot"].cuda(),
            "ego_future_xyz": data["ego_future_xyz"].cuda(),
            "ego_future_rot": data["ego_future_rot"].cuda(),
        }
        fused = model.fuse_traj_tokens(td["input_ids"].clone().cuda(), traj_data)
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
        return float(logp[is_fut].mean().item()), int(td["input_ids"].shape[1])

    def ntok(s):
        return len(tokenizer(s, add_special_tokens=False)["input_ids"])

    rows = []
    t0 = time.time()
    for c in CASES:
        ent = by_prefix.get(c["clip_prefix"])
        if ent is None:
            print(f"[reg] {c['tag']}: {c['clip_prefix']} not in manifest -- skipped", flush=True)
            continue
        data = load_physical_aiavdataset(ent["clip_id"], t0_us=c["t0_us"], avdi=avdi)
        n_gold = ntok(c["gold"])

        lp_none, seqlen_none = score_future(tokenize(data, ""), data)
        variants = [
            ("gold", c["gold"]),
            ("filler", c["filler"]),
            ("donor", c["donor"]),
            ("nonsense", NONSENSE),
        ]
        vals = {}
        for name, text in variants:
            lp, seqlen = score_future(tokenize(data, text), data)
            vals[name] = lp
            n = ntok(text)
            rows.append(
                {
                    "tag": c["tag"], "clip_id": ent["clip_id"], "clip8": c["clip_prefix"],
                    "variant": name, "text": text,
                    "n_tokens": n, "n_tokens_gold": n_gold, "len_delta": n - n_gold,
                    "seq_len": seqlen, "seq_len_none": seqlen_none,
                    "logp": lp, "logp_none": lp_none, "llr": lp - lp_none,
                }
            )
        llr_gold = vals["gold"] - lp_none
        print(f"\n[reg] === {c['tag']} {c['clip_prefix']}  llr_gold={llr_gold:+.4f} ({n_gold} tok) ===", flush=True)
        for name, text in variants:
            llr = vals[name] - lp_none
            n = ntok(text)
            flag = "" if abs(n - n_gold) <= 2 else f"  [LENGTH {n-n_gold:+d} tok]"
            ret = 100 * llr / llr_gold if llr_gold else float("nan")
            print(f"[reg]   {name:9s} {n:3d} tok  llr {llr:+.4f}  retained {ret:6.1f}%{flag}", flush=True)
            print(f"[reg]             {text!r}", flush=True)

    df = pd.DataFrame(rows)
    parent = os.path.dirname(args.out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_parquet(args.out, index=False)

    print("\n" + "=" * 92)
    print("RETAINED FRACTION OF llr_gold, by variant (mean over events)")
    print("=" * 92)
    g = df[df.variant == "gold"].set_index("tag")["llr"]
    df["retained"] = df.apply(lambda r: r["llr"] / g[r["tag"]] if g[r["tag"]] else float("nan"), axis=1)
    summ = df.groupby("variant").agg(
        mean_llr=("llr", "mean"), mean_retained=("retained", "mean"),
        mean_len_delta=("len_delta", "mean"),
    )
    print((summ.assign(mean_retained=100 * summ.mean_retained)).round(4).to_string())
    print(
        "\nHow to read it:\n"
        "  filler ~= gold  -> the LLR is a register/length effect; the prose need not describe\n"
        "                     THIS scene, so a positive LLR is not evidence of scene-specific\n"
        "                     reasoning being used.\n"
        "  filler ~= 0     -> content about this scene genuinely matters, even though the\n"
        "                     directive's DIRECTION provably does not (flip test).\n"
        f"  nonsense is the floor for 'not plausible prose at all'."
    )
    print(f"\n[reg] {len(df)} rows in {(time.time()-t0)/60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
