# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Fix for extract_llr_viz_predictions.py's without-reasoning arm.

The original script generated the "no reasoning" condition by dropping `cot` from
`components_prompt` entirely (structural omission), which put the model in a sequence
shape it never trained on -- symptomatic "Invalid token ids found" warnings and one
event (max1) predicting a wrong-direction 2.70->14.38 m/s acceleration where GT/the
with-reasoning prediction both decelerate to 0.36 m/s.

This script instead replicates the validated `coc_text=""` mechanism from
~/repos/alpamayo1.5/src/alpamayo1_5/models/alpamayo1_5.py
(Alpamayo1_5.sample_trajectories_from_data_with_vlm_rollout, used for real by
eval_pai_av_val.py --no_coc): build the prompt normally up to <|cot_start|>, then
directly splice `[cot_end_id, traj_future_start_id]` onto input_ids (genuinely empty
CoC content, but the SAME structural position the model was trained to see a CoC span
end at) and generate only the trajectory span from there.
RLWrapperReasoningVLA.sample_trajectories_from_data has no coc_text kwarg, so this
replicates its generation call manually with the spliced input_ids.

Only the without-reasoning trajectory is regenerated; the existing (validated)
with-reasoning trajectory is reused verbatim from manifest.json.
"""

from __future__ import annotations

import json
import os
import sys

import einops
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path[:0] = ["../../../src", "../.."]
import physical_ai_av  # noqa: E402
from alpamayo_r1.helper import to_device  # noqa: E402
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
from alpamayo_r1.models.token_utils import extract_traj_tokens  # noqa: E402
from alpamayo.processor.qwen_processor import (  # noqa: E402
    collate_fn_from_model_config,
    get_preprocess_data_fn_from_model_config,
)
from alpamayo1_x_rl.models.reasoning_vla.base_model import RLWrapperReasoningVLA  # noqa: E402

MODEL_PATH = "/mnt/efs/users/rod/ckpts/Alpamayo-1.5-10B-rl-training"
OUT_ROOT = (
    "/tmp/claude-1001/-mnt-efs-users-rod-repos-alpamayo-recipes/"
    "2313ddf8-517c-4f41-8959-3dd623c291a8/scratchpad/llr_viz/alpamayo1_5_v2"
)
MANIFEST_PATH = os.path.join(OUT_ROOT, "manifest.json")
MIN_LATERAL_SPAN_M = 16.0
DT = 0.1


def speed_from_xy(xy: np.ndarray, dt: float = DT) -> np.ndarray:
    n = len(xy)
    v = np.zeros(n)
    if n >= 2:
        v[0] = np.linalg.norm(xy[1] - xy[0]) / dt
        v[-1] = np.linalg.norm(xy[-1] - xy[-2]) / dt
        for i in range(1, n - 1):
            v[i] = np.linalg.norm(xy[i + 1] - xy[i - 1]) / (2 * dt)
    return v


def render_bev(history_xy, future_xy, pred_with_xy, pred_without_xy, out_path):
    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=150)
    fig.patch.set_facecolor("#0a0d10")
    ax.set_facecolor("#0a0d10")
    ax.plot(history_xy[:, 0], history_xy[:, 1], color="#4f8fd6", lw=2.2, marker="o", ms=3, label="history")
    ax.plot(future_xy[:, 0], future_xy[:, 1], color="#4fc7c7", lw=2.2, marker="o", ms=3, label="future (GT)")
    ax.plot(pred_with_xy[:, 0], pred_with_xy[:, 1], color="#e8a23a", lw=2.0, ls="--",
            marker="^", ms=3, label="pred (with reasoning)", alpha=0.95)
    ax.plot(pred_without_xy[:, 0], pred_without_xy[:, 1], color="#c86ee2", lw=2.0, ls="--",
            marker="v", ms=3, label="pred (no reasoning)", alpha=0.95)
    ax.scatter([0], [0], color="#e7ecf1", marker="*", s=140, zorder=5, label="ego @ t0")

    all_xy = np.concatenate([history_xy, future_xy, pred_with_xy, pred_without_xy], axis=0)
    x_min, x_max = all_xy[:, 0].min(), all_xy[:, 0].max()
    y_min, y_max = all_xy[:, 1].min(), all_xy[:, 1].max()
    y_center = (y_min + y_max) / 2.0
    y_span = max(y_max - y_min, MIN_LATERAL_SPAN_M)
    ax.set_ylim(y_center - y_span / 2.0, y_center + y_span / 2.0)
    x_pad = max((x_max - x_min) * 0.08, 2.0)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("forward (m)", color="#9aa7b4", fontsize=9)
    ax.set_ylabel("lateral (m)", color="#9aa7b4", fontsize=9)
    ax.tick_params(colors="#64707c", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#262f3a")
    ax.grid(True, color="#1a222b", lw=0.6)
    ax.legend(loc="upper left", fontsize=6.5, facecolor="#131920", edgecolor="#262f3a", labelcolor="#e7ecf1")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_velocity(t_hist, v_hist, t_fut, v_fut, t_pred_w, v_pred_w, t_pred_wo, v_pred_wo, out_path):
    fig, ax = plt.subplots(figsize=(4.6, 2.6), dpi=150)
    fig.patch.set_facecolor("#0a0d10")
    ax.set_facecolor("#0a0d10")
    ax.plot(t_hist, v_hist, color="#4f8fd6", lw=2.0, marker="o", ms=2.5, label="history")
    ax.plot(t_fut, v_fut, color="#4fc7c7", lw=2.0, marker="o", ms=2.5, label="future (GT)")
    ax.plot(t_pred_w, v_pred_w, color="#e8a23a", lw=1.8, ls="--", marker="^", ms=2.5,
            label="pred (with reasoning)", alpha=0.95)
    ax.plot(t_pred_wo, v_pred_wo, color="#c86ee2", lw=1.8, ls="--", marker="v", ms=2.5,
            label="pred (no reasoning)", alpha=0.95)
    ax.axvline(0, color="#e7ecf1", lw=1.0, alpha=0.5, zorder=1)
    ax.set_xlabel("time (s)  [0 = ego @ t0]", color="#9aa7b4", fontsize=9)
    ax.set_ylabel("speed (m/s)", color="#9aa7b4", fontsize=9)
    ax.tick_params(colors="#64707c", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#262f3a")
    ax.grid(True, color="#1a222b", lw=0.6)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=6, facecolor="#131920", edgecolor="#262f3a", labelcolor="#e7ecf1", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    torch.manual_seed(42)
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    print(f"[fix-nococ] loading model {MODEL_PATH} ...", flush=True)
    model = RLWrapperReasoningVLA.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).cuda().eval()

    cot_end_id = model.special_token_ids["cot_end"]
    traj_future_start_id = model.special_token_ids["traj_future_start"]
    print(f"[fix-nococ] cot_end_id={cot_end_id} traj_future_start_id={traj_future_start_id}", flush=True)

    # Same prompt construction as the with-reasoning arm (ends right before CoC content,
    # at <|cot_start|>) -- this is the shape the model was actually trained on.
    preprocess_with = get_preprocess_data_fn_from_model_config(
        components_order=["image", "traj_history", "prompt", "cot"],
        components_prompt=["cot", "traj_future"],
        label_components=[],
        generation_mode=True,
        include_camera_ids=True,
        include_frame_nums=True,
        model_config=model.config,
        chat_template_version="r1",
    )

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    def _predict_no_coc(data) -> np.ndarray:
        sample = {
            "image_frames": data["image_frames"],
            "camera_indices": data["camera_indices"],
            "relative_timestamps": data["relative_timestamps"],
            "ego_history_xyz": data["ego_history_xyz"].squeeze(0),
            "ego_history_rot": data["ego_history_rot"].squeeze(0),
        }
        sample["tokenized_data"] = preprocess_with(sample)
        batch = collate_fn_from_model_config(
            [sample], model_config=model.config,
            include_camera_ids=True, include_frame_nums=True, chat_template_version="r1",
        )
        model_inputs = to_device(batch, "cuda")

        ego_history_xyz = model_inputs["ego_history_xyz"]
        ego_history_rot = model_inputs["ego_history_rot"]
        tokenized_data = dict(model_inputs["tokenized_data"])
        input_ids = tokenized_data.pop("input_ids")
        traj_data = {"ego_history_xyz": ego_history_xyz, "ego_history_rot": ego_history_rot}
        input_ids = model.fuse_traj_tokens(input_ids, traj_data)

        # coc_text="" -> empty coc_ids -> splice straight to [cot_end, traj_future_start]
        B = input_ids.shape[0]
        suffix = torch.tensor(
            [cot_end_id, traj_future_start_id], dtype=input_ids.dtype, device=input_ids.device,
        ).unsqueeze(0).expand(B, -1)
        input_ids = torch.cat([input_ids, suffix], dim=1)
        if "attention_mask" in tokenized_data:
            ones = torch.ones(
                B, suffix.shape[1], dtype=tokenized_data["attention_mask"].dtype, device=input_ids.device,
            )
            tokenized_data["attention_mask"] = torch.cat([tokenized_data["attention_mask"], ones], dim=1)

        generation_config = model.vlm.generation_config
        generation_config.top_p = 0.98
        generation_config.temperature = 0.6
        generation_config.do_sample = True
        generation_config.num_return_sequences = 1
        generation_config.max_new_tokens = model.config.tokens_per_future_traj
        generation_config.output_logits = False
        generation_config.return_dict_in_generate = True
        generation_config.top_k = None
        generation_config.pad_token_id = model.tokenizer.pad_token_id

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = model.vlm.generate(
                input_ids=input_ids, **tokenized_data, generation_config=generation_config,
            )
        generated_tokens = generated.sequences[:, input_ids.shape[1]:]

        traj_token_ids = extract_traj_tokens(
            generated_tokens, model.special_token_ids, model.config.tokens_per_future_traj,
            model.future_token_start_idx, model.traj_tokenizer.vocab_size,
        )
        pred_xyz, pred_rot, _ = model.traj_tokenizer.decode(
            hist_xyz=einops.repeat(ego_history_xyz[:, -1], "b ... -> (b n) ...", n=1),
            hist_rot=einops.repeat(ego_history_rot[:, -1], "b ... -> (b n) ...", n=1),
            tokens=traj_token_ids,
        )
        return pred_xyz[0, :, :2].float().cpu().numpy()

    diffs = []
    for entry in manifest:
        clip_id, t0_us = entry["clip_id"], entry["t0_us"]
        out_dir = entry["out_dir"]

        data = load_physical_aiavdataset(clip_id, t0_us=t0_us, avdi=avdi)
        pred_without_xy = _predict_no_coc(data)

        pred_with_xy = np.array(entry["pred_traj_with_reasoning"])
        hist_xyz = data["ego_history_xyz"].squeeze(0).squeeze(0).numpy()
        fut_xyz = data["ego_future_xyz"].squeeze(0).squeeze(0).numpy()

        old_div = entry.get("pred_with_without_divergence_m")
        n = min(len(pred_with_xy), len(pred_without_xy))
        new_div = float(np.linalg.norm(pred_with_xy[:n] - pred_without_xy[:n], axis=-1).mean())

        entry["pred_coc_without_reasoning"] = ""
        entry["pred_traj_without_reasoning"] = pred_without_xy.tolist()
        entry["pred_with_without_divergence_m"] = new_div

        render_bev(
            hist_xyz[:, :2], fut_xyz[:, :2], pred_with_xy, pred_without_xy,
            os.path.join(out_dir, "bev.png"),
        )

        v_hist = speed_from_xy(hist_xyz[:, :2])
        v_fut = speed_from_xy(fut_xyz[:, :2])
        v_pred_w = speed_from_xy(pred_with_xy)
        v_pred_wo = speed_from_xy(pred_without_xy)
        n_hist = len(hist_xyz)
        t_hist = (np.arange(n_hist) - (n_hist - 1)) * DT
        t_fut = (np.arange(len(fut_xyz)) + 1) * DT
        t_pred_w = (np.arange(len(pred_with_xy)) + 1) * DT
        t_pred_wo = (np.arange(len(pred_without_xy)) + 1) * DT
        render_velocity(
            t_hist, v_hist, t_fut, v_fut, t_pred_w, v_pred_w, t_pred_wo, v_pred_wo,
            os.path.join(out_dir, "velocity.png"),
        )

        name = f"{entry['label']}{entry['rank']}"
        diffs.append((name, old_div, new_div))
        print(
            f"[fix-nococ] {name} llr_act={entry['llr_act']:.3f} "
            f"divergence old={old_div:.2f}m -> new={new_div:.2f}m "
            f"v_pred_wo(start/end)={v_pred_wo[0]:.2f}/{v_pred_wo[-1]:.2f} "
            f"v_fut(start/end)={v_fut[0]:.2f}/{v_fut[-1]:.2f}",
            flush=True,
        )

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[fix-nococ] wrote manifest -> {MANIFEST_PATH}")

    print("[fix-nococ] divergence comparison (old -> new):")
    for name, old_d, new_d in diffs:
        print(f"  {name}: {old_d:.2f}m -> {new_d:.2f}m")


if __name__ == "__main__":
    main()
