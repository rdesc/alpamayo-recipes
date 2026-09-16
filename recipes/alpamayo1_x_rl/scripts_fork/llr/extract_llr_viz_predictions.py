# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Behavioral-ablation layer (langforce_readme.md Sec 4.3) for the Phase-0 LLR visualization
artifact's 9 example events (alpamayo1_5_v2/manifest.json).

For each event, generates the model's actual predicted trajectory two ways:
  - with reasoning: normal generation, components_prompt=["cot","traj_future"] (cot generated
    autoregressively, then trajectory tokens) -- same call eval_baseline_ood_reasoning.py uses.
  - without reasoning: components_prompt=["traj_future"] only, mirroring alpamayo2's
    docs/pai_av_ood_eval.md `--no_coc` arm -- generation jumps straight to trajectory tokens,
    cot is never produced (not blanked after the fact; genuinely never generated), which is the
    cleanest ablation available for an autoregressive generation path (can't teacher-force a
    "blanked" span into free generation the way phase0_llr_action_direction.py does for scoring).

K=1 per condition (cheap illustration on 9 specific events, not a statistical run).

Overwrites each event's bev.png in place with GT history/future (existing 2 layers) plus the
two new predicted-trajectory overlays, and adds pred_coc_with_reasoning / pred_traj_with_reasoning
/ pred_coc_without_reasoning / pred_traj_without_reasoning fields to manifest.json (all other
fields preserved).
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path[:0] = ["../../../src", "../.."]
import physical_ai_av  # noqa: E402
from alpamayo_r1.helper import to_device  # noqa: E402
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset  # noqa: E402
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


def _cot_text(extra: dict, k: int) -> str:
    val = extra["cot"][0, 0, k]
    return str(val) if val is not None else ""


def render_bev(
    history_xy: np.ndarray,
    future_xy: np.ndarray,
    pred_with_xy: np.ndarray,
    pred_without_xy: np.ndarray,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=150)
    fig.patch.set_facecolor("#0a0d10")
    ax.set_facecolor("#0a0d10")

    ax.plot(history_xy[:, 0], history_xy[:, 1], color="#4f8fd6", lw=2.2, marker="o", ms=3, label="history")
    ax.plot(future_xy[:, 0], future_xy[:, 1], color="#4fc7c7", lw=2.2, marker="o", ms=3, label="future (GT)")
    ax.plot(
        pred_with_xy[:, 0], pred_with_xy[:, 1], color="#e8a23a", lw=2.0, ls="--",
        marker="^", ms=3, label="pred (with reasoning)", alpha=0.95,
    )
    ax.plot(
        pred_without_xy[:, 0], pred_without_xy[:, 1], color="#c86ee2", lw=2.0, ls="--",
        marker="v", ms=3, label="pred (no reasoning)", alpha=0.95,
    )
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


def main() -> None:
    torch.manual_seed(42)
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    print(f"[predict] loading model {MODEL_PATH} ...", flush=True)
    model = RLWrapperReasoningVLA.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16).cuda().eval()

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
    preprocess_without = get_preprocess_data_fn_from_model_config(
        components_order=["image", "traj_history", "prompt", "cot"],
        components_prompt=["traj_future"],
        label_components=[],
        generation_mode=True,
        include_camera_ids=True,
        include_frame_nums=True,
        model_config=model.config,
        chat_template_version="r1",
    )

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    def _predict(preprocess_fn, data) -> tuple[np.ndarray, str]:
        sample = {
            "image_frames": data["image_frames"],
            "camera_indices": data["camera_indices"],
            "relative_timestamps": data["relative_timestamps"],
            "ego_history_xyz": data["ego_history_xyz"].squeeze(0),
            "ego_history_rot": data["ego_history_rot"].squeeze(0),
        }
        sample["tokenized_data"] = preprocess_fn(sample)
        batch = collate_fn_from_model_config(
            [sample], model_config=model.config,
            include_camera_ids=True, include_frame_nums=True, chat_template_version="r1",
        )
        model_inputs = to_device(batch, "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, _, extra = model.sample_trajectories_from_data(
                data=model_inputs,
                temperature=0.6,
                top_p=0.98,
                num_traj_samples=1,
                max_generation_length=256,
                return_extra=True,
            )
        xy = pred_xyz[0, 0, 0, :, :2].float().cpu().numpy()
        coc = _cot_text(extra, 0)
        return xy, coc

    diffs = []
    for entry in manifest:
        clip_id, t0_us = entry["clip_id"], entry["t0_us"]
        subdir_name = f"{entry['label']}{entry['rank']}_{clip_id[:8]}"
        out_dir = entry.get("out_dir") or os.path.join(OUT_ROOT, subdir_name)

        data = load_physical_aiavdataset(clip_id, t0_us=t0_us, avdi=avdi)

        pred_with_xy, coc_with = _predict(preprocess_with, data)
        pred_without_xy, coc_without = _predict(preprocess_without, data)

        hist_xyz = data["ego_history_xyz"].squeeze(0).squeeze(0).numpy()
        fut_xyz = data["ego_future_xyz"].squeeze(0).squeeze(0).numpy()

        render_bev(
            hist_xyz[:, :2], fut_xyz[:, :2], pred_with_xy, pred_without_xy,
            os.path.join(out_dir, "bev.png"),
        )

        entry["pred_coc_with_reasoning"] = coc_with
        entry["pred_traj_with_reasoning"] = pred_with_xy.tolist()
        entry["pred_coc_without_reasoning"] = coc_without
        entry["pred_traj_without_reasoning"] = pred_without_xy.tolist()

        # rough divergence metric: mean displacement between the two predicted paths
        n = min(len(pred_with_xy), len(pred_without_xy))
        divergence = float(np.linalg.norm(pred_with_xy[:n] - pred_without_xy[:n], axis=-1).mean())
        entry["pred_with_without_divergence_m"] = divergence
        diffs.append((subdir_name, divergence))

        print(
            f"[predict] {subdir_name} llr_act={entry['llr_act']:.3f} "
            f"with/without divergence={divergence:.2f}m "
            f"coc_with={coc_with[:60]!r}",
            flush=True,
        )

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"[predict] wrote manifest -> {MANIFEST_PATH}")

    diffs.sort(key=lambda x: -x[1])
    print("[predict] divergence ranking (largest with/without gap first):")
    for name, d in diffs:
        print(f"  {name}: {d:.2f}m")


if __name__ == "__main__":
    main()
