"""HF Trainer callback that logs trajectory-prediction visualizations to W&B.

Adapted from alpamayo1_5_sft/callbacks/viz_callback.py, but the prediction is
sourced differently. Alpamayo 2 Super's release
``sample_trajectories_from_data`` always decodes the future trajectory through
the frozen diffusion expert (``model.expert``), never through the VLM's own
discrete trajectory tokens -- see ``Alpamayo2Super.sample_trajectories_from_data``,
which masks discrete-trajectory-token generation out entirely and stops
autoregressive generation right at ``<|traj_future_start|>``. Since this LoRA
recipe only trains the VLM-side discrete-trajectory-token loss (the expert is
untouched), plugging that release sampler in here would render a curve that
never changes with training -- a misleading "viz".

Instead, this callback uses ``rollout.py``'s TRUE autoregressive rollout:
``model.vlm.generate()`` over the (unmasked) discrete trajectory-token span,
each token conditioned on the model's own previously generated tokens, then
decoded to xyz with ``model.future_traj_tokenizer.decode(...)``. This directly
reflects what the discrete-token CE loss is training, without the
teacher-forcing optimism an earlier version of this callback had (predicting
each token from *ground-truth* previous tokens hides compounding errors --
verified empirically: teacher-forced ADE was ~0.01m on a held-out sample vs.
0.8-1.9m for the true rollout on the same sample).

Also plots the frozen future-trajectory-tokenizer's own encode->decode
reconstruction of the GT future (same "ceiling" concept as alpamayo1_5_sft's
viz: the best the discrete token space can represent, regardless of model
quality).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: training process has no display

import torch
from transformers import TrainerCallback

import alpamayo.common.constants as constants
from alpamayo.visualization.viz import visualize_data

from alpamayo2_sft.rollout import build_generation_prompt, rollout_future_trajectory

logger = logging.getLogger(__name__)

# A window whose GT travels less than this over the whole future horizon is
# labelled STANDSTILL in the figure header/caption. Display only -- it does
# not change any target. Generous vs. the dataset's standstill snap (which
# zeroes the GT outright) so near-parked windows are flagged too. Matches
# alpamayo1_5_sft/callbacks/viz_callback.py's constant of the same name.
_STANDSTILL_SPAN_M = 1.0


def decode_gt_tokenizer_reconstruction(
    model: Any,
    ego_history_xyz: torch.Tensor,
    ego_history_rot: torch.Tensor,
    ego_future_xyz: torch.Tensor,
    ego_future_rot: torch.Tensor,
) -> torch.Tensor:
    """Encode->decode the GT future through the frozen tokenizer (the "ceiling")."""
    hist_xyz = ego_history_xyz.flatten(0, 1)
    hist_rot = ego_history_rot.flatten(0, 1)
    fut_xyz = ego_future_xyz.flatten(0, 1)
    fut_rot = ego_future_rot.flatten(0, 1)
    tokens = model.future_traj_tokenizer.encode(
        hist_xyz=hist_xyz, hist_rot=hist_rot, fut_xyz=fut_xyz, fut_rot=fut_rot
    )
    rec_xyz, _rec_rot, _ = model.future_traj_tokenizer.decode(
        hist_xyz=hist_xyz, hist_rot=hist_rot, tokens=tokens
    )
    return rec_xyz


def _gt_span_m(ego_future_xyz: torch.Tensor) -> float:
    """Ground-truth XY distance from t0 to the last future waypoint (metres)."""
    return float(ego_future_xyz.reshape(-1, 3)[-1, :2].norm())


def _grid_labels(data: dict[str, Any]) -> tuple[list[str] | None, list[str] | None]:
    """Per-column camera names + per-row timestep labels for the image grid.

    Cameras are in the sample's (index-sorted) order; timesteps are re-based
    so the last frame (the t0 prediction anchor) reads ``+0.0s``. Matches
    alpamayo1_5_sft/callbacks/viz_callback.py's ``_grid_labels``.
    """
    cam_labels = None
    cam_idx = data.get("camera_indices")
    if cam_idx is not None:
        cam_labels = [
            constants.CAMERA_INDICES_TO_DISPLAY_NAMES.get(int(i), f"cam{int(i)}") for i in cam_idx
        ]
    frame_labels = None
    rel = data.get("relative_timestamps")  # (N_cam, num_frames), seconds
    if rel is not None and rel.numel():
        per_frame = rel.float().mean(dim=0)  # ~equal across cameras
        per_frame = per_frame - per_frame.max()  # end at t0 = +0.0s
        frame_labels = [f"{float(t):+.1f}s" for t in per_frame]
    return cam_labels, frame_labels


class TrajectoryVizCallback(TrainerCallback):
    """Render GT-vs-rollout trajectory figures and log them to W&B.

    Args:
        eval_dataset: An Alpamayo2TrajectoryDataset. Only its ``load_raw(idx)``
            and ``.processor`` are used here (the rollout builds its own
            generation-mode prompt from the raw clip data -- it does not reuse
            the dataset's teacher-forced ``__getitem__`` tokenization).
        collate_fn: Unused by rollout rendering; kept for interface parity /
            future batched rollout support.
        output_dir: Directory to drop rendered PNGs into (a ``viz/`` subdir).
        every_n_steps: Render cadence in optimizer steps.
        sample_indices: Which eval-dataset indices to render. Defaults to the
            first ``num_samples`` samples.
        num_samples: Number of samples to render if ``sample_indices`` is None.
        num_traj_samples: Independent rollouts sampled per input (each plotted).
        top_p, temperature: Nucleus sampling params for the rollout.
        do_sample: False for greedy rollout decoding.
        on_train_begin: If True, also render once before training (step 0
            baseline).

    Cost note: each rollout is a real ~128-step autoregressive generation
    (~15-25s/sample on a single GPU in local testing), unlike the one-forward-
    pass teacher-forced decode this replaces -- keep `num_samples` and
    `every_n_steps` conservative.
    """

    def __init__(
        self,
        eval_dataset: Any,
        collate_fn: Callable,
        output_dir: str,
        every_n_steps: int = 100,
        sample_indices: Sequence[int] | None = None,
        num_samples: int = 2,
        num_traj_samples: int = 2,
        top_p: float = 0.98,
        temperature: float = 0.6,
        do_sample: bool = True,
        on_train_begin: bool = True,
    ) -> None:
        self.eval_dataset = eval_dataset
        self.collate_fn = collate_fn
        self.viz_dir = os.path.join(output_dir, "viz")
        self.every_n_steps = max(1, int(every_n_steps))
        if sample_indices is not None:
            self.sample_indices = list(sample_indices)
        else:
            n = min(num_samples, len(eval_dataset))
            self.sample_indices = list(range(n))
        self.num_traj_samples = num_traj_samples
        self.top_p = top_p
        self.temperature = temperature
        self.do_sample = do_sample
        self.on_train_begin_render = on_train_begin

    def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
        if self.on_train_begin_render and state.is_world_process_zero:
            self._safe_render(kwargs.get("model"), state)

    def on_step_end(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
        if not state.is_world_process_zero:
            return
        if state.global_step > 0 and state.global_step % self.every_n_steps == 0:
            self._safe_render(kwargs.get("model"), state)

    def _safe_render(self, model, state) -> None:
        if model is None:
            return
        try:
            self._render(model, state)
        except Exception as exc:  # noqa: BLE001 - viz must never kill training
            logger.warning(
                "[TrajectoryVizCallback] skipped at step %s: %r",
                getattr(state, "global_step", "?"),
                exc,
            )

    def _render(self, model, state) -> None:
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return

        os.makedirs(self.viz_dir, exist_ok=True)

        was_training = model.training
        model.eval()
        images: dict[str, Any] = {}
        ades = []
        recon_errs = []
        try:
            for idx in self.sample_indices:
                data = self.eval_dataset.load_raw(idx)
                prompt = build_generation_prompt(model, data, self.eval_dataset.processor)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_xyz = rollout_future_trajectory(
                        model,
                        prompt,
                        data["ego_history_xyz"],
                        data["ego_history_rot"],
                        num_traj_samples=self.num_traj_samples,
                        top_p=self.top_p,
                        temperature=self.temperature,
                        do_sample=self.do_sample,
                    )  # [1, K, Tf, 3]
                recon_xyz = decode_gt_tokenizer_reconstruction(
                    model,
                    data["ego_history_xyz"],
                    data["ego_history_rot"],
                    data["ego_future_xyz"],
                    data["ego_future_rot"],
                )

                gt_cpu = data["ego_future_xyz"][0].flatten(0, 1).cpu()
                pred_cpu = pred_xyz.cpu()  # [1, K, Tf, 3]
                per_k_ade = torch.tensor(
                    [
                        float(torch.linalg.norm(pred_cpu[0, k, :, :2] - gt_cpu[:, :2], dim=-1).mean())
                        for k in range(pred_cpu.shape[1])
                    ]
                )
                min_ade, mean_ade = float(per_k_ade.min()), float(per_k_ade.mean())
                ades.append(min_ade)

                recon_cpu = recon_xyz.cpu()
                recon_err = torch.linalg.norm(
                    recon_cpu[..., :2] - gt_cpu[..., :2], dim=-1
                ).reshape(-1)
                recon_errs.append(recon_err)

                scene_id = data.get("scene_id") or data.get("clip_id")
                t0_us = data.get("t0_us")
                header_lines = [f"step {state.global_step}  |  eval idx {idx}"]
                if scene_id is not None:
                    meta = f"scene: {scene_id}"
                    if t0_us is not None:
                        meta += f"  |  t0={t0_us / 1e6:.2f}s"
                    header_lines.append(meta)
                is_standstill = _gt_span_m(gt_cpu) < _STANDSTILL_SPAN_M
                if is_standstill:
                    header_lines.append(
                        f"STANDSTILL (GT travels {_gt_span_m(gt_cpu):.2f}m over the horizon)"
                    )
                header_lines.append(
                    f"minADE={min_ade:.2f}m (best of {self.num_traj_samples})"
                    f"  meanADE={mean_ade:.2f}m  (true autoregressive)"
                )
                header = "\n".join(header_lines)

                caption = f"idx {idx} · step {state.global_step}"
                if scene_id is not None:
                    caption += f" · {scene_id}"
                if is_standstill:
                    caption += " · STANDSTILL"
                caption += f" · minADE {min_ade:.2f}m"

                frames = data["image_frames"].permute(1, 0, 2, 3, 4)
                cam_labels, frame_labels = _grid_labels(data)
                png_path = os.path.join(
                    self.viz_dir, f"step{state.global_step:07d}_sample{idx}.png"
                )
                visualize_data(
                    image_frames=frames,
                    ego_future_xyz_gt=gt_cpu.unsqueeze(0),
                    ego_future_xyz_pred=pred_cpu.unsqueeze(0),
                    ego_future_xyz_recon=recon_cpu.unsqueeze(0),
                    ego_history_xyz=data["ego_history_xyz"][0].flatten(0, 1).cpu(),
                    show_waypoint_pai=False,
                    save_path=png_path,
                    info_text=header,
                    camera_labels=cam_labels,
                    frame_labels=frame_labels,
                )
                images[f"viz/val_sample_{idx}"] = wandb.Image(png_path, caption=caption)
        finally:
            if was_training:
                model.train()

        # Distinct key from val_metric's val/minADE_mean (that is the
        # representative strided ~248-window subset; this is the fixed
        # curated viz set) -- both fire on the same cadence, so they must not
        # share a key. Matches alpamayo1_5_sft's naming convention.
        metrics: dict[str, float] = {}
        if ades:
            metrics["val/minADE_viz_mean"] = sum(ades) / len(ades)
        if recon_errs:
            allerr = torch.cat(recon_errs)
            metrics["viz/tokenizer_ceiling_xy_mean_m"] = float(allerr.mean())
            metrics["viz/tokenizer_ceiling_xy_max_m"] = float(allerr.max())
        if images or metrics:
            wandb.log({**images, **metrics}, step=state.global_step)
            logger.info(
                "[TrajectoryVizCallback] logged %d trajectory viz(s) at step %s",
                len(images),
                state.global_step,
            )
