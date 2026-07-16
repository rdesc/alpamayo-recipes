# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HF Trainer callback that logs trajectory-prediction visualizations to W&B.

On a fixed step interval (and optionally at train start) this renders a few
held-out validation samples: the multi-camera image grid plus a bird's-eye-view
plot of the ground-truth future trajectory (red) against the model's predicted
trajectory (blue), with the nav instruction as a caption. The prediction comes
from ``model.sample_trajectories_from_data``, which decodes the VLM's *generated
discrete trajectory tokens* into xyz — so the plot tracks Stage-1 learning
directly, rather than the diffusion expert.

It is intentionally defensive: any failure during rendering is caught and logged
as a warning so a visualization hiccup can never interrupt training. Rendering
runs on rank 0 only; with DeepSpeed ZeRO-2 (params replicated, not sharded) a
rank-0-only forward needs no parameter all-gather, so this does not deadlock.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: training process has no display

import torch
from transformers import TrainerCallback

import alpamayo.common.constants as constants
from alpamayo.visualization.viz import visualize_data

logger = logging.getLogger(__name__)


def _to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors in a (possibly nested) batch to ``device``."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


class TrajectoryVizCallback(TrainerCallback):
    """Render GT-vs-predicted trajectory figures and log them to W&B.

    Args:
        eval_dataset: Validation dataset (must be built with
            ``generation_mode: true`` so ``tokenized_data`` is generation-ready).
        collate_fn: The same data collator used by the Trainer.
        output_dir: Directory to drop rendered PNGs into (a ``viz/`` subdir).
        every_n_steps: Render cadence in optimizer steps.
        sample_indices: Which eval-dataset indices to render. Defaults to the
            first ``num_samples`` samples.
        num_samples: Number of samples to render if ``sample_indices`` is None.
        num_traj_samples: Trajectory samples drawn per input (each plotted).
        top_p: Nucleus sampling top-p for generation.
        temperature: Sampling temperature for generation.
        max_generation_length: Override for generated token budget (defaults to
            the model's ``tokens_per_future_traj``).
        on_train_begin: If True, also render once before training (step 0
            baseline) so you can see the starting point.
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
        max_generation_length: int | None = None,
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
        self.max_generation_length = max_generation_length
        self.on_train_begin_render = on_train_begin

    # --- Trainer hooks --------------------------------------------------------

    def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
        if self.on_train_begin_render and state.is_world_process_zero:
            self._safe_render(kwargs.get("model"), state)

    def on_step_end(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
        if not state.is_world_process_zero:
            return
        if state.global_step > 0 and state.global_step % self.every_n_steps == 0:
            self._safe_render(kwargs.get("model"), state)

    # --- Sample identification ------------------------------------------------

    def _sample_info(self, sample: dict, idx: int, step: int) -> tuple[str, str]:
        """Build (figure header, W&B caption) identifying this train example.

        Pulls always-present fields from the loaded sample (clip_id, t0_us,
        nav_text) plus, if the dataset exposes its annotation list, the richer
        nav_maneuver / distance_m from the annotation entry.
        """
        clip_id = sample.get("clip_id")
        t0_us = sample.get("t0_us")
        nav_text = sample.get("nav_text")

        ann = {}
        anns = getattr(self.eval_dataset, "_samples", None)
        # Some datasets (e.g. PAI) index by annotation dicts; others (TruckDrive)
        # index by (scene_id, t0) tuples. Only mine the richer fields from dicts.
        if anns is not None and 0 <= idx < len(anns) and isinstance(anns[idx], dict):
            ann = anns[idx]
        maneuver = ann.get("nav_maneuver")
        distance_m = ann.get("distance_m")

        header_lines = [f"step {step}  |  eval idx {idx}"]
        if clip_id is not None:
            header_lines.append(f"clip_id: {clip_id}")
        meta = []
        if t0_us is not None:
            meta.append(f"t0={int(t0_us)}us")
        if maneuver:
            meta.append(str(maneuver))
        if distance_m is not None:
            meta.append(f"{distance_m}m")
        if meta:
            header_lines.append("  |  ".join(meta))
        if nav_text:
            header_lines.append(f'nav: "{nav_text}"')
        header = "\n".join(header_lines)

        clip_short = clip_id[:8] if isinstance(clip_id, str) else str(clip_id)
        caption = f'idx {idx} · clip {clip_short} · step {step}'
        if nav_text:
            caption += f' · "{nav_text}"'
        return header, caption

    def _grid_labels(self, sample: dict) -> tuple[list[str] | None, list[str] | None]:
        """Per-column camera names + per-row timestep labels for the image grid.

        Cameras are in the sample's (index-sorted) order; timesteps are re-based
        so the last frame (the t0 prediction anchor) reads ``+0.0s``.
        """
        cam_labels = None
        cam_idx = sample.get("camera_indices")
        if cam_idx is not None:
            cam_labels = [
                constants.CAMERA_INDICES_TO_DISPLAY_NAMES.get(int(i), f"cam{int(i)}")
                for i in cam_idx
            ]
        frame_labels = None
        rel = sample.get("relative_timestamps")  # (N_cam, num_frames), seconds
        if rel is not None and rel.numel():
            per_frame = rel.float().mean(dim=0)          # ~equal across cameras
            per_frame = per_frame - per_frame.max()      # end at t0 = +0.0s
            frame_labels = [f"{float(t):+.1f}s" for t in per_frame]
        return cam_labels, frame_labels

    # --- Rendering ------------------------------------------------------------

    def _safe_render(self, model, state) -> None:
        """Render + log, swallowing any error so training is never interrupted."""
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
        device = next(model.parameters()).device

        samples = [self.eval_dataset[i] for i in self.sample_indices]
        batch = self.collate_fn([dict(s) for s in samples])
        batch = _to_device(batch, device)

        gen_kwargs: dict[str, Any] = {}
        if self.max_generation_length is not None:
            gen_kwargs["max_generation_length"] = self.max_generation_length

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                pred_xyz, _ = model.sample_trajectories_from_data(
                    data=batch,
                    num_traj_samples=self.num_traj_samples,
                    num_traj_sets=1,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    **gen_kwargs,
                )
        finally:
            if was_training:
                model.train()

        # pred_xyz: [B, N, K, Tf, 3]; visualize_data wants [1, 1, K, Tf, 3].
        images = {}
        for j, idx in enumerate(self.sample_indices):
            s = samples[j]
            # image_frames: (N_cams, num_frames, C, H, W) -> (num_frames, N_cams, C, H, W)
            frames = s["image_frames"]
            if frames.dim() == 5:
                frames = frames.permute(1, 0, 2, 3, 4)
            header, caption = self._sample_info(s, idx, state.global_step)
            cam_labels, frame_labels = self._grid_labels(s)
            png_path = os.path.join(
                self.viz_dir, f"step{state.global_step:07d}_sample{idx}.png"
            )
            visualize_data(
                image_frames=frames,
                ego_future_xyz_gt=s.get("ego_future_xyz"),
                ego_future_xyz_pred=pred_xyz[j].unsqueeze(0).cpu(),
                cot_text=s.get("nav_text"),
                show_waypoint_pai=False,
                save_path=png_path,
                info_text=header,
                camera_labels=cam_labels,
                frame_labels=frame_labels,
            )
            images[f"viz/sample_{idx}"] = wandb.Image(png_path, caption=caption)

        if images:
            wandb.log(images, step=state.global_step)
            logger.info(
                "[TrajectoryVizCallback] logged %d trajectory viz(s) at step %s",
                len(images),
                state.global_step,
            )

        # Frozen-tokenizer reconstruction "ceiling": encode->decode the GT future
        # and measure XY error. This is a property of (data, tokenizer), ~constant
        # over training -- a reference line the model's predicted-traj error can
        # approach but never beat. Logged next to the viz on the same cadence.
        ceiling = self._tokenizer_ceiling(model, samples)
        if ceiling:
            wandb.log(ceiling, step=state.global_step)

    def _tokenizer_ceiling(self, model, samples: list[dict]) -> dict[str, float]:
        """XY reconstruction error of the frozen future tokenizer on GT futures.

        Returns {} if the model exposes no future tokenizer. The unicycle
        tokenizer models the ground-plane path only, so error is measured in XY.
        """
        tok = getattr(model, "traj_tokenizer", None)
        if tok is None:
            return {}
        errs = []
        for s in samples:
            try:
                tokens = tok.encode(
                    hist_xyz=s["ego_history_xyz"], hist_rot=s["ego_history_rot"],
                    fut_xyz=s["ego_future_xyz"], fut_rot=s["ego_future_rot"],
                )
                rec, _, _ = tok.decode(
                    hist_xyz=s["ego_history_xyz"], hist_rot=s["ego_history_rot"],
                    tokens=tokens,
                )
                e = torch.linalg.norm(
                    rec[..., :2].cpu() - s["ego_future_xyz"][..., :2].cpu(), dim=-1
                )
                errs.append(e.reshape(-1))
            except Exception as exc:  # noqa: BLE001 - never interrupt training
                logger.warning("[TrajectoryVizCallback] tokenizer ceiling skipped: %r", exc)
                return {}
        if not errs:
            return {}
        allerr = torch.cat(errs)
        return {
            "viz/tokenizer_ceiling_xy_mean_m": float(allerr.mean()),
            "viz/tokenizer_ceiling_xy_max_m": float(allerr.max()),
        }
