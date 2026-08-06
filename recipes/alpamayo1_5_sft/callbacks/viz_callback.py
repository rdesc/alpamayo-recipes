# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""HF Trainer callback that logs trajectory-prediction visualizations to W&B.

On a fixed step interval (and optionally at train start) this renders a few
held-out validation samples: the multi-camera image grid plus a bird's-eye-view
plot of the ground-truth future trajectory (solid red) against the model's
predicted trajectory (blue) and the frozen tokenizer's encode->decode
reconstruction of that GT future (dashed light red) -- the latter is the floor
the prediction is measured against, so a recon that already looks wrong points
at the tokenizer rather than the model. The nav instruction is the caption. The
prediction comes from
``model.sample_trajectories_from_data``, which decodes the VLM's *generated
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
from contextlib import nullcontext
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: training process has no display

import torch
from transformers import TrainerCallback

import alpamayo.common.constants as constants
from alpamayo.visualization.viz import visualize_data

logger = logging.getLogger(__name__)

try:
    from alpamayo_r1.models.base_model import SPECIAL_TOKENS

    _ROUTE_START = SPECIAL_TOKENS["route_start"]
except Exception:  # noqa: BLE001 - viz must not depend on the model package
    _ROUTE_START = "<|route_start|>"


# A window whose GT travels less than this over the whole future horizon is
# labelled STANDSTILL in the figure header/caption. Display only -- it does not
# change any target. Generous vs the dataset's standstill snap (which zeroes the
# GT outright) so that near-parked windows are flagged too.
_STANDSTILL_SPAN_M = 1.0


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
        show_nav: If True, the figure header/caption always report the sample's
            navigation command (``(none)`` when unlabeled) *and* whether the
            prompt the model sees actually carries the route component -- so a
            nav-conditioned run can be told apart from one where the command is
            present in the data but never reaches the model.
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
        gen_batch_size: int = 4,
        show_nav: bool = False,
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
        # Generate in mini-batches: with ~30 fixed viz windows x K samples a
        # single VLM-rollout forward can need >20 GiB, which OOMs when it lands
        # on top of the training footprint. Chunking bounds the peak.
        self.gen_batch_size = max(1, int(gen_batch_size))
        self.num_traj_samples = num_traj_samples
        self.top_p = top_p
        self.temperature = temperature
        self.max_generation_length = max_generation_length
        self.on_train_begin_render = on_train_begin
        self.show_nav = show_nav

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
        # Flag parked windows explicitly: their GT is a dot at the origin, which
        # otherwise reads as a broken plot rather than the correct answer. With
        # the dataset's standstill snap on, the GT here is exactly zero.
        gt_span = self._gt_span_m(sample)
        if gt_span is not None and gt_span < _STANDSTILL_SPAN_M:
            header_lines.append(f"STANDSTILL (GT travels {gt_span:.2f}m over the horizon)")
        if self.show_nav:
            conditioned = self._route_in_prompt(sample)
            if conditioned is None:
                cond_tag = "conditioning: unknown"
            elif conditioned:
                cond_tag = "CONDITIONED on nav"
            else:
                cond_tag = "NOT conditioned on nav"
            nav_str = f'"{nav_text}"' if nav_text else "(none)"
            header_lines.append(f"nav: {nav_str}  [{cond_tag}]")
        elif nav_text:
            header_lines.append(f'nav: "{nav_text}"')
        header = "\n".join(header_lines)

        clip_short = clip_id[:8] if isinstance(clip_id, str) else str(clip_id)
        caption = f'idx {idx} · clip {clip_short} · step {step}'
        if gt_span is not None and gt_span < _STANDSTILL_SPAN_M:
            caption += " · STANDSTILL"
        if nav_text:
            caption += f' · "{nav_text}"'
        if self.show_nav:
            caption += " · cond" if self._route_in_prompt(sample) else " · uncond"
        return header, caption

    @staticmethod
    def _gt_span_m(sample: dict) -> float | None:
        """Ground-truth XY distance from t0 to the last future waypoint (metres).

        Returns None when the sample carries no future trajectory (e.g. a
        vqa-only batch), so callers can skip the standstill tag.
        """
        fut = sample.get("ego_future_xyz")
        if fut is None:
            return None
        try:
            return float(torch.as_tensor(fut).reshape(-1, 3)[-1, :2].norm())
        except Exception:  # noqa: BLE001 - a malformed field must not kill viz
            return None

    @staticmethod
    def _route_in_prompt(sample: dict) -> bool | None:
        """Whether the prompt the model actually sees carries the route component.

        Ground truth rather than inference: the nav instruction only reaches the
        model if ``construct_route`` emitted ``<|route_start|>`` into the
        tokenized prompt text, which requires both a ``nav_text`` on the sample
        and ``route`` in the processor's ``components_order``. Returns None when
        no tokenized text is available (dataset built without a preprocessor).
        """
        tok = sample.get("tokenized_data")
        text = tok.get("text") if isinstance(tok, dict) else None
        if not isinstance(text, str):
            return None
        return _ROUTE_START in text

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

    @staticmethod
    def _min_ade(pred_j, gt) -> tuple[float, float]:
        """XY-plane ADE of prediction vs GT future. pred_j: [N,K,Tf,3] or [K,Tf,3];
        gt: [N,Tf,3] or [Tf,3]. Returns (minADE over K, meanADE over K) in meters."""
        if pred_j.dim() == 4:
            pred_j = pred_j[0]
        if gt.dim() == 3:
            gt = gt[0]
        pred_xy = pred_j[..., :2].float().cpu()
        gt_xy = gt[..., :2].float().cpu().unsqueeze(0)
        per_k = torch.linalg.norm(pred_xy - gt_xy, dim=-1).mean(dim=-1)  # [K]
        return float(per_k.min()), float(per_k.mean())

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

        gen_kwargs: dict[str, Any] = {}
        if self.max_generation_length is not None:
            gen_kwargs["max_generation_length"] = self.max_generation_length

        # Release the training allocator's cached-but-unused blocks so the
        # rollout's large contiguous allocations don't fail on fragmentation.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        was_training = model.training
        model.eval()
        pred_chunks = []
        # Match training's bf16 autocast: the diffusion head / action projections
        # are bf16, but diffusion.sample() seeds float32 noise, so without autocast
        # the expert (stage-2) rollout hits a Float-vs-BFloat16 matmul mismatch.
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else nullcontext()
        )
        try:
            with torch.no_grad(), autocast:
                for start in range(0, len(samples), self.gen_batch_size):
                    chunk = samples[start:start + self.gen_batch_size]
                    batch = self.collate_fn([dict(s) for s in chunk])
                    batch = _to_device(batch, device)
                    # Polymorphic — the plotted trajectory depends on the model class
                    # this callback was handed (selected by the `_target_` in the model
                    # config), not on any flag here:
                    #   stage-1 (TrainableReasoningVLA): VLM.generate() -> discrete traj
                    #     tokens -> traj_tokenizer.decode() -> xyz. The tokens ARE the prediction
                    #   stage-2 (TrainableAlpamayoR1): VLM.generate() only fills the
                    #     KV-cache up to <traj_future_start> (its generated future tokens
                    #     are cropped/masked out); xyz comes from the diffusion action
                    #     head via the continuous action_space.
                    pred_c, _ = model.sample_trajectories_from_data(
                        data=batch,
                        num_traj_samples=self.num_traj_samples,
                        num_traj_sets=1,
                        top_p=self.top_p,
                        temperature=self.temperature,
                        **gen_kwargs,
                    )
                    pred_chunks.append(pred_c.cpu())
                    del batch
        finally:
            if was_training:
                model.train()
        pred_xyz = torch.cat(pred_chunks, dim=0)

        # Frozen-tokenizer encode->decode of each GT future. Plotted alongside
        # GT/prediction and reduced to the "ceiling" scalars logged below.
        recons = self._tokenizer_recons(model, samples)

        # pred_xyz: [B, N, K, Tf, 3]; visualize_data wants [1, 1, K, Tf, 3].
        images = {}
        metrics: dict[str, float] = {}
        ades = []
        for j, idx in enumerate(self.sample_indices):
            s = samples[j]
            # image_frames: (N_cams, num_frames, C, H, W) -> (num_frames, N_cams, C, H, W)
            frames = s["image_frames"]
            if frames.dim() == 5:
                frames = frames.permute(1, 0, 2, 3, 4)
            header, caption = self._sample_info(s, idx, state.global_step)
            cam_labels, frame_labels = self._grid_labels(s)
            # minADE of this step's prediction on this fixed sample (shown on the
            # figure + logged as a scalar so the viz set doubles as a metric).
            gt = s.get("ego_future_xyz")
            if gt is not None:
                mn, mean = self._min_ade(pred_xyz[j], gt)
                header = (f"{header}\nminADE={mn:.2f}m (best of {self.num_traj_samples})"
                          f"  meanADE={mean:.2f}m")
                caption += f" · minADE {mn:.2f}m"
                # Per-sample minADE is shown on the figure/caption; only the mean
                # across the viz set is logged as a scalar (below).
                ades.append(mn)
            png_path = os.path.join(
                self.viz_dir, f"step{state.global_step:07d}_sample{idx}.png"
            )
            visualize_data(
                image_frames=frames,
                ego_future_xyz_gt=s.get("ego_future_xyz"),
                ego_future_xyz_pred=pred_xyz[j].unsqueeze(0).cpu(),
                ego_future_xyz_recon=recons[j],
                ego_history_xyz=s.get("ego_history_xyz"),
                cot_text=s.get("nav_text"),
                show_waypoint_pai=False,
                save_path=png_path,
                info_text=header,
                camera_labels=cam_labels,
                frame_labels=frame_labels,
            )
            images[f"viz/val_sample_{idx}"] = wandb.Image(png_path, caption=caption)

        if ades:
            # Mean minADE across the fixed viz set. Distinct key from the
            # val_metric callback's val/minADE_mean (that is the representative
            # strided ~248-window subset; this is the 20 difficulty-stratified
            # viz windows) -- both fire on the same cadence, so they must not
            # share a key.
            metrics["val/minADE_viz_mean"] = sum(ades) / len(ades)
        if images or metrics:
            wandb.log({**images, **metrics}, step=state.global_step)
            logger.info(
                "[TrajectoryVizCallback] logged %d trajectory viz(s) at step %s",
                len(images),
                state.global_step,
            )

        # Frozen-tokenizer reconstruction "ceiling": encode->decode the GT future
        # and measure XY error. This is a property of (data, tokenizer), ~constant
        # over training -- a reference line the model's predicted-traj error can
        # approach but never beat. Logged next to the viz on the same cadence.
        ceiling = self._tokenizer_ceiling(samples, recons)
        if ceiling:
            wandb.log(ceiling, step=state.global_step)

    def _tokenizer_recons(self, model, samples: list[dict]) -> list[Any]:
        """Encode->decode each sample's GT future with the frozen tokenizer.

        Returns one reconstructed xyz tensor per sample (None where unavailable),
        so the same round-trip feeds both the plotted recon curve and the
        scalar "ceiling" metrics below.
        """
        tok = getattr(model, "traj_tokenizer", None)
        if tok is None:
            return [None] * len(samples)
        recons: list[Any] = []
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
                recons.append(rec.detach().cpu())
            except Exception as exc:  # noqa: BLE001 - never interrupt training
                logger.warning("[TrajectoryVizCallback] tokenizer round-trip skipped: %r", exc)
                recons.append(None)
        return recons

    def _tokenizer_ceiling(self, samples: list[dict], recons: list[Any]) -> dict[str, float]:
        """XY reconstruction error of the frozen future tokenizer on GT futures.

        Returns {} if no round-trip succeeded. The unicycle tokenizer models the
        ground-plane path only, so error is measured in XY.
        """
        errs = []
        for s, rec in zip(samples, recons):
            if rec is None:
                continue
            e = torch.linalg.norm(
                rec[..., :2] - s["ego_future_xyz"][..., :2].cpu(), dim=-1
            )
            errs.append(e.reshape(-1))
        if not errs:
            return {}
        allerr = torch.cat(errs)
        return {
            "viz/tokenizer_ceiling_xy_mean_m": float(allerr.mean()),
            "viz/tokenizer_ceiling_xy_max_m": float(allerr.max()),
        }
