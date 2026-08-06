"""Periodic val minADE_K over a fixed strided subset of the val split, to W&B.

Ported from alpamayo1_5_sft/callbacks/val_metric_callback.py, adapted the same
way viz_callback.py was: Alpamayo 2 Super's own ``sample_trajectories_from_data``
always decodes through the frozen (untouched-by-this-recipe) diffusion expert,
so it would render a flat curve regardless of training progress. This uses
``rollout.py``'s true autoregressive VLM-token rollout instead -- the same
per-sample ``build_generation_prompt`` + ``rollout_future_trajectory`` calls
viz_callback.py already makes, just over a larger strided subset and without
the plotting/image-saving overhead.

Also drops the alpamayo1_5 version's ``alpamayo.common.distributed`` import
(pulls in ``alpamayo_r1``, not installed in this recipe's venv) in favor of
plain ``torch.distributed`` calls.

Unlike the teacher-forced ``eval/loss`` (token likelihood), this logs a stable
*geometric* driving-accuracy curve: minADE over K generated trajectories
(K=6, Alpamayo convention), aggregated over every Nth val window. Generation
is sharded across ranks (each does subset[rank::world]) and the per-window
errors are all-gathered, so no single rank blocks the others.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from transformers import TrainerCallback

from alpamayo2_sft.rollout import build_generation_prompt, rollout_future_trajectory

logger = logging.getLogger(__name__)


def _rank() -> int:
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def _world_size() -> int:
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def _min_ade(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor) -> float:
    """XY-plane minADE over K. pred_xyz: [K,Tf,3]; gt_xyz: [Tf,3]."""
    pred_xy = pred_xyz[..., :2].float().cpu()
    gt_xy = gt_xyz[..., :2].float().cpu().unsqueeze(0)
    per_k = torch.linalg.norm(pred_xy - gt_xy, dim=-1).mean(dim=-1)  # [K]
    return float(per_k.min())


class ValMinADECallback(TrainerCallback):
    """Log ``val/minADE_{mean,median,p95}`` over a fixed strided val subset."""

    def __init__(
        self,
        eval_dataset: Any,
        every_n_steps: int = 500,
        stride: int = 10,
        num_traj_samples: int = 6,
        top_p: float = 0.98,
        temperature: float = 0.6,
        do_sample: bool = True,
        on_train_begin: bool = True,
    ) -> None:
        self.ds = eval_dataset
        self.every_n_steps = max(1, int(every_n_steps))
        self.num_traj_samples = num_traj_samples
        self.top_p = top_p
        self.temperature = temperature
        self.do_sample = do_sample
        self.on_train_begin_run = on_train_begin
        self.subset = list(range(0, len(eval_dataset), max(1, int(stride))))

    def on_train_begin(self, args, state, control, model=None, **kwargs):  # type: ignore[no-untyped-def]
        if self.on_train_begin_run:
            self._safe_run(model, state)

    def on_step_end(self, args, state, control, model=None, **kwargs):  # type: ignore[no-untyped-def]
        if state.global_step > 0 and state.global_step % self.every_n_steps == 0:
            self._safe_run(model, state)

    def _safe_run(self, model, state) -> None:
        if model is None:
            return
        try:
            self._run(model, state)
        except Exception as exc:  # noqa: BLE001 - metric must never kill training
            logger.warning("[ValMinADE] skipped at step %s: %r", getattr(state, "global_step", "?"), exc)

    def _run(self, model, state) -> None:
        rank, world = _rank(), _world_size()
        my_idxs = self.subset[rank::world]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        was_training = model.training
        model.eval()
        local_ades: list[float] = []
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else nullcontext()
        )
        try:
            with torch.no_grad(), autocast:
                for idx in my_idxs:
                    data = self.ds.load_raw(idx)
                    prompt = build_generation_prompt(model, data, self.ds.processor)
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
                    gt_xyz = data["ego_future_xyz"][0].flatten(0, 1)  # [Tf, 3]
                    local_ades.append(_min_ade(pred_xyz[0].cpu(), gt_xyz.cpu()))
        finally:
            if was_training:
                model.train()

        if world > 1 and torch.distributed.is_initialized():
            gathered: list[Any] = [None] * world
            torch.distributed.all_gather_object(gathered, local_ades)
        else:
            gathered = [local_ades]
        if rank != 0:
            return

        all_ades = [a for part in gathered for a in part]
        if not all_ades:
            return
        arr = np.array(all_ades)
        metrics = {
            "val/minADE_mean": float(arr.mean()),
            "val/minADE_median": float(np.median(arr)),
            "val/minADE_p95": float(np.percentile(arr, 95)),
            "val/minADE_n": int(arr.size),
        }
        try:
            import wandb
        except ImportError:
            wandb = None
        if wandb is not None and wandb.run is not None:
            wandb.log(metrics, step=state.global_step)
        logger.info(
            "[ValMinADE] step %s (K=%d, n=%d): mean=%.3fm median=%.3fm p95=%.3fm",
            state.global_step,
            self.num_traj_samples,
            metrics["val/minADE_n"],
            metrics["val/minADE_mean"],
            metrics["val/minADE_median"],
            metrics["val/minADE_p95"],
        )
