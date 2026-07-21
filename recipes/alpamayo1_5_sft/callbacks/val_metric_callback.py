# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Periodic val minADE_K over a fixed strided subset of the val split, to W&B.

Unlike the teacher-forced ``eval/loss`` (token likelihood) and the 20-window viz
panel, this logs a stable *geometric* driving-accuracy curve: minADE over K
generated trajectories (K=6, Alpamayo convention), aggregated over every Nth val
window. Generation is sharded across ranks (each does subset[rank::world]) and the
per-window errors are all-gathered, so no single rank blocks the others -- the
same pattern as truckdrive/score_val_ade.py.

The subset is fixed and its images are cached on first run, so the curve is
comparable step-to-step and later evals skip S3.
"""
import logging
from contextlib import nullcontext
from typing import Any, Callable

import torch
from transformers import TrainerCallback

from alpamayo.common import distributed

logger = logging.getLogger(__name__)


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def _min_ade(pred_j, gt) -> float:
    """XY-plane minADE over K. pred_j: [N,K,Tf,3] or [K,Tf,3]; gt: [N,Tf,3] or [Tf,3]."""
    if pred_j.dim() == 4:
        pred_j = pred_j[0]
    if gt.dim() == 3:
        gt = gt[0]
    pred_xy = pred_j[..., :2].float().cpu()
    gt_xy = gt[..., :2].float().cpu().unsqueeze(0)
    per_k = torch.linalg.norm(pred_xy - gt_xy, dim=-1).mean(dim=-1)  # [K]
    return float(per_k.min())


class ValMinADECallback(TrainerCallback):
    """Log ``val/minADE_{mean,median,p95}`` over a fixed strided val subset."""

    def __init__(
        self,
        eval_dataset: Any,                 # generation_mode=True val dataset
        collate_fn: Callable,
        every_n_steps: int = 500,
        stride: int = 10,
        num_traj_samples: int = 6,
        batch_size: int = 8,
        top_p: float = 0.98,
        temperature: float = 0.6,
        on_train_begin: bool = True,
        max_generation_length: int | None = None,
    ) -> None:
        self.ds = eval_dataset
        self.collate_fn = collate_fn
        self.every_n_steps = max(1, int(every_n_steps))
        self.num_traj_samples = num_traj_samples
        self.batch_size = batch_size
        self.top_p = top_p
        self.temperature = temperature
        self.on_train_begin_run = on_train_begin
        self.max_generation_length = max_generation_length
        self.subset = list(range(0, len(eval_dataset), max(1, int(stride))))
        self._cache = None  # per-rank list of (cpu_batch, [gt,...])

    # --- hooks ----------------------------------------------------------------
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.on_train_begin_run:
            self._safe_run(model, state)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.every_n_steps == 0:
            self._safe_run(model, state)

    # --- work -----------------------------------------------------------------
    def _safe_run(self, model, state) -> None:
        if model is None:
            return
        try:
            self._run(model, state)
        except Exception as exc:  # noqa: BLE001 - metric must never kill training
            logger.warning("[ValMinADE] skipped at step %s: %r",
                           getattr(state, "global_step", "?"), exc)

    def _build_cache(self) -> None:
        rank = distributed.get_global_rank()
        world = distributed.get_world_size()
        my = self.subset[rank::world]
        cache = []
        for start in range(0, len(my), self.batch_size):
            idxs = my[start:start + self.batch_size]
            samples = [dict(self.ds[i]) for i in idxs]
            batch = self.collate_fn([dict(s) for s in samples])
            gts = [s.get("ego_future_xyz") for s in samples]
            cache.append((batch, gts))
        self._cache = cache

    def _run(self, model, state) -> None:
        if self._cache is None:
            self._build_cache()
        device = next(model.parameters()).device
        gen_kwargs: dict[str, Any] = {}
        if self.max_generation_length is not None:
            gen_kwargs["max_generation_length"] = self.max_generation_length

        # Release the training allocator's cached-but-unused blocks so the
        # rollout's large contiguous allocations don't fail on fragmentation.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        was_training = model.training
        model.eval()
        local_ades = []
        # Match training's bf16 autocast (see viz_callback): the stage-2 diffusion
        # rollout seeds float32 noise into bf16 action projections otherwise.
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else nullcontext()
        )
        try:
            with torch.no_grad(), autocast:
                for batch, gts in self._cache:
                    b = _to_device(batch, device)
                    pred_xyz, _ = model.sample_trajectories_from_data(
                        data=b, num_traj_samples=self.num_traj_samples,
                        num_traj_sets=1, top_p=self.top_p, temperature=self.temperature,
                        **gen_kwargs,
                    )
                    pred_xyz = pred_xyz.cpu()
                    for j, gt in enumerate(gts):
                        if gt is not None:
                            local_ades.append(_min_ade(pred_xyz[j], gt.cpu()))
        finally:
            if was_training:
                model.train()

        # all-gather per-window minADEs so every rank contributes its shard
        world = distributed.get_world_size()
        if world > 1 and torch.distributed.is_initialized():
            gathered = [None] * world
            torch.distributed.all_gather_object(gathered, local_ades)
        else:
            gathered = [local_ades]
        if distributed.get_global_rank() != 0:
            return

        all_ades = [a for part in gathered for a in part]
        if not all_ades:
            return
        import numpy as np
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
        logger.info("[ValMinADE] step %s (K=%d, n=%d): mean=%.3fm median=%.3fm p95=%.3fm",
                    state.global_step, self.num_traj_samples, metrics["val/minADE_n"],
                    metrics["val/minADE_mean"], metrics["val/minADE_median"],
                    metrics["val/minADE_p95"])
