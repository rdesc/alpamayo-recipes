"""NOTE: new file -- not in upstream. Standalone offline eval for the Alpamayo 2
Super LoRA trajectory fine-tune.

The A2 recipe shipped only ``train_hf.py``; its two prediction paths
(``viz_callback.py``, ``val_metric_callback.py``) are ``TrainerCallback``s that
only run *inside* a training loop. This is the missing entry point: load a
trained checkpoint, run the same true-autoregressive rollout over the val split,
and write per-window predictions + a minADE summary to disk.

Deliberately mirrors ``val_metric_callback.py``'s metric so offline numbers are
comparable to the ``val/minADE_*`` curve logged during training: same
``build_generation_prompt`` + ``rollout_future_trajectory`` calls, same XY-plane
minADE-over-K reduction, same val dataset built from the same config. The one
difference is coverage -- the callback strides the split to stay cheap, this
defaults to every window.

Loading is adapter-based, NOT full-state. An HF Trainer checkpoint of a LoRA run
stores the *wrapped* model (``base_layer.*`` / ``lora_*`` keys, ~68 GB across 15
shards for this model). Loading those into a bare ``Alpamayo2Super`` is the
documented footgun: the adapted projections silently stay at their random init
because the key names no longer line up. So this instead rebuilds the base model
from the config, re-injects adapters with the run's *own* saved settings
(``adapter/alpamayo_lora.json``), and loads the 268 MB adapter file on top --
which is both correct and ~250x less I/O.

Usage (single GPU):

    python -m alpamayo2_sft.evaluate_hf --config-name sft_truckdrive_lora \\
      paths.output_dir=/mnt/efs/users/rod/eval/a2_lora_ckpt1030 \\
      eval.checkpoint=/home/rod/efs_ckpts/.../checkpoint-1030

Multi-GPU (windows are sharded across ranks and all-gathered):

    torchrun --nproc_per_node 8 -m alpamayo2_sft.evaluate_hf ... (same args)

``paths.output_dir`` is required because the config anchors Hydra's run dir to
it; point it at where the eval artifacts should go, not at the training run.

CoC (chain-of-causation) inspection run: add ``eval.use_cot=true`` to
free-generate reasoning text before the trajectory span (see
``rollout.rollout_future_trajectory_with_cot``) instead of decoding trajectory
tokens directly. Each record then also carries a ``"cot"`` string. NOTE: the
LoRA checkpoints this recipe produces are trained trajectory-only (no ``cot``
in ``label_components``), so this probes whatever reasoning ability the BASE
Alpamayo2Super checkpoint carries through the adapter -- it is not a trained
CoC ability and not a clean with/without-CoC ablation the way
alpamayo1_5_sft's cot vs non-cot checkpoints are.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import nullcontext
from typing import Any

import hydra
import hydra.utils as hyu
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from alpamayo2_sft.eval_metrics import displacement_metrics_from_records
from alpamayo2_sft.rollout import (
    build_generation_prompt,
    rollout_future_trajectory,
    rollout_future_trajectory_with_cot,
)

logger = logging.getLogger(__name__)


def _rank() -> int:
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def _world_size() -> int:
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def _min_ade(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor) -> float:
    """XY-plane minADE over K. pred_xyz: [K,Tf,3]; gt_xyz: [Tf,3].

    Identical reduction to ``val_metric_callback._min_ade`` -- keep them in sync
    or the offline number stops being comparable to the training curve.
    """
    pred_xy = pred_xyz[..., :2].float().cpu()
    gt_xy = gt_xyz[..., :2].float().cpu().unsqueeze(0)
    per_k = torch.linalg.norm(pred_xy - gt_xy, dim=-1).mean(dim=-1)  # [K]
    return float(per_k.min())


def _resolve_output_dir(configured: str | None, checkpoint: str) -> str:
    """Where eval artifacts go: ``<checkpoint>/eval``, timestamped on collision.

    Mirrors ``alpamayo1_5_sft/evaluate_hf.py``: results land next to the weights
    that produced them, and re-running never silently overwrites an earlier eval.
    Unlike that script no broadcast is needed here -- only rank 0 writes, so no
    other rank has to agree on the timestamp.
    """
    from datetime import datetime

    out_dir = configured or os.path.join(checkpoint.rstrip("/"), "eval")
    if os.path.exists(out_dir):
        out_dir = f"{out_dir.rstrip('/')}_{datetime.now():%Y%m%d_%H%M%S}"
    return out_dir


def _load_checkpoint(model: torch.nn.Module, checkpoint: str) -> torch.nn.Module:
    """Re-inject LoRA with the run's saved settings, then load adapter weights.

    Accepts either the Trainer checkpoint dir (``checkpoint-N/``) or the
    ``adapter/`` dir inside it.
    """
    from alpamayo2_sft.lora import (
        apply_lora,
        find_adapter_dir,
        load_lora_adapter,
        load_lora_settings,
    )

    adapter_dir = find_adapter_dir(checkpoint)
    if adapter_dir is None:
        # Maybe we were handed the adapter dir itself.
        if os.path.isfile(os.path.join(checkpoint, "adapter_model.safetensors")):
            adapter_dir = checkpoint
        else:
            raise FileNotFoundError(
                f"No LoRA adapter found under {checkpoint!r}. Expected "
                f"'adapter/adapter_model.safetensors'. Refusing to fall back to the "
                f"full-state shards: they store LoRA-wrapped keys, and loading them "
                f"into an un-adapted model leaves the adapted projections at random init."
            )

    settings = load_lora_settings(adapter_dir)
    logger.info("Rebuilding adapters from %s: %s", adapter_dir, settings)
    model = apply_lora(model, settings)
    model = load_lora_adapter(model, adapter_dir)
    return model


@hydra.main(version_base=None, config_path="configs", config_name="sft_base")
def evaluate(cfg: DictConfig) -> None:
    """Offline eval entry point."""
    torch.manual_seed(42)

    eval_cfg = cfg.get("eval", None)
    checkpoint = None if eval_cfg is None else eval_cfg.get("checkpoint", None)
    if not checkpoint:
        raise ValueError(
            "eval.checkpoint is required, e.g. "
            "eval.checkpoint=/path/to/checkpoint-1030 (the Trainer checkpoint dir "
            "or the adapter/ dir inside it)."
        )
    stride = int(eval_cfg.get("stride", 1))
    limit = eval_cfg.get("limit", None)
    num_traj_samples = int(eval_cfg.get("num_traj_samples", 6))
    top_p = float(eval_cfg.get("top_p", 0.98))
    temperature = float(eval_cfg.get("temperature", 0.6))
    do_sample = bool(eval_cfg.get("do_sample", True))
    save_predictions = bool(eval_cfg.get("save_predictions", True))
    # CoC (chain-of-causation) inspection run: free-generate reasoning text before
    # the trajectory span instead of decoding trajectory tokens directly. See
    # rollout.rollout_future_trajectory_with_cot. NOTE: this checkpoint format's
    # LoRA finetune was trained trajectory-only (no cot in label_components), so
    # this probes whatever reasoning the BASE Alpamayo2Super checkpoint carries
    # through the adapter, not a trained CoC ability -- not a clean ablation like
    # alpamayo1_5_sft's cot vs non-cot checkpoints.
    use_cot = bool(eval_cfg.get("use_cot", False))
    max_cot_tokens = int(eval_cfg.get("max_cot_tokens", 640))

    if torch.distributed.is_available() and "RANK" in os.environ:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    rank, world = _rank(), _world_size()
    # Resolved before the (long) rollout loop so a bad path fails immediately
    # rather than after an hour of generation.
    out_dir = _resolve_output_dir(eval_cfg.get("output_dir", None), checkpoint)
    if rank == 0:
        print(OmegaConf.to_yaml(cfg))
        os.makedirs(out_dir, exist_ok=True)
        logger.info("Eval artifacts will be written to %s", out_dir)

    model = hyu.instantiate(cfg.model, _convert_="partial")
    model = _load_checkpoint(model, checkpoint)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    eval_dataset = hyu.instantiate(
        cfg.data.val_dataset, model_config=model.config, tokenizer=model.tokenizer
    )

    subset = list(range(0, len(eval_dataset), max(1, stride)))
    if limit is not None:
        subset = subset[: int(limit)]
    my_idxs = subset[rank::world]
    if rank == 0:
        logger.info(
            "Evaluating %d val windows (stride=%d) across %d rank(s), K=%d, "
            "do_sample=%s top_p=%s temperature=%s",
            len(subset),
            stride,
            world,
            num_traj_samples,
            do_sample,
            top_p,
            temperature,
        )

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else nullcontext()
    )
    records: list[dict[str, Any]] = []
    with torch.no_grad(), autocast:
        for n, idx in enumerate(my_idxs):
            data = eval_dataset.load_raw(idx)
            cot_text = None
            if use_cot:
                prompt = build_generation_prompt(model, data, eval_dataset.processor, with_cot=True)
                pred_xyz, cot_text = rollout_future_trajectory_with_cot(
                    model,
                    prompt,
                    data["ego_history_xyz"],
                    data["ego_history_rot"],
                    num_traj_samples=num_traj_samples,
                    top_p=top_p,
                    temperature=temperature,
                    do_sample=do_sample,
                    max_cot_tokens=max_cot_tokens,
                )  # [1, K, Tf, 3]
            else:
                prompt = build_generation_prompt(model, data, eval_dataset.processor)
                pred_xyz = rollout_future_trajectory(
                    model,
                    prompt,
                    data["ego_history_xyz"],
                    data["ego_history_rot"],
                    num_traj_samples=num_traj_samples,
                    top_p=top_p,
                    temperature=temperature,
                    do_sample=do_sample,
                )  # [1, K, Tf, 3]
            gt_xyz = data["ego_future_xyz"][0].flatten(0, 1)  # [Tf, 3]
            # load_raw() does not guarantee scene_id/t0 in its payload (viz_callback
            # falls back through clip_id for the same reason). The dataset's own
            # `windows` list is the authoritative (scene_id, t0) for the index.
            window = getattr(eval_dataset, "windows", None)
            scene_id, t0 = (window[idx] if window is not None else (None, None))
            rec: dict[str, Any] = {
                "index": int(idx),
                "scene_id": scene_id or data.get("scene_id") or data.get("clip_id"),
                "t0": t0,
                "min_ade": _min_ade(pred_xyz[0].cpu(), gt_xyz.cpu()),
            }
            if cot_text is not None:
                rec["cot"] = cot_text
            if save_predictions:
                rec["pred_xyz"] = pred_xyz[0].float().cpu().tolist()
                rec["gt_xyz"] = gt_xyz.float().cpu().tolist()
            records.append(rec)
            if rank == 0 and (n + 1) % 25 == 0:
                logger.info("rank0: %d/%d windows done", n + 1, len(my_idxs))

    if world > 1 and torch.distributed.is_initialized():
        gathered: list[Any] = [None] * world
        torch.distributed.all_gather_object(gathered, records)
    else:
        gathered = [records]
    if rank != 0:
        return

    all_records = [r for part in gathered for r in part]
    all_records.sort(key=lambda r: r["index"])
    ades = np.array([r["min_ade"] for r in all_records])
    summary = {
        "checkpoint": checkpoint,
        "n_windows": int(ades.size),
        "num_traj_samples": num_traj_samples,
        "do_sample": do_sample,
        "top_p": top_p,
        "temperature": temperature,
        "stride": stride,
        "use_cot": use_cot,
        "max_cot_tokens": max_cot_tokens if use_cot else None,
        "minADE_mean": float(ades.mean()),
        "minADE_median": float(np.median(ades)),
        "minADE_p95": float(np.percentile(ades, 95)),
    }
    # Full min_ade/ade/min_fde/fde breakdown across horizons, matching the set
    # alpamayo1_5_sft's evaluate_hf.py reports (see eval_metrics.py). Needs
    # pred_xyz/gt_xyz per record, which requires save_predictions=True.
    if save_predictions:
        summary.update(displacement_metrics_from_records(all_records))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "eval_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(all_records, f)

    logger.info("Wrote eval artifacts to %s", out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    evaluate()
