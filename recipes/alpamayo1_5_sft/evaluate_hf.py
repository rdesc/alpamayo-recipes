# NOTE: modified in this fork -- see upstream NVlabs/alpamayo-recipes for the original.
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

import glob
import json
import os
import pickle
from collections import defaultdict
from datetime import datetime
from itertools import islice

import hydra
import hydra.utils as hyu
import torch

from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from alpamayo1_5_sft.trainer import ReasoningVLA_Trainer
from alpamayo1_5_sft.trainer import TrainingArguments
from alpamayo1_5_sft.models.sft_base_model import TrainableReasoningVLA
from alpamayo1_5_sft.models.sft_alpamayo_r1 import TrainableAlpamayoR1

from alpamayo.common import (
    distributed,
    misc,
    wandb_utils,
)
from alpamayo_r1.common import logging

from alpamayo_r1.common.logging import setup_logging

setup_logging()

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")


dtype_map = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

# Per-sample keys pulled out of output_batch/data_batch when dumping predictions.
# ID keys are collated into per-sample Python lists; tensor keys carry a leading
# batch dim we slice per sample. Missing keys are skipped silently.
_PRED_ID_KEYS = ("scene_id", "clip_id", "t0_us")
_PRED_TENSOR_KEYS = (
    "pred_xyz",
    "pred_rot",
    "ego_future_xyz",
    "ego_future_rot",
    "ego_history_xyz",
    "ego_history_rot",
    "sample_ade",
)


def _extract_prediction_records(data_batch, output_batch):
    """Slice one batch into a list of per-sample CPU records for offline eval."""
    batch_size = len(data_batch["image_frames"])
    # metric/* keys carry the per-sample distance metrics (ade, min_ade, ...).
    tensor_keys = list(_PRED_TENSOR_KEYS) + [
        k for k in output_batch if k.startswith("metric/")
    ]

    # gen_text/* keys carry decoded reasoning strings (cot / meta_action /
    # answer), one [B, ns, nj] array per key, present only when the sampler ran
    # with return_extra=True.
    text_keys = [k for k in output_batch if k.startswith("gen_text/")]

    records = []
    for i in range(batch_size):
        record = {}
        for key in _PRED_ID_KEYS:
            values = data_batch.get(key)
            if values is not None:
                record[key] = values[i]
        for key in tensor_keys:
            value = output_batch.get(key)
            if isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
                record[key] = value[i].detach().to(torch.float32).cpu()
        for key in text_keys:
            value = output_batch.get(key)
            if value is not None and len(value) == batch_size:
                row = value[i]
                # numpy arrays of strings -> plain python for a portable dump.
                record[key] = row.tolist() if hasattr(row, "tolist") else row
        records.append(record)
    return records


@hydra.main(version_base=None, config_path=None, config_name="config")
def evaluate(cfg: DictConfig) -> None:
    distributed.initialize_distributed_simple()

    logger.info(
        "Dataset Configs:\n"
        + misc.pformat(OmegaConf.to_container(cfg.data.val_dataset, resolve=True))
    )
    logger.info(
        "Evaluate Configs:\n" + misc.pformat(OmegaConf.to_container(cfg.evaluate, resolve=True))
    )
    training_args = TrainingArguments(**OmegaConf.to_container(cfg.trainer, resolve=True))

    if cfg.evaluate.get("eval_ckpt", None) is not None:
        logger.info(f"Loading model from {cfg.evaluate.eval_ckpt}")
        model_cls = hyu.get_class(cfg.model._target_.rsplit(".", 1)[0])
        if issubclass(model_cls, TrainableReasoningVLA):
            cfg.model.checkpoint_path = cfg.evaluate.eval_ckpt
        elif issubclass(model_cls, TrainableAlpamayoR1):
            cfg.model.pretrained_model_name_or_path = cfg.evaluate.eval_ckpt
        else:
            raise ValueError(f"Unsupported model class: {model_cls}")
    model = hyu.instantiate(cfg.model, _convert_="partial")

    eval_dataset = hyu.instantiate(
        cfg.data.val_dataset, _convert_="partial", model_config=model.config
    )
    collate_fn = hyu.instantiate(
        cfg.data.collate_fn, _convert_="partial", model_config=model.config
    )

    trainer = ReasoningVLA_Trainer(
        model=model, args=training_args, eval_dataset=eval_dataset, data_collator=collate_fn
    )
    model = trainer.accelerator.prepare_model(model, evaluation_mode=True)
    model.eval()
    accelerator = trainer.accelerator
    is_main_process = accelerator.is_main_process

    if cfg.get("wandb", None) and is_main_process:
        os.makedirs(cfg.wandb.output_dir, exist_ok=True)
        wandb_utils.init_wandb(**cfg.wandb)

    val_dataloader = trainer.get_eval_dataloader()

    metric_runner = hydra.utils.instantiate(cfg.evaluate.metric_runner)

    # Resolve where per-sample predictions get dumped. Default: <eval_ckpt>/eval
    # (or ./eval when running without an eval_ckpt), so recomputing metrics later
    # lives right next to the checkpoint that produced them. If that dir already
    # exists we append a _YYYYmmdd_HHMMSS suffix so a re-run never clobbers an
    # earlier eval. Rank 0 picks the (timestamp-dependent) dir and broadcasts it
    # so every rank writes its shard to the same place.
    save_predictions = cfg.evaluate.get("save_predictions", False)
    predictions_dir = cfg.evaluate.get("predictions_dir", None)
    if save_predictions:
        if predictions_dir is None:
            eval_ckpt = cfg.evaluate.get("eval_ckpt", None)
            predictions_dir = os.path.join(eval_ckpt, "eval") if eval_ckpt else "./eval"
        if is_main_process and os.path.exists(predictions_dir):
            predictions_dir = f"{predictions_dir.rstrip('/')}_{datetime.now():%Y%m%d_%H%M%S}"
        if torch.distributed.is_initialized():
            dir_holder = [predictions_dir]
            torch.distributed.broadcast_object_list(dir_holder, src=0)
            predictions_dir = dir_holder[0]
        os.makedirs(predictions_dir, exist_ok=True)
        if is_main_process:
            logger.info(f"Saving per-sample predictions under {predictions_dir}")
    # Stream each batch's records straight to a per-rank shard so memory stays
    # flat and completed batches survive a crash. The shard is a sequence of
    # pickled lists (one per batch); the merge step reads them back until EOF.
    pred_shard_path = None
    pred_shard_file = None
    if save_predictions:
        pred_shard_path = os.path.join(
            predictions_dir, f"predictions_rank{accelerator.process_index}.pkl"
        )
        pred_shard_file = open(pred_shard_path, "wb")

    max_eval_steps = cfg.evaluate.get("max_eval_steps", -1)
    if max_eval_steps == -1 or max_eval_steps is None:
        dataloader_iter = val_dataloader
        total = len(val_dataloader)
    else:
        dataloader_iter = islice(val_dataloader, max_eval_steps)
        total = max_eval_steps

    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)
    val_count = 0

    for data in tqdm(dataloader_iter, total=total, disable=not is_main_process):
        output_batch = {}
        with torch.autocast("cuda", dtype=dtype_map[cfg.evaluate.torch_dtype]):
            metric_runner.run(model, data, output_batch)

        if save_predictions:
            pickle.dump(_extract_prediction_records(data, output_batch), pred_shard_file)
            pred_shard_file.flush()

        batch_size = len(data["image_frames"])
        gathered_batch_size = accelerator.gather_for_metrics(
            torch.tensor([batch_size], device=accelerator.device, dtype=torch.long)
        )
        if is_main_process:
            val_count += int(gathered_batch_size.sum().item())

        for k, v in output_batch.items():
            if not k.startswith("metric/"):
                continue
            gathered_metric = accelerator.gather_for_metrics(v)
            if is_main_process:
                metric_sums[k] += gathered_metric.float().sum().item()
                metric_counts[k] += gathered_metric.numel()

    # Each rank streamed its own shard above; flush/close, then rank 0 merges.
    # Sharding avoids cross-rank gather of ragged records; we dedup on
    # (scene_id, t0_us) since the distributed sampler pads the last batch.
    if save_predictions:
        pred_shard_file.close()
        accelerator.wait_for_everyone()

    if not is_main_process:
        return

    final_metrics_dict = {}
    for key in metric_sums.keys():
        if metric_counts[key] > 0:
            final_metrics_dict["val/" + key] = metric_sums[key] / metric_counts[key]

    padding = 15
    if len(final_metrics_dict) > 0:
        padding = max(padding, *(len(k) for k in final_metrics_dict.keys()))
    logger.info(
        f"Validation @ iteration\n"
        f"{'val/count':<{padding}} {val_count:.4f}\n"
        + "\n".join(
            f"{k:<{padding}} {final_metrics_dict[k]:.4f}" for k in sorted(final_metrics_dict.keys())
        )
        + "\n"
    )

    if save_predictions:
        merged = {}
        shard_paths = sorted(glob.glob(os.path.join(predictions_dir, "predictions_rank*.pkl")))
        for shard_path in shard_paths:
            with open(shard_path, "rb") as f:
                while True:
                    try:
                        batch_records = pickle.load(f)
                    except (EOFError, pickle.UnpicklingError):
                        break  # end of shard, or truncated tail from a crash
                    for record in batch_records:
                        key = (record.get("scene_id"), record.get("t0_us"))
                        merged[key] = record  # dedup padded repeats
        merged_records = list(merged.values())
        merged_path = os.path.join(predictions_dir, "predictions.pt")
        torch.save(merged_records, merged_path)
        metrics_path = os.path.join(predictions_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({"val/count": val_count, **final_metrics_dict}, f, indent=2)
        for shard_path in shard_paths:
            os.remove(shard_path)
        logger.info(
            f"Saved {len(merged_records)} per-sample predictions to {merged_path} "
            f"and metrics to {metrics_path}"
        )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    evaluate()
