"""Trajectory-only SFT dataset + collate for Alpamayo 2 Super.

Loads PhysicalAI-AV clips through ``alpamayo2_super``'s own release loader
(``load_physical_aiavdataset`` + ``select_task_input``), builds a *training-mode*
(teacher-forced) conversation with ``build_conversation(generation_mode=False)``,
and derives a ``labels_mask`` that supervises only the requested components
(default: just ``traj_future`` -- no ground-truth Chain-of-Causation text
required, matching this fork's existing TruckDrive trajectory-only recipes for
Alpamayo-1.5).

Verified (2026-08-05) end to end against the real ``nvidia/Alpamayo2-Super``
checkpoint and a real PhysicalAI-AV clip: correct placeholder-token counts,
finite loss, and (with LoRA applied) correct gradient flow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from alpamayo2_super import helper
from alpamayo2_super.chat_template.conversation import build_conversation
from alpamayo2_super.input_profiles import select_task_input
from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo2_super.load_truckdrive import (
    DEFAULT_DATA_ROOT as TRUCKDRIVE_DEFAULT_DATA_ROOT,
    DEFAULT_VIEW_TO_ALPAMAYO,
    enumerate_val_windows,
    load_truckdrive_sample,
    prefilter_tok_recon,
    scenes_from_metainfo,
)
from alpamayo2_super.models.utils import SPECIAL_TOKENS


def fill_masks_between_special_tokens(
    start_token: str,
    end_token: str,
    input_ids: torch.Tensor,
    tokenizer: Any,
    labels_mask: torch.Tensor,
) -> torch.Tensor:
    """Set ``labels_mask`` True (inclusive) between paired start/end special tokens.

    Same technique as ``alpamayo-recipes/src/alpamayo/utils/get_label_mask.py``
    (proven there for Alpamayo-1.5); Alpamayo 2 Super's ``SPECIAL_TOKENS`` uses the
    same ``<label>_start`` / ``<label>_end`` naming convention, so the logic carries
    over directly.
    """
    start_idx = (input_ids == tokenizer.convert_tokens_to_ids(start_token)).nonzero()
    end_idx = (input_ids == tokenizer.convert_tokens_to_ids(end_token)).nonzero()
    for start, end in zip(start_idx, end_idx):
        labels_mask[start[0], start[1] : end[1] + 1] = True
    return labels_mask


def get_label_mask(
    input_ids: torch.Tensor, tokenizer: Any, label_components: list[str]
) -> torch.Tensor:
    """Build a labels_mask supervising the given components (e.g. ["traj_future"])."""
    labels_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for label in label_components:
        labels_mask = fill_masks_between_special_tokens(
            SPECIAL_TOKENS[f"{label}_start"],
            SPECIAL_TOKENS[f"{label}_end"],
            input_ids,
            tokenizer,
            labels_mask,
        )
    return labels_mask


class Alpamayo2TrajectoryDataset(Dataset):
    """Trajectory-only (no CoT) SFT samples for Alpamayo 2 Super.

    Args:
        manifest_path: JSON manifest with a top-level "samples" list of
            {"clip_id": str, "t0_us": int} entries (e.g.
            ``examples/validation_samples.json`` / ``public_golden_validation_samples.json``
            shipped with the alpamayo2 repo). Swap this for a full PAI clip-index
            enumeration once scaling past a handful of clips.
        model_config: The loaded ``Alpamayo2SuperConfig`` (drives tokens-per-traj,
            camera/frame-label settings, and checkpoint resolution for the processor).
        tokenizer: The model's tokenizer (``model.tokenizer``).
        label_components: Which assistant components to supervise. Default is
            trajectory-only; add "cot" only if samples carry real GT reasoning text
            in ``data["cot"]`` (not wired here -- see task-list follow-up).
    """

    def __init__(
        self,
        manifest_path: str,
        model_config: Any,
        tokenizer: Any,
        label_components: list[str] | None = None,
    ) -> None:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.samples = manifest["samples"]
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.processor = helper.get_processor(tokenizer, model_config)
        self.label_components = label_components or ["traj_future"]

    def __len__(self) -> int:
        return len(self.samples)

    def load_raw(self, idx: int) -> dict[str, Any]:
        """Load one manifest sample's raw (pre-tokenization) task input.

        Exposed separately from ``__getitem__`` so callers that need a
        *different* tokenization of the same clip -- e.g. rollout.py's
        generation-mode prompt (history fused, no GT future placeholder) --
        can reuse the clip-loading step without re-deriving it.
        """
        sample = self.samples[idx]
        source_data = load_physical_aiavdataset(sample["clip_id"], t0_us=int(sample["t0_us"]))
        return select_task_input(source_data, "trajectory")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        data = self.load_raw(idx)

        components_order = ["image", "traj_history", "prompt"] + list(self.label_components)
        messages = build_conversation(
            data=data,
            num_tokens_per_history_traj=self.model_config.tokens_per_history_traj,
            num_tokens_per_future_traj=self.model_config.tokens_per_future_traj,
            components_order=components_order,
            components_prompt=[],
            generation_mode=False,
            include_camera_ids=self.model_config.include_camera_ids,
            camera_ids=data["camera_indices"],
            include_frame_nums=self.model_config.frame_label == "frame_num",
        )
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            add_vision_id=False,
            continue_final_message=False,
        )
        images = data["image_frames"].flatten(0, 1)
        images = (images.float() / 255.0) if images.dtype == torch.uint8 else images.float()
        tokenized_data = dict(
            self.processor(
                text=text,
                images=images,
                videos=None,
                padding=False,
                return_tensors="pt",
                do_rescale=False,
            )
        )
        # Only input_ids/attention_mask carry a leading batch-of-1 dim to drop here
        # (collate_fn re-batches them). pixel_values/image_grid_thw are already
        # unbatched flattened-patch tensors (shape [n_patches, dim] / [n_images, 3]
        # with no per-sample batch dim at all) -- collate_fn concatenates those as-is.
        for key in ("input_ids", "attention_mask"):
            tokenized_data[key] = tokenized_data[key][0]

        labels_mask = get_label_mask(
            tokenized_data["input_ids"].unsqueeze(0), self.tokenizer, self.label_components
        )[0]

        return {
            "tokenized_data": tokenized_data,
            "traj_data": {
                "ego_history_xyz": data["ego_history_xyz"][0],
                "ego_history_rot": data["ego_history_rot"][0],
                "ego_future_xyz": data["ego_future_xyz"][0],
                "ego_future_rot": data["ego_future_rot"][0],
            },
            "labels_mask": labels_mask,
            # Raw camera frames, kept only for TrajectoryVizCallback's camera-grid
            # panel -- not part of the model's forward() input, so collate_fn
            # does not touch this key.
            "image_frames": data["image_frames"],
        }


class Alpamayo2TruckDriveDataset(Alpamayo2TrajectoryDataset):
    """Trajectory-only SFT samples sourced from TruckDrive, for Alpamayo 2 Super.

    Shares ``Alpamayo2TrajectoryDataset``'s tokenization pipeline
    (``build_conversation`` + processor + label mask) but replaces the
    PhysicalAI-AV manifest loader with ``load_truckdrive.load_truckdrive_sample``,
    with windows enumerated exactly as ``alpamayo.data.truckdrive.TruckDriveDataset
    ._build_index`` does for ``alpamayo-recipes``' TruckDrive SFT configs
    (``recipes/alpamayo1_5_sft/configs/sft_truckdrive_lora.yaml``): every
    ``t0_stride``-th pose step across each scene's valid timeline, scenes
    missing a requested view dropped, reversing/heavy-sideslip windows dropped.

    ``split="train"`` with ``split_metainfo`` set to the TruckDrive devkit's
    ``metainfo.json`` resolves to the identical scene set (union of the three
    official training-scene lists) used to LoRA-finetune Alpamayo 1.5 on
    TruckDrive -- just sourced through Alpamayo 2 Super's own loader/tokenizer/
    camera-slot mapping (``load_truckdrive.DEFAULT_VIEW_TO_ALPAMAYO``, identical
    5-view set).

    Args:
        split_metainfo: Path to the TruckDrive devkit's ``metainfo.json``.
        split: "train" (union of the 3 official training-scene lists), "val",
            or "test".
        max_scenes: Cap on number of scenes after split resolution (0 = no cap;
            useful for a quick smoke test before scaling to the full split).
        t0_stride: Sample one t0 every this many pose steps per scene (~10 Hz
            poses, so stride=10 ~= one window/second). Matches the class
            default `alpamayo.data.truckdrive.TruckDriveDataset` (and hence
            `sft_truckdrive_lora.yaml`, which does not override it) unless
            changed here.
        windows_cache: Optional path to cache the enumerated (scene_id, t0_s)
            list as JSON, AFTER the tok-recon prefilter (when enabled) has run
            -- so the cache always reflects the actual windows a training run
            would see. Enumeration is metadata-only (pose file + camera
            directory listing per scene, no image bytes) but still one S3
            round trip per scene per view -- worth caching once max_scenes=0
            (thousands of scenes). Put it on shared storage (EFS) to reuse
            across runs/instances.
        tok_recon_max_xy_m: If set, drop windows whose future round-trips
            badly through the frozen future-trajectory tokenizer (encode ->
            decode XY error above this many metres) -- matches
            ``sft_truckdrive.yaml``'s train-only ``tok_recon_max_xy_m`` filter
            for Alpamayo 1.5 (there: 1.0m). ``None`` (default) disables it.
            Deliberately leave this unset on the val split, same as that
            config -- dropping val windows would make the split non-comparable
            run to run.
        tok_recon_reduce: How to reduce the per-waypoint XY error to one
            per-window value before thresholding: "mean" (default), "max"
            (drop if ANY waypoint is bad), "median", or "p95".
    """

    def __init__(
        self,
        model_config: Any,
        tokenizer: Any,
        split_metainfo: str,
        split: str = "train",
        data_root: str = TRUCKDRIVE_DEFAULT_DATA_ROOT,
        backend: str = "s3",
        view_to_alpamayo: dict[str, str] | None = None,
        num_history_steps: int = 16,
        num_future_steps: int = 64,
        num_frames: int = 4,
        time_step: float = 0.1,
        image_time_step: float = 0.2,
        min_speed_mps: float = 0.5,
        t0_stride: int = 10,
        filter_reverse: bool = True,
        standstill_snap_mps: float | None = 0.5,
        max_scenes: int = 0,
        windows_cache: str | None = None,
        tok_recon_max_xy_m: float | None = None,
        tok_recon_reduce: str = "mean",
        label_components: list[str] | None = None,
        manifest_path: str | None = None,  # noqa: ARG002 -- absorbs sft_base's mandatory PAI-manifest key (unused here)
    ) -> None:
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.processor = helper.get_processor(tokenizer, model_config)
        self.label_components = label_components or ["traj_future"]

        self.data_root = data_root
        self.backend = backend
        self.view_to_alpamayo = dict(view_to_alpamayo or DEFAULT_VIEW_TO_ALPAMAYO)
        self.num_history_steps = num_history_steps
        self.num_future_steps = num_future_steps
        self.num_frames = num_frames
        self.time_step = time_step
        self.image_time_step = image_time_step
        self.min_speed_mps = min_speed_mps
        self.standstill_snap_mps = standstill_snap_mps

        if windows_cache is not None and Path(windows_cache).is_file():
            cached = json.loads(Path(windows_cache).read_text(encoding="utf-8"))
            self.windows: list[tuple[str, float]] = [(s, t) for s, t in cached]
        else:
            scene_ids = scenes_from_metainfo(split_metainfo, split)
            if max_scenes:
                scene_ids = scene_ids[:max_scenes]
            self.windows = enumerate_val_windows(
                scene_ids,
                data_root=data_root,
                backend=backend,
                view_to_alpamayo=self.view_to_alpamayo,
                num_history_steps=num_history_steps,
                num_future_steps=num_future_steps,
                num_frames=num_frames,
                time_step=time_step,
                image_time_step=image_time_step,
                min_speed_mps=min_speed_mps,
                t0_stride=t0_stride,
                filter_reverse=filter_reverse,
            )
            if tok_recon_max_xy_m is not None:
                import hydra.utils as hyu

                # Deterministic bin-based quantizer, no checkpoint weights --
                # cheap to build standalone from model_config, same as the
                # model's own self.future_traj_tokenizer.
                recon_tokenizer = hyu.instantiate(model_config.future_traj_tokenizer_cfg)
                self.windows = prefilter_tok_recon(
                    self.windows,
                    recon_tokenizer,
                    data_root=data_root,
                    backend=backend,
                    num_history_steps=num_history_steps,
                    num_future_steps=num_future_steps,
                    time_step=time_step,
                    min_speed_mps=min_speed_mps,
                    standstill_snap_mps=standstill_snap_mps,
                    max_xy_m=tok_recon_max_xy_m,
                    reduce=tok_recon_reduce,
                )
            if windows_cache is not None:
                Path(windows_cache).parent.mkdir(parents=True, exist_ok=True)
                Path(windows_cache).write_text(json.dumps(self.windows), encoding="utf-8")

    def __len__(self) -> int:
        return len(self.windows)

    def load_raw(self, idx: int) -> dict[str, Any]:
        scene_id, t0_s = self.windows[idx]
        return load_truckdrive_sample(
            scene_id=scene_id,
            t0_s=t0_s,
            data_root=self.data_root,
            backend=self.backend,
            num_history_steps=self.num_history_steps,
            num_future_steps=self.num_future_steps,
            num_frames=self.num_frames,
            time_step=self.time_step,
            image_time_step=self.image_time_step,
            min_speed_mps=self.min_speed_mps,
            view_to_alpamayo=self.view_to_alpamayo,
            standstill_snap_mps=self.standstill_snap_mps,
            include_calibration=False,
        )


def collate_fn(batch: list[dict[str, Any]], pad_token_id: int = 0) -> dict[str, Any]:
    """Left-pad variable-length samples into one batch.

    Alpamayo2SuperConfig defaults to ``padding_side="left"`` (see config.py), so
    padding is added on the left to match the checkpoint's expected layout.
    ``pixel_values``/``image_grid_thw`` are concatenated along dim 0 (Qwen-VL's
    "flattened patches" convention -- these are NOT per-sample fixed-size tensors).
    """
    max_len = max(sample["tokenized_data"]["input_ids"].shape[0] for sample in batch)

    input_ids, attention_mask, labels_mask = [], [], []
    pixel_values, image_grid_thw = [], []
    ego_history_xyz, ego_history_rot, ego_future_xyz, ego_future_rot = [], [], [], []

    for sample in batch:
        td = sample["tokenized_data"]
        seq_len = td["input_ids"].shape[0]
        pad_len = max_len - seq_len

        input_ids.append(
            torch.cat([torch.full((pad_len,), pad_token_id, dtype=td["input_ids"].dtype), td["input_ids"]])
        )
        attention_mask.append(
            torch.cat([torch.zeros(pad_len, dtype=td["attention_mask"].dtype), td["attention_mask"]])
        )
        labels_mask.append(
            torch.cat([torch.zeros(pad_len, dtype=torch.bool), sample["labels_mask"]])
        )
        pixel_values.append(td["pixel_values"])
        image_grid_thw.append(td["image_grid_thw"])

        traj = sample["traj_data"]
        ego_history_xyz.append(traj["ego_history_xyz"])
        ego_history_rot.append(traj["ego_history_rot"])
        ego_future_xyz.append(traj["ego_future_xyz"])
        ego_future_rot.append(traj["ego_future_rot"])

    return {
        "tokenized_data": {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "pixel_values": torch.cat(pixel_values, dim=0),
            "image_grid_thw": torch.cat(image_grid_thw, dim=0),
        },
        "traj_data": {
            "ego_history_xyz": torch.stack(ego_history_xyz),
            "ego_history_rot": torch.stack(ego_history_rot),
            "ego_future_xyz": torch.stack(ego_future_xyz),
            "ego_future_rot": torch.stack(ego_future_rot),
        },
        "labels_mask": torch.stack(labels_mask),
    }
