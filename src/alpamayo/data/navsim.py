# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""NAVSIM navtrain dataset for Alpamayo 1.5 SFT.

Loads ``navtrain_finetune_cache.pkl``, built by
``navsim/scripts/export_navtrain_finetune_cache.py`` in the NAVSIM devkit repo
(py3.9, since nuplan-devkit cannot coexist with this py3.12 environment). That
script already did the heavy lifting NAVSIM-side: token selection (one window
per ``t0_stride`` seconds per log), pose extraction straight from the raw
per-log pickles (bypassing ``SceneLoader``'s 14-frame truncation to get a full
6.4s future), resampling onto Alpamayo's native 10Hz grid, and deriving
``nav_text`` the same way ``alpamayo1.5/eval_navsim.py`` does for inference.
So this Dataset is much thinner than ``alpamayo.data.truckdrive.TruckDriveDataset``:
no scene discovery, no S3 backend, no tokenizer-reconstruction pre-filter --
just image decoding (deferred to ``__getitem__``, same as every other dataset
here) and the same sample-dict contract as TruckDrive/PAI.

Frame conventions (already baked into the cache, restated here for reference):
poses are (x, y, heading) local to t0, FLU (x-forward, y-left), rear-axle
origin -- the convention Alpamayo expects, verified end-to-end against the
NAVSIM PDM scorer (docs/alpamayo_navsim.md's GT round-trip).
"""
from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from hydra.utils import instantiate
from PIL import Image
from torch.utils.data import Dataset


def _rot_z(yaw: np.ndarray) -> np.ndarray:
    """Yaw angles (N,) -> planar rotation matrices (N, 3, 3). Matches truckdrive._rot_z."""
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.zeros((len(yaw), 3, 3), dtype=np.float64)
    rot[:, 0, 0] = c
    rot[:, 0, 1] = -s
    rot[:, 1, 0] = s
    rot[:, 1, 1] = c
    rot[:, 2, 2] = 1.0
    return rot


# Native camera-frame timestamps relative to t0: 4 frames @ 2Hz, -1.5s..0s
# (matches NUM_HIST_NATIVE in export_navtrain_finetune_cache.py and
# alpamayo1_5/load_navsim.py's _NAVSIM_PAST_TIMES). Fixed across the whole
# cache -- NOT the same grid as past_times/future_times, which are the
# resampled 10Hz POSE grid (16/64 points); camera frames stay at their native
# 4-frame/2Hz cadence.
_CAM_FRAME_TIMES = np.array([-1.5, -1.0, -0.5, 0.0])


class NavsimDataset(Dataset):
    """NAVSIM navtrain RGB + ego-trajectory dataset for Alpamayo SFT.

    Each item is one pre-selected (log, t0) window from the cache.
    ``__getitem__`` returns the standard Alpamayo sample dict
    (``image_frames``, ``camera_indices``, ``relative_timestamps``,
    ``ego_history_xyz/rot``, ``ego_future_xyz/rot``, ``nav_text``, plus
    ``tokenized_data`` when a preprocessor is configured) -- the same contract
    ``TruckDriveDataset``/``PAIDatasetWithNav`` produce.
    """

    def __init__(
        self,
        cache_path: str,
        model_config: Any | None = None,
        vla_preprocess_args: dict | None = None,
        nav_dropout_prob: float = 0.0,
        split: str = "train",
        val_fraction: float = 0.05,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialise the dataset.

        Args:
            cache_path: Path to ``navtrain_finetune_cache.pkl``.
            model_config: Model config forwarded to the VLA preprocessor.
            vla_preprocess_args: Hydra config to instantiate the preprocessor;
                its result is stored as ``tokenized_data`` per sample.
            nav_dropout_prob: Probability of dropping ``nav_text`` to None on a
                given draw (classifier-free-guidance-style route-conditioning
                dropout), applied fresh on every ``__getitem__`` call -- NOT baked into
                the cache, so the same token sees both conditioned and
                unconditioned draws across epochs. 0.0 (default) never drops;
                set e.g. 0.5 for "nav command enabled half the time". Val
                datasets should normally use 0.0 so evaluation is deterministic.
            split / val_fraction / seed: Deterministic LOG-level train/val
                split (random shuffle of log_names, seeded) -- whole logs go
                to one side or the other, never split within a log. Tokens
                within a log are only ~2s apart (t0_stride) and heavily
                overlap in history/future/scene content, so a token-level
                shuffle would leak near-duplicate windows across the split
                and inflate val metrics. val_fraction is applied to the
                number of distinct logs, not tokens, so the resulting val
                token count is close to but not exactly val_fraction of the
                total (logs have variable window counts).
        """
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)

        self.camera_views = cache["camera_views"]
        self.past_times = np.asarray(cache["past_times"], dtype=np.float64)
        self.future_times = np.asarray(cache["future_times"], dtype=np.float64)
        self.samples: dict[str, dict] = cache["tokens"]

        log_names = sorted({p["log_name"] for p in self.samples.values()})
        rng = random.Random(seed)
        rng.shuffle(log_names)
        n_val_logs = max(1, round(len(log_names) * val_fraction))
        val_logs = set(log_names[:n_val_logs])

        all_tokens = sorted(self.samples.keys())
        in_val = split == "val"
        self.tokens = [
            t for t in all_tokens if (self.samples[t]["log_name"] in val_logs) == in_val
        ]

        self.nav_dropout_prob = nav_dropout_prob

        self.vla_preprocess_func: Callable[..., Any] | None = None
        if vla_preprocess_args is not None:
            self.vla_preprocess_func = instantiate(vla_preprocess_args, model_config=model_config)

    def __len__(self) -> int:
        return len(self.tokens)

    def _load_camera_frames(self, payload: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build (image_frames, camera_indices, relative_timestamps) by decoding the cached JPEG paths."""
        cam_paths = payload["cam_paths"]  # (N_cam, 4) str
        cam_names = payload["cam_names"]  # (N_cam,) int64, already alpamayo indices

        frames = np.stack(
            [
                np.stack([np.asarray(Image.open(p).convert("RGB")) for p in cam_row])
                for cam_row in cam_paths
            ]
        )  # (N_cam, 4, H, W, 3)
        image_frames = torch.from_numpy(frames).permute(0, 1, 4, 2, 3).contiguous()  # (N_cam, 4, 3, H, W)
        camera_indices = torch.from_numpy(np.asarray(cam_names, dtype=np.int64))

        order = torch.argsort(camera_indices)
        image_frames = image_frames[order]
        camera_indices = camera_indices[order]

        relative_timestamps = (
            torch.from_numpy(_CAM_FRAME_TIMES.astype(np.float32)).unsqueeze(0).repeat(len(camera_indices), 1)
        )
        return image_frames, camera_indices, relative_timestamps

    def __getitem__(self, idx: int) -> dict[str, Any]:
        token = self.tokens[idx]
        payload = self.samples[token]

        image_frames, camera_indices, relative_timestamps = self._load_camera_frames(payload)

        past_pose = payload["past_pose"]  # (16, 3) x, y, heading
        future_pose = payload["future_pose"]  # (64, 3) x, y, heading
        hist_xyz = np.concatenate([past_pose[:, :2], np.zeros((len(past_pose), 1), dtype=np.float32)], axis=-1)
        fut_xyz = np.concatenate([future_pose[:, :2], np.zeros((len(future_pose), 1), dtype=np.float32)], axis=-1)
        hist_rot = _rot_z(past_pose[:, 2].astype(np.float64))
        fut_rot = _rot_z(future_pose[:, 2].astype(np.float64))

        nav_text = payload["nav_text"]
        if nav_text is not None and self.nav_dropout_prob > 0 and random.random() < self.nav_dropout_prob:
            nav_text = None

        sample_data: dict[str, Any] = {
            "image_frames": image_frames,
            "camera_indices": camera_indices,
            "relative_timestamps": relative_timestamps,
            "ego_history_xyz": torch.from_numpy(hist_xyz).float().unsqueeze(0),
            "ego_history_rot": torch.from_numpy(hist_rot).float().unsqueeze(0),
            "ego_future_xyz": torch.from_numpy(fut_xyz).float().unsqueeze(0),
            "ego_future_rot": torch.from_numpy(fut_rot).float().unsqueeze(0),
            "nav_text": nav_text,
            "t0_us": payload["t0_timestamp_us"],
            "clip_id": token,
            "scene_id": payload["log_name"],
        }

        if self.vla_preprocess_func is not None:
            sample_data["tokenized_data"] = self.vla_preprocess_func(data=sample_data)

        return sample_data
