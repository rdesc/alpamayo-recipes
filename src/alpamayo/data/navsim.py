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
no scene discovery, no S3 backend -- just image decoding (deferred to
``__getitem__``, same as every other dataset here) and the same sample-dict
contract as TruckDrive/PAI. It does share TruckDrive's tokenizer-reconstruction
PRE-FILTER (``tok_recon_max_xy_m``): encode->decode each window's future
through the frozen future tokenizer at init and drop windows whose XY
round-trip error exceeds the threshold. Off by default (``None``).

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
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset

from alpamayo_r1.common import logging

logger = logging.RankedLogger(__name__, rank_zero_only=True)


def _get_cfg(model_config: Any, key: str) -> Any:
    """Read a sub-config by key from a model config (OmegaConf node, dataclass,
    or plain dict), returning ``None`` if absent. Mirrors
    ``alpamayo.data.truckdrive._get_cfg`` -- duplicated rather than imported so
    this module stays independent of TruckDrive's (per the module docstring)."""
    if model_config is None:
        return None
    if isinstance(model_config, dict):
        return model_config.get(key)
    val = getattr(model_config, key, None)
    if val is None and OmegaConf.is_config(model_config):
        val = OmegaConf.select(model_config, key)
    return val


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
        tok_recon_max_xy_m: float | None = None,
        tok_recon_reduce: str = "mean",
        tok_recon_chunk: int = 512,
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
            tok_recon_max_xy_m: If set, enable the SAME trajectory-tokenizer
                PRE-FILTER as ``TruckDriveDataset``: encode->decode each
                window's future through the frozen future tokenizer and drop
                windows whose XY round-trip error exceeds this many metres.
                Requires ``model_config.traj_tokenizer_cfg`` (logs a warning
                and skips filtering otherwise). ``None`` (default) disables
                it -- every cached window is kept.
            tok_recon_reduce: How to reduce the per-waypoint XY error to one
                value per window before thresholding: mean/max/median/p95.
            tok_recon_chunk: Batch size for the vectorised encode->decode pass.
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

        if tok_recon_max_xy_m is not None:
            self._prefilter_tok_recon(
                model_config, tok_recon_max_xy_m, tok_recon_reduce, tok_recon_chunk
            )

    def __len__(self) -> int:
        return len(self.tokens)

    @torch.no_grad()
    def _prefilter_tok_recon(
        self, model_config: Any, threshold: float, reduce: str, chunk: int
    ) -> None:
        """Drop windows whose future round-trips badly through the trajectory
        tokenizer (encode -> decode XY error above ``threshold`` metres).

        Same mechanism as ``TruckDriveDataset._prefilter_tok_recon``, applied to
        whichever split (train/val) is already selected in ``self.tokens``.
        Pose-only (no image loads); the encode/decode runs vectorised in chunks.
        """
        if reduce not in ("mean", "max", "median", "p95"):
            raise ValueError(f"tok_recon_reduce must be mean/max/median/p95, got {reduce!r}")
        tok_cfg = _get_cfg(model_config, "traj_tokenizer_cfg")
        if tok_cfg is None:
            logger.warning(
                "tok_recon_max_xy_m=%s set but model_config has no traj_tokenizer_cfg -- "
                "tokenizer-reconstruction PRE-FILTER DISABLED (pass the model config so "
                "the future tokenizer can be built).",
                threshold,
            )
            return
        tokenizer = instantiate(tok_cfg)

        kept: list[str] = []
        errs: list[float] = []
        n_fail = 0
        chunk = max(1, int(chunk))

        def _flush(tok_batch: list[str]) -> None:
            nonlocal n_fail
            hx, hr, fx, fr = [], [], [], []
            for token in tok_batch:
                payload = self.samples[token]
                past_pose = payload["past_pose"]
                future_pose = payload["future_pose"]
                hx.append(
                    np.concatenate(
                        [past_pose[:, :2], np.zeros((len(past_pose), 1), dtype=np.float32)],
                        axis=-1,
                    )
                )
                fx.append(
                    np.concatenate(
                        [future_pose[:, :2], np.zeros((len(future_pose), 1), dtype=np.float32)],
                        axis=-1,
                    )
                )
                hr.append(_rot_z(past_pose[:, 2].astype(np.float64)))
                fr.append(_rot_z(future_pose[:, 2].astype(np.float64)))
            hx_t = torch.from_numpy(np.stack(hx)).float()
            hr_t = torch.from_numpy(np.stack(hr)).float()
            fx_t = torch.from_numpy(np.stack(fx)).float()
            fr_t = torch.from_numpy(np.stack(fr)).float()
            try:
                tokens = tokenizer.encode(hist_xyz=hx_t, hist_rot=hr_t, fut_xyz=fx_t, fut_rot=fr_t)
                rec, _, _ = tokenizer.decode(hist_xyz=hx_t, hist_rot=hr_t, tokens=tokens)
            except Exception:  # noqa: BLE001
                # A tokenizer failure is itself a sign of a pathological window;
                # keep them (fail-open) rather than silently nuke a whole chunk,
                # but count so a systematic problem is visible in the logs.
                n_fail += len(tok_batch)
                kept.extend(tok_batch)
                return
            e = torch.linalg.norm(rec[..., :2] - fx_t[..., :2], dim=-1)  # (B, F)
            red = self._reduce_recon_err(e, reduce)
            for token, r in zip(tok_batch, red.tolist()):
                errs.append(r)
                if r <= threshold:
                    kept.append(token)

        buf: list[str] = []
        for token in self.tokens:
            buf.append(token)
            if len(buf) >= chunk:
                _flush(buf)
                buf = []
        if buf:
            _flush(buf)

        n_drop = len(self.tokens) - len(kept)
        e_arr = np.asarray(errs, dtype=np.float64)
        stats = ""
        if e_arr.size:
            stats = (
                f" [{reduce} recon err over kept+dropped: "
                f"median={np.median(e_arr):.3f}m p95={np.quantile(e_arr, 0.95):.3f}m "
                f"max={e_arr.max():.3f}m]"
            )
        logger.info(
            "NavsimDataset: tok-recon pre-filter dropped %d/%d windows "
            "(threshold %s=%.3fm)%s",
            n_drop, len(self.tokens), reduce, threshold, stats,
        )
        if n_fail:
            logger.warning(
                "NavsimDataset: tok-recon pre-filter: %d windows errored during "
                "encode/decode and were kept (fail-open).", n_fail,
            )
        if not kept:
            raise RuntimeError(
                "All NAVSIM windows were dropped by the tokenizer-reconstruction "
                f"pre-filter -- tok_recon_max_xy_m={threshold} is likely too aggressive."
            )
        self.tokens = kept

    @staticmethod
    def _reduce_recon_err(e: torch.Tensor, reduce: str) -> torch.Tensor:
        """Reduce per-waypoint XY error ``(B, F)`` to one value per window ``(B,)``."""
        if reduce == "mean":
            return e.mean(dim=-1)
        if reduce == "max":
            return e.amax(dim=-1)
        if reduce == "median":
            return e.median(dim=-1).values
        return torch.quantile(e, 0.95, dim=-1)  # "p95"

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
