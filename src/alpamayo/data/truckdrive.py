# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""TruckDrive RGB dataset for Alpamayo trajectory fine-tuning.

Loads the ``TruckDrivePublic`` dataset (per-scene camera JPEGs + a single
``poses/gt_trajectory.txt`` per scene) and emits the exact sample dict the
Alpamayo SFT stack expects -- the same contract produced by
``load_physical_aiavdataset`` for PAI (see ``src/alpamayo/data/pai.py``) and by
``LingoQADataset`` for VQA.

Layout under each ``scene_*`` directory::

    camera/leopard/<view>/images/<SYNC_KEY>_<ts>.jpg   # per-view RGB frames
    poses/gt_trajectory.txt                            # ego pose over time
    annotations/ calibrations/ lidar/ radar/ ...       # unused here

``poses/gt_trajectory.txt`` columns (whitespace-separated, two header lines)::

    SYNC_KEY  t_sec  x  y  z  qx  qy  qz  qw

Poses are ~10 Hz in a fixed scene frame (frame-0 at the origin). We resample
them onto the frozen trajectory tokenizer's convention -- ``num_history_steps``
points ending at ``t0`` and ``num_future_steps`` points after ``t0``, spaced
``time_step`` seconds -- and re-express them in the **ego frame at t0**
(``xyz_local = R_t0^{-1} (xyz - xyz_t0)``), exactly as the PAI loader does.

Design notes
------------
* **Reads are pluggable.** ``backend="local"`` reads scene files from a local
  directory root (staged data *or* a mounted ``mount-s3`` filesystem).
  ``backend="s3"`` streams objects straight from S3 via ``s3torchconnector``
  (imported lazily; ``pip install s3torchconnector`` only if you use it). This
  scaffolding follows a colleague's fork-safe ``s3torchconnector`` loader.
* **Camera mapping is explicit.** TruckDrive's Leopard view names are mapped to
  Alpamayo's fixed camera slots (``constants.CAMERA_NAMES_TO_INDICES``), which
  drive learned per-camera embeddings, via ``camera_view_to_alpamayo``. Defaults
  to the single forward camera -> ``front_wide`` (mirrors ``LingoQADataset``).
  Multi-view training additionally requires resizing views to a common H/W.
* **The trajectory tokenizer is frozen** and expects per-``time_step`` deltas
  inside ``ego_xyz_min/max`` (default +/- 4 m). Verify the encode->decode
  round-trip on your scenes before training (see ``validate_truckdrive.py``).
"""

from __future__ import annotations

import json
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Callable, Sequence

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
from torch.utils.data import Dataset

import alpamayo.common.constants as constants
from alpamayo_r1.common import logging

logger = logging.RankedLogger(__name__, rank_zero_only=False)

_IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# Default mapping from TruckDrive Leopard view names to Alpamayo camera slots.
# The value must be a key of constants.CAMERA_NAMES_TO_INDICES. This is a
# deliberate modeling choice (it selects which pretrained camera embedding the
# view borrows) and the 15->7 assignment below is a best-effort HEURISTIC by
# rough viewing direction -- verify / override per experiment via the
# ``camera_view_to_alpamayo`` arg. Several views intentionally share a slot.
DEFAULT_VIEW_TO_ALPAMAYO: dict[str, str] = {
    # Forward
    "forward_center_medium": constants.FRONT_WIDE_CAMERA_NAME,
    "forward_left_medium": constants.CROSS_LEFT_CAMERA_NAME,
    "forward_left_narrow": constants.FRONT_TELE_CAMERA_NAME,
    "forward_left_wide": constants.CROSS_LEFT_CAMERA_NAME,
    "forward_right_medium": constants.CROSS_RIGHT_CAMERA_NAME,
    "forward_right_narrow": constants.FRONT_TELE_CAMERA_NAME,
    "forward_right_wide": constants.CROSS_RIGHT_CAMERA_NAME,
    # Rearward
    "rearward_left_bottom_medium": constants.REAR_LEFT_CAMERA_NAME,
    "rearward_left_top_medium": constants.REAR_LEFT_CAMERA_NAME,
    "rearward_right_bottom_medium": constants.REAR_RIGHT_CAMERA_NAME,
    "rearward_right_top_medium": constants.REAR_RIGHT_CAMERA_NAME,
    # Sideward
    "sideward_left_back_wide": constants.REAR_LEFT_CAMERA_NAME,
    "sideward_left_front_wide": constants.CROSS_LEFT_CAMERA_NAME,
    "sideward_right_back_wide": constants.REAR_RIGHT_CAMERA_NAME,
    "sideward_right_front_wide": constants.CROSS_RIGHT_CAMERA_NAME,
}

# All 15 TruckDrive camera views, grouped for convenience.
ALL_CAMERA_VIEWS: tuple[str, ...] = tuple(DEFAULT_VIEW_TO_ALPAMAYO)

# Official TruckDrive devkit split keys (metainfo.json). The three training
# lists differ only in 3-D *box* annotation style, which trajectory training
# doesn't use -- all three are trainable for us. Val/test hold out whole
# collection batches (scene groups), the devkit's leak protection.
_METAINFO_TRAIN_KEYS = (
    "sequentially_labelled_training_scenes",
    "non_sequentially_labelled_training_scenes",
    "unlabelled_training_scenes",
)


def _scenes_from_metainfo(path: str, split: str) -> list[str]:
    """Resolve scene IDs for ``split`` from the devkit's official metainfo.json."""
    with open(path) as f:
        meta = json.load(f)
    if split == "train":
        ids = [s for k in _METAINFO_TRAIN_KEYS for s in meta[k]]
    elif split == "val":
        ids = list(meta["validation_scenes"])
    elif split == "test":
        ids = list(meta["test_scenes"])
    else:
        raise ValueError(
            f"split={split!r} invalid with split_metainfo (use 'train'/'val'/'test')"
        )
    return sorted(set(ids))


# --------------------------------------------------------------------------- #
# Read backends
# --------------------------------------------------------------------------- #
class _Backend:
    """Reads bytes and lists files under a dataset-relative path."""

    def list_files(self, rel_dir: str) -> list[str]:
        raise NotImplementedError

    def list_subdirs(self, rel_dir: str) -> list[str]:
        raise NotImplementedError

    def read_bytes(self, rel_path: str) -> bytes:
        raise NotImplementedError

    def exists(self, rel_path: str) -> bool:
        raise NotImplementedError


class _LocalBackend(_Backend):
    """Reads from a local directory root (staged files or a FUSE mount)."""

    def __init__(self, root: str) -> None:
        self._root = root.rstrip("/")

    def _abs(self, rel: str) -> str:
        return os.path.join(self._root, rel.strip("/"))

    def list_files(self, rel_dir: str) -> list[str]:
        try:
            return sorted(e.name for e in os.scandir(self._abs(rel_dir)) if e.is_file())
        except OSError:
            return []

    def list_subdirs(self, rel_dir: str) -> list[str]:
        try:
            return sorted(e.name for e in os.scandir(self._abs(rel_dir)) if e.is_dir())
        except OSError:
            return []

    def read_bytes(self, rel_path: str) -> bytes:
        with open(self._abs(rel_path), "rb") as f:
            return f.read()

    def exists(self, rel_path: str) -> bool:
        return os.path.exists(self._abs(rel_path))


class _S3Backend(_Backend):
    """Streams objects from S3 via ``s3torchconnector`` (lazy, fork-safe).

    The native S3 client is not fork-safe, so it is created lazily per process
    and dropped from the pickled state (see ``TruckDriveDataset.__getstate__``).
    """

    def __init__(self, bucket: str, prefix: str, region: str = "us-east-1") -> None:
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._region = region
        self._client = None  # created lazily; not fork-safe

    def _get_client(self):
        if self._client is None:
            try:
                from s3torchconnector._s3client import S3Client  # noqa: E402
            except ImportError as exc:  # pragma: no cover - optional dep
                raise ImportError(
                    "backend='s3' requires s3torchconnector "
                    "(`pip install s3torchconnector`)."
                ) from exc
            self._client = S3Client(region=self._region)
        return self._client

    def _full_prefix(self, rel: str) -> str:
        rel = rel.strip("/")
        base = f"{self._prefix}/{rel}" if rel else self._prefix
        return base + "/"

    def _retrying(self, desc: str, fn):
        """Run ``fn`` with bounded retries so a transient S3/network blip cannot
        kill a long training run. The client is dropped between attempts (it may
        hold a broken connection) and rebuilt lazily."""
        delay = 0.5
        for attempt in range(4):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - retry any transient failure
                if attempt == 3:
                    raise
                logger.warning(
                    "S3 %s failed (%r); retry %d/3 in %.1fs", desc, exc, attempt + 1, delay
                )
                self._client = None
                time.sleep(delay)
                delay *= 4

    def _pages(self, prefix: str):
        for page in self._get_client().list_objects(self._bucket, prefix, "/", 1000):
            yield page

    def list_files(self, rel_dir: str) -> list[str]:
        def _go() -> list[str]:
            prefix = self._full_prefix(rel_dir)
            names: list[str] = []
            for page in self._pages(prefix):
                for oi in page.object_info:
                    base = oi.key[len(prefix):]
                    if base and "/" not in base:
                        names.append(base)
            return sorted(set(names))

        return self._retrying(f"list {rel_dir!r}", _go)

    def list_subdirs(self, rel_dir: str) -> list[str]:
        def _go() -> list[str]:
            prefix = self._full_prefix(rel_dir)
            names: list[str] = []
            for page in self._pages(prefix):
                for cp in page.common_prefixes:
                    name = cp[len(prefix):].rstrip("/")
                    if name:
                        names.append(name)
            return sorted(set(names))

        return self._retrying(f"listdirs {rel_dir!r}", _go)

    def read_bytes(self, rel_path: str) -> bytes:
        key = f"{self._prefix}/{rel_path.strip('/')}"
        return self._retrying(
            f"get {key!r}",
            lambda: self._get_client().get_object(bucket=self._bucket, key=key).read(),
        )

    def exists(self, rel_path: str) -> bool:
        rel_path = rel_path.strip("/")
        parent, _, name = rel_path.rpartition("/")
        return name in self.list_files(parent)


def _make_backend(backend: str, data_root: str, region: str) -> _Backend:
    if backend == "local":
        return _LocalBackend(data_root)
    if backend == "s3":
        uri = data_root[len("s3://"):] if data_root.startswith("s3://") else data_root
        bucket, _, prefix = uri.partition("/")
        return _S3Backend(bucket=bucket, prefix=prefix, region=region)
    raise ValueError(f"unknown backend {backend!r} (use 'local' or 's3')")


# --------------------------------------------------------------------------- #
# Scene metadata cache
# --------------------------------------------------------------------------- #
# The expensive part of dataset init over S3 is *metadata*: one pose-file GET
# plus one LIST per camera view, per scene (~seconds x 3829 scenes, per rank).
# The cache stores exactly those raw inputs -- pose-file text + per-view image
# filename listings -- in one portable pickle (put it on EFS to share across
# instances). Everything else (sync-key joins, window index, reverse filtering,
# train/val split) is *derived* locally at init, so one cache file serves any
# combination of split/filter/horizon parameters.

_META_CACHE_VERSION = 1


def build_metadata_cache(
    data_root: str,
    cache_path: str,
    camera_views: Sequence[str],
    backend: str = "s3",
    region: str = "us-east-1",
    pose_file: str = "poses/gt_trajectory.txt",
    threads: int = 32,
    max_scenes: int = 0,
) -> dict:
    """Fetch pose text + per-view image listings for every scene and pickle them.

    Threaded with a per-thread backend (the native S3 client is not thread-safe).
    Unreadable scenes are skipped with a warning, mirroring index-build behaviour.
    Returns the loaded cache dict (see ``_META_CACHE_VERSION`` block layout).
    """
    root_backend = _make_backend(backend, data_root, region)
    scenes = sorted(s for s in root_backend.list_subdirs("") if s.startswith("scene_"))
    if not scenes:
        raise RuntimeError(f"No scene_* directories found under {data_root}")
    if max_scenes and max_scenes > 0:
        scenes = scenes[:max_scenes]

    tls = threading.local()

    def _thread_backend() -> _Backend:
        if getattr(tls, "backend", None) is None:
            tls.backend = _make_backend(backend, data_root, region)
        return tls.backend

    def _fetch(scene_id: str) -> tuple[str, dict | None]:
        try:
            be = _thread_backend()
            pose_text = be.read_bytes(f"{scene_id}/{pose_file}").decode("utf-8")
            view_files: dict[str, list[str]] = {}
            for view in camera_views:
                rel = f"{scene_id}/camera/leopard/{view}/images"
                view_files[view] = [
                    f for f in be.list_files(rel) if f.lower().endswith(_IMAGE_EXTS)
                ]
            return scene_id, {"pose_text": pose_text, "view_files": view_files}
        except Exception as exc:  # noqa: BLE001 - skip unreadable scenes
            logger.warning("metadata cache: skipping scene %s: %r", scene_id, exc)
            return scene_id, None

    entries: dict[str, dict] = {}
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(_fetch, s) for s in scenes]
        for i, fut in enumerate(as_completed(futures), 1):
            scene_id, entry = fut.result()
            if entry is not None:
                entries[scene_id] = entry
            if i % 200 == 0 or i == len(scenes):
                logger.info(
                    "metadata cache: %d/%d scenes (%.0fs elapsed)",
                    i, len(scenes), time.time() - t_start,
                )

    cache = {
        "version": _META_CACHE_VERSION,
        "data_root": data_root,
        "pose_file": pose_file,
        "camera_views": sorted(camera_views),
        "scenes": entries,
    }
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    tmp = cache_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cache_path)
    logger.info(
        "metadata cache: wrote %d scenes to %s (%.1f MB)",
        len(entries), cache_path, os.path.getsize(cache_path) / 1e6,
    )
    return cache


def _load_metadata_cache(
    cache_path: str,
    camera_views: Sequence[str],
    data_root: str,
    pose_file: str,
) -> dict[str, dict]:
    """Load + validate a metadata cache; returns the per-scene entries dict."""
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    if cache.get("version") != _META_CACHE_VERSION:
        raise ValueError(
            f"metadata cache {cache_path} has version {cache.get('version')!r}, "
            f"expected {_META_CACHE_VERSION}; rebuild it (truckdrive.build_cache)."
        )
    missing = [v for v in camera_views if v not in set(cache["camera_views"])]
    if missing:
        raise ValueError(
            f"metadata cache {cache_path} lacks views {missing}; it was built for "
            f"{cache['camera_views']}. Rebuild with the views you need."
        )
    if cache.get("data_root") != data_root:
        logger.warning(
            "metadata cache was built for data_root=%s but dataset uses %s; "
            "listings are keyed by scene layout so this is OK if both point at "
            "the same TruckDrive tree.",
            cache.get("data_root"), data_root,
        )
    if cache.get("pose_file") != pose_file:
        raise ValueError(
            f"metadata cache pose_file={cache.get('pose_file')!r} != {pose_file!r}; rebuild."
        )
    return cache["scenes"]


# --------------------------------------------------------------------------- #
# Pose file parsing + interpolation
# --------------------------------------------------------------------------- #
def _sync_key(basename: str) -> str:
    """``0034_200027227.jpg`` -> ``0034`` (the temporal alignment key)."""
    return basename.split("_", 1)[0]


class _ScenePoses:
    """Parsed ego trajectory for one scene, with time interpolation.

    Positions are linearly interpolated and rotations are spherically
    interpolated (Slerp) so that samples land exactly on the requested
    timestamps, matching the PAI ``egomotion`` interpolator semantics.
    """

    def __init__(self, text: str) -> None:
        sync_keys: list[str] = []
        t: list[float] = []
        xyz: list[list[float]] = []
        quat: list[list[float]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("refined"):
                continue
            f = line.split()
            if len(f) < 9:
                continue
            try:
                row_t = float(f[1])
                row_xyz = [float(f[2]), float(f[3]), float(f[4])]
                row_quat = [float(f[5]), float(f[6]), float(f[7]), float(f[8])]  # x,y,z,w
            except ValueError:
                # Header/junk line -- some scenes ship the column header without
                # a leading '#' ("SYNC_KEY, TIMESTAMP, ..."), which has 9 tokens
                # and previously aborted the whole scene.
                continue
            sync_keys.append(f[0])
            t.append(row_t)
            xyz.append(row_xyz)
            quat.append(row_quat)

        self.sync_keys = sync_keys
        self.t = np.asarray(t, dtype=np.float64)
        self.xyz = np.asarray(xyz, dtype=np.float64)
        self._rot = Rotation.from_quat(np.asarray(quat, dtype=np.float64))
        self._slerp = Slerp(self.t, self._rot) if len(t) >= 2 else None
        # sync_key -> t_sec, for joining camera frames onto the pose timeline.
        self.key_to_t = {k: self.t[i] for i, k in enumerate(sync_keys)}

        # Per-pose reverse signal: speed, and cos(angle) between travel direction
        # (position tangent) and body-forward (quaternion x-axis). cos < 0 means
        # the vehicle is moving backward relative to where it points (reversing).
        if len(t) >= 2:
            vel = np.gradient(self.xyz[:, :2], self.t, axis=0)  # (N, 2) m/s
            self.speed = np.linalg.norm(vel, axis=1)
            vel_dir = vel / (self.speed[:, None] + 1e-9)
            body_fwd = self._rot.as_matrix()[:, :2, 0]  # body x-axis in xy
            body_fwd /= np.linalg.norm(body_fwd, axis=1, keepdims=True) + 1e-9
            self.cos_vel_body = (vel_dir * body_fwd).sum(axis=1)
        else:
            self.speed = np.zeros(len(t))
            self.cos_vel_body = np.ones(len(t))

    def is_forward_window(
        self,
        t0: float,
        horizon: float,
        min_speed: float,
        reverse_angle_deg: float,
        max_reverse_fraction: float,
    ) -> bool:
        """True if the future span [t0, t0+horizon] is forward driving.

        A window is rejected when too large a fraction of its *moving* poses are
        travelling backward relative to body orientation (reverse / large
        sideslip). Near-stationary poses are ignored (heading undefined).
        """
        sel = (self.t >= t0) & (self.t <= t0 + horizon) & (self.speed >= min_speed)
        moving = self.cos_vel_body[sel]
        if moving.size == 0:
            return True  # no confident motion to judge -> don't reject as reverse
        reverse = moving < np.cos(np.radians(reverse_angle_deg))
        return reverse.mean() <= max_reverse_fraction

    @property
    def t_min(self) -> float:
        return float(self.t[0])

    @property
    def t_max(self) -> float:
        return float(self.t[-1])

    def sample(self, timestamps: np.ndarray) -> tuple[np.ndarray, Rotation]:
        """Interpolate ego pose at ``timestamps`` -> (xyz (N, 3), Rotation)."""
        ts = np.clip(timestamps, self.t_min, self.t_max)
        xyz = np.stack(
            [np.interp(ts, self.t, self.xyz[:, d]) for d in range(3)], axis=-1
        )
        rot = self._slerp(ts) if self._slerp is not None else self._rot[[0] * len(ts)]
        return xyz, rot


def _to_t0_frame(
    xyz: np.ndarray, rot: Rotation, t0_xyz: np.ndarray, t0_rot: Rotation
) -> tuple[np.ndarray, np.ndarray]:
    """Express world poses in the ego frame at t0 (PAI convention).

    Returns (xyz_local (N, 3), rot_local (N, 3, 3)).
    """
    inv = t0_rot.inv()
    xyz_local = inv.apply(xyz - t0_xyz)
    rot_local = (inv * rot).as_matrix()
    return xyz_local, rot_local


# Optional lateral reflection about the x-z plane (negate y). TruckDrive's
# gt_trajectory poses are already in Alpamayo's frame (x-forward, y-LEFT, z-up):
# processing a lane-following future through the front-camera intrinsics +
# extrinsics places it on the correct lane once the rig's own (mirrored) camera
# frame is accounted for -- see the projection check in ``validate_truckdrive``.
# This reflection is therefore OFF by default; it is an escape hatch for a future
# dataset that ships y-right poses. ``S = diag(1, -1, 1)`` maps positions
# ``p -> S p`` and rotations ``R -> S R S`` (still a proper rotation; yaw flips).
_FLU_REFLECT = np.diag([1.0, -1.0, 1.0]).astype(np.float64)


def _reflect_lateral(xyz: np.ndarray, rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reflect positions and rotations across the x-z plane (swap y sign)."""
    xyz_r = xyz @ _FLU_REFLECT
    rot_r = np.einsum("ij,njk,kl->nil", _FLU_REFLECT, rot, _FLU_REFLECT)
    return xyz_r, rot_r


def _rot_z(yaw: np.ndarray) -> np.ndarray:
    """Yaw angles (N,) -> planar rotation matrices (N, 3, 3)."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.zeros((len(yaw), 3, 3), dtype=np.float64)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    R[:, 2, 2] = 1.0
    return R


def _heading_from_positions(
    xyz: np.ndarray, time_step: float, min_speed: float
) -> np.ndarray:
    """Derive travel heading (yaw) from the position track.

    The gt_trajectory pose *quaternions* encode vehicle body orientation, which
    does NOT match the direction of travel; feeding them to the trajectory
    tokenizer yields metre-scale round-trip error, while heading taken from the
    position tangent yields ~cm error (see ``validate_truckdrive``). Below
    ``min_speed`` the tangent is ill-defined (parking-lot / stops), so the last
    confident heading is held.
    """
    d = np.gradient(xyz[:, :2], axis=0)  # per-sample velocity * time_step
    speed = np.linalg.norm(d, axis=1) / time_step
    yaw = np.arctan2(d[:, 1], d[:, 0])

    good = speed >= min_speed
    if good.any():
        # hold the last confident heading through low-speed gaps; back-fill the lead-in
        last = -1
        for i in range(len(good)):
            if good[i]:
                last = i
            elif last >= 0:
                yaw[i] = yaw[last]
        first = int(np.argmax(good))
        yaw[:first] = yaw[first]
    else:
        yaw[:] = 0.0
    return np.unwrap(yaw)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class TruckDriveDataset(Dataset):
    """TruckDrive RGB + ego-trajectory dataset for Alpamayo SFT.

    Each item is one ``(scene, t0)`` window. ``__getitem__`` returns the
    Alpamayo sample dict (``image_frames``, ``camera_indices``,
    ``relative_timestamps``, ``ego_history_xyz/rot``, ``ego_future_xyz/rot``,
    plus ``tokenized_data`` when a preprocessor is configured).
    """

    def __init__(
        self,
        data_root: str,
        model_config: Any | None = None,
        vla_preprocess_args: dict | None = None,
        camera_views: Sequence[str] = ("forward_center_medium",),
        camera_view_to_alpamayo: dict[str, str] | None = None,
        num_history_steps: int = 16,
        num_future_steps: int = 64,
        num_frames: int = 4,
        time_step: float = 0.1,
        image_time_step: float | None = None,
        t0_stride: int = 10,
        scene_ids: Sequence[str] | None = None,
        max_scenes: int = 0,
        split: str = "all",
        val_fraction: float = 0.0,
        seed: int = 0,
        backend: str = "local",
        region: str = "us-east-1",
        pose_file: str = "poses/gt_trajectory.txt",
        metadata_cache: str | None = None,
        cache_build_threads: int = 32,
        split_metainfo: str | None = None,
        flip_lateral: bool = False,
        heading_source: str = "position",
        min_speed_mps: float = 0.5,
        filter_reverse: bool = True,
        reverse_angle_deg: float = 90.0,
        max_reverse_fraction: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialise the dataset.

        Args:
            data_root: Local dir (``backend='local'``) or ``s3://bucket/prefix``
                (``backend='s3'``) containing ``scene_*`` directories.
            model_config: Model config forwarded to the VLA preprocessor.
            vla_preprocess_args: Hydra config to instantiate the preprocessor;
                its result is stored as ``tokenized_data`` per sample.
            camera_views: Leopard view names to load (default: forward camera).
            camera_view_to_alpamayo: view -> Alpamayo camera-slot name mapping.
                Defaults to ``DEFAULT_VIEW_TO_ALPAMAYO``.
            num_history_steps: Ego history points ending at t0.
            num_future_steps: Ego future points after t0 (tokenizer horizon).
            num_frames: Camera frames per view, ending at t0.
            time_step: Seconds between trajectory samples (tokenizer convention).
            image_time_step: Seconds between camera frames (defaults to
                ``time_step``); TruckDrive cameras are ~5 Hz so ~0.2 is natural.
            t0_stride: Sample one t0 every ``t0_stride`` pose steps per scene.
            scene_ids: Explicit scene subset; otherwise all discovered scenes.
            max_scenes: Cap on number of scenes (0 = no cap).
            split / val_fraction / seed: Deterministic scene-level train/val split.
            backend: ``'local'`` or ``'s3'``.
            region: AWS region for ``backend='s3'``.
            pose_file: Per-scene pose file, relative to the scene dir.
            metadata_cache: Optional path to a scene-metadata cache pickle (see
                ``build_metadata_cache``). If the file exists, dataset init makes
                **zero** S3 metadata calls (pose reads + frame listings come from
                the cache); if it does not exist, it is built (threaded) and
                saved. Put it on shared storage (EFS) to reuse across instances.
                Only image bytes are fetched from the backend at ``__getitem__``.
            cache_build_threads: Parallelism when building a missing cache.
            split_metainfo: Optional path to the TruckDrive devkit's
                ``metainfo.json``. When set, ``split`` selects the OFFICIAL
                scene lists ("train" = all three training lists, "val" =
                validation_scenes, "test" = test_scenes) and the random
                ``val_fraction``/``seed`` split is disabled. Prefer this over
                the random split: the official val/test hold out whole
                collection batches (no same-drive leakage) and keep test
                untouched for comparability.
            flip_lateral: Reflect the ego trajectory across the x-z plane (swap y
                sign). OFF by default -- TruckDrive poses are already in
                Alpamayo's y-LEFT frame. Escape hatch for a dataset that ships
                y-right poses.
            heading_source: ``"position"`` (default) derives ego heading from the
                position tangent -- required for faithful trajectory tokenization
                (the gt_trajectory quaternions are body orientation, not travel
                direction). ``"quaternion"`` uses the pose quaternions (only for
                a dataset whose orientation truly tracks heading).
            min_speed_mps: Below this speed the position tangent is ill-defined,
                so the last confident heading is held.
            filter_reverse: Drop windows whose future is reversing / heavily
                sideslipping (not well-posed for a forward-camera model, and
                out of the tokenizer's forward-driving domain). ON by default.
            reverse_angle_deg: A pose counts as "reverse" when its travel
                direction deviates from body-forward by more than this (90 =
                moving backward).
            max_reverse_fraction: Reject a window if more than this fraction of
                its moving future poses are reverse.
            **kwargs: Absorbs unused keys shared with other dataset configs.
        """
        super().__init__()
        self._backend_name = backend
        self._region = region
        self._data_root = data_root
        self.backend = _make_backend(backend, data_root, region)

        self.camera_views = list(camera_views)
        self.view_to_alpamayo = dict(camera_view_to_alpamayo or DEFAULT_VIEW_TO_ALPAMAYO)
        for view in self.camera_views:
            if view not in self.view_to_alpamayo:
                raise ValueError(
                    f"camera view {view!r} has no Alpamayo camera-slot mapping; "
                    f"add it to camera_view_to_alpamayo (known Alpamayo slots: "
                    f"{list(constants.CAMERA_NAMES_TO_INDICES)})"
                )

        self.num_history_steps = num_history_steps
        self.num_future_steps = num_future_steps
        self.num_frames = num_frames
        self.time_step = time_step
        self.image_time_step = image_time_step if image_time_step is not None else time_step
        self.t0_stride = max(1, int(t0_stride))
        self.pose_file = pose_file
        self.flip_lateral = flip_lateral
        self.heading_source = heading_source
        self.min_speed_mps = min_speed_mps
        self.filter_reverse = filter_reverse
        self.reverse_angle_deg = reverse_angle_deg
        self.max_reverse_fraction = max_reverse_fraction

        # Trajectory sample offsets (seconds), relative to t0.
        self._hist_offsets = np.array(
            [-(num_history_steps - 1 - i) * time_step for i in range(num_history_steps)],
            dtype=np.float64,
        )
        self._fut_offsets = np.array(
            [(i + 1) * time_step for i in range(num_future_steps)], dtype=np.float64
        )
        self._img_offsets = np.array(
            [-(num_frames - 1 - i) * self.image_time_step for i in range(num_frames)],
            dtype=np.float64,
        )

        self.vla_preprocess_func: Callable[..., Any] | None = None
        if model_config is not None and isinstance(model_config, dict):
            model_config = OmegaConf.create(model_config)
        if vla_preprocess_args is not None:
            self.vla_preprocess_func = instantiate(vla_preprocess_args, model_config=model_config)

        # Scene metadata cache: makes init S3-free (see build_metadata_cache).
        self.metadata_cache = metadata_cache
        self._meta: dict[str, dict] | None = None
        if metadata_cache is not None:
            if os.path.exists(metadata_cache):
                self._meta = _load_metadata_cache(
                    metadata_cache, self.camera_views, data_root, pose_file
                )
                logger.info(
                    "TruckDriveDataset: loaded metadata cache %s (%d scenes)",
                    metadata_cache, len(self._meta),
                )
            else:
                logger.info(
                    "TruckDriveDataset: metadata cache %s missing; building it "
                    "(threads=%d) -- prefer prebuilding via truckdrive.build_cache",
                    metadata_cache, cache_build_threads,
                )
                cache = build_metadata_cache(
                    data_root=data_root, cache_path=metadata_cache,
                    camera_views=self.camera_views, backend=backend, region=region,
                    pose_file=pose_file, threads=cache_build_threads,
                )
                self._meta = cache["scenes"]

        # Official devkit split: resolve explicit scene lists and disable the
        # random val_fraction split (the metainfo lists ARE the split).
        if split_metainfo is not None:
            if scene_ids is not None:
                raise ValueError("pass either scene_ids or split_metainfo, not both")
            scene_ids = _scenes_from_metainfo(split_metainfo, split)
            if self._meta is not None:
                available = set(self._meta)
                missing = sum(1 for s in scene_ids if s not in available)
                scene_ids = [s for s in scene_ids if s in available]
                if missing:
                    logger.info(
                        "TruckDriveDataset: %d official '%s' scenes absent from the "
                        "metadata cache (missing pose files in the bucket); using %d",
                        missing, split, len(scene_ids),
                    )
            split, val_fraction = "all", 0.0  # explicit list; no random sub-split

        self._scene_ids = self._discover_scenes(scene_ids, max_scenes, split, val_fraction, seed)
        # Per-scene state is loaded lazily and cached (keeps index build cheap).
        self._scene_cache: dict[str, dict] = {}
        self._samples = self._build_index()

        logger.info(
            "TruckDriveDataset: %d samples across %d scenes (views=%s, split=%s, backend=%s)",
            len(self._samples),
            len(self._scene_ids),
            self.camera_views,
            split,
            backend,
        )

    # --- pickling (drop non-fork-safe S3 client) -----------------------------
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if self._backend_name == "s3":
            state["backend"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if self._backend_name == "s3":
            self.backend = _make_backend("s3", self._data_root, self._region)

    # --- discovery + indexing ------------------------------------------------
    def _discover_scenes(
        self,
        scene_ids: Sequence[str] | None,
        max_scenes: int,
        split: str,
        val_fraction: float,
        seed: int,
    ) -> list[str]:
        if scene_ids is not None:
            scenes = sorted(scene_ids)
        elif self._meta is not None:
            scenes = sorted(self._meta.keys())  # cache is the scene inventory
        else:
            scenes = sorted(s for s in self.backend.list_subdirs("") if s.startswith("scene_"))
        if not scenes:
            raise RuntimeError(f"No scene_* directories found under {self._data_root}")

        if split != "all" and val_fraction > 0.0:
            perm = np.random.default_rng(seed).permutation(len(scenes))
            n_val = int(round(len(scenes) * min(1.0, max(0.0, val_fraction))))
            val_idx = set(int(i) for i in perm[:n_val])
            keep = val_idx if split == "val" else set(range(len(scenes))) - val_idx
            scenes = [s for i, s in enumerate(scenes) if i in keep]

        if max_scenes and max_scenes > 0:
            scenes = scenes[:max_scenes]
        return scenes

    def _load_scene(self, scene_id: str) -> dict:
        """Parse poses + list camera frames for one scene (cached).

        With a metadata cache, pose text and frame listings come from the cache
        (no backend calls); otherwise they are fetched from the backend. The
        sync-key join to the pose timeline is always recomputed locally.
        """
        if scene_id in self._scene_cache:
            return self._scene_cache[scene_id]

        meta = self._meta.get(scene_id) if self._meta is not None else None
        if meta is not None:
            text = meta["pose_text"]
        else:
            text = self.backend.read_bytes(f"{scene_id}/{self.pose_file}").decode("utf-8")
        poses = _ScenePoses(text)

        # Per view: ordered list of (t_sec, filename) for frames whose SYNC_KEY
        # exists on the pose timeline (so camera frames share the ego clock).
        view_frames: dict[str, tuple[np.ndarray, list[str]]] = {}
        for view in self.camera_views:
            if meta is not None:
                files = meta["view_files"].get(view, [])
            else:
                rel = f"{scene_id}/camera/leopard/{view}/images"
                files = [
                    f for f in self.backend.list_files(rel) if f.lower().endswith(_IMAGE_EXTS)
                ]
            timed: list[tuple[float, str]] = []
            for name in files:
                t = poses.key_to_t.get(_sync_key(name))
                if t is not None:
                    timed.append((float(t), name))
            timed.sort()
            if not timed:
                continue
            times = np.array([t for t, _ in timed], dtype=np.float64)
            names = [n for _, n in timed]
            view_frames[view] = (times, names)

        scene = {"poses": poses, "view_frames": view_frames}
        self._scene_cache[scene_id] = scene
        return scene

    def _build_index(self) -> list[tuple[str, float]]:
        """Enumerate valid ``(scene_id, t0_time)`` samples with time margins."""
        hist_margin = (self.num_history_steps - 1) * self.time_step
        fut_margin = self.num_future_steps * self.time_step
        img_margin = (self.num_frames - 1) * self.image_time_step
        back_margin = max(hist_margin, img_margin)

        fut_horizon = self.num_future_steps * self.time_step
        samples: list[tuple[str, float]] = []
        n_reverse = 0
        n_missing_views = 0
        for scene_id in self._scene_ids:
            try:
                scene = self._load_scene(scene_id)
            except Exception as exc:  # noqa: BLE001 - skip unreadable scenes
                logger.warning("skipping scene %s: %r", scene_id, exc)
                continue
            poses = scene["poses"]
            if len(poses.t) < 2:
                continue
            # Require ALL requested views: samples must be shape-homogeneous
            # (N_cam is a batch dim -- a 2-view sample cannot collate with a
            # 5-view one). ~15% of scenes lack some views entirely.
            if any(v not in scene["view_frames"] for v in self.camera_views):
                n_missing_views += 1
                continue
            lo = poses.t_min + back_margin
            hi = poses.t_max - fut_margin
            if hi <= lo:
                continue
            # One t0 per t0_stride pose steps, within the valid window.
            for i in range(0, len(poses.t), self.t0_stride):
                t0 = float(poses.t[i])
                if not (lo <= t0 <= hi):
                    continue
                if self.filter_reverse and not poses.is_forward_window(
                    t0, fut_horizon, self.min_speed_mps,
                    self.reverse_angle_deg, self.max_reverse_fraction,
                ):
                    n_reverse += 1
                    continue
                samples.append((scene_id, t0))
        if not samples:
            raise RuntimeError(
                "No valid TruckDrive samples: check that scenes have the requested "
                "camera views and enough pose length for the history/future horizon "
                "(and that they are not all reverse/maneuver windows)."
            )
        if self.filter_reverse and n_reverse:
            logger.info("TruckDriveDataset: filtered %d reverse/maneuver windows", n_reverse)
        if n_missing_views:
            logger.info(
                "TruckDriveDataset: skipped %d scenes lacking some of the %d "
                "requested camera views (samples must be shape-homogeneous)",
                n_missing_views, len(self.camera_views),
            )
        return samples

    # --- sample assembly -----------------------------------------------------
    def __len__(self) -> int:
        return len(self._samples)

    def _load_camera_frames(
        self, scene_id: str, scene: dict, t0: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build (image_frames, camera_indices, relative_timestamps)."""
        img_times = t0 + self._img_offsets
        frames_per_view: list[torch.Tensor] = []
        cam_idx_list: list[int] = []
        abs_ts_list: list[torch.Tensor] = []

        for view in self.camera_views:
            if view not in scene["view_frames"]:
                continue
            times, names = scene["view_frames"][view]
            # Nearest camera frame to each requested image timestamp.
            sel = np.searchsorted(times, img_times)
            sel = np.clip(sel, 0, len(times) - 1)
            for j, s in enumerate(sel):
                if s > 0 and abs(times[s - 1] - img_times[j]) < abs(times[s] - img_times[j]):
                    sel[j] = s - 1

            frames = []
            frame_ts = []
            for s in sel:
                # A single corrupt JPEG among millions must not kill a training
                # run: retry the fetch+decode once (transient truncation), then
                # substitute the nearest neighbouring frame (~0.2 s apart, near
                # identical content). Raise only if a whole neighbourhood fails.
                img = None
                for cand in (s, s, s - 1, s + 1, s - 2, s + 2):
                    if not 0 <= cand < len(names):
                        continue
                    rel = f"{scene_id}/camera/leopard/{view}/images/{names[cand]}"
                    try:
                        img = Image.open(BytesIO(self.backend.read_bytes(rel))).convert("RGB")
                        if cand != s:
                            logger.warning(
                                "substituted neighbour frame %s for unreadable %s",
                                names[cand], names[s],
                            )
                        break
                    except OSError as exc:
                        logger.warning("unreadable image %s: %r", rel, exc)
                        img = None
                if img is None:
                    raise RuntimeError(
                        f"no readable frame near {scene_id}/{view}/{names[s]}"
                    )
                arr = np.array(img)  # (H, W, 3) uint8 (writable copy)
                frames.append(torch.from_numpy(arr).permute(2, 0, 1))  # (3, H, W)
                frame_ts.append(float(times[s]))
            frames_per_view.append(torch.stack(frames, dim=0))  # (num_frames, 3, H, W)
            cam_name = self.view_to_alpamayo[view]
            cam_idx_list.append(constants.CAMERA_NAMES_TO_INDICES[cam_name])
            abs_ts_list.append(torch.tensor(frame_ts, dtype=torch.float64))

        image_frames = torch.stack(frames_per_view, dim=0)  # (N_cam, num_frames, 3, H, W)
        camera_indices = torch.tensor(cam_idx_list, dtype=torch.int64)  # (N_cam,)
        abs_ts = torch.stack(abs_ts_list, dim=0)  # (N_cam, num_frames)

        # Sort by camera index for a consistent slot ordering.
        order = torch.argsort(camera_indices)
        image_frames = image_frames[order]
        camera_indices = camera_indices[order]
        abs_ts = abs_ts[order]
        relative_timestamps = (abs_ts - abs_ts.min()).float()  # seconds
        return image_frames, camera_indices, relative_timestamps

    def __getitem__(self, idx: int) -> dict[str, Any]:
        scene_id, t0 = self._samples[idx]
        scene = self._load_scene(scene_id)
        poses: _ScenePoses = scene["poses"]

        # Sample the whole history+future window on one timeline so heading can
        # be derived from the contiguous position track. History offsets end at
        # t0 (index H-1); future offsets follow.
        H = self.num_history_steps
        all_offsets = np.concatenate([self._hist_offsets, self._fut_offsets])
        xyz_w, quat_rot = poses.sample(t0 + all_offsets)

        if self.heading_source == "position":
            yaw_w = _heading_from_positions(xyz_w, self.time_step, self.min_speed_mps)
            rot_w = _rot_z(yaw_w)
        else:
            rot_w = quat_rot.as_matrix()

        # Express in the ego frame at t0 (= index H-1): xyz (T, 3), rot (T, 3, 3).
        t0_xyz = xyz_w[H - 1]
        t0_rot_inv = rot_w[H - 1].T  # orthonormal -> inverse is transpose
        xyz_local = (xyz_w - t0_xyz) @ t0_rot_inv.T
        rot_local = np.einsum("ij,njk->nik", t0_rot_inv, rot_w)
        hist_xyz, fut_xyz = xyz_local[:H], xyz_local[H:]
        hist_rot, fut_rot = rot_local[:H], rot_local[H:]

        if self.flip_lateral:  # optional y-sign swap (default off; poses are FLU)
            hist_xyz, hist_rot = _reflect_lateral(hist_xyz, hist_rot)
            fut_xyz, fut_rot = _reflect_lateral(fut_xyz, fut_rot)

        image_frames, camera_indices, relative_timestamps = self._load_camera_frames(
            scene_id, scene, t0
        )

        sample_data: dict[str, Any] = {
            "image_frames": image_frames,
            "camera_indices": camera_indices,
            "relative_timestamps": relative_timestamps,
            "ego_history_xyz": torch.from_numpy(hist_xyz).float().unsqueeze(0),
            "ego_history_rot": torch.from_numpy(hist_rot).float().unsqueeze(0),
            "ego_future_xyz": torch.from_numpy(fut_xyz).float().unsqueeze(0),
            "ego_future_rot": torch.from_numpy(fut_rot).float().unsqueeze(0),
            "t0_us": int(round(t0 * 1e6)),
            "clip_id": scene_id,
            "scene_id": scene_id,
        }

        if self.vla_preprocess_func is not None:
            sample_data["tokenized_data"] = self.vla_preprocess_func(data=sample_data)

        return sample_data
