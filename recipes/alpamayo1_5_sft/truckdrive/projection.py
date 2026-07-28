# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Project an ego trajectory onto a TruckDrive camera image.

TruckDrive ships pinhole intrinsics (`calibrations/calib_camera_*.json`, `K` +
zero `plumb_bob` distortion) and an extrinsic TF tree
(`calibrations/calib_tf_tree_full.json`, `vehicle -> cab -> camera`). This module
turns those into an image-plane projection of an ego-frame trajectory.

Frame note (important): the loader / model trajectories are **FLU** (x-forward,
y-left, z-up), but this rig's TF *vehicle* frame is **FRU** (y-right). So we
negate y (`_FLU_TO_FRU`) before applying the extrinsics. Verified: with the flip,
the projected future tracks lanes and painted turn arrows and matches the
frame-independent turn direction of the path; without it, the overlay is
mirrored.
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

# Loader/model poses are FLU (y-left); the TF vehicle frame is FRU (y-right).
_FLU_TO_FRU = np.diag([1.0, -1.0, 1.0]).astype(np.float64)


def parse_intrinsics(cam_json: dict) -> tuple[np.ndarray, tuple[int, int]]:
    """Pinhole ``K`` (3x3) and ``(width, height)`` from a parsed calib_camera json."""
    K = np.asarray(cam_json["K"], dtype=np.float64).reshape(3, 3)
    return K, (cam_json["width"], cam_json["height"])


def load_intrinsics(scene_dir: str, view: str) -> tuple[np.ndarray, tuple[int, int]]:
    """Load pinhole ``K`` (3x3) and ``(width, height)`` for a Leopard camera view."""
    path = os.path.join(scene_dir, "calibrations", f"calib_camera_leopard_{view}.json")
    with open(path) as f:
        c = json.load(f)
    return parse_intrinsics(c)


def _tf_to_matrix(entry: dict) -> np.ndarray:
    tr = entry["transform"]["translation"]
    q = entry["transform"]["rotation"]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
    T[:3, 3] = [tr["x"], tr["y"], tr["z"]]
    return T


def parse_extrinsics(tf_json: dict, view: str) -> np.ndarray:
    """``T_vehicle_camera`` (4x4) from a parsed ``calib_tf_tree_full.json``."""
    T_v_cab = _tf_to_matrix(tf_json["vehicle_cab"])
    T_cab_cam = _tf_to_matrix(tf_json[f"cab_camera_leopard_{view}"])
    return T_v_cab @ T_cab_cam


def load_extrinsics(scene_dir: str, view: str) -> np.ndarray:
    """``T_vehicle_camera`` (4x4): camera pose in the vehicle frame.

    Composed from the TF tree chain ``vehicle -> cab -> camera``.
    """
    path = os.path.join(scene_dir, "calibrations", "calib_tf_tree_full.json")
    with open(path) as f:
        tf = json.load(f)
    return parse_extrinsics(tf, view)


def project_ego_to_image(
    traj_xyz: np.ndarray,
    K: np.ndarray,
    T_vehicle_cam: np.ndarray,
    image_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project ego-frame (FLU) trajectory points onto the camera image plane.

    Args:
        traj_xyz: (N, 3) points in the ego frame at t0 (FLU, y-left).
        K: (3, 3) pinhole intrinsics.
        T_vehicle_cam: (4, 4) camera pose in the vehicle frame (``load_extrinsics``).
        image_size: optional ``(width, height)``; if given, points outside the
            image are marked invalid.

    Returns:
        uv: (N, 2) pixel coordinates.
        valid: (N,) bool — in front of the camera (and on-image if size given).
    """
    pts = np.asarray(traj_xyz, dtype=np.float64).reshape(-1, 3) @ _FLU_TO_FRU
    T_cam_v = np.linalg.inv(T_vehicle_cam)
    Xc = (T_cam_v[:3, :3] @ pts.T + T_cam_v[:3, 3:4]).T  # camera optical frame
    valid = Xc[:, 2] > 0.1
    z = np.where(valid, Xc[:, 2], 1.0)
    u = K[0, 0] * Xc[:, 0] / z + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / z + K[1, 2]
    uv = np.stack([u, v], axis=1)
    if image_size is not None:
        w, h = image_size
        valid &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return uv, valid


def load_calibration(scene_dir: str, view: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Convenience: return ``(K, T_vehicle_cam, (width, height))`` for a view."""
    K, size = load_intrinsics(scene_dir, view)
    T = load_extrinsics(scene_dir, view)
    return K, T, size


def load_calibration_via(
    read_bytes, scene_id: str, view: str
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Same as ``load_calibration`` but through a byte reader.

    ``read_bytes(rel_path) -> bytes`` lets callers reuse ``TruckDriveDataset``'s
    pluggable backend, so calibration can be pulled straight from S3 without
    staging the scene locally.
    """
    cam = json.loads(read_bytes(f"{scene_id}/calibrations/calib_camera_leopard_{view}.json"))
    tf = json.loads(read_bytes(f"{scene_id}/calibrations/calib_tf_tree_full.json"))
    K, size = parse_intrinsics(cam)
    return K, parse_extrinsics(tf, view), size
