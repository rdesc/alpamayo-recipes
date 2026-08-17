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

"""Trajectory-side kinematic bucket classifier for the CoC-action consistency reward.

**Validated as of 2026-08-16**, against ``alpamayo-coc-autolabeler``'s own 2077-event
PAI-AV OOD gold corpus (see that repo's
``scripts/cac_prototype_segmented_trajectory.py`` and
``scripts/cac_validate_rl_trajectory_classifier.py`` for the harness this was ported
from). This module used to be a fixed-window kinematic average (see git history) that
scored 0.874 gold binary / 0.834 hard-negative separation against ground-truth future
motion -- clearly above the *rejected* B0 baseline (0.79-0.83) but well below the fully
validated multi-segment E3xM5-abstain design (0.941, 0.945 @ t0+1s). This version closes
almost that entire gap: **0.932 gold binary / 0.902 hard-negative separation**, with the
7-event ``reverse`` bucket getting a documented offset exception (see
``coc_action_consistency_reward.py``) that brings the overall number to 0.933, matching
the production design's own t0+1s number.

**What changed:** the previous version averaged kinematic features over one fixed 3.0s
window. This version instead **segments** the predicted future via the same
hysteresis-based state-change cutting the production pipeline uses
(``meta_action/processor/{longitudinal,lateral}/segmentation.py``), classifies each
resulting segment with the same thresholds as before, then merges segments shorter than
a caption-specific minimum length into a neighbor
(``meta_action/data_structures/ego_meta_action.py``'s ``clean_{longitudinal,lateral}_segments``)
-- all ported to run directly on ``pred_xyz``/``pred_rot`` instead of a trajdata
``scenario`` object, since the algorithm itself only needs a per-frame feature series,
not "minutes of driving": segmenting a short 6s predicted future works the same way,
just with fewer/shorter segments. The answer for "the decision at t0" is always the
*first* resulting segment, since a rollout's predicted future starts exactly at the
decision frame (unlike the production pipeline, which has to search a full clip for
whichever segment is nearest to an arbitrary decision frame).

**What is still NOT validated:** this number is measured against each gold event's
*real recorded* future ego motion, not a rollout's own decoded prediction -- it is an
oracle ceiling on what this design can achieve, not a forecast of in-loop reward values.
A policy's actual decoded trajectory (especially early in training) will diverge from
ground truth, and this module has never been run against real rollout output.

**Stub, not implemented:** Alpamayo's actual action space is ``(accel, kappa)`` per
waypoint (``alpamayo_r1.action_space.unicycle_accel_curvature.UnicycleAccelCurvatureActionSpace``,
``dt=0.1s``, exactly ``PLANNING_FREQ``) -- a rollout's ``pred_xyz``/``pred_rot`` are
produced by rolling this out through a closed-form unicycle model
(``action_to_traj``). ``kappa`` (curvature, dθ/ds) IS this module's
``average_turn_rate_rad_per_m`` feature, and ``accel`` IS its ``average_acceleration``
feature -- exactly, no unit conversion needed. Reading them directly from the decoded
action tensor (``DiscreteTrajectoryTokenizer.decode``'s internal ``action`` variable,
currently discarded) would skip the differentiate-a-rolled-out-trajectory step this
module still does, removing a real noise source (see ``_velocity_from_xy``'s docstring
for one such artifact already found and fixed). Not built: needs
``decode_rollout_trajectory`` (or a sibling) to also surface the dequantized action
tensor, and can't be validated the way the rest of this module was, since gold events
have real recorded pose, not model-predicted control tokens. See
``trajectory_state_from_action`` below for the stub.

Frequency conventions (``PLANNING_FREQ``) are taken from ``rewards/comfort_reward.py``
to stay consistent with the comfort term already scored on the same decoded trajectory.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import torch

from alpamayo1_x_rl.rewards.coc_action_consistency_matcher import (
    LON_BUCKET,
    LAT_BUCKET,
    TOKEN_MAGNITUDE,
    AxisState,
)
from alpamayo1_x_rl.rewards.comfort_reward import PLANNING_FREQ

DT: Final[float] = 1.0 / PLANNING_FREQ

# --------------------------------------------------------------------- ported thresholds
# Verbatim from meta_action/processor/longitudinal/classification.py and
# meta_action/processor/lateral/classification.py. Order matters: first match wins,
# same as the source's `ORDERED_ACTIONS` + `classify_chunk`.

# (token, property, comparator, threshold) tuples per axis, evaluated against a
# per-segment feature dict. comparator "min" == source's MinThreshold (value >=
# threshold), "max" == source's MaxThreshold (value <= threshold).
_LON_RULES: list[tuple[str, list[tuple[str, str, float]]]] = [
    ("Reverse", [("average_speed_along_heading", "max", -0.12)]),
    ("Stop", [("average_speed", "max", 0.12)]),
    (
        "StrongAcceleration",
        [("average_acceleration", "min", 1.4), ("delta_velocity", "min", 0.8)],
    ),
    (
        "GentleAcceleration",
        [
            ("average_acceleration", "min", 0.15),
            ("average_acceleration", "max", 1.4),
            ("delta_velocity", "min", 0.8),
        ],
    ),
    (
        "StrongDeceleration",
        [("average_acceleration", "max", -1.4), ("delta_velocity", "max", -0.8)],
    ),
    (
        "GentleDeceleration",
        [
            ("average_acceleration", "max", -0.15),
            ("average_acceleration", "min", -1.4),
            ("delta_velocity", "max", -0.8),
        ],
    ),
    ("MaintainSpeed", []),  # fallback, always applicable
]

_LAT_RULES: list[tuple[str, list[tuple[str, str, float]]]] = [
    (
        "ReverseRight",
        [
            ("average_speed_along_heading", "max", -0.1),
            ("average_turn_rate_rad_per_m", "max", -0.002),
            ("total_heading_change_rad", "max", -0.2),
        ],
    ),
    (
        "ReverseLeft",
        [
            ("average_speed_along_heading", "max", -0.1),
            ("average_turn_rate_rad_per_m", "min", 0.002),
            ("total_heading_change_rad", "min", 0.2),
        ],
    ),
    (
        "SharpSteerRight",
        [
            ("average_turn_rate_rad_per_m", "max", -0.05),
            ("total_heading_change_rad", "max", -0.2),
        ],
    ),
    (
        "SharpSteerLeft",
        [
            ("average_turn_rate_rad_per_m", "min", 0.05),
            ("total_heading_change_rad", "min", 0.2),
        ],
    ),
    (
        "SteerRight",
        [
            ("average_turn_rate_rad_per_m", "max", -0.0015),
            ("total_heading_change_rad", "max", -0.015),
        ],
    ),
    (
        "SteerLeft",
        [
            ("average_turn_rate_rad_per_m", "min", 0.0015),
            ("total_heading_change_rad", "min", 0.015),
        ],
    ),
    ("GoStraight", []),  # fallback, always applicable
]

_EPS: Final[float] = 1e-6


def _passes(value: float, comparator: str, threshold: float) -> bool:
    return value >= threshold if comparator == "min" else value <= threshold


def _classify(features: dict[str, float], rules: list[tuple[str, list[tuple[str, str, float]]]]) -> str:
    for tag, thresholds in rules:
        if all(_passes(features[prop], cmp, thr) for prop, cmp, thr in thresholds):
            return tag
    raise AssertionError("unreachable: every rule list ends in an unconditional fallback")


# ------------------------------------------------------------------- segmentation thresholds
# Verbatim from meta_action/processor/longitudinal/config.py and lateral/config.py.
_LON_SEG_THRESHOLDS: list[tuple[str, float]] = [
    ("agent_speed_along_heading", 0.3),
    ("agent_speed_along_heading", -0.3),
    ("agent_acceleration", 0.1),
    ("agent_acceleration", -0.1),
    ("agent_acceleration", 1.25),
    ("agent_acceleration", -1.25),
]
_ACCEL_SMOOTH_WINDOW = 5
_HYSTERESIS_FRACTION = 0.15

_LAT_SEG_THRESHOLDS: list[tuple[str, float]] = [
    ("agent_velocity_along_heading", -0.1),
    ("agent_heading_change_rate", 0.001),
    ("agent_heading_change_rate", -0.001),
    ("agent_heading_change_rate", 0.05),
    ("agent_heading_change_rate", -0.05),
]
_LAT_DH_SMOOTH_WINDOW = 5

# Caption-specific minimum segment lengths (ticks @ 10Hz), verbatim from
# longitudinal/config.py and lateral/config.py. A segment shorter than this gets merged
# into a neighbor (see `_clean_longitudinal`/`_clean_lateral`) rather than trusted on its
# own -- this is the piece that took the unmerged prototype from 0.791 to 0.932 gold
# binary; skipping it is not a minor simplification.
_LON_MIN_TICKS: dict[str, int] = {
    "GentleAcceleration": 8, "StrongAcceleration": 8, "GentleDeceleration": 8,
    "StrongDeceleration": 8, "MaintainSpeed": 12, "Stop": 8, "Reverse": 8,
}
_LAT_MIN_TICKS: dict[str, int] = {
    "GoStraight": 10, "SteerLeft": 10, "SteerRight": 10, "SharpSteerLeft": 10,
    "SharpSteerRight": 10, "ReverseLeft": 10, "ReverseRight": 10,
}


# ------------------------------------------------------------------------- feature series
def _velocity_from_xy(x: np.ndarray, y: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference velocity: unsigned magnitude, signed along heading.

    Deliberately NOT a forward-diff-with-repeated-last-frame convention (e.g. what an
    earlier version of this module used): that convention defines velocity at the final
    tick as zero, which corrupts any segment that ends at the array's last tick --
    exactly what happens here, since segmentation runs over the whole predicted future,
    not a window that stops short of the end. `np.gradient` (central differences) has no
    such artifact and is the same convention `alpamayo-coc-autolabeler`'s own raw-pose
    scripts use for the equivalent signal.
    """
    vx, vy = np.gradient(x, DT), np.gradient(y, DT)
    speed = np.hypot(vx, vy)
    speed_along = vx * np.cos(h) + vy * np.sin(h)
    return speed, speed_along


def _heading(rot: np.ndarray) -> np.ndarray:
    return np.arctan2(rot[..., 1, 0], rot[..., 0, 0])


def _longitudinal_features(xyz: np.ndarray, rot: np.ndarray) -> dict[str, np.ndarray]:
    x, y, h = xyz[:, 0], xyz[:, 1], _heading(rot)
    speed, v_lon = _velocity_from_xy(x, y, h)
    acc = np.diff(speed) / DT
    acc = np.concatenate([[acc[0]], acc]) if acc.size else np.zeros_like(speed)
    return {"agent_speed": speed, "agent_speed_along_heading": v_lon, "agent_acceleration": acc}


def _moving_average_centered(values: np.ndarray, window_size: int) -> np.ndarray:
    """Verbatim port of lateral/segmentation.py's `_moving_average_centered`."""
    x = np.asarray(values, dtype=float)
    n = int(window_size)
    if n <= 1 or x.size == 0:
        return x.copy()
    if n % 2 == 0:
        n += 1
    pad = n // 2
    x_pad = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(x_pad, np.ones(n) / n, mode="valid")


def _lateral_features(xyz: np.ndarray, rot: np.ndarray) -> dict[str, np.ndarray]:
    x, y, h = xyz[:, 0], xyz[:, 1], _heading(rot)
    _, speed_along_full = _velocity_from_xy(x, y, h)

    dxy = np.stack([np.diff(x), np.diff(y)], axis=-1)
    dist = np.linalg.norm(dxy, axis=-1)
    dh = np.diff(h)
    dh = (dh + np.pi) % (2 * np.pi) - np.pi
    dh = _moving_average_centered(dh, _LAT_DH_SMOOTH_WINDOW)
    rate = np.zeros_like(dh)
    nz = dist > 1e-6
    rate[nz] = dh[nz] / dist[nz]

    L = len(dh)
    v_along = speed_along_full[:L] if len(speed_along_full) >= L else np.pad(
        speed_along_full, (0, L - len(speed_along_full)), mode="edge"
    )
    return {
        "agent_heading_change_rate": rate,
        "agent_heading_change_rad": dh,
        "agent_distance_m": dist,
        "agent_velocity_along_heading": v_along,
    }


# ------------------------------------------------------------------------------ segmentation
def _smooth_series(x: np.ndarray, window: int) -> np.ndarray:
    """Verbatim port of longitudinal/segmentation.py's `_smooth_series`."""
    import pandas as pd

    window = int(max(1, window))
    if window == 1 or x.size <= 1:
        return x.astype(float)
    return pd.Series(x, dtype=float).rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)


def _hysteresis_boolean(series: np.ndarray, threshold: float, frac: float) -> np.ndarray:
    """Verbatim port of longitudinal/segmentation.py's `_hysteresis_boolean`."""
    T = float(threshold)
    f = float(max(0.0, frac))
    T_on, T_off = (T * (1 + f), T * (1 - f)) if T >= 0.0 else (T * (1 - f), T * (1 + f))
    out = np.zeros_like(series, dtype=bool)
    if series.size == 0:
        return out
    state = bool(series[0] >= T)
    out[0] = state
    for i in range(1, series.size):
        val = float(series[i])
        if not state and val >= T_on:
            state = True
        elif state and val < T_off:
            state = False
        out[i] = state
    return out


def _cut(threshold_states: list[np.ndarray]) -> list[tuple[int, int]]:
    """Hysteresis state-change cutting loop, verbatim from both segmentation.py files
    (identical modulo the state arrays' dtype)."""
    if not threshold_states:
        return [(0, 1)]
    T = min(len(s) for s in threshold_states)
    segments = []
    start = 0
    current = tuple(bool(s[0]) for s in threshold_states)
    i = 1
    while i < T:
        nxt = tuple(bool(s[i]) for s in threshold_states)
        if nxt != current:
            if (T - i) <= 3:
                break
            segments.append((start, i))
            start = i
            current = nxt
            i += 3
            continue
        i += 1
    segments.append((start, T))
    return segments


def _segment_longitudinal(xyz: np.ndarray, rot: np.ndarray) -> list[dict]:
    feats = _longitudinal_features(xyz, rot)
    speeds, speeds_along = feats["agent_speed"], feats["agent_speed_along_heading"]
    acc_smooth = _smooth_series(feats["agent_acceleration"], _ACCEL_SMOOTH_WINDOW)

    states = [
        _hysteresis_boolean(acc_smooth, thr, _HYSTERESIS_FRACTION) if prop == "agent_acceleration"
        else (feats[prop] >= thr)
        for prop, thr in _LON_SEG_THRESHOLDS
    ]

    segs = []
    for s0, s1 in _cut(states):
        s1c = min(s1, len(speeds) - 1)
        if s1c <= s0:
            continue
        duration = (s1c - s0) * DT
        dv = float(speeds[s1c] - speeds[s0])
        segs.append({
            "duration": duration,
            "delta_velocity": dv,
            "average_acceleration": dv / duration if duration > 0 else 0.0,
            "average_speed": float(np.mean(speeds[s0:s1c])) if s1c > s0 else float(speeds[s0]),
            "average_speed_along_heading": float(np.mean(speeds_along[s0:s1c])) if s1c > s0 else float(speeds_along[s0]),
        })
    return segs


def _segment_lateral(xyz: np.ndarray, rot: np.ndarray) -> list[dict]:
    feats = _lateral_features(xyz, rot)
    dh, dist = feats["agent_heading_change_rad"], feats["agent_distance_m"]

    states = [feats[prop] >= thr for prop, thr in _LAT_SEG_THRESHOLDS]

    segs = []
    for s0, s1 in _cut(states):
        s0c, s1c = max(0, min(s0, len(dh))), min(s1, len(dh))
        if s1c <= s0c:
            continue
        duration = (s1c - s0c) * DT
        total_heading = float(np.sum(dh[s0c:s1c]))
        total_dist = float(np.sum(dist[s0c:s1c]))
        segs.append({
            "duration": duration,
            "total_heading_change_rad": total_heading,
            "total_distance_m": total_dist,
            "average_turn_rate_rad_per_m": total_heading / total_dist if total_dist > 1e-6 else 0.0,
            "average_speed_along_heading": float(np.mean(feats["agent_velocity_along_heading"][s0c:s1c])),
        })
    return segs


# ---------------------------------------------------------------------------------- cleaning
# Verbatim port of ego_meta_action.py's clean_longitudinal_segments / clean_lateral_segments.
# Captions are assigned ONCE, before any merging, and simply inherited by whichever segment
# survives a merge -- never recomputed from the merged/pooled features.
def _merge_same_caption_lon(rows: list[dict]) -> list[dict]:
    if len(rows) <= 1:
        return rows
    merged, cur = [], dict(rows[0])
    for nxt in rows[1:]:
        nxt = dict(nxt)
        if nxt["caption"] == cur["caption"]:
            cur_dur, nxt_dur = cur["duration"], nxt["duration"]
            new_dur = cur_dur + nxt_dur
            cur["delta_velocity"] += nxt["delta_velocity"]
            cur["average_speed"] = (
                (cur["average_speed"] * cur_dur + nxt["average_speed"] * nxt_dur) / new_dur if new_dur > 0 else 0.0
            )
            cur["duration"] = new_dur
            cur["average_acceleration"] = cur["delta_velocity"] / new_dur if new_dur > 0 else 0.0
        else:
            merged.append(cur)
            cur = nxt
    merged.append(cur)
    return merged


def _clean_longitudinal(segs: list[dict]) -> list[dict]:
    rows = _merge_same_caption_lon([dict(s) for s in segs])  # pre-merge
    i = 0
    while i < len(rows):
        seg = rows[i]
        min_len_s = _LON_MIN_TICKS.get(seg["caption"], 8) * DT
        if seg["duration"] >= min_len_s:
            i += 1
            continue
        has_prev, has_next = i > 0, i < len(rows) - 1
        if has_prev and has_next:
            prev, nxt = rows[i - 1], rows[i + 1]
            if prev["caption"] and prev["caption"] == nxt["caption"]:
                prev_dur, mid_dur, nxt_dur = prev["duration"], seg["duration"], nxt["duration"]
                new_dur = prev_dur + mid_dur + nxt_dur
                prev["delta_velocity"] += seg["delta_velocity"] + nxt["delta_velocity"]
                prev["average_speed"] = (
                    (prev["average_speed"] * prev_dur + seg["average_speed"] * mid_dur + nxt["average_speed"] * nxt_dur) / new_dur
                    if new_dur > 0 else 0.0
                )
                prev["duration"] = new_dur
                prev["average_acceleration"] = prev["delta_velocity"] / new_dur if new_dur > 0 else 0.0
                rows.pop(i); rows.pop(i)
                i = max(i - 1, 0)
                continue
            else:
                prev_dur, nxt_dur, mid_dur = prev["duration"], nxt["duration"], seg["duration"]
                half_dur = 0.5 * mid_dur
                mid_dv = seg["delta_velocity"]
                prev["delta_velocity"] += 0.5 * mid_dv
                nxt["delta_velocity"] += 0.5 * mid_dv
                new_prev_dur, new_nxt_dur = prev_dur + half_dur, nxt_dur + half_dur
                prev["average_speed"] = (
                    (prev["average_speed"] * prev_dur + seg["average_speed"] * half_dur) / new_prev_dur
                    if new_prev_dur > 0 else 0.0
                )
                nxt["average_speed"] = (
                    (nxt["average_speed"] * nxt_dur + seg["average_speed"] * half_dur) / new_nxt_dur
                    if new_nxt_dur > 0 else 0.0
                )
                prev["duration"], nxt["duration"] = new_prev_dur, new_nxt_dur
                prev["average_acceleration"] = prev["delta_velocity"] / new_prev_dur if new_prev_dur > 0 else 0.0
                nxt["average_acceleration"] = nxt["delta_velocity"] / new_nxt_dur if new_nxt_dur > 0 else 0.0
                rows.pop(i)
                continue
        elif has_prev:
            prev = rows[i - 1]
            prev_dur, seg_dur = prev["duration"], seg["duration"]
            new_dur = prev_dur + seg_dur
            prev["delta_velocity"] += seg["delta_velocity"]
            prev["average_speed"] = (
                (prev["average_speed"] * prev_dur + seg["average_speed"] * seg_dur) / new_dur if new_dur > 0 else 0.0
            )
            prev["duration"] = new_dur
            prev["average_acceleration"] = prev["delta_velocity"] / new_dur if new_dur > 0 else 0.0
            rows.pop(i)
            continue
        elif has_next:
            nxt = rows[i + 1]
            nxt_dur, seg_dur = nxt["duration"], seg["duration"]
            new_dur = nxt_dur + seg_dur
            nxt["delta_velocity"] += seg["delta_velocity"]
            nxt["average_speed"] = (
                (nxt["average_speed"] * nxt_dur + seg["average_speed"] * seg_dur) / new_dur if new_dur > 0 else 0.0
            )
            nxt["duration"] = new_dur
            nxt["average_acceleration"] = nxt["delta_velocity"] / new_dur if new_dur > 0 else 0.0
            rows.pop(i)
            continue
        else:
            i += 1
    return _merge_same_caption_lon(rows)  # post-merge


def _merge_same_caption_lat(rows: list[dict]) -> list[dict]:
    if len(rows) <= 1:
        return rows
    merged, cur = [], dict(rows[0])
    for nxt in rows[1:]:
        nxt = dict(nxt)
        if nxt["caption"] == cur["caption"]:
            new_dur = cur["duration"] + nxt["duration"]
            new_dist = cur["total_distance_m"] + nxt["total_distance_m"]
            new_head = cur["total_heading_change_rad"] + nxt["total_heading_change_rad"]
            cur["duration"], cur["total_distance_m"], cur["total_heading_change_rad"] = new_dur, new_dist, new_head
            cur["average_turn_rate_rad_per_m"] = new_head / new_dist if new_dist > 1e-6 else 0.0
        else:
            merged.append(cur)
            cur = nxt
    merged.append(cur)
    return merged


def _clean_lateral(segs: list[dict]) -> list[dict]:
    rows = _merge_same_caption_lat([dict(s) for s in segs])  # pre-merge
    i = 0
    while i < len(rows):
        seg = rows[i]
        min_len_s = _LAT_MIN_TICKS.get(seg["caption"], 10) * DT
        if seg["duration"] >= min_len_s:
            i += 1
            continue
        has_prev, has_next = i > 0, i < len(rows) - 1
        if has_prev and has_next:
            prev, nxt = rows[i - 1], rows[i + 1]
            if prev["caption"] and prev["caption"] == nxt["caption"]:
                new_dur = prev["duration"] + seg["duration"] + nxt["duration"]
                new_dist = prev["total_distance_m"] + seg["total_distance_m"] + nxt["total_distance_m"]
                new_head = prev["total_heading_change_rad"] + seg["total_heading_change_rad"] + nxt["total_heading_change_rad"]
                prev["duration"], prev["total_distance_m"], prev["total_heading_change_rad"] = new_dur, new_dist, new_head
                prev["average_turn_rate_rad_per_m"] = new_head / new_dist if new_dist > 1e-6 else 0.0
                rows.pop(i); rows.pop(i)
                i = max(i - 1, 0)
                continue
            else:
                half_dist, half_head = 0.5 * seg["total_distance_m"], 0.5 * seg["total_heading_change_rad"]
                half_dur = 0.5 * seg["duration"]
                prev["total_distance_m"] += half_dist
                prev["total_heading_change_rad"] += half_head
                nxt["total_distance_m"] += half_dist
                nxt["total_heading_change_rad"] += half_head
                prev["duration"] += half_dur
                nxt["duration"] += half_dur
                prev["average_turn_rate_rad_per_m"] = (
                    prev["total_heading_change_rad"] / prev["total_distance_m"] if prev["total_distance_m"] > 1e-6 else 0.0
                )
                nxt["average_turn_rate_rad_per_m"] = (
                    nxt["total_heading_change_rad"] / nxt["total_distance_m"] if nxt["total_distance_m"] > 1e-6 else 0.0
                )
                rows.pop(i)
                continue
        elif has_prev:
            prev = rows[i - 1]
            prev["duration"] += seg["duration"]
            prev["total_distance_m"] += seg["total_distance_m"]
            prev["total_heading_change_rad"] += seg["total_heading_change_rad"]
            prev["average_turn_rate_rad_per_m"] = (
                prev["total_heading_change_rad"] / prev["total_distance_m"] if prev["total_distance_m"] > 1e-6 else 0.0
            )
            rows.pop(i)
            continue
        elif has_next:
            nxt = rows[i + 1]
            nxt["duration"] += seg["duration"]
            nxt["total_distance_m"] += seg["total_distance_m"]
            nxt["total_heading_change_rad"] += seg["total_heading_change_rad"]
            nxt["average_turn_rate_rad_per_m"] = (
                nxt["total_heading_change_rad"] / nxt["total_distance_m"] if nxt["total_distance_m"] > 1e-6 else 0.0
            )
            rows.pop(i)
            continue
        else:
            i += 1
    return _merge_same_caption_lat(rows)  # post-merge


# --------------------------------------------------------------------------- public entry
def _clamp_offset(T: int, offset_s: float) -> int:
    f0 = int(round(offset_s * PLANNING_FREQ))
    return max(0, min(f0, max(0, T - 5)))  # keep at least 5 frames so smoothing/segmentation can run


def segments_from_pred(
    pred_xyz: torch.Tensor,
    pred_rot: torch.Tensor,
    *,
    offset_s: float = 0.0,
) -> dict[str, list[dict]]:
    """The full post-merge segment list per axis, in time order -- what
    `trajectory_state_from_pred` reduces down to just its first segment. Exposed
    separately for visualization (a segmentation timeline strip needs every segment,
    not just the decision-frame one); each segment dict carries `duration` (seconds)
    and `caption` (fine token), plus the raw kinematic features that produced it.
    Cumulative start time for plotting is just the running sum of prior durations --
    segments are contiguous and already in order, so no tick indices are needed.
    """
    assert pred_xyz.dim() == 2 and pred_xyz.shape[-1] == 3, f"expected [T,3], got {tuple(pred_xyz.shape)}"
    assert pred_rot.dim() == 3 and pred_rot.shape[-2:] == (3, 3), (
        f"expected [T,3,3], got {tuple(pred_rot.shape)}"
    )
    xyz_np = pred_xyz.detach().float().cpu().numpy()
    rot_np = pred_rot.detach().float().cpu().numpy()

    f0 = _clamp_offset(xyz_np.shape[0], offset_s)
    xyz_np, rot_np = xyz_np[f0:], rot_np[f0:]

    lon_segs = _segment_longitudinal(xyz_np, rot_np)
    lat_segs = _segment_lateral(xyz_np, rot_np)
    for s in lon_segs:
        s["caption"] = _classify(s, _LON_RULES)
    for s in lat_segs:
        s["caption"] = _classify(s, _LAT_RULES)
    return {
        "longitudinal": _clean_longitudinal(lon_segs),
        "lateral": _clean_lateral(lat_segs),
    }


def trajectory_state_from_pred(
    pred_xyz: torch.Tensor,
    pred_rot: torch.Tensor,
    *,
    offset_s: float = 0.0,
) -> dict[str, AxisState]:
    """Classify a rollout's predicted future trajectory into the matcher's taxonomy.

    Segments the future (from `offset_s` into it, in case a caller needs a later read --
    see `coc_action_consistency_reward.py`'s reverse-claim handling for why offset_s
    exists at all: it is NOT a general knob to sweep, offset=0 is correct for every
    bucket except `reverse`) via the same hysteresis cutting + min-length merge the
    production pipeline uses, classifies each resulting segment, and reads off whichever
    segment covers the start of the (possibly offset) window -- always the first one,
    since a rollout's predicted future starts exactly at the decision frame.

    Args:
        pred_xyz: ``[T, 3]`` predicted future ego positions (no batch dim -- caller
            indexes ``[0]`` first, matching how ``aggregated_reward*.py`` already calls
            ``calculate_ade(predicted_fut_xyz[0], ...)``).
        pred_rot: ``[T, 3, 3]`` predicted future ego rotation matrices.
        offset_s: seconds into the predicted future to start reading from. 0.0 for every
            claim except a longitudinal `reverse` claim, which should be read at 1.0s
            (validated: 7/7 vs 5/7 correct on the gold corpus's reverse events -- reverse
            maneuvers in this corpus often begin from a brief stop, so the backward-motion
            signal is genuinely absent at t0, not just noisy).

    Returns:
        ``{"longitudinal": {...}, "lateral": {...}}``, each an ``AxisState`` with
        ``bucket`` (coarse), ``token`` (fine autolabeler tag), ``magnitude``.
    """
    segs = segments_from_pred(pred_xyz, pred_rot, offset_s=offset_s)
    lon_segs, lat_segs = segs["longitudinal"], segs["lateral"]

    lon_token, lat_token = lon_segs[0]["caption"], lat_segs[0]["caption"]
    return {
        "longitudinal": {
            "bucket": LON_BUCKET[lon_token],
            "token": lon_token,
            "magnitude": TOKEN_MAGNITUDE.get(lon_token),
        },
        "lateral": {
            "bucket": LAT_BUCKET[lat_token],
            "token": lat_token,
            "magnitude": TOKEN_MAGNITUDE.get(lat_token),
        },
    }


def trajectory_state_from_action(action: torch.Tensor, v0: float, offset_s: float = 0.0) -> dict[str, AxisState]:
    """STUB -- not implemented. See module docstring's "Stub, not implemented" section.

    Would classify directly from the model's own decoded ``(accel, kappa)`` action
    tensor (``action[..., 0]`` = accel m/s^2, ``action[..., 1]`` = kappa rad/m, both
    already de-normalized -- see
    ``alpamayo_r1.action_space.unicycle_accel_curvature.UnicycleAccelCurvatureActionSpace``),
    skipping the differentiate-a-rolled-out-trajectory step ``trajectory_state_from_pred``
    still does. Needs ``decode_rollout_trajectory`` (or a sibling) to surface the
    dequantized action tensor, which it currently discards. Not validated -- gold events
    have real recorded pose, not model-predicted control tokens, so there is no gold
    corpus to check this against the way the rest of this module was checked.
    """
    raise NotImplementedError(
        "trajectory_state_from_action is a documented stub -- see module docstring. "
        "Use trajectory_state_from_pred (the validated path) instead."
    )


# --------------------------------------------------------------------------- self-test
def _selftest() -> bool:
    """Smoke tests with synthetic trajectories -- not a substitute for gold-label
    validation (see module docstring), just a check that geometry/signs/segmentation
    aren't flipped.
    """
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        ok &= cond
        print(f"  [{'OK ' if cond else 'BAD'}] {name}")

    freq = PLANNING_FREQ
    dt = 1.0 / freq
    n = 61  # 6s @ 10Hz, matching a realistic predicted-future length

    def straight_traj(speed_mps: float) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(n, dtype=torch.float32) * dt
        x = speed_mps * t
        y = torch.zeros(n)
        xyz = torch.stack([x, y, torch.zeros(n)], dim=-1)
        rot = torch.eye(3).unsqueeze(0).repeat(n, 1, 1)  # identity heading (facing +x)
        return xyz, rot

    # Constant 10 m/s, straight -> MaintainSpeed / GoStraight.
    xyz, rot = straight_traj(10.0)
    ts = trajectory_state_from_pred(xyz, rot)
    check(
        "constant speed, straight line -> MaintainSpeed/GoStraight",
        ts["longitudinal"]["token"] == "MaintainSpeed" and ts["lateral"]["token"] == "GoStraight",
    )

    # Near-zero speed throughout -> Stop.
    xyz, rot = straight_traj(0.0)
    ts = trajectory_state_from_pred(xyz, rot)
    check("stationary -> Stop", ts["longitudinal"]["token"] == "Stop")

    # Strong deceleration: from 15 m/s to ~0 early, then holds.
    t = torch.arange(n, dtype=torch.float32) * dt
    v0 = 15.0
    decel = 4.0  # m/s^2, strong
    v = torch.clamp(v0 - decel * t, min=0.0)
    x = torch.cumsum(v * dt, dim=0)
    xyz = torch.stack([x, torch.zeros(n), torch.zeros(n)], dim=-1)
    rot = torch.eye(3).unsqueeze(0).repeat(n, 1, 1)
    ts = trajectory_state_from_pred(xyz, rot)
    check(
        f"strong braking (15->~0 m/s) -> StrongDeceleration first (got {ts['longitudinal']['token']})",
        ts["longitudinal"]["token"] == "StrongDeceleration",
    )

    # Left turn: heading rotates positively (counter-clockwise) at constant forward speed.
    yaw_rate = 0.3  # rad/s, well past the SteerLeft/SharpSteerLeft thresholds
    yaw = torch.cumsum(torch.full((n,), yaw_rate * dt), dim=0)
    speed = 8.0
    vx = speed * torch.cos(yaw)
    vy = speed * torch.sin(yaw)
    x = torch.cumsum(vx * dt, dim=0)
    y = torch.cumsum(vy * dt, dim=0)
    xyz = torch.stack([x, y, torch.zeros(n)], dim=-1)
    cos_h, sin_h = torch.cos(yaw), torch.sin(yaw)
    rot = torch.zeros(n, 3, 3)
    rot[:, 0, 0], rot[:, 0, 1] = cos_h, -sin_h
    rot[:, 1, 0], rot[:, 1, 1] = sin_h, cos_h
    rot[:, 2, 2] = 1.0
    ts = trajectory_state_from_pred(xyz, rot)
    check(
        f"left turn -> lateral bucket 'left' (got token {ts['lateral']['token']})",
        ts["lateral"]["bucket"] == "left",
    )

    # offset_s mechanism: brake hard for the first 1s then cruise steady -> offset=0 should
    # catch the braking, offset=1.5s should see only the post-braking cruise.
    t = torch.arange(n, dtype=torch.float32) * dt
    v = torch.where(t < 1.0, 12.0 - 6.0 * t, 6.0 * torch.ones_like(t))
    x = torch.cumsum(v * dt, dim=0)
    xyz = torch.stack([x, torch.zeros(n), torch.zeros(n)], dim=-1)
    rot = torch.eye(3).unsqueeze(0).repeat(n, 1, 1)
    ts0 = trajectory_state_from_pred(xyz, rot, offset_s=0.0)
    ts1 = trajectory_state_from_pred(xyz, rot, offset_s=1.5)
    check(
        f"offset_s shifts the read: offset=0 sees braking (got {ts0['longitudinal']['token']}), "
        f"offset=1.5s sees the settled cruise (got {ts1['longitudinal']['token']})",
        ts0["longitudinal"]["token"] in ("StrongDeceleration", "GentleDeceleration")
        and ts1["longitudinal"]["token"] == "MaintainSpeed",
    )

    print(f"\nSelf-test {'PASSED' if ok else 'FAILED'}.")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if _selftest() else 1)
