# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Score val windows by ego-trajectory GEOMETRY to surface "interesting" scenarios.

TruckDrive ships no scenario/maneuver/weather labels (metainfo.json is only an
object-detection taxonomy + split lists), so "interesting" must be derived from
the ego future trajectory. This does a cheap, GPU-free, IMAGE-free sweep over
every val window -- reusing ``TruckDriveDataset._ego_traj`` (the same ego@t0
future the model sees, no image loads, exactly the trick the tok-recon
pre-filter uses) -- and computes per-window motion descriptors:

  * turn_deg      : net heading change over the future horizon (|yaw_end|, deg)
  * abs_turn_deg  : max |yaw| over the horizon (peak turn magnitude, deg)
  * max_lat_m     : max |y| lateral excursion in the ego@t0 frame (m)
  * max_kappa     : peak path curvature 1/m (tight-turn detector)
  * mean_speed    : future path length / horizon (m/s)
  * path_len_m    : future arc length (m)

Windows are indexed identically to score_val_ade.py (same dataset, same order),
so the emitted ``idx`` joins directly onto the ADE manifest's ``per_window``.
Pass ``+score.ade_manifest=<the ADE json>`` to join ``min_ade`` in and rank the
most informative qualitative examples: high curvature AND high error.

Emits stratified picks (via the shared ``select_stratified``) over whichever key
``+score.rank_by`` names (default ``abs_turn_deg``) plus a full per-window table.

Launch (single CPU box is fine -- no GPU, no model, no images):

    python -m alpamayo1_5_sft.truckdrive.score_val_interesting \
      --config-path pkg://alpamayo1_5_sft/configs --config-name sft_truckdrive \
      data.val_dataset.data_root=s3://torc-data/datasets/TruckDrivePublic \
      data.val_dataset.backend=s3 \
      data.val_dataset.metadata_cache=/mnt/efs/users/rod/truckdrive_cache/meta_5views.pkl \
      data.val_dataset.split_metainfo=/mnt/efs/users/rod/repos/alpamayo-recipes/recipes/alpamayo1_5_sft/truckdrive/truckdrive_metainfo.json \
      +score.out=/mnt/efs/users/rod/truckdrive_val_interesting.json \
      +score.ade_manifest=/mnt/efs/users/rod/truckdrive_val_ade_selection.json \
      +score.rank_by=abs_turn_deg +score.num_bins=4 +score.per_bin=3
"""
import json
import os

import hydra
import hydra.utils as hyu
import numpy as np
from omegaconf import DictConfig, OmegaConf


def _get(cfg: DictConfig, path: str, default):
    v = OmegaConf.select(cfg, path)
    return default if v is None else v


def _yaw_from_rot(fut_rot: np.ndarray) -> np.ndarray:
    """(F,3,3) ego@t0 rotations -> unwrapped yaw (F,) radians."""
    yaw = np.arctan2(fut_rot[:, 1, 0], fut_rot[:, 0, 0])
    return np.unwrap(yaw)


def _window_metrics(fut_xyz: np.ndarray, fut_rot: np.ndarray, horizon: float) -> dict:
    """Motion descriptors for one future window (ego@t0 frame, x-fwd, y-left)."""
    xy = fut_xyz[:, :2]
    yaw = _yaw_from_rot(fut_rot)  # yaw at t0 is ~0 by construction (ego frame)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)  # (F-1,) step arc lengths
    path_len = float(seg.sum())
    dyaw = np.diff(yaw)
    kappa = np.abs(dyaw) / (seg + 1e-6)  # per-step curvature 1/m
    return {
        "turn_deg": float(np.degrees(abs(yaw[-1]))),
        "abs_turn_deg": float(np.degrees(np.abs(yaw).max())),
        "max_lat_m": float(np.abs(xy[:, 1]).max()),
        "max_kappa": float(kappa.max()) if kappa.size else 0.0,
        "path_len_m": path_len,
        "mean_speed": path_len / horizon if horizon > 0 else 0.0,
    }


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    out_path = _get(cfg, "score.out", "truckdrive_val_interesting.json")
    ade_manifest = _get(cfg, "score.ade_manifest", None)
    rank_by = _get(cfg, "score.rank_by", "abs_turn_deg")
    num_bins = int(_get(cfg, "score.num_bins", 4))
    per_bin = int(_get(cfg, "score.per_bin", 3))
    max_windows = int(_get(cfg, "score.max_windows", -1))

    # Geometry only: instantiate the val dataset (no model, no preprocessor, no
    # image loads). Same params as scoring/training so idx matches the ADE run.
    OmegaConf.update(cfg, "data.val_dataset.vla_preprocess_args", None, force_add=True)
    ds = hyu.instantiate(cfg.data.val_dataset, _convert_="partial")
    horizon = ds.num_future_steps * ds.time_step

    n = len(ds) if max_windows < 0 else min(max_windows, len(ds))
    print(f"[interesting] scoring {n} val windows (horizon={horizon:.1f}s, "
          f"rank_by={rank_by})", flush=True)

    records = []
    for idx in range(n):
        scene_id, t0 = ds._samples[idx]
        poses = ds._load_scene(scene_id)["poses"]
        _, _, fut_xyz, fut_rot = ds._ego_traj(poses, t0)
        rec = {"idx": idx, "scene_id": scene_id, "t0": t0}
        rec.update(_window_metrics(fut_xyz, fut_rot, horizon))
        records.append(rec)
        if idx and idx % 500 == 0:
            print(f"[interesting] {idx}/{n}", flush=True)

    # Optional join with the ADE manifest so we can rank by "hard AND curvy".
    if ade_manifest is not None and os.path.exists(ade_manifest):
        ade = {r["idx"]: r["min_ade"]
               for r in json.load(open(ade_manifest)).get("per_window", [])}
        for r in records:
            r["min_ade"] = ade.get(r["idx"])
        print(f"[interesting] joined min_ade for "
              f"{sum(r.get('min_ade') is not None for r in records)}/{n} windows",
              flush=True)

    if rank_by not in records[0]:
        raise ValueError(f"rank_by={rank_by!r} not a metric; choose from "
                         f"{[k for k in records[0] if k not in ('idx','scene_id','t0')]}")

    # Stratified picks over the chosen key (reuse the ADE binning helper, which
    # keys on 'min_ade' -- so alias the ranking metric onto that field).
    from alpamayo1_5_sft.truckdrive._ade_binning import select_stratified
    for r in records:
        r["min_ade_backup"] = r.get("min_ade")
        r["min_ade"] = r[rank_by]
    sample_indices, bins_summary = select_stratified(records, num_bins, per_bin)
    for r in records:
        r["min_ade"] = r.pop("min_ade_backup")

    by_rank = sorted(records, key=lambda r: r[rank_by], reverse=True)
    manifest = {
        "rank_by": rank_by, "n_windows": int(n),
        "num_bins": num_bins, "per_bin": per_bin,
        "sample_indices": sample_indices,
        "bins": bins_summary,
        "top_by_rank": by_rank[:30],
        "per_window": by_rank,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=========== TOP {rank_by} (most interesting) ===========")
    for r in by_rank[:20]:
        ade = f"  minADE={r['min_ade']:.2f}m" if r.get("min_ade") is not None else ""
        print(f" idx {r['idx']:>5}  {r['scene_id']:<16}  {rank_by}={r[rank_by]:7.2f}"
              f"  lat={r['max_lat_m']:5.1f}m  v={r['mean_speed']:4.1f}m/s{ade}")
    print(f"\n viz.sample_indices (stratified by {rank_by}): {sample_indices}")
    print(f"\n wrote manifest -> {out_path}")


if __name__ == "__main__":
    main()
