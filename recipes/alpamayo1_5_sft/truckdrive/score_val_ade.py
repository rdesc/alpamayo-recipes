# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Score the val split by per-window ADE and pick a fixed, difficulty-stratified
set of examples for the W&B trajectory viz.

Runs generation over every val window with a given checkpoint, records per-window
minADE (min over K sampled trajectories, matching what the viz shows), bins the
windows by ADE, and picks a few examples per bin. Emits the resulting
``sample_indices`` list (+ a JSON manifest) to paste into ``sft_truckdrive.yaml``'s
``viz.sample_indices`` so future runs always render the same easy->hard spread.

Single-GPU on purpose: iterating the dataset by index (no DDP gather) keeps the
index<->window mapping exact, which is the whole point.

IMPORTANT: score a checkpoint in the SAME processor format it was trained with.
The step-8374 checkpoint was trained with include_camera_ids/include_frame_nums
= false, so score it with those false (override on the CLI) even though the
current run uses true -- the difficulty *ranking* still transfers.

Launch (on the box that has the checkpoint + training venv + a GPU), e.g.:

    python -m alpamayo1_5_sft.truckdrive.score_val_ade \
      --config-path pkg://alpamayo1_5_sft/configs --config-name sft_truckdrive \
      data.val_dataset.data_root=s3://torc-data/datasets/TruckDrivePublic \
      data.val_dataset.backend=s3 \
      data.val_dataset.metadata_cache=/mnt/efs/users/rod/truckdrive_cache/meta_5views.pkl \
      data.val_dataset.split_metainfo=/mnt/efs/users/rod/repos/alpamayo-recipes/recipes/alpamayo1_5_sft/truckdrive/truckdrive_metainfo.json \
      data.val_dataset.vla_preprocess_args.include_camera_ids=false \
      data.val_dataset.vla_preprocess_args.include_frame_nums=false \
      model.attn_implementation=sdpa \
      +score.eval_ckpt=/home/rod/ckpts/output_truckdrive_full_checkpoint-8374 \
      +score.out=/mnt/efs/users/rod/truckdrive_val_ade_selection.json \
      +score.num_bins=4 +score.per_bin=3 +score.batch_size=8
"""
import json
import os

import hydra
import hydra.utils as hyu
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from alpamayo.common import distributed


def _get(cfg: DictConfig, path: str, default):
    v = OmegaConf.select(cfg, path)
    return default if v is None else v


def _min_ade(pred_j: torch.Tensor, gt_j: torch.Tensor) -> tuple[float, float]:
    """pred_j: [N,K,Tf,3] or [K,Tf,3]; gt_j: [N,Tf,3] or [Tf,3]. XY-plane ADE.

    Returns (minADE over K, meanADE over K), in meters.
    """
    if pred_j.dim() == 4:      # [N,K,Tf,3] -> take first future set
        pred_j = pred_j[0]
    if gt_j.dim() == 3:        # [N,Tf,3] -> first future set
        gt_j = gt_j[0]
    pred_xy = pred_j[..., :2].float()          # [K,Tf,2]
    gt_xy = gt_j[..., :2].float().unsqueeze(0)  # [1,Tf,2]
    per_k = torch.linalg.norm(pred_xy - gt_xy, dim=-1).mean(dim=-1)  # [K]
    return float(per_k.min()), float(per_k.mean())


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    distributed.initialize_distributed_simple()
    eval_ckpt = _get(cfg, "score.eval_ckpt", None)
    out_path = _get(cfg, "score.out", "truckdrive_val_ade_selection.json")
    num_bins = int(_get(cfg, "score.num_bins", 5))
    per_bin = int(_get(cfg, "score.per_bin", 4))
    batch_size = int(_get(cfg, "score.batch_size", 8))
    num_workers = int(_get(cfg, "score.num_workers", 8))
    num_traj_samples = int(_get(cfg, "score.num_traj_samples", 6))  # Alpamayo convention
    top_p = float(_get(cfg, "score.top_p", 0.98))
    temperature = float(_get(cfg, "score.temperature", 0.6))
    max_windows = int(_get(cfg, "score.max_windows", -1))  # -1 = all (debug knob)

    if eval_ckpt is not None:
        cfg.model.checkpoint_path = eval_ckpt
    print(f"[score] checkpoint = {cfg.model.checkpoint_path}", flush=True)

    model = hyu.instantiate(cfg.model, _convert_="partial")
    model = model.to(f"cuda:{distributed.get_local_rank()}").eval()

    # val split must be generation-ready (tokenized_data for rollout).
    OmegaConf.update(cfg, "data.val_dataset.vla_preprocess_args.generation_mode", True,
                     force_add=True)
    ds = hyu.instantiate(cfg.data.val_dataset, _convert_="partial", model_config=model.config)
    collate_fn = hyu.instantiate(cfg.data.collate_fn, _convert_="partial",
                                 model_config=model.config)

    n = len(ds) if max_windows < 0 else min(max_windows, len(ds))
    rank = distributed.get_global_rank()
    world = distributed.get_world_size()
    local_rank = distributed.get_local_rank()
    device = f"cuda:{local_rank}"
    if rank == 0:
        print(f"[score] scoring {n} val windows across {world} rank(s)  "
              f"(bins={num_bins} x {per_bin}/bin, K={num_traj_samples}, "
              f"batch={batch_size})", flush=True)

    def _dev(obj):
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: _dev(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_dev(v) for v in obj)
        return obj

    # Shard windows across ranks by stride; each record carries its true global
    # index, so the index<->window mapping stays exact after merging. shuffle=False
    # over the Subset keeps order, so batch b holds my_indices[b*B : b*B+bs].
    my_indices = list(range(rank, n, world))
    subset = torch.utils.data.Subset(ds, my_indices)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn, drop_last=False,
    )
    smp = getattr(ds, "_samples", None)
    records = []
    ptr = 0
    for batch in loader:
        bs = len(batch["image_frames"])
        gidx = my_indices[ptr:ptr + bs]
        ptr += bs
        gt_all = batch.get("ego_future_xyz")
        batch = _dev(batch)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, _ = model.sample_trajectories_from_data(
                data=batch, num_traj_samples=num_traj_samples, num_traj_sets=1,
                top_p=top_p, temperature=temperature,
            )
        pred_xyz = pred_xyz.cpu()
        for j, idx in enumerate(gidx):
            if gt_all is None:
                continue
            mn, mean = _min_ade(pred_xyz[j], gt_all[j].cpu())
            if smp is not None and isinstance(smp[idx], tuple):
                sid, t0 = smp[idx][0], smp[idx][1]
            else:
                sid, t0 = None, None
            records.append({"idx": int(idx), "scene_id": sid, "t0": t0,
                            "min_ade": mn, "mean_ade": mean})
        if rank == 0 and (ptr % (batch_size * 5) == 0 or ptr >= len(my_indices)):
            ades = np.array([r["min_ade"] for r in records])
            print(f"[score] rank0 {ptr}/{len(my_indices)} (~{ptr * world}/{n} total)  "
                  f"running minADE median={np.median(ades):.3f}m "
                  f"mean={ades.mean():.3f}m", flush=True)

    # ---- gather shards: each rank writes its records, rank 0 merges + bins ----
    shard_path = f"{out_path}.rank{rank}.json"
    with open(shard_path, "w") as f:
        json.dump(records, f)
    if world > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if rank != 0:
        return

    records = []
    for r in range(world):
        sp = f"{out_path}.rank{r}.json"
        records.extend(json.load(open(sp)))
        os.remove(sp)
    records.sort(key=lambda x: x["idx"])
    print(f"[score] merged {len(records)} windows from {world} shard(s)", flush=True)

    # ---- bin by minADE quantiles, pick per_bin spread within each bin ----
    from alpamayo1_5_sft.truckdrive._ade_binning import select_stratified
    ades = np.array([r["min_ade"] for r in records])
    sample_indices, bins_summary = select_stratified(records, num_bins, per_bin)
    manifest = {
        "checkpoint": str(cfg.model.checkpoint_path),
        "n_windows": int(n),
        "num_bins": num_bins, "per_bin": per_bin, "num_traj_samples": num_traj_samples,
        "overall_min_ade_mean": float(ades.mean()),
        "overall_min_ade_median": float(np.median(ades)),
        "bins": bins_summary,
        "sample_indices": sample_indices,
        "per_window": sorted(records, key=lambda r: r["min_ade"]),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n=========== VAL ADE SELECTION ===========")
    for bs in bins_summary:
        print(f" bin {bs['bin']}  ADE [{bs['ade_lo']:.2f}, {bs['ade_hi']:.2f}] m  "
              f"(n={bs['n_in_bin']}):")
        for p in bs["picked"]:
            print(f"    idx {p['idx']:>5}  {p['scene_id']}  minADE={p['min_ade']}m")
    print(f"\n overall minADE: mean={ades.mean():.3f}m  median={np.median(ades):.3f}m")
    print(f"\n viz.sample_indices: {sample_indices}")
    print(f"\n wrote manifest -> {out_path}")


if __name__ == "__main__":
    main()
