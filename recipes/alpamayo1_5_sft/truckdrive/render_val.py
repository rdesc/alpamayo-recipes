# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Offline render of the W&B trajectory-viz figures for a fixed set of val ids.

Same picture the ``TrajectoryVizCallback`` logs during training -- multi-camera
image grid + BEV GT (red) vs. K generated trajectories (blue), minADE in the
header -- but produced offline for a chosen checkpoint and written to PNGs on
disk instead of W&B. Reuses the callback's own helpers (``_sample_info``,
``_grid_labels``, ``_min_ade``) and ``alpamayo.visualization.viz.visualize_data``
so the output is byte-for-byte the same renderer.

Indices default to ``viz.sample_indices`` from the config (the 20 geometry-aware
val windows); override with ``+render.indices=[...]``. Inference is sharded
across ranks (one figure per index, no cross-rank gather needed), so launch it
under torchrun to use all 8 GPUs.

Score/render a checkpoint in the SAME processor format it was trained with (the
step-8374 ckpt used include_camera_ids/frame_nums=false -- override on the CLI).

Launch (8 GPUs)::

    torchrun --nproc_per_node=8 -m alpamayo1_5_sft.truckdrive.render_val \
      --config-path pkg://alpamayo1_5_sft/configs --config-name sft_truckdrive \
      data.val_dataset.data_root=s3://torc-data/datasets/TruckDrivePublic \
      data.val_dataset.backend=s3 \
      data.val_dataset.metadata_cache=/mnt/efs/users/rod/truckdrive_cache/meta_5views.pkl \
      data.val_dataset.split_metainfo=/mnt/efs/users/rod/repos/alpamayo-recipes/recipes/alpamayo1_5_sft/truckdrive/truckdrive_metainfo.json \
      model.attn_implementation=sdpa \
      +render.eval_ckpt=/home/rod/ckpts/output_truckdrive_full_checkpoint-8374 \
      +render.out_dir=/mnt/efs/users/rod/truckdrive_val_figs
"""
import os

import hydra
import hydra.utils as hyu
import torch
from omegaconf import DictConfig, OmegaConf

from alpamayo.common import distributed
from alpamayo.visualization.viz import visualize_data
from alpamayo1_5_sft.callbacks.viz_callback import TrajectoryVizCallback, _to_device


def _get(cfg: DictConfig, path: str, default):
    v = OmegaConf.select(cfg, path)
    return default if v is None else v


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: DictConfig) -> None:
    distributed.initialize_distributed_simple()
    rank = distributed.get_global_rank()
    world = distributed.get_world_size()
    local_rank = distributed.get_local_rank()
    device = f"cuda:{local_rank}"

    eval_ckpt = _get(cfg, "render.eval_ckpt", None)
    out_dir = _get(cfg, "render.out_dir", "truckdrive_val_figs")
    # Default to the config's viz set; allow an override list.
    indices = _get(cfg, "render.indices", None)
    if indices is None:
        indices = list(_get(cfg, "viz.sample_indices", []))
    indices = list(indices)
    num_traj_samples = int(_get(cfg, "render.num_traj_samples",
                                _get(cfg, "viz.num_traj_samples", 6)))
    top_p = float(_get(cfg, "render.top_p", _get(cfg, "viz.top_p", 0.98)))
    temperature = float(_get(cfg, "render.temperature", _get(cfg, "viz.temperature", 0.6)))

    if eval_ckpt is not None:
        cfg.model.checkpoint_path = eval_ckpt
    if rank == 0:
        print(f"[render] checkpoint = {cfg.model.checkpoint_path}", flush=True)
        print(f"[render] rendering {len(indices)} ids across {world} rank(s) -> {out_dir}",
              flush=True)
        os.makedirs(out_dir, exist_ok=True)
    if world > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()

    model = hyu.instantiate(cfg.model, _convert_="partial").to(device).eval()

    # val split must be generation-ready (tokenized_data for rollout).
    OmegaConf.update(cfg, "data.val_dataset.vla_preprocess_args.generation_mode", True,
                     force_add=True)
    ds = hyu.instantiate(cfg.data.val_dataset, _convert_="partial", model_config=model.config)
    collate_fn = hyu.instantiate(cfg.data.collate_fn, _convert_="partial",
                                 model_config=model.config)

    # Reuse the callback purely for its render helpers (never touches W&B here).
    helper = TrajectoryVizCallback(eval_dataset=ds, collate_fn=collate_fn,
                                   output_dir=out_dir, sample_indices=indices,
                                   num_traj_samples=num_traj_samples,
                                   top_p=top_p, temperature=temperature)

    # Shard the ids across ranks (one figure per id; no gather needed).
    my_ids = indices[rank::world]
    for idx in my_ids:
        s = ds[idx]
        batch = _to_device(collate_fn([dict(s)]), device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, _ = model.sample_trajectories_from_data(
                data=batch, num_traj_samples=num_traj_samples, num_traj_sets=1,
                top_p=top_p, temperature=temperature,
            )
        frames = s["image_frames"]
        if frames.dim() == 5:  # (N_cam, F, C, H, W) -> (F, N_cam, C, H, W)
            frames = frames.permute(1, 0, 2, 3, 4)
        header, _ = helper._sample_info(s, idx, step=0)
        cam_labels, frame_labels = helper._grid_labels(s)
        gt = s.get("ego_future_xyz")
        if gt is not None:
            mn, mean = helper._min_ade(pred_xyz[0], gt)
            header = (f"{header}\nminADE={mn:.2f}m (best of {num_traj_samples})"
                      f"  meanADE={mean:.2f}m")
        png_path = os.path.join(out_dir, f"val_sample{idx:05d}.png")
        visualize_data(
            image_frames=frames,
            ego_future_xyz_gt=gt,
            ego_future_xyz_pred=pred_xyz[0].unsqueeze(0).cpu(),
            cot_text=s.get("nav_text"),
            show_waypoint_pai=False,
            save_path=png_path,
            info_text=header,
            camera_labels=cam_labels,
            frame_labels=frame_labels,
        )
        print(f"[render] rank{rank} wrote {png_path}", flush=True)

    if world > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if rank == 0:
        print(f"[render] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
