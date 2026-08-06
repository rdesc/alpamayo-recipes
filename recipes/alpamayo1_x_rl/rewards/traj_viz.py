# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""W&B bird's-eye-view plots of RL rollout trajectories vs reference policy vs GT.

Motivation: the shipped motion reward logs only scalars (`reward`, `traj_L2`,
`comfort_reward`). A collapse run on 2026-07-30 showed those scalars are not
enough to see *what* the policy is doing -- reward/ADE looked merely noisy while
the policy had actually collapsed to emitting one constant trajectory token
(`<i1498>` x128, CoT dropped entirely). Entropy was the only warning, and no
scalar showed the geometry. These plots make that failure visible immediately.

Four curves per plot, all in the ego frame:

* **GT** (red) -- human ground truth (``reference["ego_future_xyz"]``).
* **rollout** (blue) -- the sampled completion being scored right now.
* **reference policy** (light blue, dashed) -- the *frozen base policy* that the
  KL anchor pulls toward (``[train.train_policy].kl_beta``).
* **history** (grey) -- the past motion that produced the window.

**Consistency with SFT.** The figure is built from the SFT viz callback's own
primitives (:mod:`alpamayo.visualization.viz`: ``rotate_90cc``,
``_plot_trajectory_with_fade``, ``_set_tight_trajectory_limits``) and reuses its
colour vocabulary, marker idiom (``x`` start / ``o`` end), axis labels, monospace
header and bottom-of-figure CoT, so an RL plot reads the same as an SFT one. Keys
live under ``viz/`` for the same reason -- see :data:`_WANDB_KEY_PREFIX`. What is
*not* reproduced is SFT's camera grid: rewards are scored off
``get_reference_answer``, which carries no ``image_frames``.

On the reference curve: the KL reference model inside Cosmos-RL's
``rl_worker`` is only ever used for **logprobs**, never for generation, so there
is no reference *trajectory* to read out of it. Instead we cache the first
decoded trajectory seen for each clip. Training starts from the base weights, so
the first-seen rollout is a sample from the reference policy -- the same weights
the KL term anchors to. It is a sample rather than the mode (rollouts are drawn
at ``[rollout.sampling_config].temperature``), which is worth remembering when
reading the plot, but it costs no extra forward passes.

**Which process this runs in.** Rewards are computed in the ROLLOUT replica, not
the controller -- verified on a real run: 2480 ``compute_reward`` calls in
``rollout_0.log``, zero in ``controller.log``. Only the controller calls
``init_wandb``, so ``wandb.run`` is None here and :func:`_ensure_wandb` attaches
to the controller's run (same ``id`` = ``config.train.timestamp``) as a
non-primary writer in W&B *shared* mode. If that fails, plots are written to
``<output_dir>/viz/`` instead, so they are never silently lost.

Step alignment: the run-level step counter belongs to the controller, which logs
scalars with an explicit ``step=``. Rather than contend with it, these plots
declare their own x-axis (``viz/step`` via ``define_metric``). That value is
derived from the reward-call count and ``train_batch_per_replica`` (see
:func:`_estimate_step`), so it tracks the true training step closely but is an
estimate -- treat the x-axis as approximate.

Enable via ``[custom.alpamayo.traj_viz]`` in the TOML::

    [custom.alpamayo.traj_viz]
    enable = true
    every_n_steps = 10     # plot cadence, in estimated training steps
    max_clips = 4          # distinct clips to plot per flush
"""

from __future__ import annotations

import io
import os
import threading
from typing import Any

_LOCK = threading.Lock()

# W&B key namespace. Deliberately mirrors the SFT callback's `viz/val_sample_{idx}`
# (recipes/alpamayo1_5_sft/callbacks/viz_callback.py) so SFT and RL trajectory
# plots sort together in the W&B sidebar under one `viz/` section. `rollout_clip`
# rather than `val_sample` because these are on-policy training rollouts scored
# by the reward, not held-out validation samples.
_WANDB_KEY_PREFIX = "viz/rollout_clip_"

# W&B attach state. The reward -- and therefore this module -- runs in the
# ROLLOUT process, which does not call cosmos_rl's init_wandb(); only the
# controller does. So wandb.run is None here and we must attach to the
# controller's run ourselves, as a secondary writer in shared mode.
_WANDB_TRIED = False
_WANDB_RUN: Any = None
# One-shot flag so a persistent failure warns once, then goes quiet.
_WARNED = False

# clip idx -> reference-policy trajectory, [T, 2] numpy array (first seen)
_REF_TRAJ: dict[int, Any] = {}
# monotonic count of reward-function invocations
_CALL_COUNT = 0
# clip idx -> last estimated step at which that clip was plotted. Per-clip rather
# than global: with one global counter whichever completion happens to arrive at
# a due step claims it, so one clip monopolises the plots and the others are
# never drawn at a comparable step.
_LAST_PLOTTED: dict[int, int] = {}


def _cfg(config: object | None) -> dict:
    """Read ``[custom.alpamayo.traj_viz]``; returns {} when absent."""
    try:
        return dict(getattr(config, "custom")["alpamayo"]["traj_viz"])
    except (TypeError, KeyError, AttributeError):
        return {}


def _prompts_per_step(config: object | None) -> int:
    """train_batch_per_replica counts ROLLOUTS, so divide by n_generation."""
    try:
        tbpr = int(config.train.train_batch_per_replica)
        ngen = int(config.rollout.n_generation)
        return max(1, tbpr // max(1, ngen))
    except Exception:
        return 1


def _calls_per_step(config: object | None) -> int:
    """One reward call per completion => train_batch_per_replica calls/step."""
    try:
        return max(1, int(config.train.train_batch_per_replica))
    except Exception:
        return 1


def _estimate_step(config: object | None) -> int:
    return _CALL_COUNT // _calls_per_step(config)


def _ensure_wandb(config: object | None) -> Any:
    """Attach this process to the controller's W&B run, once.

    ``cosmos_rl.utils.report.wandb_logger.init_wandb`` is only called by the
    controller (``dispatcher/controller.py:109``), but rewards are computed in
    the rollout replica -- verified: 2480 ``compute_reward`` calls in
    ``rollout_0.log``, zero in ``controller.log``. So ``wandb.run`` is None here.

    We re-init with the controller's run ``id`` (``config.train.timestamp``,
    matching what ``init_wandb`` uses) in **shared mode** as a non-primary
    writer, which is W&B's supported multi-process pattern. Returns None if the
    attach fails, in which case the caller falls back to writing PNGs to disk.
    """
    global _WANDB_TRIED, _WANDB_RUN
    import wandb

    if wandb.run is not None:
        return wandb.run
    if _WANDB_TRIED:
        return _WANDB_RUN
    _WANDB_TRIED = True

    from cosmos_rl.utils.logging import logger

    try:
        timestamp = str(config.train.timestamp)
        experiment = getattr(config.logging, "experiment_name", None)
        name = os.path.join(experiment, timestamp) if experiment else timestamp
        run = wandb.init(
            project=config.logging.project_name,
            name=name,
            group=getattr(config.logging, "group_name", None),
            id=timestamp,
            dir=config.train.output_dir,
            resume="allow",
            settings=wandb.Settings(
                mode="shared",
                x_primary=False,  # the controller is the primary writer
                x_label="rollout",
                x_update_finish_state=False,  # don't finish the run when we exit
            ),
        )
        # Custom x-axis so these plots never contend with the controller's
        # explicit `step=` on the global step counter.
        run.define_metric("viz/step")
        run.define_metric(f"{_WANDB_KEY_PREFIX}*", step_metric="viz/step")
        _WANDB_RUN = run
        logger.info(f"[traj_viz] attached to W&B run '{name}' as secondary writer (shared mode)")
    except Exception as e:
        logger.warning(
            f"[traj_viz] could not attach to W&B ({type(e).__name__}: {e}); "
            "falling back to PNGs under <output_dir>/viz/"
        )
        _WANDB_RUN = None
    return _WANDB_RUN


def _to_xy(arr: Any) -> Any:
    """Squeeze a trajectory tensor/array down to [T, 2] (XY only)."""
    import numpy as np
    import torch

    if isinstance(arr, torch.Tensor):
        arr = arr.detach().float().cpu().numpy()
    arr = np.asarray(arr, dtype=float)
    while arr.ndim > 2:  # drop leading batch/mode dims
        arr = arr[0]
    return arr[:, :2]


def _plen(xy) -> float:
    """Arc length of an [T, 2] path, in metres."""
    import numpy as np

    if xy is None or len(xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=-1).sum())


def _render(
    idx: int,
    gt_xy,
    pred_xy,
    ref_xy,
    hist_xy,
    cot_text,
    step: int,
    meta: dict,
    cam_images: Any = None,
    cam_labels: list[str] | None = None,
) -> bytes:
    """Build a camera-grid + BEV figure and return it as PNG bytes.

    Deliberately built from the SFT callback's own primitives
    (:mod:`alpamayo.visualization.viz`) so an RL plot can be read side by side
    with an SFT one: same 90-degree-CC ego rotation, same fade-in markers
    (``x`` = start, ``o`` = end), same colour vocabulary, same tight equal-scale
    limits, same monospace ``info_text`` header and bottom-of-figure CoT, and
    (when ``cam_images`` is given) the same left-camera-grid / right-BEV
    gridspec layout as SFT's ``visualize_data``.

    The colour scheme maps onto SFT's as follows:

    ===================== ================= ==========================
    SFT curve             colour            RL counterpart here
    ===================== ================= ==========================
    Ground Truth          ``r``             human GT (identical)
    Predicted Trajectory  ``b``             the rollout being scored
    GT Tokenizer Recon    ``#ff8080`` --    (n/a -- no tokenizer here)
    History (past)        ``#777777``       ego history (identical)
    ===================== ================= ==========================

    The reference policy has no SFT counterpart, so it takes a *light dashed
    blue* (``#80b0ff``): light+dashed is SFT's idiom for "derived companion of
    the solid curve above it", and blue keeps it in the prediction family.

    ``cam_images`` (current-timestep camera views, one per element of
    ``cam_labels``) is optional and best-effort -- the reward runs in the
    rollout replica off ``get_reference_answer``, which does not itself carry
    ``image_frames``; the caller fetches these separately straight from the
    dataset already resident in-process. When unavailable, only the BEV panel
    is drawn, same as before.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from alpamayo.visualization.viz import (
        _plot_trajectory_with_fade,
        _set_tight_trajectory_limits,
        make_image_grid,
        rotate_90cc,
    )

    has_cams = cam_images is not None and len(cam_images) > 0

    if has_cams:
        # 2 columns rather than 1 row of N: half as wide per tile => each tile
        # renders roughly 2x larger for a typical 4-camera rig.
        import math

        n_cams = len(cam_images)
        cols = min(2, n_cams)
        rows = math.ceil(n_cams / cols)
        tile_h_px, tile_w_px = cam_images.shape[1], cam_images.shape[2]
        grid_aspect = (rows * tile_h_px) / (cols * tile_w_px)  # height / width

        fig_width = 12.0
        grid_width_frac = 1.3 / (1.3 + 1)  # matches width_ratios below
        grid_height = fig_width * grid_width_frac * grid_aspect
        # Leave headroom for the header text above and CoT caption below;
        # never shrink below what the BEV legend/axes need to stay readable.
        fig_height = max(grid_height + 1.5, 6.0)

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=110)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1])
        ax_grid = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[0, 1])

        grid = make_image_grid(cam_images, columns=cols)
        ax_grid.imshow(grid)
        ax_grid.axis("off")
        if cam_labels:
            tile_h, tile_w = cam_images.shape[1], cam_images.shape[2]
            _lbl_bbox = dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.55)
            for c, lbl in enumerate(cam_labels):
                r, col = divmod(c, cols)
                ax_grid.text(
                    (col + 0.5) * tile_w, r * tile_h + 4, lbl, ha="center", va="top",
                    fontsize=7, color="white", bbox=_lbl_bbox,
                )
    else:
        fig = plt.figure(figsize=(7.0, 6.0), dpi=110)
        ax = fig.add_subplot(1, 1, 1)

    # viz.py's helpers take (2, N) and apply the same ego rotation the SFT panel
    # uses, so identical inputs land at identical pixels in both plots.
    def _prep(xy):
        return None if xy is None else rotate_90cc(xy.T)

    plotted = []

    def _draw(xy, color, label, linestyle="-"):
        rot = _prep(xy)
        if rot is None:
            return
        plotted.append(rot)
        _plot_trajectory_with_fade(
            ax, rot, color=color, label=label, fade_in=True, linestyle=linestyle
        )

    # One deliberate departure from SFT, which draws GT last: here the rollout
    # goes on top. A *collapsed* rollout is a zero-length path sitting exactly on
    # the origin, and under SFT's order it hides beneath the GT start marker --
    # which is precisely the failure these plots exist to make visible.
    _draw(hist_xy, "#777777", "History (past)")
    _draw(ref_xy, "#80b0ff", "Reference Policy (KL anchor)", linestyle="--")
    _draw(gt_xy, "r", "Ground Truth Trajectory")
    _draw(pred_xy, "b", "Rollout Trajectory")

    # SFT's axis labels verbatim -- the marker legend lives in the ylabel there.
    ax.set_ylabel("y coordinate (meters) x for start, o for end")
    ax.set_xlabel("x coordinate (meters)")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        borderaxespad=0.0,
    )
    _set_tight_trajectory_limits(ax, plotted)

    # Header in SFT's info_text style: monospace, top-left, fontsize 8.
    header = [f"step {step}  |  clip idx {idx}"]
    scalars = []
    if "traj_L2" in meta:
        scalars.append(f"ADE={meta['traj_L2']:.2f}m")
    if "reward" in meta:
        scalars.append(f"reward={meta['reward']:.3f}")
    if "reasoning_score" in meta:
        scalars.append(f"reasoning={meta['reasoning_score']:.3f}")
    if "comfort_reward" in meta:
        scalars.append(f"comfort={meta['comfort_reward']:.3f}")
    if scalars:
        header.append("  ".join(scalars))
    # Path length makes the degenerate "predict a constant trajectory" mode
    # numerically explicit: rollout length collapses to ~0 while GT keeps moving.
    header.append(f"path len: GT {_plen(gt_xy):.1f}m  rollout {_plen(pred_xy):.1f}m")
    fig.text(
        0.01, 0.985, "\n".join(header),
        ha="left", va="top", fontsize=8, family="monospace", color="#222222",
    )

    if cot_text:
        import textwrap

        wrapped = textwrap.fill(str(cot_text), width=110)
        fig.text(
            0.5, 0.01, wrapped,
            ha="center", va="bottom", fontsize=9,
        )
        # Reserve enough bottom margin for however many lines the caption wrapped
        # to, so it never collides with the BEV panel's x-axis label above it.
        n_lines = wrapped.count("\n") + 1
        bottom_margin = 0.05 + 0.035 * n_lines
        fig.tight_layout(rect=[0, bottom_margin, 1, 0.94])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.94])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _fetch_current_frame_images(idx: int, split: str) -> tuple[Any, Any] | None:
    """Best-effort fetch of the current-timestep (most recent history frame)
    camera images for clip ``idx``, straight from the dataset already resident
    in this process -- no extra shm/prefetch bandwidth for every rollout,
    since this only runs for the handful of clips actually rendered.

    Returns ``(images, camera_indices)`` -- ``images``: [N_cams, H, W, C]
    uint8/float numpy array; ``camera_indices``: list[int], len N_cams -- or
    None on any failure (dataset not carrying images, index error, etc).

    Handles both dataset layouts:
      * unflattened: ``image_frames`` [N_cams, num_frames, C, H, W],
        ``camera_indices`` [N_cams].
      * ``reshape_tensors_for_rl=True`` (this recipe's default): flattened to
        ``[N_cams * num_frames, 1, C, H, W]`` with ``camera_indices``
        ``repeat_interleave``'d to ``[N_cams * num_frames]`` -- in which case
        the most-recent frame per camera is that camera id's LAST occurrence.
    """
    try:
        import alpamayo1_x_rl.state as alp_state

        # Mirrors base_dataset.py's own resolution exactly (single `.dataset`
        # hop): the loader's `.dataset` is the raw PAIDataset, or the
        # NodePrefetchDatasetWrapper around it when node-prefetch is enabled
        # ([custom.alpamayo.prefetch].capacity > 0) -- both support `[idx]`.
        raw_dataset = alp_state.get_dataloaders()[split].dataset
        sample = raw_dataset[idx]
        frames = sample.get("image_frames")
        cam_idx = sample.get("camera_indices")
        if frames is None or cam_idx is None:
            from cosmos_rl.utils.logging import logger

            logger.debug(
                f"[traj_viz] no image_frames/camera_indices in sample for clip {idx} "
                f"(keys={list(sample.keys())})"
            )
            return None

        cam_idx_list = [int(c) for c in cam_idx.reshape(-1).tolist()]

        if frames.dim() == 5 and frames.shape[1] != 1:
            # Unflattened: one cam id per camera already, just take the last frame.
            current = frames[:, -1]
            images = current.permute(0, 2, 3, 1).detach().cpu().numpy()
            return images, cam_idx_list

        # Flattened [N_cams*num_frames, 1, C, H, W]: keep each camera id's LAST
        # occurrence (its most recent frame), in first-seen order.
        flat = frames.reshape(frames.shape[0], *frames.shape[2:])
        last_pos: dict[int, int] = {}
        order: list[int] = []
        for i, c in enumerate(cam_idx_list):
            if c not in last_pos:
                order.append(c)
            last_pos[c] = i
        sel = [last_pos[c] for c in order]
        images = flat[sel].permute(0, 2, 3, 1).detach().cpu().numpy()
        return images, order
    except Exception as e:
        from cosmos_rl.utils.logging import logger

        logger.debug(f"[traj_viz] camera fetch failed for clip {idx}: {type(e).__name__}: {e}")
        return None


def _png_to_wandb_image(png: bytes, caption: str | None = None) -> Any:
    """wandb.Image() takes a path, ndarray or PIL Image -- NOT a file-like
    object (it calls .ndim on whatever it is given). Decode via PIL, and
    .load() so the image survives the buffer going out of scope."""
    import wandb
    from PIL import Image as _PILImage

    pil = _PILImage.open(io.BytesIO(png))
    pil.load()
    return wandb.Image(pil.convert("RGB"), caption=caption)


def record_and_maybe_log(
    idx: int | None,
    gt_xyz: Any,
    pred_xyz: Any,
    reward_dict: dict[str, float],
    config: object | None,
    hist_xyz: Any = None,
    cot_text: str | None = None,
) -> None:
    """Cache the reference trajectory and periodically log BEV plots to W&B.

    Best-effort by design: any failure here is logged at debug level and
    swallowed, because a plotting bug must never take down a training run.
    """
    global _CALL_COUNT

    cfg = _cfg(config)
    if not cfg.get("enable", False) or idx is None:
        return

    from cosmos_rl.utils.logging import logger

    try:
        import numpy as np  # noqa: F401

        every_n = int(cfg.get("every_n_steps", 10))
        max_clips = int(cfg.get("max_clips", 4))

        pred_xy = _to_xy(pred_xyz)
        gt_xy = _to_xy(gt_xyz)
        hist_xy = _to_xy(hist_xyz) if hist_xyz is not None else None

        with _LOCK:
            _CALL_COUNT += 1
            # First sighting of this clip == a sample from the reference policy,
            # since training has not yet moved the weights.
            if idx not in _REF_TRAJ:
                _REF_TRAJ[idx] = pred_xy.copy()
            ref_xy = _REF_TRAJ.get(idx)

            # Only plot the first `max_clips` clips we ever see, so the panel
            # stays a stable set across the run instead of drifting. Dicts
            # preserve insertion order, so list() (not sorted(), which reorders
            # by numeric idx and silently breaks "first-seen") gives that set.
            if idx not in list(_REF_TRAJ)[:max_clips]:
                return

            step = _estimate_step(config)
            if step < _LAST_PLOTTED.get(idx, -(10**9)) + every_n:
                return
            _LAST_PLOTTED[idx] = step  # claim for this clip only

        # Camera views at the current timestep, folded into the same figure as
        # the BEV plot. Isolated in its own try/except so a camera-fetch
        # failure (e.g. this dataset variant carries no images) never disables
        # the BEV plotting below, which is the module's outer failure boundary.
        cam_images, cam_labels = None, None
        try:
            fetched = _fetch_current_frame_images(idx, split="train")
            if fetched is not None:
                from alpamayo.common.constants import CAMERA_INDICES_TO_DISPLAY_NAMES

                cam_images, cam_ids = fetched
                cam_labels = [
                    CAMERA_INDICES_TO_DISPLAY_NAMES.get(int(i), f"cam{int(i)}") for i in cam_ids
                ]
        except Exception as cam_e:
            logger.debug(f"[traj_viz] camera grid skipped for clip {idx}: {cam_e}")

        png = _render(
            idx, gt_xy, pred_xy, ref_xy, hist_xy, cot_text, step, reward_dict,
            cam_images=cam_images, cam_labels=cam_labels,
        )

        # Caption in the SFT callback's format ("idx N · clip X · step S · ...").
        caption = f"clip {idx} · step {step}"
        if "traj_L2" in reward_dict:
            caption += f" · ADE {reward_dict['traj_L2']:.2f}m"

        run = _ensure_wandb(config)
        if run is not None:
            # No explicit step=: the run-level step belongs to the controller.
            # "viz/step" is our own x-axis, declared via define_metric().
            run.log({
                "viz/step": step,
                f"{_WANDB_KEY_PREFIX}{idx}": _png_to_wandb_image(png, caption),
            })
            logger.info(f"[traj_viz] logged clip {idx} to W&B at viz/step {step}")
        else:
            out_dir = os.path.join(str(config.train.output_dir), "viz")
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"step{step:07d}_clip{idx}.png")
            with open(path, "wb") as fh:
                fh.write(png)
            logger.info(f"[traj_viz] wrote {path}")
    except Exception as e:  # never break training for a plot
        # First failure at WARNING, the rest at DEBUG. Logging these at debug
        # only is how two real bugs in this module stayed invisible for a whole
        # run -- silence is not the same as success.
        global _WARNED
        try:
            from cosmos_rl.utils.logging import logger as _lg

            if not _WARNED:
                _WARNED = True
                import traceback

                _lg.warning(f"[traj_viz] disabled after error: {type(e).__name__}: {e}")
                _lg.warning(f"[traj_viz] {traceback.format_exc()}")
            else:
                _lg.debug(f"[traj_viz] skipped ({type(e).__name__}: {e})")
        except Exception:
            pass
