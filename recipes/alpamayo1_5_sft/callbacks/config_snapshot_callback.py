# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""HF Trainer callback that snapshots the run config into each checkpoint dir.

``train_hf.py`` writes the resolved Hydra config once to ``paths.output_dir``,
but individual ``checkpoint-N/`` dirs carry no config of their own. Once a
checkpoint is copied elsewhere (or the original run dir on local NVMe is wiped)
there is no longer any record of what produced it -- recovering the settings
means reading ``training_args.bin`` for ``output_dir`` and hunting through
``outputs/<date>/<time>/.hydra/``.

This callback copies the config (plus the ``.hydra/`` sidecars, when the run was
launched through Hydra) into every checkpoint as it is written, so a checkpoint
directory is self-describing.
"""

from __future__ import annotations

import logging
import os
import shutil

from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

logger = logging.getLogger(__name__)


class ConfigSnapshotCallback(TrainerCallback):
    """Copy the run's config.yaml (and .hydra/) into each checkpoint directory.

    Args:
        run_output_dir: The run root -- where ``train_hf.py`` saved ``config.yaml``.
        hydra_dir: Optional ``.hydra`` directory to copy alongside it. Defaults to
            the Hydra run dir of the current process, if any.
    """

    def __init__(self, run_output_dir: str, hydra_dir: str | None = None) -> None:
        self.run_output_dir = run_output_dir
        if hydra_dir is None:
            hydra_dir = self._detect_hydra_dir()
        self.hydra_dir = hydra_dir

    @staticmethod
    def _detect_hydra_dir() -> str | None:
        try:
            from hydra.core.hydra_config import HydraConfig

            if not HydraConfig.initialized():
                return None
            candidate = os.path.join(HydraConfig.get().runtime.output_dir, ".hydra")
        except Exception:  # hydra not initialized / unavailable
            return None
        return candidate if os.path.isdir(candidate) else None

    def on_save(self, args, state, control, **kwargs):  # noqa: D102
        if not state.is_world_process_zero:
            return
        ckpt_dir = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        if not os.path.isdir(ckpt_dir):
            return
        # Never let a bookkeeping copy take down a training run.
        try:
            src_cfg = os.path.join(self.run_output_dir, "config.yaml")
            if os.path.isfile(src_cfg):
                shutil.copy2(src_cfg, os.path.join(ckpt_dir, "run_config.yaml"))
            if self.hydra_dir is not None:
                shutil.copytree(
                    self.hydra_dir,
                    os.path.join(ckpt_dir, "hydra"),
                    dirs_exist_ok=True,
                )
        except Exception as e:
            logger.warning(f"Failed to snapshot config into {ckpt_dir}: {e}")
