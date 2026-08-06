"""Minimal W&B init, mirroring alpamayo.common.wandb_utils.init_wandb's run-id
persistence (resume the same run across restarts) without pulling in
alpamayo_r1 (that module's only reason to depend on it is a RankedLogger).
"""

import logging
import os

import wandb

logger = logging.getLogger(__name__)


def _is_rank_zero() -> bool:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


def _save_wandb_id(output_dir: str, wandb_id: str) -> None:
    with open(os.path.join(output_dir, ".wandb_id"), "w", encoding="utf-8") as f:
        f.write(wandb_id)


def _check_wandb_id(output_dir: str) -> str | None:
    path = os.path.join(output_dir, ".wandb_id")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    return None


def init_wandb(
    key: str | None,
    team: str,
    project: str,
    group: str,
    name: str,
    output_dir: str,
    **kwargs,
) -> None:
    """Initialize wandb on rank 0 only, resuming the same run id across restarts."""
    if not _is_rank_zero():
        return
    if key is not None:
        wandb.login(key=key)

    os.makedirs(output_dir, exist_ok=True)
    wandb_id = _check_wandb_id(output_dir)
    if wandb_id is None:
        wandb_id = wandb.util.generate_id()
        _save_wandb_id(output_dir, wandb_id)
        logger.info("Starting a new wandb run with ID %s", wandb_id)
    else:
        logger.info("Resuming wandb run with ID %s", wandb_id)

    wandb.init(
        id=wandb_id,
        entity=team,
        project=project,
        group=group,
        name=name,
        dir=output_dir,
        resume="allow",
        **kwargs,
    )
