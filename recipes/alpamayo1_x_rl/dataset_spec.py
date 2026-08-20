# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""Single source of truth for which dataset an RL entry point trains on.

Before this module, each entry point in ``models/reasoning_vla/`` resolved its
own dataset and built its own ``hydra_overrides`` list. Three copies drifted
apart in exactly the way copies do:

* the motion entry required ``ALPAMAYO_PAI_LOCAL_DIR`` and failed loudly without
  it;
* the reasoning entry did the same for ``ALPAMAYO_PAI_REASONING_LOCAL_DIR``;
* the CoC entry defaulted to a hardcoded ``/home/rod/...`` path, so a missing
  env var silently trained on whatever happened to be at that path -- with no
  error, and no record in the run of which dataset was actually used.

That last one is the failure this module exists to make impossible.

**Design.** A profile names the *shape* of a dataset selection (which index
files, which keyframe mode, whether reasoning labels are expected). The machine
path is NOT part of the profile -- it comes from an environment variable with
no default, so every launch states it explicitly. Filenames inside the root
(``clip_index_mini.parquet``) stay literals here: they identify the profile
rather than the machine, and pushing them into env vars would move the drift
problem rather than remove it.

**Fail fast, and fail completely.** ``resolve()`` validates that the root, the
train index, the val index, and any reasoning metadata all exist on disk, and
reports every problem at once. Checking the env var alone is not enough -- the
`pai_nav_subset_50chunks` download had a valid root for hours before its
``clip_index_mini.parquet`` was curated, and a run started in that window would
have died deep inside dataset construction with a much worse error.

**Provenance.** The resolved selection is stashed in :data:`LAST_RESOLVED` so
``utils/cosmos_patches.apply_dataset_config_logging_patch`` can inject it into
the W&B config. Cosmos-RL's W&B snapshot is ``config.model_dump()`` of the TOML
(``utils/report/wandb_logger.py:88``), and dataset selection reaches the run
through Hydra overrides instead -- so without that injection the single most
important fact about a run is absent from its permanent record.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class DatasetProfile:
    """A named dataset *shape*; the machine path comes from ``root_env``."""

    name: str
    # Environment variable holding the dataset root. Required, never defaulted.
    root_env: str
    # Train-split clip index, relative to the root.
    clip_index: str
    features: str = "features.csv"
    # Alpamayo keyframe selection. The motion subsets have no per-clip
    # `event_t0s`, so they must use the default keyframe; the reasoning subset
    # has them and must not.
    use_default_keyframe: bool = True
    # Path (relative to root) of the CoT label table, or None for `null`.
    reasoning_metadata: Optional[str] = None
    # Held-out val index, relative to the root. None means "no override", which
    # leaves `data.val` interpolating from `data.train` -- i.e. validation on a
    # strided subsample of the TRAINING windows. That is in-distribution and
    # cannot separate real improvement from overfitting, so it is declared
    # explicitly per profile rather than inferred from whether a file happens to
    # be present: a silent fallback to in-distribution val is precisely the
    # failure mode worth refusing.
    val_clip_index: Optional[str] = None
    # Optional env var permitted to override `clip_index` (e.g. to swap a smoke
    # subset for the full one). Unset env var keeps the profile default.
    clip_index_env: Optional[str] = None


PROFILES: dict[str, DatasetProfile] = {
    "motion": DatasetProfile(
        name="motion",
        root_env="ALPAMAYO_PAI_LOCAL_DIR",
        clip_index="clip_index_mini.parquet",
        use_default_keyframe=True,
        reasoning_metadata=None,
        val_clip_index="clip_index_mini_val.parquet",
    ),
    "reasoning": DatasetProfile(
        name="reasoning",
        root_env="ALPAMAYO_PAI_REASONING_LOCAL_DIR",
        clip_index="clip_index_reasoning_mini.parquet",
        use_default_keyframe=False,
        reasoning_metadata="reasoning/ood_reasoning.parquet",
        # No held-out index in this root: the CoT-labelled subset is 16 clips
        # (14 train / 2 val) and both splits live in the same file. Val
        # therefore mirrors train here, which is a real limitation of the
        # reasoning data, recorded rather than hidden.
        val_clip_index=None,
    ),
    "coc": DatasetProfile(
        name="coc",
        root_env="ALPAMAYO_PAI_LOCAL_DIR",
        # Was `clip_index_smoke16.parquet`: the 16-clip set the CoC reward was
        # first validated on. Now the same full train index the motion profile
        # uses, so CoC scales to whatever root the env var names (2213 clips on
        # pai_nav_subset_50chunks). The smoke set is still one env var away --
        # `ALPAMAYO_PAI_CLIP_INDEX=clip_index_smoke16.parquet` -- which is the
        # right direction for the default to point: a scaled run that silently
        # uses 16 clips is a much worse failure than a smoke test that has to
        # ask for itself.
        clip_index="clip_index_mini.parquet",
        clip_index_env="ALPAMAYO_PAI_CLIP_INDEX",
        use_default_keyframe=True,
        reasoning_metadata=None,
        val_clip_index="clip_index_mini_val.parquet",
    ),
}


@dataclass
class ResolvedDataset:
    """What :func:`resolve` produced: the overrides plus a provenance record."""

    profile: str
    root: str
    clip_index: str
    features: str
    use_default_keyframe: bool
    reasoning_metadata: Optional[str]
    val_clip_index: Optional[str]
    overrides: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


LAST_RESOLVED: Optional[ResolvedDataset] = None


def _clip_count(path: str) -> Optional[int]:
    """Row count of a clip index, or None if it cannot be read.

    Best-effort: a provenance nicety must never be able to fail a launch.
    """
    try:
        import pandas as pd

        return int(len(pd.read_parquet(path)))
    except Exception:
        return None


def resolve(profile_name: str) -> ResolvedDataset:
    """Resolve *profile_name* into Hydra overrides, or raise explaining why not.

    Raises:
        KeyError: unknown profile name.
        RuntimeError: the root env var is unset, or a resolved path is missing.
            Every problem found is reported in one message rather than one per
            launch attempt.
    """
    global LAST_RESOLVED

    if profile_name not in PROFILES:
        raise KeyError(
            f"Unknown dataset profile {profile_name!r}. Known: {sorted(PROFILES)}"
        )
    p = PROFILES[profile_name]

    root = os.getenv(p.root_env)
    if not root:
        raise RuntimeError(
            f"Missing required env var {p.root_env} (dataset root for the "
            f"{p.name!r} profile). There is deliberately no default: a hardcoded "
            "fallback path is how a run silently trains on the wrong dataset. "
            f"Set it, e.g. `export {p.root_env}=/home/rod/datasets/pai_nav_subset_50chunks`"
            " (or the stage-dataset location, /opt/dlami/nvme/rod/datasets/<name>)."
        )

    clip_index = p.clip_index
    if p.clip_index_env:
        clip_index = os.getenv(p.clip_index_env) or clip_index

    problems: list[str] = []
    if not os.path.isdir(root):
        problems.append(f"root does not exist or is not a directory: {root}")
    else:
        for label, rel in (
            ("clip_index_metadata", clip_index),
            ("features_metadata", p.features),
            ("reasoning_metadata", p.reasoning_metadata),
            ("val clip_index_metadata", p.val_clip_index),
        ):
            if rel is None:
                continue
            full = os.path.join(root, rel)
            if not os.path.exists(full):
                problems.append(f"{label} not found: {full}")
    if problems:
        raise RuntimeError(
            f"Dataset profile {p.name!r} rooted at {root} is incomplete:\n  - "
            + "\n  - ".join(problems)
            + "\nResolved from $"
            + p.root_env
            + ". Curate the missing index (scripts/curate_pai_samples.py) or "
            "point the env var at a complete root."
        )

    overrides = [
        f"data.train.dataset.local_dir={root}",
        f"data.train.dataset.clip_index_metadata={clip_index}",
        f"data.train.dataset.features_metadata={p.features}",
        f"data.train.dataset.use_default_keyframe={p.use_default_keyframe}",
        "data.train.dataset.reasoning_metadata="
        + (p.reasoning_metadata if p.reasoning_metadata else "null"),
    ]
    if p.val_clip_index:
        overrides.append(
            f"data.val.dataset.clip_index_metadata={p.val_clip_index}"
        )

    summary = {
        "profile": p.name,
        "root": root,
        "resolved_from_env": p.root_env,
        "clip_index": clip_index,
        "features": p.features,
        "use_default_keyframe": p.use_default_keyframe,
        "reasoning_metadata": p.reasoning_metadata or "null",
        # Explicit about the in-distribution case rather than omitting the key,
        # so a W&B config panel distinguishes "held out" from "unknown".
        "val_clip_index": p.val_clip_index or "(mirrors train: IN-DISTRIBUTION)",
        "n_train_clips": _clip_count(os.path.join(root, clip_index)),
        "n_val_clips": (
            _clip_count(os.path.join(root, p.val_clip_index))
            if p.val_clip_index
            else None
        ),
    }

    resolved = ResolvedDataset(
        profile=p.name,
        root=root,
        clip_index=clip_index,
        features=p.features,
        use_default_keyframe=p.use_default_keyframe,
        reasoning_metadata=p.reasoning_metadata,
        val_clip_index=p.val_clip_index,
        overrides=overrides,
        summary=summary,
    )
    LAST_RESOLVED = resolved
    return resolved
