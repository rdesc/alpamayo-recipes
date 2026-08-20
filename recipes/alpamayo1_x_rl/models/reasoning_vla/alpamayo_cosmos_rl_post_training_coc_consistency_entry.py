# NOTE: new file in this fork -- not in upstream NVlabs/alpamayo-recipes.
"""RL entry point for the CoC-action consistency reward (2026-08-16).

NOTE: fork addition, not upstream. A near-copy of
``alpamayo_cosmos_rl_post_training_reasoning_entry.py`` (same reward function --
``aggregated_reward_with_reasoning.compute_reward``, which is where
``coc_consistency_weight`` lives), but pointed at the plain motion-only PAI
dataset via the ``coc`` profile in ``dataset_spec.py`` instead of the
reasoning-labeled one. That profile follows ``$ALPAMAYO_PAI_LOCAL_DIR``; as of
2026-08-17 it defaults to the full ``clip_index_mini.parquet`` (2213 clips on
``pai_nav_subset_50chunks``) rather than the original 16-clip smoke set, which
is still reachable via ``ALPAMAYO_PAI_CLIP_INDEX=clip_index_smoke16.parquet``.
Paired TOML for the scaled run: ``toml/alpamayo_rvla_rl_coc_consistency_50chunks.toml``.

Why not the reasoning dataset: this run's whole point is to isolate and validate
coc_consistency in isolation (``traj_l2_weight = comfort_weight = reasoning_weight =
0.0``, only ``coc_consistency_weight`` nonzero -- see the paired TOML,
``toml/alpamayo_rvla_rl_local_test_coc_consistency.toml``). coc_consistency needs
nothing but the model's OWN predicted CoC + predicted trajectory -- no ground-truth
CoC required -- so the reasoning-labeled dataset brings no benefit here, and the
Lingo-Judge checkpoint it would otherwise need is not available on this box right now
(lost to an ephemeral-storage reset, matching a prior known failure mode on this
instance). See the paired FIX in ``aggregated_reward_with_reasoning.py``: the
reasoning-score gate that used to clamp the ENTIRE reward to -1.0 whenever no
ground-truth CoC was present is now skipped whenever ``reasoning_weight == 0.0``.

Once Lingo-Judge is re-fetched, prefer switching back to
``alpamayo_cosmos_rl_post_training_reasoning_entry.py`` + the reasoning-labeled
dataset for a true joint reasoning+coc_consistency run.
"""

# ruff: noqa: E402

import os


os.environ.setdefault("COSMOS_HEARTBEAT_TIMEOUT", "600")
os.environ.setdefault("COSMOS_LOG_LEVEL", "DEBUG")

# NOTE: fork change. Dataset selection moved to `dataset_spec.resolve()` so all
# entry points share one definition. Resolution happens at import: a missing env
# var or a missing index file fails here, before any GPU is touched.
from alpamayo1_x_rl.dataset_spec import resolve as _resolve_dataset

_DATASET = _resolve_dataset("coc")


# ---------------------------------------------------------------------------
# vLLM registration
# ---------------------------------------------------------------------------
from cosmos_rl.utils.logging import logger

try:
    from vllm import ModelRegistry as vllm_model_registry

    from alpamayo1_x_rl.models.reasoning_vla.vllm_wrapper import ReasoningVLAModelForVLLM

    vllm_model_registry.register_model("ReasoningVLA", ReasoningVLAModelForVLLM)
except Exception as e:
    logger.warning(f"Failed to register ReasoningVLA model with vLLM: {e}")

# ---------------------------------------------------------------------------
# Model spec components
# ---------------------------------------------------------------------------
from alpamayo1_x_rl.models._spec import ModelSpec
from alpamayo1_x_rl.models.reasoning_vla.cosmos_wrapper import ReasoningVLACosmos
from alpamayo1_x_rl.models.reasoning_vla.data_packer import RVLADataPacker
from alpamayo1_x_rl.models.reasoning_vla.rollout import ReasoningVLAVllmRollout  # noqa: F401 (Cosmos registry)
from alpamayo1_x_rl.models.reasoning_vla.trainer import ReasoningVLAGRPOTrainer  # noqa: F401 (Cosmos registry)
from alpamayo1_x_rl.models.reasoning_vla.weight_mapper import ReasoningVLAWeightMapper


def _reasoning_vla_reward_fn(to_be_evaluated, reference=None, *args, config=None, **kwargs):
    """Compute aggregated reward for one ReasoningVLA rollout, or a whole group.

    Same reward path as the reasoning entry point (`aggregated_reward_with_reasoning`)
    -- coc_consistency lives there, not in the plain motion-only `aggregated_reward.py`.
    Dispatches on the shape of `to_be_evaluated`, same pattern as the motion-only entry
    point: a single string when `[train.train_policy].group_reward_calculation = false`
    (the default), or a list/tuple of every completion sharing one prompt when it's
    `true` -- group mode batches the CoC claim extraction across the group in one Qwen
    call, see `compute_group_reward`'s docstring for why that's the actual point of
    turning the flag on for this reward.
    """
    import alpamayo1_x_rl.state as alp_state
    from alpamayo1_x_rl.rewards.aggregated_reward_with_reasoning import (
        compute_group_reward,
        compute_reward,
    )

    assert isinstance(reference, dict) and reference, (
        f"Expected a non-empty dict for reference, got {type(reference).__name__}: {reference!r}"
    )
    fn = compute_group_reward if isinstance(to_be_evaluated, (list, tuple)) else compute_reward
    return fn(
        to_be_evaluated,
        reference,
        tokenizer=alp_state.get_tokenizer(),
        traj_tokenizer=alp_state.get_traj_tokenizer(),
        config=config,
        model_config=alp_state.get_ckpt_cfg(),
    )


REASONING_VLA_SPEC = ModelSpec(
    cosmos_wrapper=ReasoningVLACosmos,
    weight_mapper=ReasoningVLAWeightMapper,
    data_packer_cls=RVLADataPacker,
    reward_fn=_reasoning_vla_reward_fn,
    hydra_config_path="hydra_configs",
    hydra_config_name="alpamayo1_5_rvla_rl_pai",
    hydra_overrides=_DATASET.overrides,
)

if __name__ == "__main__":
    REASONING_VLA_SPEC.launch()
