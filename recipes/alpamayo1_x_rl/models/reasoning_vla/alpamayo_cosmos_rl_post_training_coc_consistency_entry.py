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

"""First RL smoke test for the CoC-action consistency reward (2026-08-16).

NOTE: fork addition, not upstream. A near-copy of
``alpamayo_cosmos_rl_post_training_reasoning_entry.py`` (same reward function --
``aggregated_reward_with_reasoning.compute_reward``, which is where
``coc_consistency_weight`` lives), but pointed at the plain motion-only PAI mini
dataset (``pai_nav_subset_19chunks``, the 16-clip smoke set already proven to load
correctly by prior local runs -- see ``/opt/dlami/nvme/rod/ckpts/alpamayo_rl/``)
instead of the reasoning-labeled dataset.

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

# Proven-working 16-clip smoke set (copied from clip_index_mini_16smoke.parquet.bak to
# a real .parquet name -- see module docstring). Override via env var if needed.
_PAI_LOCAL_DIR = os.getenv("ALPAMAYO_PAI_LOCAL_DIR", "/home/rod/datasets/pai_nav_subset_19chunks")

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
    """Compute aggregated reward for a single ReasoningVLA rollout.

    Same reward path as the reasoning entry point (`aggregated_reward_with_reasoning.
    compute_reward`) -- coc_consistency lives there, not in the plain motion-only
    `aggregated_reward.py`.
    """
    import alpamayo1_x_rl.state as alp_state
    from alpamayo1_x_rl.rewards.aggregated_reward_with_reasoning import compute_reward

    assert isinstance(reference, dict) and reference, (
        f"Expected a non-empty dict for reference, got {type(reference).__name__}: {reference!r}"
    )
    assert not isinstance(to_be_evaluated, (list, tuple)), (
        "The reasoning reward does not support group_reward_calculation=true; "
        "set it to false in the TOML for this entry point."
    )
    return compute_reward(
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
    hydra_overrides=[
        f"data.train.dataset.local_dir={_PAI_LOCAL_DIR}",
        "data.train.dataset.clip_index_metadata=clip_index_smoke16.parquet",
        "data.train.dataset.features_metadata=features.csv",
        "data.train.dataset.use_default_keyframe=True",
        "data.train.dataset.reasoning_metadata=null",
    ],
)

if __name__ == "__main__":
    REASONING_VLA_SPEC.launch()
