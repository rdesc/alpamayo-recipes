# NOTE: modified in this fork -- see upstream NVlabs/alpamayo-recipes for the original.
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

"""ReasoningVLA RL post-training entry point (Cosmos-RL / GRPO).

Sets up env vars and registers the ReasoningVLA model with vLLM before
assembling a ``ModelSpec`` with all Cosmos-RL components — model wrapper,
weight mapper, data packer, and reward function — and launching GRPO
training on the PAI dataset.
"""

# ruff: noqa: E402

import os


os.environ.setdefault("COSMOS_HEARTBEAT_TIMEOUT", "600")
os.environ.setdefault("COSMOS_LOG_LEVEL", "DEBUG")

# NOTE: fork change. Dataset selection moved to `dataset_spec.resolve()` so all
# entry points share one definition. Resolution happens at import: a missing env
# var or a missing index file fails here, before any GPU is touched.
from alpamayo1_x_rl.dataset_spec import resolve as _resolve_dataset

_DATASET = _resolve_dataset("reasoning")


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

    NOTE: fork addition -- dispatches on the shape of `to_be_evaluated`, same pattern
    as the motion-only entry point's `_reasoning_vla_reward_fn`
    (`alpamayo_cosmos_rl_post_training_entry.py`): a single string when
    `[train.train_policy].group_reward_calculation = false` (the default), or a
    list/tuple of every completion sharing one prompt when it's `true`. Group mode
    batches the CoC claim extraction across the group in one Qwen call -- see
    `aggregated_reward_with_reasoning.compute_group_reward`'s docstring. Previously
    this asserted group mode was unsupported; that's no longer true now that
    compute_group_reward exists.
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