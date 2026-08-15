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

"""Cosmos-RL dataset wrapper for Alpamayo models."""

from __future__ import annotations

from typing import Any

from cosmos_rl.policy.config import Config
from cosmos_rl.utils.logging import logger
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import alpamayo1_x_rl.state as alp_state
from alpamayo1_x_rl.prefetch.server import set_custom_cfg


class AlpamayoCosmosDataset(Dataset):
    """Lightweight dataset adapter between Alpamayo and Cosmos-RL.

    Returns only index identifiers; actual data loading is handled by
    the DataPacker. Both ReasoningVLA and ExpertModel use this class.
    """

    def __init__(self, split: str = "train"):
        self.split = split
        # Positional index -> underlying dataset index. Identity for train; a
        # strided subset for val (see _build_indices).
        self.indices: list[int] | None = None

    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        """Initialize config, dataloader, and tokenizer from global state."""
        self.config = config
        set_custom_cfg(config)
        self.dataset = alp_state.get_dataloaders()[self.split].dataset
        self.tokenizer = alp_state.get_tokenizer()
        self._build_indices(config)

    # NOTE: fork addition -- validation subsetting.
    #
    # Cosmos-RL validates over the WHOLE val dataset every `[validation].freq`
    # steps (`len(val_dataset.val_set)` in dispatcher/data/data_fetcher.py:436),
    # generating `[validation].n_generation` rollouts per window. On the 891-window
    # PAI nav subset at n_generation=6 that is 5346 rollouts, i.e. tens of minutes
    # of pure eval per validation -- more than the training it interrupts. A fixed
    # stride keeps the curve comparable step-to-step while making it affordable,
    # and matches how the SFT recipe samples its val split
    # (callbacks/val_metric_callback.py takes `range(0, len(ds), stride)`).
    def _build_indices(self, config: object) -> None:
        """Resolve `[custom.alpamayo.validation]` stride/max_samples into indices."""
        if self.split != "val":
            self.indices = None
            return
        try:
            cfg = dict(getattr(config, "custom")["alpamayo"]["validation"])
        except (TypeError, KeyError, AttributeError):
            cfg = {}
        stride = max(1, int(cfg.get("stride", 1)))
        max_samples = int(cfg.get("max_samples", 0) or 0)

        indices = list(range(0, len(self.dataset), stride))
        if max_samples > 0:
            indices = indices[:max_samples]
        self.indices = indices
        logger.info(
            f"[AlpamayoCosmosDataset] val split: {len(indices)} of {len(self.dataset)} "
            f"windows (stride={stride}, max_samples={max_samples or 'unset'})"
        )

    def _underlying(self, idx: int) -> int:
        """Map a positional index to the underlying dataset index."""
        return idx if self.indices is None else self.indices[idx]

    def __len__(self) -> int:
        """Return the number of samples this split exposes."""
        return len(self.dataset) if self.indices is None else len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, str]:
        """Return an index-only stub; real data is loaded by the DataPacker.

        ``idx`` is positional; the emitted ``idx`` is the *underlying* dataset
        index, since the DataPacker resolves it against the raw dataset.
        """
        return {"idx": str(self._underlying(idx)), "split": self.split}

    def get_reference_answer(self, idx: int) -> dict[str, Any]:
        """Retrieve ground-truth trajectory and metadata for reward computation.

        Returns an empty dict when the sample cannot be fetched or lacks
        required trajectory keys. The reward function asserts on this and
        will fail loudly if the dataset is misconfigured.
        """
        # Cosmos-RL passes the POSITIONAL index here (dispatcher/data/__init__.py:38
        # sets prompt_idx from the dataloader position), so it must be mapped the
        # same way __getitem__ maps it or val rewards would score the wrong clip.
        idx = self._underlying(idx)
        try:
            sample = self.dataset[idx]
        except Exception as e:
            logger.error(f"[AlpamayoCosmosDataset] Error getting reference answer: {e}")
            return {}
        if not isinstance(sample, dict) or "ego_future_xyz" not in sample:
            return {}
        return {
            # NOTE: fork addition -- clip identity so the reward hook can key a
            # per-clip cache (used by rewards/traj_viz.py for W&B plotting).
            "idx": int(idx),
            # Lets the reward tell a validation rollout from a training one --
            # both splits index the same underlying dataset, so without this the
            # traj_viz per-clip cache would mix them under one key.
            "split": self.split,
            "ego_future_xyz": sample["ego_future_xyz"],
            "ego_future_rot": sample["ego_future_rot"],
            "ego_history_xyz": sample["ego_history_xyz"],
            "ego_history_rot": sample["ego_history_rot"],
            "egomotion_road_boundaries": sample.get("egomotion_road_boundaries", None),
            "egomotion_lanelines": sample.get("egomotion_lanelines", None),
            "ego_lwh": sample.get("ego_lwh", None),
            "ego_length_offset": sample.get("ego_length_offset", None),
            "obstacle_bbox_history": sample.get("obstacle_bbox_history", None),
            "obstacle_bbox_future": sample.get("obstacle_bbox_future", None),
            "cot": sample.get("cot", ""),
        }
