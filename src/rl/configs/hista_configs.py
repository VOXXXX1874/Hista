# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from rl.configs.grpo_configs import GRPOPlusConfig

@dataclass
class HistaConfig(GRPOPlusConfig):
    """
    args for callbacks, benchmarks etc
    """
    # Hidden state related arguments
    gae_lambda: float = field(
        default=1.0,
        metadata={"help": "Lambda value for Generalized Advantage Estimation (GAE)"},
    )
    hista_min_interval: int = field(
        default=50,
        metadata={"help": "Minimum interval between hidden state sampled."},
    )
    hista_alpha: float = field(
        default=0.97,
        metadata={"help": "Alpha value for hidden state exponential moving average."},
    )
    hista_mean_window: int = field(
        default=150,
        metadata={"help": "Window size for calculating the mean of hidden states."},
    )
    hista_min_d: int = field(
        default=3,
        metadata={"help": "Minimum distance between two states."},
    )
    hista_max_k: int = field(
        default=66,
        metadata={"help": "Number of nodes to consider for distance calculation in prefix."},
    )
    hista_min_k: int = field(
        default=6,
        metadata={"help": "Number of nodes to consider for distance calculation in suffix."},
    )