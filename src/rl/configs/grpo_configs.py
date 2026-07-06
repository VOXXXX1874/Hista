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
from typing import Optional

import trl

from trl import ScriptArguments


@dataclass
class GRPOPlusConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None, metadata={"help": "The optional system prompt to use for benchmarking."}
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    silence: bool = field(
        default=False,
        metadata={"help": "Whether to silence verification outputs during training."},
    )

    # Dynamic sampling, overlong filtering, and length related arguments
    dynamic_sampling_scale: int = field(
        default=1,
        metadata={"help": "Scale for dynamic sampling. We will multiply the original batch size by this factor and only keep the top samples."},
    )
    overlong_punishment_threshold: float = field(
        default=1.0,
        metadata={"help": "Threshold for overlong punishment. If the length of a generation is longer than this ratio of the max length, it will be punished."},
    )
    manage_verifier_vllm_sleep: bool = field(
        default=False,
        metadata={"help": "Whether the main process should wake/sleep the verifier vLLM proxy around reward calculation."},
    )
    verifier_vllm_base_url: str = field(
        default="http://localhost:8000",
        metadata={"help": "Base URL of the verifier vLLM proxy control endpoint."},
    )
    verifier_vllm_sleep_level: int = field(
        default=1,
        metadata={"help": "Sleep level passed to the verifier vLLM /sleep endpoint."},
    )
    verifier_vllm_control_timeout: float = field(
        default=120.0,
        metadata={"help": "Timeout in seconds for verifier vLLM wake/sleep control requests."},
    )

    # For model after qwen3
    enable_thinking: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to enable thinking mode. If not specified, no parameter will be passed to tokenizer."},
    )

@dataclass
class GRPOPlusScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'thinking', and'format'"
        },
    )
    quick_eval_dataset: str = field(
        default=None,
        metadata={"help": "Quick evaluation dataset"},
    )
    quick_eval_dataset_size: int = field(
        default=320,
        metadata={"help": "Number of samples to use from the quick evaluation dataset"},
    )
    use_default_system_prompt: bool = field(
        default=False,
        metadata={"help": "Whether to use the default system prompt for generation."},
    )
    distributed_training: bool = field(
        default=False,
        metadata={"help": "Whether to use distributed training."},
    )

