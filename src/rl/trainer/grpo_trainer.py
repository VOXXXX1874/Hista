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

import asyncio
import atexit
import importlib.resources as pkg_resources
import inspect
import os
import sys
import warnings
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any, Protocol, TYPE_CHECKING
from accelerate import logging
import inspect
import numpy as np

import datasets
import torch
import torch.nn as nn
from packaging.version import Version
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from huggingface_hub import CommitScheduler, DatasetCard, DatasetCardData, create_repo
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.utils import is_datasets_available, is_peft_available
from transformers.trainer_utils import SaveStrategy

from trl.chat_template_utils import add_response_schema, get_training_chat_template
from trl.models.utils import _ForwardRedirection, disable_gradient_checkpointing
from trl.trainer import GRPOTrainer
from trl.trainer.base_trainer import BaseTrainer
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    prepare_multimodal_messages,
)
from trl.trainer.utils import(
    RepeatSampler,
    create_model_from_path,
    disable_dropout_in_model,
    get_config_model_id,
    identity,
    nanmax,
    nanmin,
    nanstd,
    pad,
    start_event_loop_in_daemon,
    shutdown_event_loop_in_daemon,
    use_adapter,
)
from torch.utils.data import DataLoader, Sampler
from trl.data_utils import apply_chat_template, is_conversational
from trl.models import prepare_deepspeed, prepare_fsdp
from trl.import_utils import is_jmespath_available, is_liger_kernel_available
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer import SyncRefModelCallback
from trl.trainer.utils import  pad
from utils.vllm_generation import VLLMGenerationPatched
from vllm_verifier.control import control_verifier_vllm

from datasets import Dataset, IterableDataset
import datasets
from functools import partial
from transformers.trainer_utils import seed_worker

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

if TYPE_CHECKING:
    import optuna

class _SupportsReset(Protocol):
    def reset(self, **kwargs) -> str | None: ...

EnvironmentFactory = Callable[[], _SupportsReset]

logger = logging.get_logger(__name__)

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = str | PreTrainedModel | Callable[[list, list], list[float]]

# What we call a rollout function is a callable that takes prompts (list) and the trainer instance as parameters and
# returns a dict of generation results. Those results must include "prompt_ids", "completion_ids", and "logprobs"
# fields. Any extra fields (per-completion) are forwarded to the reward functions.
RolloutFunc = Callable[[list[str], "GRPOTrainer"], dict[str, Any]]

class GRPOPlusTrainer(GRPOTrainer):
    # base trl GRPO_trainer


    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: "str | PreTrainedModel | PeftModel",
        reward_funcs: RewardFunc | list[RewardFunc],
        args: GRPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: "PeftConfig | None" = None,
        tools: list[Callable] | None = None,
        rollout_func: RolloutFunc | None = None,
        environment_factory: EnvironmentFactory | None = None,
    ):
        
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Model
        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            # Distributed training requires device_map=None ("auto" fails)
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            model = create_model_from_path(model, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )

        # Some models (SmolVLM/Idefics3) don't support `logits_to_keep` argument and error out if we pass it
        # Inspect the forward method before we wrap the model with PEFT
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(
                get_config_model_id(model.config), truncation_side="left", padding_side="left"
            )

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        if is_peft_available() and is_peft_model(model) and peft_config is not None:
            raise ValueError(
                "You passed a `PeftModel` instance together with a `peft_config` to the trainer. Please first merge "
                "and unload the existing adapter, save the resulting base model, and then pass that base model along "
                "with the new `peft_config` to the trainer."
            )

        if is_peft_available() and is_peft_model(model) and args.beta != 0.0:
            # If the model is a PEFT model with a pretrained adapter, we need to create a "ref" adapter that is a copy
            # of the "default" adapter, so that we can use it as the reference model during GRPO training.
            model.add_adapter("ref", model.peft_config["default"])
            for name, param in model.named_parameters():
                if ".default." in name:
                    ref_name = name.replace(".default.", ".ref.")
                    ref_param = model.get_parameter(ref_name)
                    ref_param.data.copy_(param.data)

        # Create PEFT model
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # When using gradient checkpointing with PEFT, we need to enable input gradients. transformers.Trainer normally
        # handles this, but a bug currently prevents it; see https://github.com/huggingface/transformers/issues/42489
        if is_peft_available() and is_peft_model(model) and args.gradient_checkpointing:
            model.enable_input_require_grads()

        # When using QLoRA, the PEFT adapter weights are converted to bf16 to follow the recommendations from the
        # original paper (see https://huggingface.co/papers/2305.14314, paragraph 3). Normally, this can be done by
        # passing `autocast_adapter_dtype=False` to `get_peft_model`, but this option is not yet supported for
        # quantized models. See: https://github.com/huggingface/peft/issues/2889
        # Non-quantized models do not have the `is_loaded_in_{8,4}bit` attributes, whereas quantized models do
        if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.to(torch.bfloat16)

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                model_init_kwargs = args.model_init_kwargs or {}
                # Distributed training requires device_map=None ("auto" fails)
                if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                    model_init_kwargs["device_map"] = None
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(reward_funcs[i], nn.Module):  # Use Module over PretrainedModel for compat w/ compiled models
                self.reward_func_names.append(get_config_model_id(reward_funcs[i].config).split("/")[-1])
            else:
                self.reward_func_names.append(reward_funcs[i].__name__)
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        if len(reward_processing_classes) != len(reward_funcs):
            raise ValueError(
                f"The number of reward processing classes ({len(reward_processing_classes)}) must match the number of "
                f"reward functions ({len(reward_funcs)})."
            )

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs, strict=True)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(get_config_model_id(reward_func.config))
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class

        self.reward_processing_classes = reward_processing_classes

        # Rollout function
        if rollout_func is not None and os.environ.get("TRL_EXPERIMENTAL_SILENCE", "0") != "1":
            warnings.warn(
                "You are using 'rollout_func', which is an experimental feature. This API may change or be removed at "
                "any time without prior notice. Silence this warning by setting environment variable "
                "TRL_EXPERIMENTAL_SILENCE=1.",
                UserWarning,
                stacklevel=2,
            )
        self.rollout_func = rollout_func
        if environment_factory is not None and os.environ.get("TRL_EXPERIMENTAL_SILENCE", "0") != "1":
            warnings.warn(
                "You are using 'environment_factory', which is an experimental feature. This API may change or be "
                "removed at any time without prior notice. Silence this warning by setting environment variable "
                "TRL_EXPERIMENTAL_SILENCE=1.",
                UserWarning,
                stacklevel=2,
            )

        # Tools
        if tools:
            if not Version(transformers.__version__) >= Version("5.0.0"):
                raise ImportError(
                    "Using tools with GRPOTrainer requires transformers version 5.0.0 or higher. Please upgrade "
                    "transformers with `pip install --upgrade transformers` to use this feature."
                )
        if environment_factory:
            if not Version(transformers.__version__) >= Version("5.2.0"):
                raise ImportError(
                    "Using `environment_factory` with GRPOTrainer requires transformers version 5.2.0 or higher. "
                    "Please install transformers from the main branch with `pip install "
                    "git+https://github.com/huggingface/transformers.git@main` to use this feature."
                )
        if tools or environment_factory:
            if not is_jmespath_available():
                raise ImportError(
                    "Using tools with GRPOTrainer requires the jmespath library for response parsing. Please install "
                    "it with `pip install jmespath` to use this feature."
                )

        # Create the environments and extract their methods to be used as tools. We create one environment per rollout
        generation_batch_size = args.per_device_train_batch_size * args.steps_per_generation
        if environment_factory is not None:
            self.environments = [environment_factory() for _ in range(generation_batch_size)]
            environment_methods = [[] for _ in range(generation_batch_size)]
            for i, environment in enumerate(self.environments):
                has_reset = False
                for name, member in inspect.getmembers(environment, predicate=inspect.ismethod):
                    if name == "reset":
                        has_reset = True
                    elif not name.startswith("_"):
                        environment_methods[i].append(member)
                if not has_reset:
                    raise ValueError(
                        "Each environment instance returned by `environment_factory` must define a callable `reset` "
                    )
        else:
            self.environments = None

        tools = tools or []
        self._sync_tool_dicts = [{} for _ in range(generation_batch_size)]
        self._async_tool_dicts = [{} for _ in range(generation_batch_size)]
        for i in range(generation_batch_size):
            for tool in tools + (environment_methods[i] if self.environments is not None else []):
                if asyncio.iscoroutinefunction(tool):
                    self._async_tool_dicts[i][tool.__name__] = tool
                else:
                    self._sync_tool_dicts[i][tool.__name__] = tool

        self.tools = tools + (environment_methods[0] if self.environments is not None else [])

        # Check for async functions to start an event loop on a daemon thread
        self._has_async_funcs = any(asyncio.iscoroutinefunction(func) for func in self.reward_funcs + self.tools)

        if self._has_async_funcs:
            self.async_loop_thread, self.async_loop, self.async_loop_ready_event = start_event_loop_in_daemon(
                name="GRPOTrainer-AsyncLoop"
            )
            # wait until the event loop is running in the daemon thread
            self.async_loop_ready_event.wait()
            atexit.register(shutdown_event_loop_in_daemon, self.async_loop_thread, self.async_loop)

        # At the time of initial implementation, most tokenizers do not have built-in support for response schemas.
        # While waiting for broader adoption, we provide this utility function to manually set the response schema for
        # known chat templates.
        # We need `getattr`` until the base class sets a default None value for response_schema
        if self.tools and not getattr(processing_class, "response_schema", None):
            processing_class = add_response_schema(processing_class)
        # In multi-turn training, the chat template *must* be prefix-preserving. If the tokenizer's original template
        # isn't, we replace it at initialization with a training-safe, prefix-preserving template.
        if self.tools:
            self.chat_template = get_training_chat_template(processing_class)
        else:
            self.chat_template = None

        # Training arguments
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.max_tool_calling_iterations = args.max_tool_calling_iterations or sys.maxsize
        self.num_generations_eval = args.num_generations_eval or self.num_generations
        self.chat_template_kwargs = args.chat_template_kwargs or {}
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_transformers_paged = args.use_transformers_paged
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.vllm_importance_sampling_correction = args.vllm_importance_sampling_correction
        self.vllm_importance_sampling_mode = args.vllm_importance_sampling_mode
        self.vllm_importance_sampling_cap = args.vllm_importance_sampling_cap
        self.use_liger_kernel = args.use_liger_kernel
        self.loss_type = args.loss_type
        self.multi_objective_aggregation = args.multi_objective_aggregation
        self.scale_rewards = args.scale_rewards
        self.importance_sampling_level = args.importance_sampling_level
        self.off_policy_mask_threshold = args.off_policy_mask_threshold
        if self.use_liger_kernel and self.off_policy_mask_threshold is not None:
            raise ValueError("Liger kernel does not support off-policy sequence masking yet.")
        self.mask_truncated_completions = args.mask_truncated_completions
        self.top_entropy_quantile = args.top_entropy_quantile
        if self.use_liger_kernel and self.top_entropy_quantile < 1.0:
            raise NotImplementedError(
                "Liger Kernels don't currently support masking token positions based on entropy."
            )
        if self.use_liger_kernel and not self.importance_sampling_level == "token":
            raise NotImplementedError(
                "Liger Kernels currently only support token-level importance sampling. Please set"
                "`importance_sampling_level` to 'token'."
            )

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if train_dataset is None:
            raise ValueError("`train_dataset` is required")
        elif (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )

        if args.loss_type == "luspo" and args.importance_sampling_level != "sequence":
            logger.warning(
                "When using `'luspo'` loss, `importance_sampling_level` should be set to `'sequence'` to mirror the "
                "paper's setup."
            )

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # Transformers explicitly set use_reentrant=True in the past to silence a PyTorch warning, but the default was
        # never updated once PyTorch switched to recommending use_reentrant=False. Until that change lands upstream
        # (see https://github.com/huggingface/transformers/pull/43203) and is released (most likely in 5.0.0), we
        # default to the recommended non-reentrant behavior here, while preserving any user-provided value.
        if args.gradient_checkpointing and Version(transformers.__version__) < Version("5.0.0"):
            args.gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
            args.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

        BaseTrainer.__init__(
            self,
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in GRPO
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            # In Trainer, `training_step` scales the loss by `gradient_accumulation_steps` only if `compute_loss_func`
            # is None. For DAPO, loss scaling instead depends on the total number of completions tokens across the
            # global accumulated batch. To control scaling ourselves, we must disable Trainer’s built-in scaling. The
            # simplest (though a bit hacky) way is to set `compute_loss_func` to any non-None value, which bypasses
            # that behavior without rewriting `training_step`.
            compute_loss_func="non-None value to disable scaling",
        )

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # For deepspeed, fsdp or non-distributed models, create a reference model from scratch
            model_init_kwargs = args.model_init_kwargs or {}
            # Distributed training requires device_map=None ("auto" fails)
            if self.args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            self.ref_model = create_model_from_path(get_config_model_id(self.model.config), **model_init_kwargs)

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Cast LM Head To FP32
        if args.cast_lm_head_to_fp32:

            def _cast_lm_head_to_fp32(target_model: PreTrainedModel):
                """Cast lm_head to fp32 while preserving embedding output dtype if tied."""

                def cast_inputs_to_fp32(module, inputs):
                    # Preserve other positional args and kwargs untouched
                    if not inputs:
                        return inputs
                    return (inputs[0].to(torch.float32),) + inputs[1:]

                original_dtype_local = target_model.lm_head.weight.dtype
                target_model.lm_head = target_model.lm_head.float()
                target_model.lm_head.register_forward_pre_hook(cast_inputs_to_fp32)

                if target_model.config.tie_word_embeddings:

                    def cast_outputs_to_original_dtype(module, args, output):
                        return output.to(original_dtype_local)

                    # Only cast activations; weights are now fp32 (intentional for numerical stability of logits)
                    target_model.model.embed_tokens.register_forward_hook(cast_outputs_to_original_dtype)

            _cast_lm_head_to_fp32(model)
            if self.ref_model is not None:
                _cast_lm_head_to_fp32(self.ref_model)

        # Liger loss
        if self.use_liger_kernel:
            if not is_liger_kernel_available():
                raise ImportError(
                    "Liger is required to use `use_liger_kernel` as the GRPO loss. Run `pip install liger-kernel`."
                )
            # redirect the model.module forward to the model forward to ensure pre-forward hooks are called
            self._forward_redirection = _ForwardRedirection()

            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                use_ref_model=self.beta != 0.0,
                loss_type=self.loss_type,
                max_completion_length=self.max_completion_length,
            )

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self._current_train_step_time = 0.0
        self.log_completions = args.log_completions
        self.log_unique_prompts = args.log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # Keep logs sized to the generation batch to record only outputs from the latest model update.
        self._logs = {
            "images": deque(maxlen=args.generation_batch_size),
            "prompt": deque(maxlen=args.generation_batch_size),
            "completion": deque(maxlen=args.generation_batch_size),
            "rewards": defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
            "advantages": deque(maxlen=args.generation_batch_size),
        }

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if args.vllm_mode == "server" and args.dynamic_sampling_scale != 1.0:
                raise NotImplementedError("Dynamic sampling is not currently supported in vLLM server mode.")
            # Initialize vLLM generation backend
            # Wrap rollout_func to capture trainer context if provided
            rollout_func = None
            if self.rollout_func is not None:

                def rollout_func(prompts):
                    return self.rollout_func(prompts, self)

            self.vllm_generation = VLLMGenerationPatched(
                model=self.model,
                accelerator=self.accelerator,
                is_fsdp_enabled=self.is_fsdp_enabled,
                processing_class=self.processing_class,
                # vLLM configuration
                mode=args.vllm_mode,
                structured_outputs_regex=args.vllm_structured_outputs_regex,
                # Server mode configuration
                server_base_url=args.vllm_server_base_url,
                server_host=args.vllm_server_host,
                server_port=args.vllm_server_port,
                group_port=args.vllm_group_port,
                server_timeout=args.vllm_server_timeout,
                # Colocate mode configuration
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_length=args.vllm_max_model_length,
                max_num_seqs=args.per_device_train_batch_size
                    * args.vllm_tensor_parallel_size
                    * args.steps_per_generation
                    * args.dynamic_sampling_scale,  # Scale the number of sequences to generate based on the dynamic sampling scale to ensure enough valid samples after filtering
                enable_sleep_mode=args.vllm_enable_sleep_mode,
                model_impl=args.vllm_model_impl,
                # Generation configuration
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                max_completion_length=self.max_completion_length,
                logprobs=0,  # we only need the generated token logprobs for the importance sampling correction
                generation_kwargs=args.generation_kwargs,
                # Chat/tool configuration
                chat_template=self.chat_template,
                chat_template_kwargs=self.chat_template_kwargs,
                tools=self.tools,
                rollout_func=rollout_func,
            )
            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation
        else:
            generation_kwargs = {
                "max_new_tokens": self.max_completion_length,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
                "cache_implementation": args.cache_implementation,
            }
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            self.generation_config = GenerationConfig(**generation_kwargs)
            # Keep training-specific generation kwargs to overwrite model's original generation config
            self.generation_kwargs = generation_kwargs

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            if self.beta == 0.0:
                raise ValueError(
                    "You passed `sync_ref_model=True` while `beta=0.0`, which means the reference model is not used "
                    "during training. Consequently, GRPOTrainer does not create a `ref_model` instance, and there is "
                    "nothing to synchronize. Please set `sync_ref_model=False`, or set `beta` to a non-zero value."
                )
            if is_peft_model(model):
                raise NotImplementedError(
                    "You passed `sync_ref_model=True` while using a PEFT model, which is currently not supported. "
                    "With PEFT, GRPOTrainer does not keep a separate reference model in memory; instead, it recovers "
                    "reference behavior by temporarily disabling the adapter. As a result, there is no standalone "
                    "`ref_model` instance to synchronize. Use `sync_ref_model=False`, or opt for full fine-tuning if "
                    "you need a synced reference model. If you need `sync_ref_model` to work with PEFT, please open a "
                    "feature request at https://github.com/huggingface/trl/issues."
                )
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    # set device placement to True to make `prepare_model` move `reward_func` to device when using fsdp
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True
                    )

        if self.accelerator.is_main_process and self.log_completions:
            os.makedirs(os.path.join(self.args.output_dir, "completions"), exist_ok=True)
            if self.args.log_completions_hub_repo is not None:
                repo_id = self.args.log_completions_hub_repo
                create_repo(repo_id, private=self.args.hub_private_repo, repo_type="dataset", exist_ok=True)
                template_path = pkg_resources.files("trl").joinpath("templates/completions_dataset_card.md")
                card_data = DatasetCardData(
                    pretty_name="TRL Completion logs",
                    tags=["trl", "trl-logs", "completions"],
                )
                card = DatasetCard.from_template(
                    card_data=card_data,
                    template_path=str(template_path),
                    repo_id=repo_id,
                    hub_model_id=self.args.hub_model_id,
                )
                card.push_to_hub(repo_id)
                self.commit_scheduler = CommitScheduler(
                    repo_id=repo_id,
                    repo_type="dataset",
                    folder_path=f"{self.args.output_dir}/completions",
                    every=2,  # minutes
                    allow_patterns=["*.parquet"],
                )

        assert self.use_vllm == True, "GRPOPlusTrainer currently only supports vLLM for rollout."

        # Handle verifier vllm
        self.manage_verifier_vllm_sleep = args.manage_verifier_vllm_sleep
        self.verifier_vllm_base_url = args.verifier_vllm_base_url
        self.verifier_vllm_sleep_level = args.verifier_vllm_sleep_level
        self.verifier_vllm_control_timeout = args.verifier_vllm_control_timeout
        verifier_reward_base_url = self.verifier_vllm_base_url.rstrip("/")
        if verifier_reward_base_url.endswith("/v1"):
            verifier_reward_base_url = verifier_reward_base_url[:-3]
        os.environ["VERIFIER_BASE_URL"] = f"{verifier_reward_base_url}/v1"

        # Check whether the eval_dataset can be loaded in one batch
        if len(self.eval_dataset) * self.num_generations_eval != self.args.per_device_eval_batch_size * self.accelerator.num_processes:
            raise ValueError(
                f"The evaluation dataset size ({len(self.eval_dataset)}) multiplied by num_generations_eval ({self.num_generations_eval}) must be equal to per_device_eval_batch_size ({self.args.per_device_eval_batch_size}) multiplied by the number of processes ({self.accelerator.num_processes}) so that we can load the eval dataset in one batch for evaluation. Please adjust the dataset size, num_generations_eval, or per_device_eval_batch_size accordingly."
            )

        # Dynamic sampling and overlong filtering attributes
        self.dynamic_sampling_scale = args.dynamic_sampling_scale
        # if self.args.generation_batch_size // self.num_generations * self.dynamic_sampling_scale has decimal part, raise error because it will cause issues in dataloader construction
        if (self.args.generation_batch_size // self.num_generations * self.dynamic_sampling_scale) % 1 != 0:
            raise ValueError(f"generation_batch_size ({self.args.generation_batch_size}) divided by num_generations ({self.num_generations}) and multiplied by dynamic_sampling_scale ({self.dynamic_sampling_scale}) must be an integer.")
        if self.dynamic_sampling_scale < 1.0:
            raise ValueError(f"dynamic_sampling_scale must be greater than or equal to 1.0, but got {self.dynamic_sampling_scale}.")

    
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": int(self._train_batch_size * self.args.steps_per_generation * self.dynamic_sampling_scale),
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )

            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def get_eval_dataloader(self, eval_dataset: Dataset | dict[str, Dataset] | None = None):
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )

            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset: Dataset | None = None) -> Sampler:
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=int(self.args.generation_batch_size // self.num_generations * self.dynamic_sampling_scale),
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        if self.environments:
            for prompt, environment, reset_kwargs in zip(prompts, self.environments, inputs, strict=True):
                observation = environment.reset(**reset_kwargs)
                if observation is None:
                    continue
                prompt[-1]["content"] += observation

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image", "image": <Image>}, {"type": "text", "text": "What color is the sky?"}]}]
        if images is not None:
            if not is_conversational(inputs[0]):
                raise ValueError(
                    "Multimodal training requires conversational prompts. It looks like the dataset contains "
                    "non-conversational inputs, likely because a chat template was applied before passing the dataset "
                    "to the trainer. Please provide the raw conversational prompts and let the trainer apply the chat "
                    "template internally."
                )
            prompts = [
                prepare_multimodal_messages(prompt, image_list)
                for prompt, image_list in zip(prompts, images, strict=True)
            ]

        (
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            completions,
            num_items_in_batch,
            sampling_per_token_logps_list,
            extra_fields,
        ) = self._generate(prompts)

        # Merge extra_fields from rollout_func into inputs for reward functions
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # Dynamical sampling and overlong filtering logic
        self._wake_verifier_vllm_for_rewards()
        try:
            if mode == "train":
                (
                    inputs,
                    prompts,
                    prompt_ids_list,
                    completion_ids_list,
                    tool_mask_list,
                    completions,
                    sampling_per_token_logps_list,
                    extra_fields,
                    rewards_per_func,
                ) = self._dynamic_sampling_overlong_filter(
                    inputs,
                    prompts,
                    prompt_ids_list,
                    completion_ids_list,
                    tool_mask_list,
                    completions,
                    sampling_per_token_logps_list,
                    extra_fields,
                )
            else:
                rewards_per_func = self._calculate_rewards(
                    inputs,
                    prompts,
                    completions,
                    completion_ids_list
                )
        finally:
            self._sleep_verifier_vllm_after_rewards()

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None
        if tool_mask_list is not None:
            tool_mask = [torch.tensor(mask, device=device) for mask in tool_mask_list]
            tool_mask = pad(tool_mask, padding_value=1, padding_side="right")
        else:
            tool_mask = None

        # If mask_truncated_completions is enabled, zero out truncated completions for attention and loss masking
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            # Mask completion_mask for attention masking
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
            # Also mask tool_mask for consistency in multi-turn training
            if tool_mask is not None:
                tool_mask = tool_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        num_images = [len(img_list) for img_list in images] if images is not None else None

        # Get forward_kwargs for models with multimodal inputs
        if mode == "train":
            if images is not None:
                prompts_text = [
                    apply_chat_template(
                        {"prompt": prompt}, self.processing_class, tools=self.tools, **self.chat_template_kwargs
                    )["prompt"]
                    for prompt in prompts
                ]
                prompt_inputs = self.processing_class(images=images, text=prompts_text, padding=True, return_tensors="pt")
                prompt_inputs = BaseTrainer._prepare_inputs(prompt_inputs)
                forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
            else:
                forward_kwargs = {}

            # If token_type_ids are used, extend them with zeros for the completion part
            if "token_type_ids" in forward_kwargs:
                token_type_ids = forward_kwargs["token_type_ids"]
                forward_kwargs["token_type_ids"] = torch.cat(
                    [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
                )

            # When gradient checkpointing is enabled with use_reentrant=True (non default), calling the model inside a
            # torch.no_grad() block triggers a harmless PyTorch warning ("None of the inputs have requires_grad=True").
            # Temporarily disable checkpointing to avoid this warning during inference.
            with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
                # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
                # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
                # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
                # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
                # old_per_token_logps to None.
                # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
                # distribution mismatch between vLLM and the training model can be large and harm the training.
                generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
                if self.args.gradient_accumulation_steps % generate_every != 0 or (
                    self.use_vllm and self.vllm_importance_sampling_correction
                ):
                    old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size,
                        num_images=num_images,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    old_per_token_logps = None

                # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
                if self.use_vllm and self.vllm_importance_sampling_correction:
                    mask = completion_mask if tool_mask is None else completion_mask * tool_mask
                    per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask

                    sequence_level_is = self.vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]
                    if sequence_level_is:
                        per_sequence_logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
                        logps_diff = per_sequence_logps_diff
                    else:
                        logps_diff = per_token_logps_diff

                    vllm_importance_sampling_ratio = torch.exp(logps_diff)

                    # vllm_importance_sampling_ratio.shape:
                    #   token_* modes:     (B, T)  (per-token ratio)
                    #   sequence_* modes:  (B, 1)  (per-sequence ratio)

                    if self.vllm_importance_sampling_mode in ["sequence_truncate", "token_truncate"]:
                        vllm_importance_sampling_ratio = torch.clamp(
                            vllm_importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                        )
                    elif self.vllm_importance_sampling_mode in ["sequence_mask", "token_mask"]:
                        vllm_importance_sampling_ratio = vllm_importance_sampling_ratio.masked_fill(
                            vllm_importance_sampling_ratio > self.vllm_importance_sampling_cap, value=0.0
                        )
                    else:
                        raise ValueError(
                            f"Unknown vLLM importance sampling level: {self.vllm_importance_sampling_mode}. Possible values are 'token_truncate', 'token_mask', 'sequence_truncate', and 'sequence_mask'."
                        )

                # Compute the per-token log probabilities for the reference model
                if self.beta != 0.0:
                    if self.ref_model is not None:
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.ref_model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                        )
                    else:
                        # When training a PEFT adapter, how we obtain the reference depends on the setup:
                        # - New adapter: disabling adapters yields the base model.
                        # - Re-training an existing adapter: an initial copy is loaded under the name "ref".
                        model = self.accelerator.unwrap_model(self.model)
                        with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
                            ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                                self.model,
                                prompt_completion_ids,
                                attention_mask,
                                logits_to_keep,
                                batch_size=batch_size,
                                num_images=num_images,
                                **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                            )
                else:
                    ref_per_token_logps = None
        else:
            old_per_token_logps = None
            ref_per_token_logps = None
            vllm_importance_sampling_ratio = None
            forward_kwargs = {}

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Reward calculation is finished previously in _dynamic_sampling_overlong_filter,
        # We only need to gather rewards_per_func here
        if mode == "train":
            rewards_per_func = gather(rewards_per_func)

        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        if self.multi_objective_aggregation == "sum_then_normalize":
            # Apply weights to each reward function's output and sum
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
            if self.scale_rewards in ["group", "none"]:
                # If self.scale_rewards = "none", we'll only use std_rewards to check for zero std for logging
                if num_generations > 1:
                    std_rewards = rewards.view(-1, num_generations).std(dim=1)
                    std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
                else:  # doesn't occur during training, but could occur in eval when num_generations_eval=1
                    std_rewards = torch.zeros_like(rewards)
            elif self.scale_rewards == "batch":
                # Compute global std
                if rewards.numel() > 1:
                    std_rewards = rewards.std().expand_as(rewards)
                else:  # doesn't occur during training, but could occur in eval when num_generations_eval=batch_size=1
                    std_rewards = torch.zeros_like(rewards)
            else:
                raise ValueError(
                    f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
                )

            advantages = rewards - mean_grouped_rewards
            if self.scale_rewards != "none":
                advantages = advantages / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))  # for logging

        elif self.multi_objective_aggregation == "normalize_then_sum":
            grouped = rewards_per_func.view(-1, num_generations, len(self.reward_funcs))
            mean_k = torch.nanmean(grouped, dim=1, keepdim=True)
            std_k = nanstd(grouped, dim=1, keepdim=True) if num_generations > 1 else torch.zeros_like(mean_k)
            reward_k = (grouped - mean_k) / (std_k + 1e-4)
            reward_k = reward_k.view(-1, len(self.reward_funcs))
            rewards = (reward_k * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            std_rewards = rewards.std().expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
            advantages = (rewards - rewards.mean()) / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))  # for logging

        else:
            raise ValueError(
                f"Invalid multi_objective_aggregation: {self.multi_objective_aggregation}. Must be "
                "'sum_then_normalize' or 'normalize_then_sum'."
            )

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        reward_name = "ds_rewards" if mode == "train" else "rewards"

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"{reward_name}/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"{reward_name}/{reward_func_name}/std"].append(std_func_rewards)
        rewards = rewards_per_func.nansum(dim=1)
        self._metrics[mode][f"{reward_name}/mean"].append(rewards.mean().item())
        self._metrics[mode][f"{reward_name}/std"].append(rewards.std().item())
        self._metrics[mode][f"{reward_name}/frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        if self.use_vllm and self.vllm_importance_sampling_correction and mode == "train":
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            mask = completion_mask.bool() if tool_mask is None else (completion_mask * tool_mask).bool()
            delta = delta[mask]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )
            if sequence_level_is:
                flat_is_ratio = vllm_importance_sampling_ratio.flatten()
            else:
                flat_is_ratio = vllm_importance_sampling_ratio[mask]

            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = vllm_importance_sampling_ratio
        if sampling_per_token_logps is not None:
            output["sampling_per_token_logps"] = sampling_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = num_images
        if tool_mask is not None:
            output["tool_mask"] = tool_mask
        return output

    def _control_verifier_vllm_for_rewards(self, action: str):
        if not self.manage_verifier_vllm_sleep:
            return

        status = [None]
        if self.accelerator.is_main_process:
            try:
                control_verifier_vllm(
                    action,
                    self.verifier_vllm_base_url,
                    timeout=self.verifier_vllm_control_timeout,
                    sleep_level=self.verifier_vllm_sleep_level,
                )
                status[0] = {"ok": True}
            except Exception as e:
                status[0] = {"ok": False, "error": repr(e)}

        if self.accelerator.num_processes > 1:
            status = broadcast_object_list(status, from_process=0)
        if not status[0]["ok"]:
            raise RuntimeError(f"Failed to {action} verifier vLLM: {status[0]['error']}")

    def _wake_verifier_vllm_for_rewards(self):
        self._control_verifier_vllm_for_rewards("wake_up")
        if self.manage_verifier_vllm_sleep:
            self.accelerator.wait_for_everyone()

    def _sleep_verifier_vllm_after_rewards(self):
        if not self.manage_verifier_vllm_sleep:
            return
        self.accelerator.wait_for_everyone()
        self._control_verifier_vllm_for_rewards("sleep")
        self.accelerator.wait_for_everyone()

    def _dynamic_sampling_overlong_filter(
        self,
        inputs: list[dict[str, torch.Tensor | Any]],
        prompts: list[str] | list[list[dict]],
        prompt_ids_list: list[list[int]],
        completion_ids_list: list[list[int]],
        tool_mask_list: list[list[int]] | None,
        completions: list[str] | list[dict] | list[list[dict]],
        sampling_per_token_logps_list: list[list[float]] | None,
        extra_fields: dict[str, list[Any]],
    ) -> tuple[list[list[int]], list[list[int]], list[list[int]] | None, list[str] | list[dict] | list[list[dict]], list[list[float]] | None, dict[str, list[Any]], torch.tensor]:
        mode = "train" if self.model.training else "eval"
        device = self.accelerator.device
        # Try to sample the prompts and completions that has accuracy closed to 0.5
        # We first calculate the reward of each completion
        rewards_per_func = self._calculate_rewards(
            inputs,
            prompts,
            completions,
            completion_ids_list
        )  # (B, num_reward_funcs)

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        rewards = rewards_per_func.nansum(dim=1)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(rewards.std().item())

        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())

        # Find the index of reward function "accuracy_reward"
        accuracy_reward_index = None
        for i, reward_func_name in enumerate(self.reward_func_names):
            if reward_func_name == "accuracy_reward":
                accuracy_reward_index = i
                break
        assert accuracy_reward_index is not None, "accuracy_reward function not found in reward_funcs"

        # Find the accuracy reward from rewards_per_func using the accuracy_reward_index
        accuracy_rewards = rewards_per_func[:, accuracy_reward_index]

        all_inputs = gather_object(inputs)
        all_prompts = gather_object(prompts)
        all_prompt_ids = gather_object(prompt_ids_list)
        all_completion_ids = gather_object(completion_ids_list)
        all_completions = gather_object(completions)
        
        all_tool_mask_list = None
        if tool_mask_list is not None:
             all_tool_mask_list = gather_object(tool_mask_list)

        all_sampling_logps = None
        if sampling_per_token_logps_list is not None:
             all_sampling_logps = gather_object(sampling_per_token_logps_list)
             
        all_extra_fields = {}
        for k, v in extra_fields.items():
            if isinstance(v, list):
                 all_extra_fields[k] = gather_object(v)
            else:
                 all_extra_fields[k] = v 
        
        # --- Dynamic Sampling ---
        num_problems_initial = accuracy_rewards.shape[0] // self.num_generations
        accuracy_per_problem = accuracy_rewards.view(num_problems_initial, self.num_generations).mean(dim=1)
        
        distance_to_0_5 = torch.abs(accuracy_per_problem - 0.5)
        
        num_problems_to_keep = self.args.generation_batch_size // self.num_generations
        # Select problems closest to 0.5
        _, selected_problem_indices = torch.topk(distance_to_0_5, k=min(num_problems_to_keep, num_problems_initial), largest=False)
        selected_problem_indices = torch.sort(selected_problem_indices).values
        
        final_global_indices = []

        for p_idx in selected_problem_indices:
            start_idx = int(p_idx * self.num_generations)
            end_idx = int(start_idx + self.num_generations)
            final_global_indices.extend(range(start_idx, end_idx))

        # --- Filter and Distribute to Local Processes ---
        world_size = self.accelerator.num_processes
        rank = self.accelerator.process_index
        
        total_items = len(final_global_indices)
        items_per_process = total_items // world_size
        
        start_slice = rank * items_per_process
        end_slice = (rank + 1) * items_per_process
        
        my_global_indices = final_global_indices[start_slice:end_slice]
        
        # Update local lists based on my_global_indices
        inputs[:] = [all_inputs[i] for i in my_global_indices]
        prompts[:] = [all_prompts[i] for i in my_global_indices]
        prompt_ids_list = [all_prompt_ids[i] for i in my_global_indices]
        completion_ids_list = [all_completion_ids[i] for i in my_global_indices]
        completions = [all_completions[i] for i in my_global_indices]
        
        if all_tool_mask_list is not None:
             tool_mask_list = [all_tool_mask_list[i] for i in my_global_indices]
             
        if all_sampling_logps is not None:
             sampling_per_token_logps_list = [all_sampling_logps[i] for i in my_global_indices]
             
        new_extra_fields = {}
        for k, v in all_extra_fields.items():
            if isinstance(v, list) and len(v) == len(all_completion_ids):
                 new_extra_fields[k] = [v[i] for i in my_global_indices]
            else:
                 new_extra_fields[k] = v
        extra_fields = new_extra_fields
        
        # Slicing the rewards tensor to match the distributed batch
        rewards_per_func = rewards_per_func[my_global_indices]

        return (
            inputs,
            prompts,
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            completions,
            sampling_per_token_logps_list,
            extra_fields,
            rewards_per_func,
        )

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        if self.args.metric_for_best_model is not None and mode == "eval":
            try:
                self.metric_value = self._metrics["eval"][f"{self.args.metric_for_best_model}"][-1]
            except KeyError as exc:
                raise KeyError(
                    f"The `metric_for_best_model` training argument is set to '{self.args.metric_for_best_model}', which is not found in the evaluation metrics. "
                    f"The available evaluation metrics are: {list(self._metrics['eval'].keys())}. Consider changing the `metric_for_best_model` via the TrainingArguments."
                ) from exc
        
        # Call the default log function to log metrics to the accelerator's logger
        super().log(logs, start_time)
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:

        inputs = self._prepare_inputs(inputs)
        return (None, None, None)

    def _determine_best_metric(self, metrics: dict[str, float], trial: "optuna.Trial | dict[str, Any] | None") -> bool:
        """
        Determine if the model should be saved based on the evaluation metrics.

        Returns:
            bool: True if a new best metric was found, else False
        """
        is_new_best_metric = False

        if self.args.metric_for_best_model is not None:

            metric_value = self.metric_value

            operator = np.greater if self.args.greater_is_better else np.less

            if self.state.best_metric is None:
                self.state.best_metric = float("-inf") if self.args.greater_is_better else float("inf")

            if operator(metric_value, self.state.best_metric):
                self.state.best_metric = metric_value

                if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH]:
                    self.state.best_global_step = self.state.global_step

                is_new_best_metric = True

        return is_new_best_metric
