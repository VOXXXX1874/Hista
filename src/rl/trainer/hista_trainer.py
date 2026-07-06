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

import time
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from accelerate import logging
from accelerate.utils import gather, gather_object
from datasets import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase, ProcessorMixin, TrainerCallback

from trl.data_utils import apply_chat_template, is_conversational, prepare_multimodal_messages
from trl.extras.profiling import profiling_decorator
from trl.models.utils import disable_gradient_checkpointing
from trl.trainer.base_trainer import BaseTrainer
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import nanmax, nanmin, nanstd, pad, selective_log_softmax, use_adapter

from rl.trainer.grpo_trainer import EnvironmentFactory, GRPOPlusTrainer, RewardFunc, RolloutFunc
from rl.utils.hista_utils import *

logger = logging.get_logger(__name__)


class HistaTrainer(GRPOPlusTrainer):
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
        
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            tools=tools,
            rollout_func=rollout_func,
            environment_factory=environment_factory,
        )

        # HISTA related attributes
        self.gae_lambda = self.args.gae_lambda
        self.hista_min_interval = self.args.hista_min_interval
        self.hista_alpha = self.args.hista_alpha
        self.hista_mean_window = self.args.hista_mean_window
        self.hista_min_d = self.args.hista_min_d
        self.hista_max_k = self.args.hista_max_k
        self.hista_min_k = self.args.hista_min_k

        # Get the start index of each question in the batch though dividing generation_batch_size by num_generations
        if self.args.generation_batch_size % self.num_generations != 0:
            raise ValueError(
                f"generation_batch_size ({self.args.generation_batch_size}) must be a multiple of num_generations ({self.num_generations})"
            )
        self.global_problems_start_idx = [i * self.num_generations for i in range(self.args.generation_batch_size // self.num_generations)]
        # Get the index of questions in each gpu process
        if self.args.generation_batch_size % self.accelerator.num_processes != 0:
            raise ValueError(
                f"generation_batch_size ({self.args.generation_batch_size}) must be a multiple of num_processes ({self.accelerator.num_processes})"
            )
        self.local_problems_start_idx = [i * self.args.generation_batch_size // self.accelerator.num_processes for i in range(self.accelerator.num_processes)]
        self.embeddings_gather_groups = None
        self.embeddings_group_size = None
        self.embeddings_group_index = None
        self.embeddings_in_group_index = None
        # Check whether the global_problem_start_idx should be a subset of local_problem_start_idx or local_problem_start_idx a subset of global_problem_start_idx
        if all(idx in self.global_problems_start_idx for idx in self.local_problems_start_idx):
            self.gather_embeddings = False
        elif all(idx in self.local_problems_start_idx for idx in self.global_problems_start_idx):
            self.gather_embeddings = True
            # For the processes sharing same questions, we need to create a group for them
            if len(self.local_problems_start_idx) % len(self.global_problems_start_idx) != 0:
                raise ValueError(
                    f"Number of local problem partitions must be divisible by number of global problems to group embeddings.\n Global: {self.global_problems_start_idx}\n Local: {self.local_problems_start_idx}"
                )
            group_size = len(self.local_problems_start_idx) // len(self.global_problems_start_idx)
            self.embeddings_gather_groups = []
            for i in range(len(self.global_problems_start_idx)):
                group = dist.new_group(ranks=[j for j in range(i * group_size, (i + 1) * group_size)])
                self.embeddings_gather_groups.append(group)
            self.embeddings_group_size = group_size
            self.embeddings_group_index = self.accelerator.process_index // group_size
            self.embeddings_in_group_index = self.accelerator.process_index % group_size
        else:
            raise ValueError(f"Inconsistent problem start indices between global and local.\n global: {self.global_problems_start_idx}\n local: {self.local_problems_start_idx}")


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

        
            # When gradient checkpointing is enabled with use_reentrant=True (default), calling the model inside a
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
                    old_per_token_logps, hidden_states = self._get_per_token_logps_and_hidden_states(
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
            hidden_states = None
            old_per_token_logps = None
            ref_per_token_logps = None
            vllm_importance_sampling_ratio = None
            forward_kwargs = {}
        
        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        if mode == "train":
            # Calculate rewards and advantage
            torch.cuda.empty_cache()  # ensure enough memory for reward calculation, which can be memory intensive
            advantages, rewards_per_func = self._calculate_Hista_advantages(
                inputs, 
                prompts, 
                completions, 
                completion_ids, 
                attention_mask,
                prompt_ids,
                prompt_mask,
                hidden_states, 
                completion_ids_list,
                rewards_per_func
            )
            # Delete hidden states to free up memory, since they are no longer needed after reward calculation
            del hidden_states
            torch.cuda.empty_cache()  # ensure enough memory for the rest of the computations
        else:
            advantages = None

        # If mask_truncated_completions is enabled, zero out truncated completions for attention and loss masking
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            # Mask completion_mask for attention masking
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
            # Also mask tool_mask for consistency in multi-turn training
            if tool_mask is not None:
                tool_mask = tool_mask * (~is_truncated).unsqueeze(1).int()

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
            
            # Slice to keep only the local part of the std_rewards
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            std_rewards = std_rewards[process_slice].unsqueeze(1)
            
            if self.scale_rewards != "none" and mode == "train":
                advantages = advantages / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))  # for logging

        elif self.multi_objective_aggregation == "normalize_then_sum":
            raise NotImplementedError("normalize_then_sum aggregation is not implemented yet. Please use sum_then_normalize.")

        else:
            raise ValueError(
                f"Invalid multi_objective_aggregation: {self.multi_objective_aggregation}. Must be "
                "'sum_then_normalize' or 'normalize_then_sum'."
            )

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
        #self._logs["advantages"].extend(all_process_advantages.tolist())

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

    @profiling_decorator
    def _calculate_Hista_advantages(
        self, 
        inputs: list[dict[str, torch.Tensor | Any]],
        prompts: list[str] | list[list[dict]],
        completions: list[str],
        completion_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        hidden_states: torch.Tensor, 
        completion_ids_list: list[torch.Tensor],
        rewards_per_func: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.accelerator.device
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        time_start = time.time()
        # First process the hidden states individually to select the embeddings we want
        representation_embeddings_list, response_indices_list, representation_indices_list = steps_selection(
            hidden_states = hidden_states,
            start_token_index = [prompt_ids.size(1)] * hidden_states.size(0),
            attention_mask = attention_mask,
            prompt_mask = prompt_mask,
            min_interval = self.hista_min_interval,
            alpha = self.hista_alpha,
            mean_window = self.hista_mean_window,
            selection_method=embedding_selection_uniform,
            average_method=exponential_running_average,
        )

        # Attach the ids to each embeddings list
        id_embeddings_indices_reward_pair = []
        distinct_ids = set()
        for representation_embeddings, representation_indices, one_input, reward in zip(representation_embeddings_list, representation_indices_list, inputs, rewards):
            id_embeddings_indices_reward_pair.append( (one_input["id"], representation_embeddings, representation_indices, reward) )
            distinct_ids.add(one_input["id"])
        logger.debug(f"Embedding selection time: {time.time() - time_start:.2f} seconds")
        time_start = time.time()
        # Gather embeddings across grouped processes if needed
        if self.gather_embeddings:
            if self.embeddings_gather_groups is None or self.embeddings_group_index is None or self.embeddings_group_size is None:
                raise RuntimeError("Embedding gather groups are not initialized correctly.")
            gather_group = self.embeddings_gather_groups[self.embeddings_group_index]
            gathered_lists = [None for _ in range(self.embeddings_group_size)]
            dist.all_gather_object(gathered_lists, id_embeddings_indices_reward_pair, group=gather_group)
            gathered_id_embeddings_indices_reward_pair = []
            for gathered in gathered_lists:
                if gathered is None:
                    continue
                gathered_id_embeddings_indices_reward_pair.extend(gathered)
        else:
            gathered_id_embeddings_indices_reward_pair = id_embeddings_indices_reward_pair

        logger.debug(f"Embedding gathering time: {time.time() - time_start:.2f} seconds")
        time_start = time.time()

        # Then group the required embeddings by id
        id_embeddings, id_representation_indices, id_rewards = {}, {}, {}
        for one_id, representation_embeddings, representation_indices, reward in gathered_id_embeddings_indices_reward_pair:
            if one_id not in distinct_ids:
                continue
            if one_id not in id_embeddings:
                id_embeddings[one_id] = []
                id_representation_indices[one_id] = []
                id_rewards[one_id] = []
            id_embeddings[one_id].append(representation_embeddings)
            id_representation_indices[one_id].append(representation_indices)
            id_rewards[one_id].append(reward.cpu())

        id_average_rewards = {}
        for id in id_embeddings.keys():
            id_average_rewards[id] = sum(id_rewards[id]) / len(id_rewards[id])

        # Group embeddings according to id
        id_embeddings_sequence, id_nodes_sequence, id_representations_sequence = to_embeddings_indices_sequence(
            id_embeddings, id_representation_indices, id_rewards)
        in_group_base = self.num_generations // self.embeddings_group_size * self.embeddings_in_group_index if self.gather_embeddings else 0
        logger.debug(f"Embedding grouping time: {time.time() - time_start:.2f} seconds")
        time_start = time.time()
        # Calculate advantages for each sample
        advantages_list = []
        for i, (embeddings, representation_indices, response_indices) in enumerate(zip(representation_embeddings_list, representation_indices_list, response_indices_list)):
            id = inputs[i]["id"]
            final_reward = rewards[i].item()
            online_nodes = [(0, idx+1) for idx in representation_indices[1:-1]]
            online_representations = id_representations_sequence[id][in_group_base + i % self.num_generations]

            existing_nodes = id_nodes_sequence[id]

            # Calculate per-node estimated value
            per_nodes_estimated_value = calculate_weighted_distance_for_nodes(
                offline_embeddings = id_embeddings_sequence[id],
                existing_nodes = existing_nodes,
                online_embeddings = embeddings,
                online_nodes = online_nodes,
                online_representations = online_representations,
                max_k = self.hista_max_k,
                min_k = self.hista_min_k,
                t = 1.0,
                min_distance = self.hista_min_d,
            )
            per_nodes_estimated_value = [id_average_rewards[id]] + per_nodes_estimated_value  + [final_reward]
            advantages = calculate_GAE_advantages(
                per_nodes_estimated_value,
                response_indices,
                self.gae_lambda,
                completion_ids.size(1),
            )
            advantages = torch.tensor(advantages, device=device)
            advantages_list.append(advantages)
            logger.debug(f"Advantage calculation time (one sample): {time.time() - time_start:.2f} seconds")
            time_start = time.time()

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            # Log the completions, advantages distribution in decoded token, and reward
            if self.accelerator.is_main_process:
                for i, advantages in enumerate(advantages_list):
                    print('*' * 100)
                    print(f"Prompt: {prompts[i]}")
                    print('-' * 100)
                    print('Initial state value function:', id_average_rewards[inputs[i]["id"]])
                    last_advantage_pos = 0
                    for j, advantage in enumerate(advantages):
                        if advantage != advantages[last_advantage_pos]:
                            print(f"In the {last_advantage_pos} to {j} tokens, the advantage is {advantages[last_advantage_pos]}.")
                            decoded_segment = self.processing_class.decode(
                                completion_ids[i][last_advantage_pos:j], skip_special_tokens=True
                            )
                            print("In that position, the decoded completion is:")
                            print(decoded_segment)
                            print('-' * 100)
                            last_advantage_pos = j
                    print(f"In the {last_advantage_pos} to {len(advantages)} position, the advantage is {advantages[last_advantage_pos]}.")
                    decoded_segment = self.processing_class.decode(
                        completion_ids[i][last_advantage_pos:], skip_special_tokens=True
                    )
                    print(f"In that position, the decoded completion is: {decoded_segment}")
                    print('-' * 100)
                    print(f"The final reward for this completion is {rewards[i].item()}.")
                    print('*' * 100)

        advantages_list = torch.stack(advantages_list).to(device)
        rewards_per_func = gather(rewards_per_func)
        return advantages_list, rewards_per_func

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        BaseTrainer.log(self, logs, start_time)
        if self.args.metric_for_best_model is not None and mode == "eval":
            try:
                self.metric_value = self._metrics["eval"][f"{self.args.metric_for_best_model}"][-1]
            except KeyError as exc:
                raise KeyError(
                    f"The `metric_for_best_model` training argument is set to '{self.args.metric_for_best_model}', which is not found in the evaluation metrics. "
                    f"The available evaluation metrics are: {list(self._metrics['eval'].keys())}. Consider changing the `metric_for_best_model` via the TrainingArguments."
                ) from exc
            
        self._metrics[mode].clear()

    @profiling_decorator
    def _get_per_token_logps_and_hidden_states(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log-probs and (optionally) entropies for each token."""
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        all_hidden_states = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start : start + batch_size]

            # Only add logits_to_keep if the model supports it
            if "logits_to_keep" in self.model_kwarg_keys:
                # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings
            model_inputs["output_hidden_states"] = True
            outputs = model(**model_inputs)
            logits = outputs.logits
            hidden_states = outputs.hidden_states[-1]
            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)  # compute logprobs
            all_logps.append(logps)
            all_hidden_states.append(hidden_states)

        logps = torch.cat(all_logps, dim=0)
        hidden_states = torch.cat(all_hidden_states, dim=0)
            
        return logps, hidden_states
