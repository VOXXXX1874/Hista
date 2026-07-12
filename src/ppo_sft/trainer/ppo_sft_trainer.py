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

import contextlib
import os
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BaseImageProcessor,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainingArguments,
)
from trl import SFTTrainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available
from trl import ModelConfig, get_kbit_device_map, get_quantization_config
from packaging.version import Version
from trl.trainer.sft_config import SFTConfig
from trl.trainer.utils import get_config_model_id
from trl.chat_template_utils import clone_chat_template
from trl.models import get_act_offloading_ctx_manager

from transformers.utils import (
    is_safetensors_available,
    logging,
)

if is_peft_available():
    from peft import PeftConfig

if is_safetensors_available():
    import safetensors.torch

logger = logging.get_logger(__name__)

class CriticModelWrapper(nn.Module):
    """
    Wrapper that combines a frozen action model (language model) with a critic value head.
    The action model extracts hidden states, and the critic head predicts values.
    """
    def __init__(
        self,
        action_model: PreTrainedModel,
        critic_model: PreTrainedModel,
        gae_lambda: float = 0.95,
    ):
        super().__init__()
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError(f"`gae_lambda` must be in [0, 1], got {gae_lambda}.")

        self.action_model = action_model
        self.critic_model = critic_model
        self.config = critic_model.config
        self.gae_lambda = gae_lambda
        
        # Freeze the action model
        for param in self.action_model.parameters():
            param.requires_grad = False
        
        # Add a value head to the critic model
        hidden_size = critic_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, bias=False)
        
        # Initialize value head
        nn.init.zeros_(self.value_head.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass through the critic model.
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            labels: Sequence-level terminal rewards with shape ``[batch_size]``
            
        Returns:
            Dictionary containing loss, values, and hidden states
        """
        batch_size, seq_length = input_ids.shape
        
        # Run only the transformer backbones. Calling AutoModelForCausalLM here
        # would also materialize unused [batch, sequence, vocabulary] logits for
        # both models, which is prohibitively expensive for long sequences.
        action_backbone = self.action_model.base_model
        with torch.no_grad():
            action_outputs = action_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
            hidden_states = action_outputs.last_hidden_state  # [batch_size, seq_length, hidden_size]
        
        critic_backbone = self.critic_model.base_model
        critic_outputs = critic_backbone(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Implementation 1: Predict values for each position
        # Get the last layer hidden states from critic
        critic_hidden_states = critic_outputs.last_hidden_state  # [batch_size, seq_length, hidden_size]
        # Predict values for each position
        values = self.value_head(critic_hidden_states).squeeze(-1)  # [batch_size, seq_length]

        # Implementation 2: Predict value for the last position only
        #critic_hidden_states = critic_outputs.hidden_states[-1][:, -1, :]  # [batch_size, hidden_size]
        #values = self.value_head(critic_hidden_states).squeeze(-1)  # [batch_size]
        
        loss = None
        if labels is not None:
            if attention_mask is None:
                value_mask = torch.ones_like(values, dtype=torch.bool)
            else:
                value_mask = attention_mask.to(device=values.device, dtype=torch.bool)

            target_rewards = labels.to(device=values.device, dtype=values.dtype)
            if target_rewards.ndim == 2 and target_rewards.shape[-1] == 1:
                target_rewards = target_rewards.squeeze(-1)
            if target_rewards.ndim != 1 or target_rewards.shape[0] != batch_size:
                raise ValueError(
                    "`labels` must contain one terminal reward per sequence and have shape "
                    f"[batch_size] (or [batch_size, 1]); got {tuple(labels.shape)}."
                )
            if not value_mask.any(dim=1).all():
                raise ValueError("Every sequence must contain at least one non-padding token.")

            # The dataset supplies one outcome reward per trajectory. PPO token rewards are
            # therefore zero except at the final valid token. Gamma is fixed to 1.
            rewards = torch.zeros_like(values)
            last_token_indices = value_mask.long().sum(dim=1) - 1
            rewards.scatter_(1, last_token_indices.unsqueeze(1), target_rewards.unsqueeze(1))

            # GAE: delta_t = r_t + V(s_{t+1}) - V(s_t),
            # A_t = delta_t + lambda * A_{t+1}. Detaching here makes the full
            # target V(s_t) + A_t a stop-gradient target, as in PPO critic loss.
            detached_values = values.detach()
            advantages = torch.zeros_like(detached_values)
            next_advantage = torch.zeros(batch_size, device=values.device, dtype=values.dtype)
            for token_idx in range(seq_length - 1, -1, -1):
                valid = value_mask[:, token_idx]
                if token_idx + 1 < seq_length:
                    next_valid = value_mask[:, token_idx + 1]
                    next_value = torch.where(
                        next_valid, detached_values[:, token_idx + 1], torch.zeros_like(next_advantage)
                    )
                else:
                    next_value = torch.zeros_like(next_advantage)

                delta = rewards[:, token_idx] + next_value - detached_values[:, token_idx]
                next_advantage = torch.where(
                    valid, delta + self.gae_lambda * next_advantage, torch.zeros_like(next_advantage)
                )
                advantages[:, token_idx] = next_advantage

            value_targets = (detached_values + advantages).detach()
            loss = F.mse_loss(values[value_mask], value_targets[value_mask])
        
        return {
            "loss": loss,
            "values": values,
            "value_targets": value_targets if labels is not None else None,
            "value_mask": value_mask if labels is not None else None,
            "hidden_states": critic_hidden_states,
        }
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the critic model."""
        self.critic_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the critic model."""
        self.critic_model.gradient_checkpointing_disable()

    @classmethod
    def load_critic_wrapper(
        cls,
        action_model_path: str,
        critic_model_path: str,
        value_head_path: str,
        device_map: Optional[str] = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        attn_implementation: Optional[str] = None,
        **model_kwargs
    ) -> "CriticModelWrapper":
        """
        Load a CriticModelWrapper from saved checkpoints.

        Args:
            action_model_path: Path to the action model (frozen language model)
            critic_model_path: Path to the saved critic model checkpoint
            value_head_path: Path to the directory containing value_head weights
            device_map: Device mapping for model loading
            torch_dtype: Data type for model weights
            **model_kwargs: Additional keyword arguments for model loading
            
        Returns:
            CriticModelWrapper instance with loaded weights
            
        Example:
            ```python
            wrapper = CriticModelWrapper.load_critic_wrapper(
                action_model_path="Qwen/Qwen2-0.5B-Instruct",
                critic_model_path="./output/checkpoint-100/",
                value_head_path="./output/checkpoint-100",
                torch_dtype=torch.bfloat16
            )
            ```
        """
        # Load action model (frozen)
        logger.info(f"Loading action model from {action_model_path}")
        action_model = AutoModelForCausalLM.from_pretrained(
            action_model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            **model_kwargs
        )

        # Load critic model
        logger.info(f"Loading critic model from {critic_model_path}")
        critic_model = AutoModelForCausalLM.from_pretrained(
            critic_model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            **model_kwargs
        )

        # Create wrapper
        wrapper = cls(action_model, critic_model)

        # Load value head weights
        value_head_safetensors_path = os.path.join(value_head_path, "value_head.safetensors")
        value_head_pt_path = os.path.join(value_head_path, "value_head.pt")

        if os.path.exists(value_head_safetensors_path):
            logger.info(f"Loading value head from {value_head_safetensors_path}")
            if is_safetensors_available():
                value_head_state_dict = safetensors.torch.load_file(value_head_safetensors_path)
            else:
                raise ImportError("safetensors is not available but .safetensors file exists")
        elif os.path.exists(value_head_pt_path):
            logger.info(f"Loading value head from {value_head_pt_path}")
            value_head_state_dict = torch.load(value_head_pt_path, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"Value head weights not found at {value_head_path}. "
                f"Expected either value_head.safetensors or value_head.pt"
            )

        # Load value head weights
        wrapper.value_head.load_state_dict(value_head_state_dict, strict=True)
        wrapper.value_head.to(torch.bfloat16)
        wrapper.value_head.to("cuda:0")
        logger.info("Successfully loaded value head weights")

        return wrapper


class PPOSFTDataCollator:
    """
    Data collator for PPO critic training.
    Expects dataset with 'input_ids', 'attention_mask', and 'reward' fields.
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Extract rewards
        rewards = [feature.pop("reward") for feature in features]
        
        # Find max length
        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of is not None:
            max_length = ((max_length + self.pad_to_multiple_of - 1) // 
                         self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        # Pad sequences
        input_ids = []
        attention_mask = []
        
        for feature in features:
            seq_length = len(feature["input_ids"])
            padding_length = max_length - seq_length
            
            # Pad input_ids
            padded_input_ids = feature["input_ids"] + [self.tokenizer.pad_token_id] * padding_length
            input_ids.append(padded_input_ids)
            
            # Pad attention_mask
            if "attention_mask" in feature:
                padded_attention_mask = feature["attention_mask"] + [0] * padding_length
            else:
                padded_attention_mask = [1] * seq_length + [0] * padding_length
            attention_mask.append(padded_attention_mask)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            # The wrapper places each sequence-level reward on its final valid token.
            "labels": torch.tensor(rewards, dtype=torch.float32),
        }


class PPOSFTTrainer(SFTTrainer):
    """
    Trainer for PPO Critic Model Training.
    
    This trainer is designed to train a critic model for PPO by:
    1. Using a frozen language model (action model) to extract hidden states
    2. Training a critic model with a value head to predict rewards
    3. Optimizing using MSE loss between predicted values and actual rewards
    
    Example:
    ```python
    from datasets import load_dataset
    from trl import PPOSFTTrainer
    
    # Dataset should contain responses and rewards
    dataset = load_dataset("your_dataset")
    
    trainer = PPOSFTTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        train_dataset=dataset
    )
    trainer.train()
    ```
    
    Args:
        model (`Union[str, PreTrainedModel]`):
            Language model to be used as both action model (frozen) and critic model (trainable).
            Can be either a model path or a PreTrainedModel instance.
        args ([`SFTConfig`], *optional*, defaults to `None`):
            Configuration for this trainer.
        data_collator (`DataCollator`, *optional*):
            Data collator for batching. If None, uses PPOSFTDataCollator.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset for training. Should contain 'input_ids', 'attention_mask', and 'reward' fields.
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Dataset]`):
            Dataset for evaluation.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Tokenizer for processing. If None, loaded from model.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            Callbacks for customizing training.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            Custom optimizer and scheduler.
        optimizer_cls_and_kwargs (`Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]`, *optional*, defaults to `None`):
            Custom optimizer class and kwargs.
        preprocess_logits_for_metrics (`Callable`, *optional*, defaults to `None`):
            Function to preprocess logits before metrics.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration for the critic model.
    """
    
    _tag_names = ["trl", "ppo-critic"]
    
    def __init__(
        self,
        model: Union[str, nn.Module, PreTrainedModel],
        args: Optional[Union[SFTConfig, TrainingArguments]] = None,
        model_args = None,
        data_collator = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[tuple[Type[torch.optim.Optimizer], dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Optional[Union[Callable[[dict], str], Callable[[dict], list[str]]]] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = SFTConfig(f"{model_name}-SFT")
        elif isinstance(args, TrainingArguments) and not isinstance(args, SFTConfig):
            dict_args = args.to_dict()
            dict_args["hub_token"] = args.hub_token  # to_dict hides the hub_token
            if Version(transformers.__version__) < Version("5.0.0"):
                dict_args.pop("push_to_hub_token")
            args = SFTConfig(**dict_args)

        if train_dataset is None:
            raise ValueError("`train_dataset` is required")
        elif isinstance(train_dataset, IterableDataset):
            # IterableDataset requires dispatch_batches=False because Accelerate's dispatch mode may try to concatenate
            # batches from multiple processes, leading to mismatch errors.
            if args.accelerator_config.dispatch_batches is True:
                logger.warning(
                    "You are using an `IterableDataset` for training with `dispatch_batches=True`. `dispatch_batches` "
                    "is forced to `False` when using an `IterableDataset`. To remove this warning, unset "
                    "`dispatch_batches` in `SFTConfig` or set it to `False`."
                )
            args.accelerator_config.dispatch_batches = False
        
        # Create or load models
        if isinstance(model, str):
            action_model = self.get_model(model_args, args)
            critic_model = self.get_model(model_args, args)
        else:
            raise ValueError("When using PPOSFTTrainer, the `model` argument must be a model path string.")
        
        # Create the wrapper model
        wrapper_model = CriticModelWrapper(action_model, critic_model, gae_lambda=args.gae_lambda)
        
        # PEFT configuration for critic model only
        if peft_config is not None:
            wrapper_model.critic_model = self._prepare_peft_model(wrapper_model.critic_model, peft_config, args)
        
        # Handle the tokenizer
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path)
            if processing_class.pad_token is None:
                processing_class.pad_token = processing_class.eos_token  # required for padding when collating data

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
            self._is_vlm = True
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
            self._is_vlm = False
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if args.eos_token is not None:
            eos_token = args.eos_token
            eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
            if eos_token_id is None:
                raise ValueError(
                    f"The specified `eos_token` ('{eos_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `eos_token` exists "
                    "in the vocabulary before using it as an EOS token."
                )
            tokenizer.eos_token_id = eos_token_id

        if args.chat_template_path is not None:
            if os.path.isfile(args.chat_template_path) and args.chat_template_path.endswith((".jinja", ".j2")):
                with open(args.chat_template_path, encoding="utf-8") as chat_template_file:
                    processing_class.chat_template = chat_template_file.read()
                added_tokens = []
            else:
                model, processing_class, added_tokens = clone_chat_template(
                    model, processing_class, args.chat_template_path
                )
        else:
            added_tokens = []

        # Catch some wrong configurations related to VLMs
        if self._is_vlm and args.packing:
            raise ValueError(
                "Packing is not supported for vision-language models. Please set `packing=False` in the SFTConfig."
            )
        if self._is_vlm and args.padding_free:
            raise ValueError(
                "Padding-free training is yet not supported for vision-language models. Please set "
                "`padding_free=False` in the `SFTConfig`."
            )
        if self._is_vlm and args.assistant_only_loss:
            raise ValueError(
                "Assistant-only loss is not yet supported for vision-language models. Please set "
                "`assistant_only_loss=False` in the `SFTConfig`."
            )
        
        # Dataset
        preprocess_dataset = args.dataset_kwargs is None or not args.dataset_kwargs.get("skip_prepare_dataset", False)
        if preprocess_dataset:
            train_dataset = self._prepare_dataset(
                train_dataset, processing_class, args, args.packing, formatting_func, "train"
            )
            if eval_dataset is not None:
                packing = args.packing if args.eval_packing is None else args.eval_packing
                if isinstance(eval_dataset, dict):
                    eval_dataset = {
                        key: self._prepare_dataset(dataset, processing_class, args, packing, formatting_func, key)
                        for key, dataset in eval_dataset.items()
                    }
                else:
                    eval_dataset = self._prepare_dataset(
                        eval_dataset, processing_class, args, packing, formatting_func, "eval"
                    )

        # Data collator
        if data_collator is None:
            data_collator = PPOSFTDataCollator(tokenizer=processing_class)
        
        # Initialize the Trainer. Parent class will handle:
        # - DeepSpeed configuration (through create_accelerator_and_postprocess)
        # - FSDP setup
        # - Distributed training setup
        # - Optimizer and scheduler creation
        # Some arguments are only available for transformers>=4.47.0. Can be removed when the min version is bumped.
        super_init_kwargs = {}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super_init_kwargs["optimizer_cls_and_kwargs"] = optimizer_cls_and_kwargs
        else:
            if optimizer_cls_and_kwargs is not None:
                warnings.warn(
                    "The `optimizer_cls_and_kwargs` argument is only available for `transformers>=4.47.0`. "
                    "The default optimizer will be used. "
                    "Remove the `optimizer_cls_and_kwargs` or upgrade to `transformers>=4.47.0`."
                )

        Trainer.__init__(
            self,
            model=wrapper_model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **super_init_kwargs,
        )
        
        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        # Initialize activation offloading context
        if self.args.activation_offloading:
            self.maybe_activation_offload_context = get_act_offloading_ctx_manager(model=self.model)
        else:
            self.maybe_activation_offload_context = contextlib.nullcontext()

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: SFTConfig,
        packing: bool,
        formatting_func: Optional[Callable[[dict], str]],
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        dataset = super()._prepare_dataset(
            dataset=dataset,
            processing_class=processing_class,
            args=args,
            packing=packing,
            formatting_func=formatting_func,
            dataset_name=dataset_name,
        )

        # Drop the "text" column if it exists, as we only need input_ids for training
        if isinstance(dataset, Dataset) and "text" in dataset.column_names:
            dataset = dataset.remove_columns("text")

        return dataset

    def get_model(self, model_args: ModelConfig, training_args: SFTConfig) -> AutoModelForCausalLM:
        """Get the model"""
        torch_dtype = (
            model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
        )
        quantization_config = get_quantization_config(model_args)
        model_kwargs = dict(
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        )
        return model
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute MSE loss for value function prediction.
        """
        mode = "train" if self.model.training else "eval"
        outputs = model(**inputs)
        loss = outputs["loss"]

        # Report metrics against exactly the same stop-gradient GAE targets used
        # by the critic loss, excluding padding tokens.
        value_mask = outputs["value_mask"]
        predicted_values = outputs["values"][value_mask].detach().float()
        value_targets = outputs["value_targets"][value_mask].detach().float()
        mae = torch.abs(predicted_values - value_targets).mean()
        batch_metrics = torch.stack(
            (
                loss.detach().float(),
                mae,
                predicted_values.mean(),
                value_targets.mean(),
                inputs["labels"].detach().float().mean(),
            )
        )
        # All ranks participate in compute_loss. Aggregate them so checkpoint
        # selection is based on the full distributed evaluation set.
        batch_metrics = self.accelerator.gather(batch_metrics.unsqueeze(0)).mean(dim=0)
        metric_names = (
            "mse_loss",
            "mean_absolute_error",
            "mean_predicted_value",
            "mean_value_target",
            "mean_terminal_reward",
        )
        for name, metric in zip(metric_names, batch_metrics):
            self._metrics[mode][name].append(metric.item())
        
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # Extract the critic model and value head from the wrapper
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        if isinstance(unwrapped_model, CriticModelWrapper):
            if state_dict is None:
                raise ValueError("State dict must be provided for saving the critic model.")

            value_head_state_dict = {}
            # Only save the critic model and value head
            for key in list(state_dict.keys()):
                if not key.startswith("critic_model."):
                    if key.startswith("value_head."):
                        value_head_state_dict[key[len("value_head.") :]] = state_dict.pop(key)
                        continue
                    del state_dict[key]
                else:
                    # Remove the "critic_model." prefix
                    new_key = key[len("critic_model.") :]
                    state_dict[new_key] = state_dict.pop(key)

            # Save the critic model
            unwrapped_model.critic_model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors
            )
            
            if self.args.save_safetensors:
                safetensors.torch.save_file(
                    value_head_state_dict,
                    os.path.join(output_dir, "value_head.safetensors"),
                    metadata={"format": "pt"}
                )
            else:
                value_head_path = os.path.join(output_dir, "value_head.pt")
                torch.save(value_head_state_dict, value_head_path)
            
            logger.info(f"Saved critic model to {output_dir}")
            logger.info(f"Saved value head to {output_dir}")
        else:
            raise ValueError("The unwrapped model is not an instance of CriticModelWrapper.")

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

    def _save_optimizer_and_scheduler(self, output_dir):
        """
        Override to prevent saving optimizer and scheduler state.
        This saves disk space when we only care about model checkpoints.
        """
        logger.info("Skipping optimizer and scheduler state saving.")
        # Optionally, you can still save them if needed by uncommenting below:
        # super()._save_optimizer_and_scheduler(output_dir)
        pass
