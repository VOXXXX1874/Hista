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

import logging
import os
import sys

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint


from rl.configs.numca_configs import NumcaConfig
from rl.configs.grpo_configs import GRPOPlusScriptArguments
from rl.utils.rewards import (
    accuracy_reward,
    format_reward,
    length_reward_threshold
)
from rl.utils.prepare_dataset import prepare_dataset
from rl.utils.wandb_logging import init_wandb_training
from rl.trainer.numca_trainer import NumcaTrainer
from trl import ModelConfig, TrlParser, get_peft_config


logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    train_dataset = prepare_dataset(script_args.dataset_name, script_args.dataset_train_split, training_args.silence, script_args.use_default_system_prompt)
    if script_args.quick_eval_dataset:
        quick_eval_dataset = prepare_dataset(script_args.quick_eval_dataset, 'test', training_args.silence, script_args.use_default_system_prompt)
        # Select a subset of the quick evaluation dataset if quick_eval_dataset_size is specified
        if script_args.quick_eval_dataset_size is not None and len(quick_eval_dataset) > script_args.quick_eval_dataset_size:
            quick_eval_dataset = quick_eval_dataset.select(range(script_args.quick_eval_dataset_size))
        elif len(quick_eval_dataset) <= script_args.quick_eval_dataset_size:
            quick_eval_scale = script_args.quick_eval_dataset_size // len(quick_eval_dataset) + 1
            quick_eval_dataset = quick_eval_dataset.repeat(quick_eval_scale).select(range(script_args.quick_eval_dataset_size))
        print(f"Quick eval dataset loaded with {len(quick_eval_dataset)} samples.")

    # Get reward functions
    for reward_func in script_args.reward_funcs:
        if reward_func not in ["accuracy", "thinking", "format", "length"]:
            raise ValueError(
                f"Reward function {reward_func} is not supported. Please use 'accuracy', 'thinking', 'format', or 'length'."
            )
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "length": length_reward_threshold(training_args.max_completion_length, training_args.overlong_punishment_threshold),
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    logger.info("*** Initializing model kwargs ***")
    device_map = None if script_args.distributed_training else "auto"
    logger.info(f"Distributed training: {script_args.distributed_training}, setting device_map to {device_map}")
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=device_map,  # Disable device_map for distributed training
        dtype=model_args.dtype,  # Set dtype for model loading
    )
    training_args.model_init_kwargs = model_kwargs

    # chat_template for model after qwen3
    if training_args.enable_thinking != None:
        # Add chat_template_kwargs to training_args
        training_args.chat_template_kwargs = {
            "enable_thinking": training_args.enable_thinking,
        }

    #############################
    # Initialize the XGRPO trainer
    #############################
    trainer = NumcaTrainer(
        model=model_args.model_name_or_path,
        # model = model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=quick_eval_dataset,
        peft_config=get_peft_config(model_args), # LoRA parameter
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["Numca"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOPlusScriptArguments, NumcaConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args )
