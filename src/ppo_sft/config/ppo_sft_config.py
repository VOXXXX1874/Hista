
from dataclasses import dataclass, field
from typing import Optional

import trl

@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
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
    trainer_type: str = field(
        default="SFTTrainer",
        metadata={"help": "Type of trainer to use"},
    )
    trainable_layers: Optional[list[str]] = field(
        default=None,
        metadata={"help": "List of layer name patterns to keep trainable. All other layers will be frozen. Example: ['model.layers.23', 'lm_head']"},
    )
    gae_lambda: float = field(
        default=0.95,
        metadata={"help": "Lambda used by GAE for PPO critic targets (discount factor is fixed to 1)."},
    )
