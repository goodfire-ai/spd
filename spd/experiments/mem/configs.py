from typing import Literal

from pydantic import Field, PositiveInt

from spd.base_config import BaseConfig


class MemModelConfig(BaseConfig):
    """Configuration for the MemTransformer model."""

    vocab_size: PositiveInt = Field(description="Size of the vocabulary (number of unique tokens)")
    d_model: PositiveInt = Field(description="Residual stream width / embedding dimension")
    d_mlp: PositiveInt = Field(description="Hidden width of the MLP block")
    n_heads: PositiveInt = Field(description="Number of attention heads")
    seq_len: PositiveInt = Field(default=3, description="Sequence length (default 3 for mem task)")
    use_layer_norm: bool = Field(
        default=True, description="Whether to use LayerNorm (disable for easier interpretability)"
    )
    device: str = "cpu"


class MemTrainConfig(BaseConfig):
    """Configuration for training a MemTransformer model."""

    wandb_project: str | None = None  # The name of the wandb project (if None, don't log to wandb)
    seed: int = 0
    mem_model_config: MemModelConfig
    n_facts: PositiveInt = Field(description="Number of facts to memorize")
    batch_size: PositiveInt
    steps: PositiveInt
    print_freq: PositiveInt = 100
    lr: float
    lr_schedule: Literal["linear", "cosine", "constant"] = "constant"


class MemTaskConfig(BaseConfig):
    """Task configuration for the mem decomposition task."""

    task_name: Literal["mem"] = Field(
        default="mem",
        description="Identifier for the mem decomposition task",
    )
