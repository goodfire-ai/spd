from typing import Any, Literal

from pydantic import Field, PositiveInt, model_validator

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

    @model_validator(mode="before")
    @classmethod
    def migrate_norm_type(cls, data: Any) -> Any:
        """Migrate old `norm_type` field to `use_layer_norm`."""
        if isinstance(data, dict) and "norm_type" in data:
            norm_type = data.pop("norm_type")
            if "use_layer_norm" not in data:
                data["use_layer_norm"] = norm_type == "layernorm"
        return data


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

    @model_validator(mode="before")
    @classmethod
    def migrate_nested_norm_type(cls, data: Any) -> Any:
        """Migrate old `norm_type` field in nested mem_model_config to `use_layer_norm`."""
        if isinstance(data, dict) and "mem_model_config" in data:
            mem_cfg = data.get("mem_model_config")
            if isinstance(mem_cfg, dict) and "norm_type" in mem_cfg:
                norm_type = mem_cfg.pop("norm_type")
                if "use_layer_norm" not in mem_cfg:
                    mem_cfg["use_layer_norm"] = norm_type == "layernorm"
        return data


class MemTaskConfig(BaseConfig):
    """Task configuration for the mem decomposition task."""

    task_name: Literal["mem"] = Field(
        default="mem",
        description="Identifier for the mem decomposition task",
    )
