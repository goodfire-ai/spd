"""Config classes of various types"""

import warnings
from typing import Any, ClassVar, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from spd.log import logger
from spd.spd_types import ModelPath, Probability


class TMSTaskConfig(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)  # Changed from "forbid" to "allow"
    task_name: Literal["tms"] = Field(
        default="tms",
        description="Task identifier for TMS",
    )
    feature_probability: Probability = Field(
        ...,
        description="Probability that a given feature is active in generated data",
    )
    data_generation_type: Literal["exactly_one_active", "at_least_zero_active"] = Field(
        default="at_least_zero_active",
        description="Strategy for generating synthetic data for TMS training",
    )


class ResidualMLPTaskConfig(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)  # Changed from "forbid" to "allow"
    task_name: Literal["residual_mlp"] = Field(
        default="residual_mlp",
        description="Identifier for the residual-MLP decomposition task",
    )
    feature_probability: Probability = Field(
        ...,
        description="Probability that a given feature is active in generated data",
    )
    data_generation_type: Literal[
        "exactly_one_active", "exactly_two_active", "at_least_zero_active"
    ] = Field(
        default="at_least_zero_active",
        description="Strategy for generating synthetic data for residual-MLP training",
    )


class LMTaskConfig(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)  # Changed from "forbid" to "allow"
    task_name: Literal["lm"] = Field(
        default="lm",
        description="Identifier for the language-model decomposition task",
    )
    max_seq_len: PositiveInt = Field(
        default=512,
        description="Maximum sequence length to truncate or pad inputs to",
    )
    buffer_size: PositiveInt = Field(
        default=1000,
        description="Buffered sample count for streaming dataset shuffling",
    )
    dataset_name: str = Field(
        default="lennart-finke/SimpleStories",
        description="HuggingFace dataset identifier to use for the LM task",
    )
    column_name: str = Field(
        default="story",
        description="Dataset column that contains the text to train on",
    )
    train_data_split: str = Field(
        default="train",
        description="Name of the dataset split used for training",
    )
    eval_data_split: str = Field(
        default="test",
        description="Name of the dataset split used for evaluation",
    )


class Config(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)  # Changed from "forbid" to "allow"
    # --- WandB
    wandb_project: str | None = Field(
        default=None,
        description="Weights & Biases project name (set to None to disable WandB logging)",
    )
    wandb_run_name: str | None = Field(
        default=None,
        description="Explicit name for the WandB run (None generates an automatic name)",
    )
    wandb_run_name_prefix: str = Field(
        default="",
        description="Prefix prepended to an auto-generated WandB run name",
    )

    # --- General ---
    seed: int = Field(default=0, description="Random seed for reproducibility")
    C: PositiveInt = Field(
        ...,
        description="The number of subcomponents per layer",
    )
    task: TMSTaskConfig | ResidualMLPTaskConfig | LMTaskConfig = Field(
        ...,
        description="Task-specific configuration",
    )
    target_model: ModelPath = Field(
        ...,
        description="Model to decompose",
    )

    # --- Training ---
    train_steps: PositiveInt = Field(
        default=2000,
        description="Total number of training steps",
    )
    eval_freq: PositiveInt = Field(
        default=100,
        description="How often to evaluate on held-out data",
    )
    print_freq: PositiveInt = Field(
        default=100,
        description="How often to print training progress",
    )
    save_freq: PositiveInt = Field(
        default=5000,
        description="How often to save checkpoints",
    )
    batch_size: PositiveInt = Field(
        default=8,
        description="Training batch size",
    )
    buffer_size: PositiveInt = Field(
        default=1000,
        description="Buffer size for shuffling during streaming",
    )
    lr: PositiveFloat = Field(
        default=1e-3,
        description="Learning rate",
    )
    weight_decay: NonNegativeFloat = Field(
        default=0.0,
        description="L2 regularization coefficient",
    )
    lr_schedule: Literal["constant", "linear", "cosine"] = Field(
        default="constant",
        description="Learning rate schedule",
    )

    # --- Model ---
    d_hidden: PositiveInt = Field(
        default=512,
        description="Hidden dimension of the decomposition model",
    )
    k: PositiveInt = Field(
        default=16,
        description="Top-k components to keep active",
    )
    init_scale: PositiveFloat = Field(
        default=1.0,
        description="Initialization scale for model parameters",
    )

    # --- Evaluation ---
    eval_metrics: list[str] = Field(
        default_factory=lambda: ["CI_L0"],
        description="List of evaluation metrics to compute",
    )

    @model_validator(mode="after")
    def validate_eval_metrics(self) -> Self:
        """Validate eval metrics with warnings instead of errors for missing classes."""
        # Import here to avoid circular imports
        try:
            from spd.metrics import AVAILABLE_METRICS
            
            valid_metrics = []
            for metric in self.eval_metrics:
                if metric in AVAILABLE_METRICS:
                    valid_metrics.append(metric)
                else:
                    warnings.warn(
                        f"Metric class '{metric}' not found in current codebase. "
                        f"Available classes: {list(AVAILABLE_METRICS.keys())}. "
                        f"This metric will be skipped.",
                        UserWarning
                    )
            
            # Update eval_metrics to only include valid ones
            self.eval_metrics = valid_metrics
            
        except ImportError:
            # If metrics module doesn't exist, just warn and continue
            warnings.warn(
                "Could not import metrics module for validation. "
                "Eval metrics validation skipped.",
                UserWarning
            )
        
        return self

    n_eval_seqs: PositiveInt = Field(
        default=1000,
        description="Number of sequences to evaluate on",
    )
    n_eval_components: PositiveInt = Field(
        default=8,
        description="Number of components to evaluate",
    )
