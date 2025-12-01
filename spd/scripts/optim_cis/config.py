"""Configuration for CI optimization on single prompts."""

from typing import Annotated, Literal, Self

from pydantic import Field, NonNegativeFloat, PositiveFloat, PositiveInt, model_validator

from spd.base_config import BaseConfig
from spd.configs import LossMetricConfigType, SamplingType
from spd.spd_types import Probability


class OptimCIConfig(BaseConfig):
    """Configuration for optimizing CI values on a single prompt."""

    seed: int = Field(
        ...,
        description="Random seed for reproducibility",
    )
    # Model and prompt
    wandb_path: str = Field(
        ...,
        description="Wandb path to load model from, e.g. 'wandb:goodfire/spd/runs/jyo9duz5'",
    )
    prompt: str = Field(
        ...,
        description="The prompt to optimize CI values for",
    )
    label: str = Field(
        ...,
        description="The label to optimize CI values for",
    )

    # Optimization hyperparameters
    lr: PositiveFloat = Field(
        ...,
        description="Learning rate for AdamW optimizer",
    )
    steps: PositiveInt = Field(
        ...,
        description="Number of optimization steps",
    )
    weight_decay: NonNegativeFloat = Field(
        ...,
        description="Weight decay for AdamW optimizer",
    )
    lr_schedule: Literal["linear", "constant", "cosine", "exponential"] = Field(
        ...,
        description="Type of learning-rate schedule to apply",
    )
    lr_exponential_halflife: PositiveFloat | None = Field(
        ...,
        description="Half-life parameter when using an exponential LR schedule",
    )
    lr_warmup_pct: Probability = Field(
        ...,
        description="Fraction of total steps to linearly warm up the learning rate",
    )
    log_freq: PositiveInt = Field(
        ...,
        description="Frequency of logging during optimization",
    )

    # Loss configuration
    loss_metric_configs: list[Annotated[LossMetricConfigType, Field(discriminator="classname")]] = (
        Field(
            ...,
            description="List of loss metric configs (must have coeff set)",
        )
    )

    # CI thresholds and sampling
    ci_threshold: PositiveFloat = Field(
        ...,
        description="Threshold for considering a component alive in original CI values. "
        "Only components with CI > ci_threshold will be optimized.",
    )
    sampling: SamplingType = Field(
        ...,
        description="Sampling mode for stochastic losses: 'continuous' or 'binomial'",
    )
    n_mask_samples: PositiveInt = Field(
        ...,
        description="Number of stochastic masks to sample for recon losses",
    )
    output_loss_type: Literal["mse", "kl"] = Field(
        ...,
        description="Loss type for reconstruction: 'kl' for LMs, 'mse' for vectors",
    )

    # Delta component
    use_delta_component: bool = Field(
        ...,
        description="Whether to use delta component in reconstruction losses",
    )

    # CE/KL metrics
    ce_loss_coeff: float = Field(
        ...,
        description="Coefficient for the CE loss",
    )
    ce_kl_rounding_threshold: float = Field(
        ...,
        description="Threshold for rounding CI values in CE/KL metric computation",
    )

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if self.lr_schedule == "exponential":
            assert self.lr_exponential_halflife is not None, (
                "lr_exponential_halflife must be set if lr_schedule is exponential"
            )
        return self
