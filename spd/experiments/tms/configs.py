from typing import Literal, Self

from pydantic import Field, NonNegativeInt, PositiveInt, model_validator

from spd.base_config import BaseConfig
from spd.spd_types import Probability


class TMSModelConfig(BaseConfig):
    n_features: PositiveInt
    n_hidden: PositiveInt
    n_hidden_layers: NonNegativeInt
    tied_weights: bool
    init_bias_to_zero: bool
    device: str


class TMSTrainConfig(BaseConfig):
    wandb_project: str | None = None  # The name of the wandb project (if None, don't log to wandb)
    tms_model_config: TMSModelConfig
    feature_probability: float
    batch_size: PositiveInt
    steps: PositiveInt
    seed: int = 0
    lr: float
    lr_schedule: Literal["linear", "cosine", "constant"] = "linear"
    data_generation_type: Literal["at_least_zero_active", "exactly_one_active"]
    fixed_identity_hidden_layers: bool = False
    fixed_random_hidden_layers: bool = False
    synced_inputs: list[list[int]] | None = None

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if self.fixed_identity_hidden_layers and self.fixed_random_hidden_layers:
            raise ValueError(
                "Cannot set both fixed_identity_hidden_layers and fixed_random_hidden_layers to True"
            )
        if self.synced_inputs is not None:
            # Ensure that the synced_inputs are non-overlapping with eachother
            all_indices = [item for sublist in self.synced_inputs for item in sublist]
            if len(all_indices) != len(set(all_indices)):
                raise ValueError("Synced inputs must be non-overlapping")
        return self


class TMSTaskConfig(BaseConfig):
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
