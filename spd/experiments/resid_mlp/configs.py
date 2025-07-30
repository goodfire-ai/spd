from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

from spd.spd_types import Probability


class ResidMLPModelConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
    n_features: PositiveInt
    d_embed: PositiveInt
    d_mlp: PositiveInt
    n_layers: PositiveInt
    act_fn_name: Literal["gelu", "relu"] = Field(
        description="Defines the activation function in the model. Also used in the labeling "
        "function if label_type is act_plus_resid."
    )
    in_bias: bool
    out_bias: bool


class ResidMLPTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_project: str | None = None  # The name of the wandb project (if None, don't log to wandb)
    seed: int = 0
    resid_mlp_model_config: ResidMLPModelConfig
    label_fn_seed: int = 0
    label_type: Literal["act_plus_resid", "abs"] = "act_plus_resid"
    loss_type: Literal["readoff", "resid"] = "readoff"
    use_trivial_label_coeffs: bool = False
    feature_probability: PositiveFloat
    synced_inputs: list[list[int]] | None = None
    importance_val: float | None = None
    data_generation_type: Literal[
        "exactly_one_active", "exactly_two_active", "at_least_zero_active"
    ] = "at_least_zero_active"
    batch_size: PositiveInt
    steps: PositiveInt
    print_freq: PositiveInt
    lr: PositiveFloat
    lr_schedule: Literal["linear", "constant", "cosine", "exponential"] = "constant"
    fixed_random_embedding: bool = False
    fixed_identity_embedding: bool = False
    n_batches_final_losses: PositiveInt = 1

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        assert not (self.fixed_random_embedding and self.fixed_identity_embedding), (
            "Can't have both fixed_random_embedding and fixed_identity_embedding"
        )
        if self.fixed_identity_embedding:
            assert self.resid_mlp_model_config.n_features == self.resid_mlp_model_config.d_embed, (
                "n_features must equal d_embed if we are using an identity embedding matrix"
            )
        if self.synced_inputs is not None:
            # Ensure that the synced_inputs are non-overlapping with eachother
            all_indices = [item for sublist in self.synced_inputs for item in sublist]
            if len(all_indices) != len(set(all_indices)):
                raise ValueError("Synced inputs must be non-overlapping")
        return self


class ResidMLPTaskConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
    task_name: Literal["resid_mlp"] = Field(
        default="resid_mlp",
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
