"""Config classes of various types"""

from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import (
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from spd.experiments.ih.configs import IHTaskConfig
from spd.experiments.lm.configs import LMTaskConfig
from spd.experiments.resid_mlp.configs import ResidMLPTaskConfig
from spd.experiments.tms.configs import TMSTaskConfig
from spd.log import logger
from spd.models.components import CiFnType
from spd.models.sigmoids import SigmoidTypes
from spd.spd_types import ModelPath, Probability
from spd.utils.general_utils import BaseModel


#### Train Metric Configs ####
class TrainMetricConfig(BaseModel):
    coeff: float = Field(
        ...,
        description="Coefficient used for weighting into loss/total.",
    )


class CIMaskedReconSubsetLossTrainConfig(TrainMetricConfig):
    classname: Literal["CIMaskedReconSubsetLoss"] = "CIMaskedReconSubsetLoss"


class CIMaskedReconLayerwiseLossTrainConfig(TrainMetricConfig):
    classname: Literal["CIMaskedReconLayerwiseLoss"] = "CIMaskedReconLayerwiseLoss"


class CIMaskedReconLossTrainConfig(TrainMetricConfig):
    classname: Literal["CIMaskedReconLoss"] = "CIMaskedReconLoss"


class FaithfulnessLossTrainConfig(TrainMetricConfig):
    classname: Literal["FaithfulnessLoss"] = "FaithfulnessLoss"


class ImportanceMinimalityLossTrainConfig(TrainMetricConfig):
    classname: Literal["ImportanceMinimalityLoss"] = "ImportanceMinimalityLoss"
    pnorm: float
    p_anneal_start_frac: float = 1.0
    p_anneal_final_p: float | None = None
    p_anneal_end_frac: float = 1.0
    eps: float = 1e-12


class StochasticReconLayerwiseLossTrainConfig(TrainMetricConfig):
    classname: Literal["StochasticReconLayerwiseLoss"] = "StochasticReconLayerwiseLoss"


class StochasticReconLossTrainConfig(TrainMetricConfig):
    classname: Literal["StochasticReconLoss"] = "StochasticReconLoss"


class StochasticReconSubsetLossTrainConfig(TrainMetricConfig):
    classname: Literal["StochasticReconSubsetLoss"] = "StochasticReconSubsetLoss"


#### Eval Metric Configs ####
class CEandKLLossesConfig(BaseModel):
    classname: Literal["CEandKLLosses"] = "CEandKLLosses"
    rounding_threshold: float


class CIHistogramsConfig(BaseModel):
    classname: Literal["CIHistograms"] = "CIHistograms"
    n_batches_accum: int | None


class CI_L0Config(BaseModel):
    classname: Literal["CI_L0"] = "CI_L0"
    groups: dict[str, list[str]] | None


class CIMeanPerComponentConfig(BaseModel):
    classname: Literal["CIMeanPerComponent"] = "CIMeanPerComponent"


class ComponentActivationDensityConfig(BaseModel):
    classname: Literal["ComponentActivationDensity"] = "ComponentActivationDensity"


class IdentityCIErrorConfig(BaseModel):
    classname: Literal["IdentityCIError"] = "IdentityCIError"
    identity_ci: list[dict[str, str | int]] | None
    dense_ci: list[dict[str, str | int]] | None


class PermutedCIPlotsConfig(BaseModel):
    classname: Literal["PermutedCIPlots"] = "PermutedCIPlots"
    sigmoid_type: SigmoidTypes
    identity_patterns: list[str] | None
    dense_patterns: list[str] | None


class StochasticReconSubsetCEAndKLConfig(BaseModel):
    classname: Literal["StochasticReconSubsetCEAndKL"] = "StochasticReconSubsetCEAndKL"
    include_patterns: dict[str, list[str]] | None
    exclude_patterns: dict[str, list[str]] | None


class StochasticHiddenActsReconConfig(BaseModel):
    classname: Literal["StochasticHiddenActsRecon"] = "StochasticHiddenActsRecon"


class UVPlotsConfig(BaseModel):
    classname: Literal["UVPlots"] = "UVPlots"
    identity_patterns: list[str] | None
    dense_patterns: list[str] | None


TrainMetricConfigType = (
    CIMaskedReconSubsetLossTrainConfig
    | CIMaskedReconLayerwiseLossTrainConfig
    | CIMaskedReconLossTrainConfig
    | FaithfulnessLossTrainConfig
    | ImportanceMinimalityLossTrainConfig
    | StochasticReconLayerwiseLossTrainConfig
    | StochasticReconLossTrainConfig
    | StochasticReconSubsetLossTrainConfig
)
EvalMetricConfigType = (
    CEandKLLossesConfig
    | CIHistogramsConfig
    | CI_L0Config
    | CIMeanPerComponentConfig
    | ComponentActivationDensityConfig
    | IdentityCIErrorConfig
    | PermutedCIPlotsConfig
    | UVPlotsConfig
    | StochasticReconSubsetCEAndKLConfig
    | StochasticHiddenActsReconConfig
)
MetricConfigType = TrainMetricConfigType | EvalMetricConfigType

TaskConfig = TMSTaskConfig | ResidMLPTaskConfig | LMTaskConfig | IHTaskConfig


class Config(BaseModel):
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
    n_mask_samples: PositiveInt = Field(
        ...,
        description="Number of stochastic masks to sample when using stochastic recon losses",
    )
    ci_fn_type: CiFnType = Field(
        default="vector_mlp",
        description="Type of causal importance function used to calculate the causal importance.",
    )
    ci_fn_hidden_dims: list[NonNegativeInt] = Field(
        default=[8],
        description="Hidden dimensions for the causal importance function used to calculate the causal importance",
    )
    sampling: Literal["continuous", "binomial"] = Field(
        default="continuous",
        description="Sampling mode for stochastic elements: 'continuous' (default) or 'binomial'",
    )
    sigmoid_type: Literal["normal", "hard", "leaky_hard", "upper_leaky_hard", "swish_hard"] = Field(
        default="leaky_hard",
        description="Type of sigmoid to use for causal importance calculation",
    )
    use_abs_inner_acts: bool = Field(
        default=False,
        description="If True, take the absolute value of inner activations before passing them to MLP CI functions",
    )
    target_module_patterns: list[str] = Field(
        ...,
        description="List of fnmatch-style patterns that select modules to decompose",
    )
    identity_module_patterns: list[str] | None = Field(
        default=None,
        description="List of fnmatch-style patterns that select modules in which an identity "
        "matrix should be inserted and decomposed beforehand",
    )

    @property
    def all_module_patterns(self):
        if self.identity_module_patterns is None:
            return self.target_module_patterns
        identity_final_patterns = [f"{p}.pre_identity" for p in self.identity_module_patterns]
        return self.target_module_patterns + identity_final_patterns

    use_delta_component: bool = Field(
        default=True,
        description="If True, use an extra component containing the difference between the target "
        "model and component weights. This allows for removing the faithfulness loss.",
    )

    loss_metric_configs: list[
        Annotated[TrainMetricConfigType, Field(discriminator="classname")]
    ] = Field(
        default=[],
        description=(
            "List of configs for loss metrics to compute (used for both training logs and eval); "
            "coefficients provided here are also used for weighting the training loss and eval loss/total."
        ),
    )
    output_loss_type: Literal["mse", "kl"] = Field(
        ...,
        description="Metric used to measure recon error between model outputs and targets",
    )

    # --- Training ---
    lr: PositiveFloat = Field(..., description="Learning rate for optimiser")
    steps: NonNegativeInt = Field(..., description="Total number of optimisation steps")
    batch_size: PositiveInt = Field(
        ...,
        description=(
            "The effective batch size used for optimisation. Depending on gradient accumulation "
            "steps, it may be processed as multiple micro-batches."
        ),
    )
    gradient_accumulation_steps: PositiveInt = Field(
        default=1,
        description="Number of steps to accumulate gradients over before updating parameters",
    )

    # --- Faithfulness Warmup ---
    faithfulness_warmup_steps: NonNegativeInt = Field(
        default=0,
        description="Number of warmup steps to optimize faithfulness loss before main training",
    )
    faithfulness_warmup_lr: PositiveFloat = Field(
        default=0.001,
        description="Learning rate for warmup phase (optimizing faithfulness loss only)",
    )
    faithfulness_warmup_weight_decay: NonNegativeFloat = Field(
        default=0.0,
        description="Weight decay for warmup phase optimizer",
    )

    @property
    def microbatch_size(self) -> PositiveInt:
        return self.batch_size // self.gradient_accumulation_steps

    lr_schedule: Literal["linear", "constant", "cosine", "exponential"] = Field(
        default="constant",
        description="Type of learning-rate schedule to apply",
    )
    lr_exponential_halflife: PositiveFloat | None = Field(
        default=None,
        description="Half-life parameter when using an exponential LR schedule",
    )
    lr_warmup_pct: Probability = Field(
        default=0.0,
        description="Fraction of total steps to linearly warm up the learning rate",
    )

    # --- Logging & Saving ---
    out_dir: Path | None = Field(
        default=None,
        description="Directory to save output to. If None, creates a dir using the wandb run id or "
        "randomly generates one",
    )
    train_log_freq: PositiveInt = Field(
        ...,
        description="Interval (in steps) at which to log training metrics",
    )
    eval_freq: PositiveInt = Field(
        ...,
        description="Interval (in steps) at which to log evaluation metrics",
    )
    eval_batch_size: PositiveInt = Field(
        ...,
        description="Batch size used for evaluation. If None, uses the same as `batch_size`.",
    )
    slow_eval_freq: PositiveInt = Field(
        ...,
        description="Interval (in steps) at which to run slow evaluation metrics. Must be a multiple of `eval_freq`.",
    )
    n_eval_steps: PositiveInt = Field(
        ...,
        description="Number of steps to run evaluation for",
    )
    slow_eval_on_first_step: bool = Field(
        default=True,
        description="Whether to run slow evaluation on the first step",
    )
    save_freq: PositiveInt | None = Field(
        default=None,
        description="Interval (in steps) at which to save model checkpoints (None disables saving "
        "until the end of training).",
    )
    eval_metric_configs: list[Annotated[EvalMetricConfigType, Field(discriminator="classname")]] = (
        Field(
            default=[],
            description="List of configs for metrics to use for evaluation",
        )
    )

    # --- Component Tracking ---
    ci_alive_threshold: Probability = Field(
        default=0.0,
        description="Causal importance threshold above which a component is considered 'firing'",
    )
    n_examples_until_dead: PositiveInt = Field(
        ...,
        description="Number of examples without firing before a component is considered dead. "
        "Note that in LMs, an example is a token, not a sequence.",
    )

    # --- Pretrained model info ---
    pretrained_model_class: str = Field(
        ...,
        description="Fully-qualified class name of the pretrained model to load. Can be defined "
        "locally or an in external package (e.g. 'transformers.LlamaForCausalLM' or "
        "'spd.experiments.resid_mlp.models.ResidMLP').",
    )
    pretrained_model_path: ModelPath | None = Field(
        default=None,
        description="Model identifier. Local path or wandb reference "
        "(e.g. 'wandb:goodfire/spd/runs/otxwx80v' or 'mnt/my_model/checkpoint.pth')",
    )
    pretrained_model_name: str | None = Field(
        default=None,
        description="hf model identifier. E.g. 'SimpleStories/SimpleStories-1.25M'",
    )
    pretrained_model_output_attr: str | None = Field(
        default=None,
        description="Name of the attribute on the forward output that contains logits or activations",
    )
    tokenizer_name: str | None = Field(
        default=None,
        description="Name or path of the tokenizer to use when loading an LM",
    )

    # --- Task Specific ---
    task_config: TaskConfig = Field(
        ...,
        discriminator="task_name",
        description="Nested task-specific configuration selected by the `task_name` discriminator",
    )

    DEPRECATED_CONFIG_KEYS: ClassVar[list[str]] = [
        "image_on_first_step",
        "image_freq",
        "metrics_fns",
        "figures_fns",
        "schatten_coeff",
        "embedding_recon_coeff",
        "is_embed_unembed_recon",
        "out_recon_coeff",
        "faithfulness_coeff",
        "stochastic_recon_coeff",
        "stochastic_recon_layerwise_coeff",
        "recon_coeff",
        "recon_layerwise_coeff",
        "ci_recon_coeff",
        "ci_recon_layerwise_coeff",
        "pnorm",
        "p_anneal_start_frac",
        "p_anneal_final_p",
        "p_anneal_end_frac",
        "importance_minimality_coeff",
        "dist_backend",
    ]
    RENAMED_CONFIG_KEYS: ClassVar[dict[str, str]] = {
        "print_freq": "eval_freq",
        "pretrained_model_name_hf": "pretrained_model_name",
        "recon_coeff": "ci_recon_coeff",
        "recon_layerwise_coeff": "ci_recon_layerwise_coeff",
        "gate_type": "ci_fn_type",
        "gate_hidden_dims": "ci_fn_hidden_dims",
    }

    @model_validator(mode="before")
    def handle_deprecated_config_keys(cls, config_dict: dict[str, Any]) -> dict[str, Any]:
        """Remove deprecated config keys and change names of any keys that have been renamed."""

        # We don't bother mapping the old ``eval_metrics`` to the new ``eval_metric_configs``.
        config_dict.pop("eval_metrics", None)

        for key in list(config_dict.keys()):
            val = config_dict[key]
            if key in cls.DEPRECATED_CONFIG_KEYS:
                logger.warning(f"{key} is deprecated, but has value: {val}. Removing from config.")
                del config_dict[key]

            elif key in cls.RENAMED_CONFIG_KEYS:
                logger.info(f"Renaming {key} to {cls.RENAMED_CONFIG_KEYS[key]}")
                config_dict[cls.RENAMED_CONFIG_KEYS[key]] = val
                del config_dict[key]

        if "eval_batch_size" not in config_dict:
            config_dict["eval_batch_size"] = config_dict["batch_size"]
        if "train_log_freq" not in config_dict:
            config_dict["train_log_freq"] = 50
        if "slow_eval_freq" not in config_dict:
            config_dict["slow_eval_freq"] = config_dict["eval_freq"]
        return config_dict

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        # Check that lr_exponential_halflife is not None if lr_schedule is "exponential"
        if self.lr_schedule == "exponential":
            assert self.lr_exponential_halflife is not None, (
                "lr_exponential_halflife must be set if lr_schedule is exponential"
            )

        assert self.batch_size % self.gradient_accumulation_steps == 0, (
            "batch_size must be divisible by gradient_accumulation_steps"
        )

        assert self.slow_eval_freq % self.eval_freq == 0, (
            "slow_eval_freq must be a multiple of eval_freq"
        )
        assert self.slow_eval_freq // self.eval_freq >= 1, (
            "slow_eval_freq must be at least eval_freq"
        )

        return self
