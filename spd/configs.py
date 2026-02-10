"""Config classes of various types"""

from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import (
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from spd.base_config import BaseConfig
from spd.log import logger
from spd.spd_types import CiFnType, ModelPath, Probability


class ScheduleConfig(BaseConfig):
    """Configuration for a schedule with warmup and decay. Can be used for LR or other values."""

    start_val: PositiveFloat = Field(..., description="Starting/peak value (after warmup)")
    warmup_pct: Probability = Field(
        default=0.0, description="Fraction of total steps for linear warmup"
    )
    final_val_frac: NonNegativeFloat = Field(
        default=1.0,
        description="End value as fraction of start_val. Can be <1 (decay), =1 (no decay), or >1 (increase)",
    )
    fn_type: Literal["constant", "cosine", "linear"] = Field(
        default="constant", description="Decay function type after warmup"
    )

    @model_validator(mode="after")
    def validate_constant_schedule(self) -> Self:
        if self.fn_type == "constant" and self.final_val_frac != 1.0:
            raise ValueError("constant schedule requires final_val_frac == 1.0")
        return self


def migrate_to_lr_schedule_config(config_dict: dict[str, Any]) -> None:
    """Migrate old LR config format (lr + lr_schedule + lr_warmup_pct) to ScheduleConfig.

    Modifies config_dict in place.
    """
    if "lr" not in config_dict:
        return

    logger.info("Migrating old LR config format to ScheduleConfig")

    old_lr = config_dict.pop("lr")
    old_fn_type = config_dict.pop("lr_schedule", "constant")
    old_warmup_pct = config_dict.pop("lr_warmup_pct", 0.0)

    # Old cosine decayed to 0, old constant stayed at 1
    final_val_frac = 0.0 if old_fn_type == "cosine" else 1.0

    config_dict["lr_schedule"] = {
        "start_val": old_lr,
        "fn_type": old_fn_type,
        "warmup_pct": old_warmup_pct,
        "final_val_frac": final_val_frac,
    }


# Task configs - these define task-specific parameters for SPD decomposition
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


class ResidMLPTaskConfig(BaseConfig):
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


class IHTaskConfig(BaseConfig):
    task_name: Literal["ih"]
    prefix_window: PositiveInt | None = Field(
        default=None,
        description="Number of tokens to use as a prefix window for the induction head. If none, uses the full sequence length.",
    )


class LMTaskConfig(BaseConfig):
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
    shuffle_each_epoch: bool = Field(
        default=True,
        description="Whether to reshuffle data at each epoch. Set False in tests to keep fixed "
        "order across dp modes.",
    )
    is_tokenized: bool = Field(
        default=False,
        description="Whether the dataset is already tokenized",
    )
    streaming: bool = Field(
        default=False,
        description="Whether to use a streaming dataset",
    )


class ModulePatternInfoConfig(BaseConfig):
    """Configuration for a module pattern with its number of components.

    Used in config files to specify which modules to decompose and how many
    components (C) to use for each module matching the pattern.
    """

    module_pattern: str = Field(..., description="fnmatch-style pattern to match module names")
    C: PositiveInt = Field(
        ..., description="Number of components for modules matching this pattern"
    )


#### Metrics that can be used as losses in training or eval ####
class LossMetricConfig(BaseConfig):
    coeff: float | None = Field(
        default=None,
        description="Loss coefficient. Used when metric is in loss_metric_configs.",
    )


class FaithfulnessLossConfig(LossMetricConfig):
    classname: Literal["FaithfulnessLoss"] = "FaithfulnessLoss"


class ImportanceMinimalityLossConfig(LossMetricConfig):
    classname: Literal["ImportanceMinimalityLoss"] = "ImportanceMinimalityLoss"
    pnorm: NonNegativeFloat
    beta: NonNegativeFloat
    p_anneal_start_frac: Probability = 1.0
    p_anneal_final_p: NonNegativeFloat | None = None
    p_anneal_end_frac: Probability = 1.0
    eps: NonNegativeFloat = 1e-12

    @model_validator(mode="before")
    @classmethod
    def default_beta(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "beta" not in data:
            logger.warning("beta not in ImportanceMinimalityLossConfig, defaulting to 0.0")
            data["beta"] = 0.0
        return data


class UniformKSubsetRoutingConfig(BaseConfig):
    type: Literal["uniform_k_subset"] = "uniform_k_subset"


class StaticProbabilityRoutingConfig(BaseConfig):
    type: Literal["static_probability"] = "static_probability"
    p: Probability


SubsetRoutingType = UniformKSubsetRoutingConfig | StaticProbabilityRoutingConfig


class CIMaskedReconSubsetLossConfig(LossMetricConfig):
    classname: Literal["CIMaskedReconSubsetLoss"] = "CIMaskedReconSubsetLoss"
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]


class CIMaskedReconLayerwiseLossConfig(LossMetricConfig):
    classname: Literal["CIMaskedReconLayerwiseLoss"] = "CIMaskedReconLayerwiseLoss"


class CIMaskedReconLossConfig(LossMetricConfig):
    classname: Literal["CIMaskedReconLoss"] = "CIMaskedReconLoss"


class StochasticReconLossConfig(LossMetricConfig):
    classname: Literal["StochasticReconLoss"] = "StochasticReconLoss"


class StochasticReconSubsetLossConfig(LossMetricConfig):
    classname: Literal["StochasticReconSubsetLoss"] = "StochasticReconSubsetLoss"
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]


class StochasticReconLayerwiseLossConfig(LossMetricConfig):
    classname: Literal["StochasticReconLayerwiseLoss"] = "StochasticReconLayerwiseLoss"


class UnmaskedReconLossConfig(LossMetricConfig):
    classname: Literal["UnmaskedReconLoss"] = "UnmaskedReconLoss"


PGDInitStrategy = Literal["random", "ones", "zeroes"]

MaskScope = Literal["unique_per_datapoint", "shared_across_batch"]


class PGDConfig(LossMetricConfig):
    init: PGDInitStrategy
    step_size: float
    n_steps: int
    mask_scope: MaskScope


class PGDReconLossConfig(PGDConfig):
    classname: Literal["PGDReconLoss"] = "PGDReconLoss"


class PGDReconSubsetLossConfig(PGDConfig):
    classname: Literal["PGDReconSubsetLoss"] = "PGDReconSubsetLoss"
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]


class PGDReconLayerwiseLossConfig(PGDConfig):
    classname: Literal["PGDReconLayerwiseLoss"] = "PGDReconLayerwiseLoss"


class PGDMultiBatchConfig(LossMetricConfig):
    init: PGDInitStrategy
    step_size: float
    n_steps: int
    gradient_accumulation_steps: int


class PGDMultiBatchReconLossConfig(PGDMultiBatchConfig):
    classname: Literal["PGDMultiBatchReconLoss"] = "PGDMultiBatchReconLoss"


class PGDMultiBatchReconSubsetLossConfig(PGDMultiBatchConfig):
    classname: Literal["PGDMultiBatchReconSubsetLoss"] = "PGDMultiBatchReconSubsetLoss"
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]


class StochasticHiddenActsReconLossConfig(LossMetricConfig):
    classname: Literal["StochasticHiddenActsReconLoss"] = "StochasticHiddenActsReconLoss"


#### Metrics that can only be used in eval ####
class CEandKLLossesConfig(BaseConfig):
    classname: Literal["CEandKLLosses"] = "CEandKLLosses"
    rounding_threshold: float


class CIHistogramsConfig(BaseConfig):
    classname: Literal["CIHistograms"] = "CIHistograms"
    n_batches_accum: int | None


class CI_L0Config(BaseConfig):
    classname: Literal["CI_L0"] = "CI_L0"
    groups: dict[str, list[str]] | None


class CIMeanPerComponentConfig(BaseConfig):
    classname: Literal["CIMeanPerComponent"] = "CIMeanPerComponent"


class ComponentActivationDensityConfig(BaseConfig):
    classname: Literal["ComponentActivationDensity"] = "ComponentActivationDensity"


class IdentityCIErrorConfig(BaseConfig):
    classname: Literal["IdentityCIError"] = "IdentityCIError"
    identity_ci: list[dict[str, str | int]] | None
    dense_ci: list[dict[str, str | int]] | None


class PermutedCIPlotsConfig(BaseConfig):
    classname: Literal["PermutedCIPlots"] = "PermutedCIPlots"
    identity_patterns: list[str] | None
    dense_patterns: list[str] | None

    @model_validator(mode="before")
    def handle_deprecated_config_keys(cls, config_dict: dict[str, Any]) -> dict[str, Any]:
        """Remove deprecated config keys and change names of any keys that have been renamed."""
        config_dict.pop("sigmoid_type", None)
        return config_dict


class StochasticReconSubsetCEAndKLConfig(BaseConfig):
    classname: Literal["StochasticReconSubsetCEAndKL"] = "StochasticReconSubsetCEAndKL"
    include_patterns: dict[str, list[str]] | None
    exclude_patterns: dict[str, list[str]] | None


class UVPlotsConfig(BaseConfig):
    classname: Literal["UVPlots"] = "UVPlots"
    identity_patterns: list[str] | None
    dense_patterns: list[str] | None


ReconLossConfigType = (
    UnmaskedReconLossConfig
    | CIMaskedReconLossConfig
    | CIMaskedReconSubsetLossConfig
    | CIMaskedReconLayerwiseLossConfig
    | StochasticReconLossConfig
    | StochasticReconSubsetLossConfig
    | StochasticReconLayerwiseLossConfig
    | PGDReconLossConfig
    | PGDReconSubsetLossConfig
    | PGDReconLayerwiseLossConfig
    | StochasticHiddenActsReconLossConfig
)

LossMetricConfigType = FaithfulnessLossConfig | ImportanceMinimalityLossConfig | ReconLossConfigType

EvalOnlyMetricConfigType = (
    CEandKLLossesConfig
    | CIHistogramsConfig
    | CI_L0Config
    | CIMeanPerComponentConfig
    | ComponentActivationDensityConfig
    | IdentityCIErrorConfig
    | PermutedCIPlotsConfig
    | UVPlotsConfig
    | StochasticReconSubsetCEAndKLConfig
    | PGDMultiBatchReconLossConfig
    | PGDMultiBatchReconSubsetLossConfig
)
MetricConfigType = LossMetricConfigType | EvalOnlyMetricConfigType

TaskConfig = TMSTaskConfig | ResidMLPTaskConfig | LMTaskConfig | IHTaskConfig

SamplingType = Literal["continuous", "binomial"]


class Config(BaseConfig):
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
    autocast_bf16: bool = Field(
        default=True,
        description="Whether to use torch.autocast with bfloat16 mixed precision",
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
    sampling: SamplingType = Field(
        default="continuous",
        description="Sampling mode for stochastic elements: 'continuous' (default) or 'binomial'",
    )
    sigmoid_type: Literal["normal", "hard", "leaky_hard", "upper_leaky_hard", "swish_hard"] = Field(
        default="leaky_hard",
        description="Type of sigmoid to use for causal importance calculation",
    )
    module_info: list[ModulePatternInfoConfig] = Field(
        ...,
        description="List of module patterns with C values specifying which modules to decompose. "
        "Example: [{module_pattern: 'h.*.mlp.c_fc', C: 10}, {module_pattern: 'h.*.attn.*', C: 20}]",
    )
    identity_module_info: list[ModulePatternInfoConfig] | None = Field(
        default=None,
        description="List of identity module patterns with C values. "
        "Identity operations will be inserted at these modules.",
    )

    @property
    def all_module_info(self) -> list[ModulePatternInfoConfig]:
        """Combine target and identity patterns with their C values.

        Returns list of ModulePatternInfoConfig with .pre_identity suffix added to identity patterns.
        """
        result = list(self.module_info)

        if self.identity_module_info is not None:
            for info in self.identity_module_info:
                result.append(
                    ModulePatternInfoConfig(
                        module_pattern=f"{info.module_pattern}.pre_identity", C=info.C
                    )
                )

        return result

    use_delta_component: bool = Field(
        default=True,
        description="If True, use an extra component containing the difference between the target "
        "model and component weights. This allows for removing the faithfulness loss.",
    )

    loss_metric_configs: list[Annotated[LossMetricConfigType, Field(discriminator="classname")]] = (
        Field(
            default=[],
            description=(
                "List of configs for loss metrics to compute (used for both training logs and eval); "
                "coefficients provided here are also used for weighting the training loss and eval loss/total."
            ),
        )
    )
    # --- Training ---
    lr_schedule: ScheduleConfig = Field(..., description="Learning rate schedule configuration")
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
    grad_clip_norm_components: PositiveFloat | None = Field(
        default=None,
        description="If set, apply grad norm clipping to the parameters of the components",
    )
    grad_clip_norm_ci_fns: PositiveFloat | None = Field(
        default=None,
        description="If set, apply grad norm clipping to the parameters of the CI functions",
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

    # --- Logging & Saving ---
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
    eval_metric_configs: list[Annotated[MetricConfigType, Field(discriminator="classname")]] = (
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
    output_extract: Literal["first_element", "logits_attr"] | None = Field(
        default=None,
        description="How to extract tensor from model output. "
        "None = raw output, 'first_element' = output[0], 'logits_attr' = output.logits",
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
        "lr_exponential_halflife",
        "out_dir",
        "n_examples_until_dead",
        "output_loss_type",
    ]
    RENAMED_CONFIG_KEYS: ClassVar[dict[str, str]] = {
        "grad_clip_norm": "grad_clip_norm_components",
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

        cls._migrate_to_module_info(config_dict)
        migrate_to_lr_schedule_config(config_dict)

        for key in list(config_dict.keys()):
            val = config_dict[key]
            if key in cls.DEPRECATED_CONFIG_KEYS:
                logger.warning(f"{key} is deprecated, but has value: {val}. Removing from config.")
                del config_dict[key]

            elif key in cls.RENAMED_CONFIG_KEYS:
                logger.info(f"Renaming {key} to {cls.RENAMED_CONFIG_KEYS[key]}")
                config_dict[cls.RENAMED_CONFIG_KEYS[key]] = val
                del config_dict[key]

            elif key in ("loss_metric_configs", "eval_metric_configs"):
                # We used to have an extra_init_kwargs field. This is hard to map. Just remove all
                # configs with it
                new_vals = [cfg for cfg in val if "extra_init_kwargs" not in cfg]
                config_dict[key] = new_vals

        # Remap simple_stories_train → spd.pretrain (models moved in-tree)
        pmc = config_dict.get("pretrained_model_class", "")
        if pmc.startswith("simple_stories_train.models."):
            config_dict["pretrained_model_class"] = pmc.replace(
                "simple_stories_train.models.", "spd.pretrain.models.", 1
            )

        # Migrate old pretrained_model_output_attr to output_extract
        if "pretrained_model_output_attr" in config_dict:
            old_val = config_dict.pop("pretrained_model_output_attr")
            match old_val:
                case None:
                    pass
                case "idx_0":
                    config_dict["output_extract"] = "first_element"
                case "logits":
                    config_dict["output_extract"] = "logits_attr"
                case _:
                    raise ValueError(f"Unknown pretrained_model_output_attr: {old_val!r}")

        # Migrate extract_tensor_output to output_extract
        if "extract_tensor_output" in config_dict:
            old_val = config_dict.pop("extract_tensor_output")
            match old_val:
                case None:
                    pass
                case "[0]":
                    config_dict["output_extract"] = "first_element"
                case ".logits":
                    config_dict["output_extract"] = "logits_attr"
                case _:
                    raise ValueError(f"Unknown extract_tensor_output: {old_val!r}")

        if "eval_batch_size" not in config_dict:
            config_dict["eval_batch_size"] = config_dict["batch_size"]
        if "train_log_freq" not in config_dict:
            config_dict["train_log_freq"] = 50
        if "slow_eval_freq" not in config_dict:
            config_dict["slow_eval_freq"] = config_dict["eval_freq"]
        return config_dict

    @classmethod
    def _migrate_to_module_info(cls, config_dict: dict[str, Any]) -> None:
        """Migrate old config format (C + target_module_patterns) to new module_info format."""
        cond = "C" in config_dict or "target_module_patterns" in config_dict
        if not cond:
            return

        logger.warning(
            "Found old config keys for C definition, mapping old structure to new module_info structure"
        )
        global_c = config_dict["C"]
        config_dict["module_info"] = [
            {"module_pattern": p, "C": global_c} for p in config_dict["target_module_patterns"]
        ]
        del config_dict["C"]
        del config_dict["target_module_patterns"]

        identity_patterns = config_dict.pop("identity_module_patterns", None)
        if identity_patterns is not None:
            config_dict["identity_module_info"] = [
                {"module_pattern": p, "C": global_c} for p in identity_patterns
            ]

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        assert self.batch_size % self.gradient_accumulation_steps == 0, (
            "batch_size must be divisible by gradient_accumulation_steps"
        )

        assert self.slow_eval_freq % self.eval_freq == 0, (
            "slow_eval_freq must be a multiple of eval_freq"
        )
        assert self.slow_eval_freq // self.eval_freq >= 1, (
            "slow_eval_freq must be at least eval_freq"
        )

        for cfg in self.loss_metric_configs:
            assert cfg.coeff is not None, "All loss_metric_configs must have a coeff"

        if any(
            isinstance(cfg, PGDConfig) and cfg.mask_scope == "shared_across_batch"
            for cfg in self.loss_metric_configs
        ):
            assert self.gradient_accumulation_steps == 1, (
                "gradient_accumulation_steps must be 1 if we are using PGD losses with "
                "mask_scope='shared_across_batch'"
            )

        return self
