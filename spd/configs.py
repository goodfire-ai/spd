"""Config classes of various types"""

import importlib
import inspect
from pathlib import Path
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

from spd.experiments.ih.configs import IHTaskConfig
from spd.experiments.lm.configs import LMTaskConfig
from spd.experiments.resid_mlp.configs import ResidMLPTaskConfig
from spd.experiments.tms.configs import TMSTaskConfig
from spd.log import logger
from spd.models.components import GateType
from spd.spd_types import ModelPath, Probability


class EvalMetricConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
    classname: str = Field(
        ...,
        description="Name of the class to instantiate",
    )
    extra_init_kwargs: dict[str, Any] = Field(
        default={},
        description="Extra keyword arguments to pass to the class constructor besides `model: ComponentModel` and `config: Config`",
    )

    def _get_metric_class(self) -> type | None:
        available_classes = importlib.import_module("spd.eval").EVAL_CLASSES
        cls = available_classes.get(self.classname)
        if cls is None:
            logger.warning(
                f"Metric class {self.classname!r} not found. Available classes: {available_classes.keys()}"
            )
        return cls

    @model_validator(mode="after")
    def validate_class_kwargs(self) -> Self:
        """Check that the classname and kwargs are valid.

        If the classname is not found, we warn instead of raising an error. This allows us to
        load checkpoints from a run where a user might have added custom metrics.
        """
        cls = self._get_metric_class()
        if cls is None:
            return self
        sig = inspect.signature(cls.__init__)
        # Skip 'self' plus the first two actual parameters (model: ComponentModel, config: Config)
        params_after_required = list(sig.parameters.values())[3:]
        sig_extra_only = inspect.Signature(params_after_required)

        # Check that kwargs are valid
        try:
            sig_extra_only.bind(**self.extra_init_kwargs)
        except TypeError as e:
            # Raise a warning instead of an error
            # e.g. "unexpected parameter 'foo'" or "missing a required argument: 'bar'"
            logger.warning(f"Invalid kwargs for {self.classname!r}: {e}")

        return self


TaskConfig = TMSTaskConfig | ResidMLPTaskConfig | LMTaskConfig | IHTaskConfig


class Config(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
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
    gate_type: GateType = Field(
        default="vector_mlp",
        description="Type of gate used to calculate the causal importance.",
    )
    gate_hidden_dims: list[NonNegativeInt] = Field(
        default=[8],
        description="Hidden dimensions for the gate used to calculate the causal importance",
    )
    sigmoid_type: Literal["normal", "hard", "leaky_hard", "upper_leaky_hard", "swish_hard"] = Field(
        default="leaky_hard",
        description="Type of sigmoid to use for causal importance calculation",
    )
    target_module_patterns: list[str] = Field(
        ...,
        description="List of fnmatch-style patterns that select modules to decompose",
    )

    # --- Loss Coefficients
    faithfulness_coeff: NonNegativeFloat | None = Field(
        default=1.0,
        description="Coefficient for matching parameters between components and target weights",
    )
    recon_coeff: NonNegativeFloat | None = Field(
        default=None,
        description="Coefficient for recon loss with a causal importance mask",
    )
    stochastic_recon_coeff: NonNegativeFloat | None = Field(
        default=None,
        description="Coefficient for recon loss with stochastically sampled masks",
    )
    recon_layerwise_coeff: NonNegativeFloat | None = Field(
        default=None,
        description="Coefficient for per-layer recon loss with a causal importance mask",
    )
    stochastic_recon_layerwise_coeff: NonNegativeFloat | None = Field(
        default=None,
        description="Coefficient for per-layer recon loss with stochastically sampled masks",
    )
    importance_minimality_coeff: NonNegativeFloat = Field(
        ...,
        description="Coefficient for importance minimality loss",
    )
    schatten_coeff: NonNegativeFloat | None = Field(
        default=None,
        description="Coefficient for Schatten-norm regularisation (LM only)",
    )
    out_recon_coeff: NonNegativeFloat | None = Field(
        default=None,
        description="Coefficient for output recon loss",
    )
    embedding_recon_coeff: float | None = Field(
        default=None,
        description="Coefficient for additional embedding recon loss (LM only)",
    )
    is_embed_unembed_recon: bool = Field(
        default=False,
        description="If True, apply embedding recon jointly to embed & unembed matrices",
    )
    pnorm: PositiveFloat = Field(
        ...,
        description="The p-value used for the importance minimality loss",
    )
    p_anneal_start_frac: Probability = Field(
        default=1.0,
        description="Fraction of training after which to start annealing p (1.0 = no annealing)",
    )
    p_anneal_final_p: PositiveFloat | None = Field(
        default=None,
        description="Final p value to anneal to (None = no annealing)",
    )
    p_anneal_end_frac: Probability = Field(
        default=1.0,
        description="Fraction of training when annealing ends. We stay at the final p value from "
        "this point onward (default 1.0 = anneal until end)",
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
        description="Batch size used for evaluation",
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
    eval_metrics: list[EvalMetricConfig] = Field(
        default=[],
        description="List of metrics to use for evaluation",
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
    pretrained_model_name_hf: str | None = Field(
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

    # --- Distributed ---
    dist_backend: Literal["nccl", "gloo"] | None = Field(
        default=None,
        description="Backend for distributed training (nccl for GPU, gloo for CPU). If None, "
        "uses the default backend for the current device.",
    )

    DEPRECATED_CONFIG_KEYS: ClassVar[list[str]] = [
        "image_on_first_step",
        "image_freq",
        "metrics_fns",
        "figures_fns",
    ]
    RENAMED_CONFIG_KEYS: ClassVar[dict[str, str]] = {"print_freq": "eval_freq"}

    @model_validator(mode="before")
    def handle_deprecated_config_keys(cls, config_dict: dict[str, Any]) -> dict[str, Any]:
        """Remove deprecated config keys and change names of any keys that have been renamed."""
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
        # If any of the coeffs are 0, raise a warning
        msg = "is 0, you may wish to instead set it to null to avoid calculating the loss"
        if self.recon_coeff == 0:
            logger.warning(f"recon_coeff {msg}")
        if self.importance_minimality_coeff == 0:
            logger.warning(f"importance_minimality_coeff {msg}")
        if self.faithfulness_coeff == 0:
            logger.warning(f"faithfulness_coeff {msg}")

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
