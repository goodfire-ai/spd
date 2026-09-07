"""Authored metric schemas whose semantics require an LM target."""

from typing import ClassVar, Literal

from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt

from param_decomp.core.base_config import BaseConfig, Probability
from param_decomp.core.configs import HiddenActsReconstruction


class CEandKLLossesConfig(BaseConfig):
    """Token-level CE/KL metrics for categorical LM outputs."""

    slow: ClassVar[bool] = False
    type: Literal["CEandKLLosses"] = "CEandKLLosses"
    rounding_threshold: Probability


class CIMaskedAttnPatternsReconLossConfig(BaseConfig):
    slow: ClassVar[bool] = False
    type: Literal["CIMaskedAttnPatternsReconLoss"] = "CIMaskedAttnPatternsReconLoss"


class StochasticAttnPatternsReconLossConfig(BaseConfig):
    slow: ClassVar[bool] = False
    type: Literal["StochasticAttnPatternsReconLoss"] = "StochasticAttnPatternsReconLoss"
    n_mask_samples: PositiveInt = 1


class WellTemperednessConfig(BaseConfig):
    """Whether higher causal importance preactivations mean greater ablation effects.

    Components are ablated one at a time at sampled token positions of an LM. `groups`
    maps names to fnmatch-style site patterns. Every region always schedules
    `n_locations * n_components_per_region` solo ablations: a sparse region pads its quota
    with out-of-region components whose damage is computed and discarded.
    """

    slow: ClassVar[bool] = True
    type: Literal["WellTemperedness"] = "WellTemperedness"
    groups: dict[str, list[str]] | None
    n_locations: PositiveInt
    n_components_per_region: PositiveInt
    ablations_per_forward: PositiveInt


class ArithmeticCEKLConfig(BaseConfig):
    rounding_threshold: Probability


class ArithmeticCIL0Config(BaseConfig):
    ci_alive_threshold: Probability
    groups: dict[str, list[str]] | None


class ArithmeticFreshPGDConfig(BaseConfig):
    name: str | None = None
    n_steps: NonNegativeInt
    step_size: PositiveFloat
    hidden_acts_reconstruction: HiddenActsReconstruction | None = None


class ArithmeticProbeMetrics(BaseConfig):
    """Scalar operations evaluated on the arithmetic grid rather than corpus batches."""

    ce_kl: ArithmeticCEKLConfig
    ci_l0: ArithmeticCIL0Config
    fresh_pgd: ArithmeticFreshPGDConfig | None


class ArithmeticCIGridConfig(BaseConfig):
    """Per-component causal-importance heatmaps over an arithmetic operand grid."""

    slow: ClassVar[bool] = True
    type: Literal["ArithmeticCIGrid"] = "ArithmeticCIGrid"
    probe_metrics: ArithmeticProbeMetrics
    operation: Literal["add", "sub", "mul"] = "add"
    a_range: tuple[int, int] = (1, 100)
    b_range: tuple[int, int] = (1, 100)
    thresholds: list[Probability] = Field(default_factory=lambda: [0.1])
    top_k: PositiveInt = 24
