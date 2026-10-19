"""TMS (Toy Model of Superposition) experiment config schema — torch-free.

The JAX trainer reads this directly (`param_decomp.core.built_run`), the same way it reads
`LMExperimentConfig`. Unlike the LM target there is no HuggingFace/pretrain-cache weight
source: the tiny TMS target is pretrained from scratch, deterministically from
`target.pretrain` (the original Anthropic `mean((|x| - relu_out)^2)` objective on the
synthetic sparse-feature data), so a run is fully reproducible from the config alone.
"""

from typing import Literal

from param_decomp_formats.config import BaseConfig, Probability
from pydantic import NonNegativeInt, PositiveInt

from param_decomp.experiments.toy_config import (
    ToyDecompositionConfig,
    ToyExperimentConfig,
)
from param_decomp.targets.tms import TMSGenerationType

TMSDataGenerationType = TMSGenerationType
"""The schema spells the target's own generation vocabulary — one Literal, defined next
to the count table it enumerates (`targets/tms.py`)."""

TMSHiddenLayerInit = Literal["identity", "random"]


class TMSPretrainConfig(BaseConfig):
    """How the frozen TMS target is pretrained from scratch (Anthropic TMS objective)."""

    steps: PositiveInt
    batch_size: PositiveInt
    lr: float
    seed: int = 0


class TMSTargetConfig(BaseConfig):
    """The TMS target architecture + its from-scratch pretraining.

    The target has tied weights (`linear2 = linear1ᵀ`); the *decomposition* is untied
    (`decomposition.sites` lists `linear1` and `linear2` — each site gets its own
    components).

    `n_hidden_layers > 0` inserts that many FROZEN `n_hidden -> n_hidden` layers between
    `linear1` and `linear2` (`hidden_layer_init` = `identity` for the `-id` variant or
    `random` for frozen-random), each an extra dense decomposition site `hidden_layers.{i}`.
    `init_bias_to_zero` zeroes `linear2`'s bias at init (else `nn.Linear`'s default uniform
    bias)."""

    n_features: PositiveInt
    n_hidden: PositiveInt
    n_hidden_layers: NonNegativeInt = 0
    hidden_layer_init: TMSHiddenLayerInit = "identity"
    init_bias_to_zero: bool = True
    pretrain: TMSPretrainConfig


class TMSDataConfig(BaseConfig):
    """Synthetic sparse-feature data for the PD decomposition step."""

    feature_probability: Probability
    data_generation_type: TMSDataGenerationType = "at_least_zero_active"


class TMSExperimentConfig(ToyExperimentConfig):
    target: TMSTargetConfig
    decomposition: ToyDecompositionConfig
    data: TMSDataConfig
