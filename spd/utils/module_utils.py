import fnmatch
import math
from typing import Literal

import torch
import torch.nn as nn
from simple_stories_train.models.gpt2_simple import LayerNorm as SSLayerNorm
from torch import Tensor
from torch.nn.init import calculate_gain

# This is equivalent to `torch.nn.init._NonlinearityType`, but for some reason this is not always
# importable. see https://github.com/goodfire-ai/spd/actions/runs/16927877557/job/47967138342
_NonlinearityType = Literal[
    "linear",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "sigmoid",
    "tanh",
    "relu",
    "leaky_relu",
    "selu",
]


def init_param_(
    param: Tensor,
    fan_val: float,
    mean: float = 0.0,
    nonlinearity: _NonlinearityType = "linear",
    generator: torch.Generator | None = None,
) -> None:
    """Fill in param with values sampled from a Kaiming normal distribution.

    Args:
        param: The parameter to initialize
        fan_val: The squared denominator of the std used for the kaiming normal distribution
        mean: The mean of the normal distribution
        nonlinearity: The nonlinearity of the activation function
        generator: The generator to sample from
    """
    gain: float = calculate_gain(nonlinearity)
    std: float = gain / math.sqrt(fan_val)
    with torch.no_grad():
        param.normal_(mean, std, generator=generator)


def replace_std_values_in_layernorm(
    component_model: nn.Module, std_values: dict[str, float]
) -> None:
    for name, std in std_values.items():
        module = component_model.get_submodule("patched_model." + name)
        assert isinstance(module, SSLayerNorm), (
            f"Expected {name} to be a simple_stories_train LayerNorm instance, got {type(module)}"
        )
        module.std = std


def get_target_module_paths(model: nn.Module, target_module_patterns: list[str]) -> list[str]:
    """Find the target_module_patterns that match real modules in the target model.

    e.g. `["layers.*.mlp_in"]` ->  `["layers.1.mlp_in", "layers.2.mlp_in"]`.
    """

    names_out: list[str] = []
    matched_patterns: set[str] = set()
    for name, _ in model.named_modules():
        for pattern in target_module_patterns:
            if fnmatch.fnmatch(name, pattern):
                matched_patterns.add(pattern)
                names_out.append(name)

    unmatched_patterns = set(target_module_patterns) - matched_patterns
    if unmatched_patterns:
        raise ValueError(
            f"The following patterns in target_module_patterns did not match any modules: "
            f"{sorted(unmatched_patterns)}"
        )

    return names_out
