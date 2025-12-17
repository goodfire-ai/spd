from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
from simple_stories_train.models.gpt2_simple import LayerNorm as SSLayerNorm
from torch import Tensor
from torch.nn.init import calculate_gain


@dataclass
class ModulePathInfo:
    """Expanded module path with its number of components.

    Created by expanding ModulePatternInfo patterns against actual module names
    in the target model. Used internally after pattern expansion.
    """

    module_path: str
    C: int


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


def expand_module_patterns(model: nn.Module, module_info: list[Any]) -> list[ModulePathInfo]:
    """Expand module patterns to concrete module paths with their C values.

    For modules matching multiple patterns, the most specific pattern wins
    (pattern with fewest wildcards). Equal specificity is an error.

    Args:
        model: The target model
        module_info: List of ModulePatternInfo specifying patterns and C values

    Returns:
        List of ModulePathInfo with expanded concrete module paths

    Raises:
        ValueError: If any pattern doesn't match any modules, or if two patterns
            with equal specificity match the same module

    Example:
        More specific pattern (fewer wildcards) wins over less specific:

        >>> module_info = [
        ...     ModulePatternInfo(module_pattern="h.*.mlp.*", C=100),
        ...     ModulePatternInfo(module_pattern="h.*.mlp.c_fc", C=50),
        ... ]
        >>> expand_module_patterns(model, module_info)
        [ModulePathInfo(module_path='h.0.mlp.c_fc', C=50), ...]

        Here h.0.mlp.c_fc gets C=50 (1 wildcard) instead of C=100 (2 wildcards).
    """
    # module -> (wildcard_count, pattern, C)
    module_to_info: dict[str, tuple[int, str, int]] = {}

    for info in module_info:
        pattern = info.module_pattern
        c = info.C
        wildcard_count = pattern.count("*")
        matched_any = False

        for name, _ in model.named_modules():
            if fnmatch.fnmatch(name, pattern):
                matched_any = True

                if name not in module_to_info:
                    module_to_info[name] = (wildcard_count, pattern, c)
                else:
                    current_wc, current_pattern, _ = module_to_info[name]
                    if wildcard_count == current_wc:
                        raise ValueError(
                            f"Module '{name}' matches patterns '{current_pattern}' and '{pattern}' "
                            f"with equal specificity ({wildcard_count} wildcards). "
                            "Use more specific patterns to resolve the conflict."
                        )
                    # More specific (fewer wildcards) wins
                    if wildcard_count < current_wc:
                        module_to_info[name] = (wildcard_count, pattern, c)

        if not matched_any:
            raise ValueError(f"Pattern '{pattern}' in module_info did not match any modules")

    return [ModulePathInfo(module_path=name, C=c) for name, (_, _, c) in module_to_info.items()]
