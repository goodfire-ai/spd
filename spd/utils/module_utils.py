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


def get_target_module_paths_with_c(
    model: nn.Module, target_module_patterns: list[tuple[str, int]]
) -> dict[str, int]:
    """Find modules matching patterns and return mapping of module_path -> C value.

    For modules matching multiple patterns, the most specific pattern wins
    (pattern with fewest wildcards). Equal specificity is an error.

    Args:
        model: The target model
        target_module_patterns: List of (pattern, C_value) tuples

    Returns:
        Dictionary mapping module paths to their C values

    Raises:
        ValueError: If any pattern doesn't match any modules, or if two patterns
            with equal specificity match the same module

    Example:
        More specific pattern (fewer wildcards) wins over less specific:

        >>> patterns = [("h.*.mlp.*", 100), ("h.*.mlp.c_fc", 50)]
        >>> get_target_module_paths_with_c(model, patterns)
        {'h.0.mlp.c_fc': 50, 'h.0.mlp.down_proj': 100, ...}

        Here h.0.mlp.c_fc gets C=50 (1 wildcard) instead of C=100 (2 wildcards).
    """
    # module -> (wildcard_count, pattern, C)
    module_to_info: dict[str, tuple[int, str, int]] = {}

    for pattern, c in target_module_patterns:
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
            raise ValueError(
                f"Pattern '{pattern}' in target_module_patterns did not match any modules"
            )

    # Return just module name -> C mapping
    return {name: c for name, (_, _, c) in module_to_info.items()}
