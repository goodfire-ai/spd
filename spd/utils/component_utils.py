from collections.abc import Mapping
from typing import cast

import einops
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader

from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponent, GateMLP, LinearComponent, StarGraphGate, VectorGateMLP
from spd.models.sigmoids import SIGMOID_TYPES, SigmoidTypes
from spd.utils.general_utils import extract_batch_data


def calc_stochastic_masks(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    n_mask_samples: int,
) -> list[dict[str, Float[Tensor, "... C"]]]:
    """Calculate n_mask_samples stochastic masks with the formula `ci + (1 - ci) * rand_unif(0,1)`.

    Args:
        causal_importances: The causal importances to use for the stochastic masks.
        n_mask_samples: The number of stochastic masks to calculate.

    Return:
        A list of n_mask_samples dictionaries, each containing the stochastic masks for each layer.
    """
    stochastic_masks = []
    for _ in range(n_mask_samples):
        stochastic_masks.append(
            {layer: ci + (1 - ci) * torch.rand_like(ci) for layer, ci in causal_importances.items()}
        )
    return stochastic_masks


def calc_ci_l_zero(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    cutoff: float = 1e-2,
) -> dict[str, float]:
    """Calculate the L0 loss on the causal importances, summed over the C dimension."""
    ci_l_zero = {}
    for layer_name, ci in causal_importances.items():
        mean_dims = tuple(range(ci.ndim - 1))
        ci_l_zero[layer_name] = (ci > cutoff).float().mean(dim=mean_dims).sum().item()
    return ci_l_zero


def component_activation_statistics(
    model: ComponentModel,
    dataloader: DataLoader[Int[Tensor, "..."]]
    | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    n_steps: int,
    device: str,
    threshold: float = 0.1,
) -> tuple[dict[str, float], dict[str, Float[Tensor, " C"]]]:
    """Get the number and strength of the masks over the full dataset."""
    # We used "-" instead of "." as module names can't have "." in them
    gates: dict[str, GateMLP | VectorGateMLP] = {
        k.removeprefix("gates.").replace("-", "."): cast(GateMLP | VectorGateMLP, v)
        for k, v in model.gates.items()
    }
    components: dict[str, LinearComponent | EmbeddingComponent] = {
        k.removeprefix("components.").replace("-", "."): cast(
            LinearComponent | EmbeddingComponent, v
        )
        for k, v in model.components.items()
    }

    n_tokens = {module_name.replace("-", "."): 0 for module_name in components}
    total_n_active_components = {module_name.replace("-", "."): 0 for module_name in components}
    component_activation_counts = {
        module_name.replace("-", "."): torch.zeros(model.C, device=device)
        for module_name in components
    }
    data_iter = iter(dataloader)
    for _ in range(n_steps):
        # --- Get Batch --- #
        batch = extract_batch_data(next(data_iter))
        batch = batch.to(device)

        _, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            batch, module_names=list(components.keys())
        )
        Vs = {module_name: v.V for module_name, v in components.items()}

        causal_importances, _ = calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            Vs=Vs,
            gates=gates,
            detach_inputs=False,
        )
        for module_name, ci in causal_importances.items():
            # mask (batch, pos, C) or (batch, C)
            n_tokens[module_name] += ci.shape[:-1].numel()

            # Count the number of components that are active above the threshold
            active_components = ci > threshold
            total_n_active_components[module_name] += int(active_components.sum().item())

            sum_dims = tuple(range(ci.ndim - 1))
            component_activation_counts[module_name] += active_components.sum(dim=sum_dims)

    # Show the mean number of components
    mean_n_active_components_per_token: dict[str, float] = {
        module_name: (total_n_active_components[module_name] / n_tokens[module_name])
        for module_name in components
    }
    mean_component_activation_counts: dict[str, Float[Tensor, " C"]] = {
        module_name: component_activation_counts[module_name] / n_tokens[module_name]
        for module_name in components
    }

    return mean_n_active_components_per_token, mean_component_activation_counts


def calc_causal_importances(
    pre_weight_acts: dict[str, Float[Tensor, "... d_in"] | Int[Tensor, "... pos"]],
    Vs: Mapping[str, Float[Tensor, "d_in C"]],
    gates: Mapping[str, GateMLP | VectorGateMLP | StarGraphGate],
    detach_inputs: bool = False,
    sigmoid_type: SigmoidTypes = "leaky_hard",
) -> tuple[dict[str, Float[Tensor, "... C"]], dict[str, Float[Tensor, "... C"]]]:
    """Calculate component activations and causal importances in one pass to save memory.

    Args:
        pre_weight_acts: The activations before each layer in the target model.
        Vs: The V matrix at each layer.
        gates: The gates to use for the mask.
        detach_inputs: Whether to detach the inputs to the gates.
        sigmoid_type: Type of sigmoid to use.

    Returns:
        Tuple of (causal_importances, causal_importances_upper_leaky) dictionaries for each layer.
    """
    causal_importances = {}
    causal_importances_upper_leaky = {}

    if len(gates) == 1 and isinstance(next(iter(gates.values())), StarGraphGate):
        acts = {}
        gate_inputs = {}
        for param_name in pre_weight_acts:
            acts = pre_weight_acts[param_name]
            if not acts.dtype.is_floating_point:
                # Embedding layer
                gate_input = Vs[param_name][acts]
            else:
                # Linear layer
                gate_input = einops.einsum(acts, Vs[param_name], "... d_in, d_in C -> ... C")
            if detach_inputs:
                gate_input = gate_input.detach()
            gate_inputs[param_name] = gate_input
        gate_output = next(iter(gates.values()))(gate_inputs)
        if sigmoid_type == "leaky_hard":
            causal_importances = {
                param_name: SIGMOID_TYPES["lower_leaky_hard"](gate_output[param_name])
                for param_name in gate_output
            }
            causal_importances_upper_leaky = {
                param_name: SIGMOID_TYPES["upper_leaky_hard"](gate_output[param_name])
                for param_name in gate_output
            }
        else:
            sigmoid_fn = SIGMOID_TYPES[sigmoid_type]
            causal_importances = {
                param_name: sigmoid_fn(gate_output[param_name]) for param_name in gate_output
            }
            causal_importances_upper_leaky = {
                param_name: sigmoid_fn(gate_output[param_name]).abs() for param_name in gate_output
            }
    else:
        for param_name in pre_weight_acts:
            acts = pre_weight_acts[param_name]
            gate = gates[param_name]

            if isinstance(gate, GateMLP):
                # need to get the inner activation for GateMLP
                if not acts.dtype.is_floating_point:
                    # Embedding layer
                    inner_acts = Vs[param_name][acts]
                else:
                    # Linear layer
                    inner_acts = einops.einsum(acts, Vs[param_name], "... d_in, d_in C -> ... C")
                gate_input = inner_acts
            else:
                gate_input = acts

            if detach_inputs:
                gate_input = gate_input.detach()

            gate_output = gate(gate_input)

            if sigmoid_type == "leaky_hard":
                causal_importances[param_name] = SIGMOID_TYPES["lower_leaky_hard"](gate_output)
                causal_importances_upper_leaky[param_name] = SIGMOID_TYPES["upper_leaky_hard"](
                    gate_output
                )
            else:
                # For other sigmoid types, use the same function for both
                sigmoid_fn = SIGMOID_TYPES[sigmoid_type]
                causal_importances[param_name] = sigmoid_fn(gate_output)
                # Use absolute value to ensure upper_leaky values are non-negative for importance minimality loss
                causal_importances_upper_leaky[param_name] = sigmoid_fn(gate_output).abs()

    return causal_importances, causal_importances_upper_leaky
