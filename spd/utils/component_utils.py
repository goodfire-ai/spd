import torch
from einops import einsum
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader

from spd.models.component_model import ComponentModel
from spd.models.sigmoids import SigmoidTypes
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
    sigmoid_type: SigmoidTypes,
    device: str,
    threshold: float,
) -> tuple[dict[str, float], dict[str, Float[Tensor, " C"]], dict[str, Float[Tensor, " C C"]]]:
    """Get the number and strength of the masks over the full dataset."""
    n_tokens = {module_name: 0 for module_name in model.components}
    total_n_active_components = {module_name: 0 for module_name in model.components}
    component_activation_counts = {
        module_name: torch.zeros(model.C, device=device) for module_name in model.components
    }
    component_co_activation_counts = {
        module_name: torch.zeros(model.C, model.C, device=device)
        for module_name in model.components
    }
    data_iter = iter(dataloader)
    for _ in range(n_steps):
        # --- Get Batch --- #
        batch = extract_batch_data(next(data_iter))
        batch = batch.to(device)

        _, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            batch, module_names=model.target_module_paths
        )

        causal_importances, _ = model.calc_causal_importances(
            pre_weight_acts, sigmoid_type=sigmoid_type, detach_inputs=False
        )
        for module_name, ci in causal_importances.items():
            # mask (batch, pos, C) or (batch, C)
            n_tokens[module_name] += ci.shape[:-1].numel()

            # Count the number of components that are active above the threshold
            active_components = ci > threshold
            total_n_active_components[module_name] += int(active_components.sum().item())

            sum_dims = tuple(range(ci.ndim - 1))
            component_activation_counts[module_name] += active_components.sum(dim=sum_dims)
            component_co_activation_counts[module_name] += einsum(
                active_components, active_components, "b C, b C2 -> b C C2"
            ).sum(dim=sum_dims)

    # Show the mean number of components
    mean_n_active_components_per_token: dict[str, float] = {
        module_name: (total_n_active_components[module_name] / n_tokens[module_name])
        for module_name in model.components
    }
    mean_component_activation_counts: dict[str, Float[Tensor, " C"]] = {
        module_name: component_activation_counts[module_name] / n_tokens[module_name]
        for module_name in model.components
    }
    sorted_activation_inds = {
        module_name: torch.argsort(
            mean_component_activation_counts[module_name], dim=-1, descending=True
        )
        for module_name in model.components
    }

    # Calculate frac components co-activated with each other conditioned on the activation of the other
    component_co_activation_counts_denom = {
        module_name: torch.ones(model.C, model.C, device=device)
        * component_activation_counts[module_name]
        for module_name in model.components
    }

    component_co_activation_fractions = {
        module_name: component_co_activation_counts[module_name]
        / component_co_activation_counts_denom[module_name]
        for module_name in model.components
    }
    # Convert nans to 0
    component_co_activation_fractions = {
        module_name: torch.where(
            torch.isnan(component_co_activation_fractions[module_name]),
            torch.zeros_like(component_co_activation_fractions[module_name]),
            component_co_activation_fractions[module_name],
        )
        for module_name in model.components
    }

    sorted_co_activation_fractions = {
        module_name: component_co_activation_fractions[module_name][
            sorted_activation_inds[module_name], :
        ][:, sorted_activation_inds[module_name]]
        for module_name in model.components
    }

    return (
        mean_n_active_components_per_token,
        mean_component_activation_counts,
        sorted_co_activation_fractions,
    )
