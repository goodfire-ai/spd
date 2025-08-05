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

    # Show the mean number of component activations
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


def component_abs_left_singular_vectors_cosine_similarity(
    model: ComponentModel,
    sorted_activation_inds: dict[str, Float[Tensor, " C"]],
) -> dict[str, Float[Tensor, " C"]]:
    """Get the cosine similarity between the absolute left singular vectors of the components."""
    eps = 1e-8
    component_abs_left_singular_vectors = {
        module_name: torch.abs(model.components[module_name].U.data)
        for module_name in model.components
    }
    # First calculate the inner products
    component_abs_left_singular_vectors_inner_products = {
        module_name: einsum(
            component_abs_left_singular_vectors[module_name],
            component_abs_left_singular_vectors[module_name],
            "C d, C2 d -> C C2",
        )
        for module_name in model.components
    }
    # Then calculate the norms
    component_abs_left_singular_vectors_norms_vecs = {
        module_name: torch.norm(component_abs_left_singular_vectors[module_name], dim=-1)
        for module_name in model.components
    }
    component_abs_left_singular_vectors_norms_mats = {
        module_name: einsum(
            component_abs_left_singular_vectors_norms_vecs[module_name],
            component_abs_left_singular_vectors_norms_vecs[module_name],
            "C, C2 -> C C2",
        )
        for module_name in model.components
    }

    # Then calculate the cosine similarity
    component_abs_left_singular_vectors_cosine_similarity_matrices = {
        module_name: component_abs_left_singular_vectors_inner_products[module_name]
        / (component_abs_left_singular_vectors_norms_mats[module_name] + eps)
        for module_name in model.components
    }

    # Then sort the cosine similarity matrices by the component counts
    component_abs_left_singular_vectors_cosine_similarity_matrices = {
        module_name: component_abs_left_singular_vectors_cosine_similarity_matrices[module_name][
            sorted_activation_inds[module_name], :
        ][:, sorted_activation_inds[module_name]]
        for module_name in model.components
    }

    return component_abs_left_singular_vectors_cosine_similarity_matrices


def create_cosine_sim_coactivation_correlation_dataset(
    model: ComponentModel,
    dataloader: DataLoader[Int[Tensor, "..."]]
    | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    n_steps: int,
    sigmoid_type: SigmoidTypes,
    device: str,
    threshold: float,
) -> dict[str, tuple[Float[Tensor, " C C"], Float[Tensor, " C C"]]]:
    (
        _,
        mean_component_activation_counts,
        sorted_co_activation_fractions,
    ) = component_activation_statistics(
        model=model,
        dataloader=dataloader,
        n_steps=n_steps,
        sigmoid_type=sigmoid_type,
        device=device,
        threshold=threshold,
    )

    sorted_activation_inds = {
        module_name: torch.argsort(
            mean_component_activation_counts[module_name], dim=-1, descending=True
        )
        for module_name in model.components
    }

    n_alive_components = {
        module_name: torch.sum(mean_component_activation_counts[module_name] > 0.0001)
        for module_name in model.components
    }  # TODO unhardcode

    component_abs_left_singular_vectors_cosine_similarity_matrices = (
        component_abs_left_singular_vectors_cosine_similarity(
            model=model,
            sorted_activation_inds=sorted_activation_inds,
        )
    )

    alive_co_activation_fractions = {
        module_name: sorted_co_activation_fractions[module_name][
            : n_alive_components[module_name], : n_alive_components[module_name]
        ]
        for module_name in model.components
    }

    alive_cosine_sim_matrices = {
        module_name: component_abs_left_singular_vectors_cosine_similarity_matrices[module_name][
            : n_alive_components[module_name], : n_alive_components[module_name]
        ]
        for module_name in model.components
    }

    # Flatten the matrices
    alive_co_activation_fractions_flattened = {
        module_name: alive_co_activation_fractions[module_name].flatten()
        for module_name in model.components
    }

    alive_cosine_sim_matrices_flattened = {
        module_name: alive_cosine_sim_matrices[module_name].flatten()
        for module_name in model.components
    }

    # Concatenate the flattened matrices per module
    alive_cosine_sim_and_coacts = {
        module_name: (
            alive_cosine_sim_matrices_flattened[module_name],
            alive_co_activation_fractions_flattened[module_name],
        )
        for module_name in model.components
    }

    return alive_cosine_sim_and_coacts
