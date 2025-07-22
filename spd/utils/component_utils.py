from collections.abc import Callable
from typing import override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader

from spd.configs import SampleConfig
from spd.models.component_model import ComponentModel
from spd.utils.general_utils import extract_batch_data


def sample_uniform_to_1(x: Tensor) -> Tensor:
    return x + (1 - x) * torch.rand_like(x)


class BernoulliSTE(torch.autograd.Function):
    @override
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        sigma: Tensor,
    ) -> Tensor:
        ctx.save_for_backward(sigma)
        return torch.bernoulli(sigma)

    @override
    @staticmethod
    def backward(  # pyright: ignore [reportIncompatibleMethodOverride]
        ctx: torch.autograd.function.FunctionCtx,
        grad_outputs: Tensor,
    ) -> tuple[Tensor]:
        return (grad_outputs.clone(),)


def rescaled_bernoulli_ste(x: Tensor) -> Tensor:
    input = x * 0.5 + 0.5
    return BernoulliSTE.apply(input)  # pyright: ignore [reportReturnType]


def binary_concrete(
    prob: Tensor,
    temp: float,
    eps: float = 1e-6,
) -> Tensor:
    prob = prob.clamp(min=eps, max=1 - eps)
    logit = torch.log(prob / (1 - prob))
    u = torch.rand_like(logit).clamp(min=eps, max=1 - eps)
    logistic_noise = torch.log(u) - torch.log1p(-u)  # logistic noise ~ log(u) - log(1-u)
    y = torch.sigmoid((logit + logistic_noise) / temp)
    return y


def binary_hard_concrete(
    prob: Tensor,
    temp: float,
    bounds: tuple[float, float],
    eps: float = 1e-6,
) -> Tensor:
    low, high = bounds
    stretched = low + (high - low) * binary_concrete(prob, temp, eps)
    return stretched.clamp(0, 1)


def linear_interpolate(a: float, b: float, pc: float) -> float:
    return a * (1 - pc) + b * pc


def get_sample_fn(sample_config: SampleConfig, training_pct: float) -> Callable[[Tensor], Tensor]:
    if sample_config.sample_type == "uniform":
        return sample_uniform_to_1
    elif sample_config.sample_type == "bernoulli_ste":
        return rescaled_bernoulli_ste
    elif sample_config.sample_type == "concrete":

        def sample_fn(x: Tensor) -> Tensor:
            temp = linear_interpolate(
                sample_config.temp_start, sample_config.temp_end, training_pct
            )
            reprojected_x = x * 0.5 + 0.5
            return binary_concrete(reprojected_x, temp)

        return sample_fn
    else:
        assert sample_config.sample_type == "hard_concrete_anneal"

        def sample_fn(x: Tensor) -> Tensor:
            temp = linear_interpolate(
                sample_config.temp_start, sample_config.temp_end, training_pct
            )
            reprojected_x = x * 0.5 + 0.5
            return binary_hard_concrete(reprojected_x, temp, sample_config.bounds)

        return sample_fn


def calc_stochastic_masks(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    n_mask_samples: int,
    sample_config: SampleConfig,
    training_pct: float,
) -> list[dict[str, Float[Tensor, "... C"]]]:
    """Calculate n_mask_samples stochastic masks with the formula `ci + (1 - ci) * rand_unif(0,1)`.

    Args:
        causal_importances: The causal importances to use for the stochastic masks.
        n_mask_samples: The number of stochastic masks to calculate.

    Return:
        A list of n_mask_samples dictionaries, each containing the stochastic masks for each layer.
    """
    sample = get_sample_fn(sample_config, training_pct)
    stochastic_masks = []
    for _ in range(n_mask_samples):
        stochastic_masks.append({layer: sample(ci) for layer, ci in causal_importances.items()})
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
    n_tokens = {module_name: 0 for module_name in model.components}
    total_n_active_components = {module_name: 0 for module_name in model.components}
    component_activation_counts = {
        module_name: torch.zeros(model.C, device=device) for module_name in model.components
    }
    data_iter = iter(dataloader)
    for _ in range(n_steps):
        # --- Get Batch --- #
        batch = extract_batch_data(next(data_iter))
        batch = batch.to(device)

        _, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            batch, module_names=model.target_module_paths
        )

        causal_importances, _ = model.calc_causal_importances(pre_weight_acts, detach_inputs=False)
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
        for module_name in model.components
    }
    mean_component_activation_counts: dict[str, Float[Tensor, " C"]] = {
        module_name: component_activation_counts[module_name] / n_tokens[module_name]
        for module_name in model.components
    }

    return mean_n_active_components_per_token, mean_component_activation_counts
