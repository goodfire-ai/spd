from collections.abc import Callable
from functools import partial
from typing import cast, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader

from spd.configs import (
    BernoulliSampleConfig,
    ConcreteSampleConfig,
    HardConcreteSampleConfig,
    UniformSampleConfig,
)
from spd.models.component_model import ComponentModel
from spd.utils.general_utils import extract_batch_data

SampleConfig = (
    UniformSampleConfig | BernoulliSampleConfig | ConcreteSampleConfig | HardConcreteSampleConfig
)


def sample_uniform_to_1(min: Tensor) -> Tensor:
    return min + (1 - min) * torch.rand_like(min)


class BernoulliSTE(torch.autograd.Function):
    @override
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        sigma: Tensor,
        stochastic: bool,
    ) -> Tensor:
        ctx.save_for_backward(sigma)
        z = torch.bernoulli(sigma) if stochastic else (sigma >= 0.5).to(sigma.dtype)

        return z

    @override
    @staticmethod
    def backward(  # pyright: ignore [reportIncompatibleMethodOverride]
        ctx: torch.autograd.function.FunctionCtx,
        grad_outputs: Tensor,
    ) -> tuple[Tensor, None]:
        return grad_outputs.clone(), None


def bernoulli_ste(x: Tensor, min: float) -> Tensor:
    input = x * (1 - min) + min
    return BernoulliSTE.apply(input, True)  # pyright: ignore [reportReturnType]


class HeavisideSTE(torch.autograd.Function):
    @override
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: Tensor,
    ) -> Tensor:
        return (x > 0).to(x.dtype)

    @override
    @staticmethod
    def backward(  # pyright: ignore [reportIncompatibleMethodOverride]
        ctx: torch.autograd.function.FunctionCtx,
        grad_outputs: Tensor,
    ) -> Tensor:
        return grad_outputs.clone()


def _concrete(log_alpha: Tensor, temperature: float) -> Tensor:
    """Sample from the concrete distribution."""
    uniform = torch.rand_like(log_alpha).clamp(1e-8, 1 - 1e-8)
    gumbel_noise = -torch.log(-torch.log(uniform))
    concrete_sample = torch.sigmoid((log_alpha + gumbel_noise) / temperature)
    return concrete_sample


def concrete_gate(x: Tensor, temperature: float) -> Tensor:
    """Sample from the concrete distribution for gates."""
    return _concrete(x, temperature)


def hard_concrete_gate(x: Tensor, temperature: float, stretch: float) -> Tensor:
    """Sample from the hard concrete distribution with stretch."""
    concrete_sample = _concrete(x, temperature)
    # Apply stretch and clamp
    stretched = concrete_sample * stretch
    hard_concrete = torch.clamp(stretched, 0, 1)
    # Apply straight-through estimator for hard thresholding
    hard_mask = cast(Tensor, HeavisideSTE.apply(stretched - 1))
    return hard_concrete * (1 - hard_mask) + hard_mask


def get_sample_fn(sample_config: SampleConfig) -> Callable[[Tensor], Tensor]:
    """Get the appropriate sampling function based on the sample config."""
    if sample_config.sample_type == "uniform":
        return sample_uniform_to_1
    elif sample_config.sample_type == "bernoulli":
        return partial(bernoulli_ste, min=sample_config.min)
    elif sample_config.sample_type == "concrete":
        return partial(concrete_gate, temperature=sample_config.temperature)
    else:
        assert sample_config.sample_type == "hard_concrete"
        return partial(
            hard_concrete_gate, temperature=sample_config.temperature, stretch=sample_config.stretch
        )


def calc_stochastic_masks(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    n_mask_samples: int,
    sample_config: SampleConfig,
) -> list[dict[str, Float[Tensor, "... C"]]]:
    """Calculate n_mask_samples stochastic masks with the formula `ci + (1 - ci) * rand_unif(0,1)`.

    Args:
        causal_importances: The causal importances to use for the stochastic masks.
        n_mask_samples: The number of stochastic masks to calculate.

    Return:
        A list of n_mask_samples dictionaries, each containing the stochastic masks for each layer.
    """
    sample = get_sample_fn(sample_config)
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
    threshold: float,
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
