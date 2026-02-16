"""SPD-specific activation functions for the generic decomposition pipeline."""

import torch
from jaxtyping import Float
from torch import Tensor

from spd.configs import SamplingType
from spd.models.component_model import ComponentModel
from spd.utils.general_utils import extract_batch_data

from ..types import BatchLike


def _as_token_tensor(batch: BatchLike) -> Tensor:
    return extract_batch_data(batch)


def make_spd_activation_fn(sampling: SamplingType):
    """Build ActivationFn for SPD causal importances."""

    def _activation_fn(
        model: ComponentModel,
        batch: BatchLike,
    ) -> dict[str, Float[Tensor, "batch seq"]]:
        tokens = _as_token_tensor(batch)
        out = model(tokens, cache_type="input")
        ci_dict = model.calc_causal_importances(
            pre_weight_acts=out.cache,
            detach_inputs=True,
            sampling=sampling,
        ).lower_leaky

        result: dict[str, Float[Tensor, "batch seq"]] = {}
        for layer in model.target_module_paths:
            layer_ci = ci_dict[layer]
            for comp_idx in range(layer_ci.shape[2]):
                result[f"{layer}:{comp_idx}"] = layer_ci[:, :, comp_idx]
        return result

    return _activation_fn


def spd_output_probs_fn(
    model: ComponentModel,
    batch: BatchLike,
) -> Float[Tensor, "batch seq vocab"]:
    """Compute next-token probabilities from SPD model output logits."""
    tokens = _as_token_tensor(batch)
    logits = model(tokens, cache_type="none")
    return torch.softmax(logits, dim=-1)


def make_spd_component_acts_fn():
    """Build ComponentActsFn using normalized SPD component activations."""

    def _component_acts_fn(
        model: ComponentModel,
        batch: BatchLike,
    ) -> dict[str, Float[Tensor, "batch seq"]]:
        tokens = _as_token_tensor(batch)
        out = model(tokens, cache_type="input")
        per_layer_acts = model.get_all_component_acts(out.cache)
        u_norms = {layer: component.U.norm(dim=1) for layer, component in model.components.items()}

        result: dict[str, Float[Tensor, "batch seq"]] = {}
        for layer in model.target_module_paths:
            layer_acts = per_layer_acts[layer] * u_norms[layer].to(per_layer_acts[layer].device)
            for comp_idx in range(layer_acts.shape[2]):
                result[f"{layer}:{comp_idx}"] = layer_acts[:, :, comp_idx]
        return result

    return _component_acts_fn
