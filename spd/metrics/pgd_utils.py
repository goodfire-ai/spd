from typing import Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import PGDConfig, PGDInitStrategy
from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos
from spd.utils.component_utils import (
    RoutingType,
    sample_uniform_k_subset_routing_masks,
)
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import calc_sum_recon_loss_lm


def pgd_masked_recon_loss_update(
    *,
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    target_out: Float[Tensor, "... vocab"],
    output_loss_type: Literal["mse", "kl"],
    routing: RoutingType,
    pgd_config: PGDConfig,
) -> tuple[Float[Tensor, ""], int]:
    """Central implementation of PGD masked reconstruction loss.

    Optimizes adversarial stochastic masks and optionally weight deltas for the given objective function.
    """

    match routing:
        case "all":
            routing_masks = "all"
        case "uniform_k-stochastic":
            routing_masks = sample_uniform_k_subset_routing_masks(
                mask_shape=next(iter(ci.values())).shape[:-1],
                module_names=list(ci.keys()),
                device=batch.device,
            )

    *batch_dims, C = next(iter(ci.values())).shape
    n_layers = len(ci)
    # C2 represents the total number of components including the optional weight delta
    C2 = C if weight_deltas is None else C + 1

    def objective_fn(
        adv_sources: Float[Tensor, "n_layers *batch_dims C2"],
    ) -> tuple[Float[Tensor, ""], int]:
        adv_sources_components: Float[Tensor, "n_layers *batch_dims C"]
        match weight_deltas:
            case None:
                weight_deltas_and_masks = None
                adv_sources_components = adv_sources
            case dict():
                assert adv_sources.shape[-1] == C + 1
                weight_deltas_and_masks = {
                    k: (weight_deltas[k], adv_sources[i, ..., -1])
                    for i, k in enumerate(weight_deltas)
                }
                adv_sources_components = adv_sources[..., :-1]

        mask_infos = make_mask_infos(
            component_masks=_interpolate_component_mask(ci, adv_sources_components),
            weight_deltas_and_masks=weight_deltas_and_masks,
            routing_masks=routing_masks,
        )
        out = model(batch, mask_infos=mask_infos)
        total_loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=output_loss_type)
        n_examples = out.shape.numel() if output_loss_type == "mse" else out.shape[:-1].numel()
        return total_loss, n_examples

    # We create a single adv_source and index into it for each layer
    match pgd_config.mask_scope:
        case "unique_per_datapoint":
            adv_source_shape = torch.Size([n_layers, *batch_dims, C2])
        case "shared_across_batch":
            adv_source_shape = torch.Size([n_layers] + [1 for _ in batch_dims] + [C2])

    adv_sources: Float[Tensor, "n_layers ... C2"] = _get_pgd_init_tensor(
        pgd_config.init, adv_source_shape, batch.device
    ).requires_grad_(True)

    # PGD ascent
    for _ in range(int(pgd_config.n_steps)):
        assert adv_sources.grad is None

        with torch.enable_grad():
            total_loss, _ = objective_fn(adv_sources=adv_sources.expand(n_layers, *batch_dims, C2))
            adv_sources_grads = torch.autograd.grad(total_loss, adv_sources)[0]

        reduced_adv_sources_grads = all_reduce(adv_sources_grads.clone(), op=ReduceOp.SUM)
        with torch.no_grad():
            adv_sources.add_(pgd_config.step_size * reduced_adv_sources_grads.sign())
            adv_sources.clamp_(0.0, 1.0)

    assert not adv_sources.requires_grad
    total_loss, n_examples = objective_fn(adv_sources=adv_sources.expand(n_layers, *batch_dims, C2))

    # no need to all-reduce total_loss or n_examples bc consumers handle this
    return total_loss, n_examples


def _get_pgd_init_tensor(
    init: PGDInitStrategy,
    shape: tuple[int, ...],
    device: torch.device | str,
) -> Float[Tensor, "... shape"]:
    match init:
        case "random":
            return torch.rand(shape, device=device)
        case "ones":
            return torch.ones(shape, device=device)
        case "zeroes":
            return torch.zeros(shape, device=device)


def _interpolate_component_mask(
    ci: dict[str, Float[Tensor, "*batch_dims C"]],
    component_sample_points: Float[Tensor, "n_layers *batch_dims C"],
) -> dict[str, Float[Tensor, "*batch_dims C"]]:
    assert torch.all(component_sample_points <= 1.0) and torch.all(component_sample_points >= 0.0)
    component_masks: dict[str, Float[Tensor, "*batch_dims C"]] = {}
    for i, module_name in enumerate(ci):
        scaled_noise_to_add = (1 - ci[module_name]) * component_sample_points[i]
        component_masks[module_name] = ci[module_name] + scaled_noise_to_add
    return component_masks
