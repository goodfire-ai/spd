from functools import partial
from typing import Any, ClassVar, Literal, override

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import PGDConfig
from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.models.components import RoutingMasks, make_mask_infos
from spd.utils.component_utils import RoutingType, sample_uniform_k_subset_routing_masks
from spd.utils.distributed_utils import all_reduce


def pgd_ce_diff_loss_update(
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    target_out: Float[Tensor, "... vocab"],
    routing: RoutingType,
    pgd_config: PGDConfig,
) -> tuple[Float[Tensor, ""], int]:
    """PGD optimization for CE difference metric.

    Optimizes adversarial stochastic masks to maximize CE loss against true labels.
    Only applicable for language model tasks with 3D outputs (batch, seq_len, vocab).
    """
    if target_out.ndim != 3:
        return torch.tensor(0.0, device=batch.device), 0

    assert batch.ndim == 2, "Batch must be 2D (batch, seq_len)"

    C = model.C
    batch_dims = next(iter(ci.values())).shape[:-1]
    n_layers = len(ci)
    C2 = C if weight_deltas is None else C + 1

    masked_batch = batch.clone()
    masked_batch[:, 0] = -100
    flat_masked_batch = masked_batch.flatten()

    match routing:
        case "all":
            routing_masks = "all"
        case "uniform_k-stochastic":
            routing_masks = sample_uniform_k_subset_routing_masks(
                mask_shape=batch_dims,
                module_names=model.target_module_paths,
                device=batch.device,
            )

    match pgd_config.mask_scope:
        case "unique_per_datapoint":
            adv_source_shape = torch.Size([n_layers, *batch_dims, C2])
        case "shared_across_batch":
            singleton_batch_dims = [1 for _ in batch_dims]
            adv_source_shape = torch.Size([n_layers, *singleton_batch_dims, C2])

    adv_sources: Float[Tensor, "n_layers *batch_dims C2"] | Float[Tensor, "n_layers *1 C2"] = (
        _get_pgd_init_tensor(pgd_config.init, adv_source_shape, batch.device).requires_grad_(True)
    )

    fwd_pass = partial(
        _forward_with_adv_sources_ce,
        model=model,
        batch=batch,
        adv_sources=adv_sources,
        ci=ci,
        weight_deltas=weight_deltas,
        routing_masks=routing_masks,
        flat_masked_batch=flat_masked_batch,
        batch_dims=batch_dims,
    )

    for _ in range(pgd_config.n_steps):
        assert adv_sources.grad is None
        with torch.enable_grad():
            ce_loss = fwd_pass()
        (adv_sources_grads,) = torch.autograd.grad(ce_loss, adv_sources)
        adv_sources_grads = all_reduce(adv_sources_grads, op=ReduceOp.SUM)
        with torch.no_grad():
            adv_sources.add_(pgd_config.step_size * adv_sources_grads.sign())
            adv_sources.clamp_(0.0, 1.0)

    final_ce_loss = fwd_pass()

    flat_target_logits = einops.rearrange(target_out, "b seq_len vocab -> (b seq_len) vocab")
    target_ce_loss = F.cross_entropy(
        flat_target_logits[:-1], flat_masked_batch[1:], ignore_index=-100, reduction="sum"
    )

    n_positions = batch.shape[0] * batch.shape[1]
    ce_diff = final_ce_loss - target_ce_loss

    return ce_diff, n_positions


def _forward_with_adv_sources_ce(
    model: ComponentModel,
    batch: Int[Tensor, "..."],
    adv_sources: Float[Tensor, "n_layers *batch_dim_or_ones C2"],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    routing_masks: RoutingMasks,
    flat_masked_batch: Int[Tensor, "..."],
    batch_dims: tuple[int, ...],
) -> Float[Tensor, ""]:
    expanded_adv_sources = adv_sources.expand(-1, *batch_dims, -1)
    adv_sources_components: Float[Tensor, "n_layers *batch_dims C"]
    match weight_deltas:
        case None:
            weight_deltas_and_masks = None
            adv_sources_components = expanded_adv_sources
        case dict():
            weight_deltas_and_masks = {
                k: (weight_deltas[k], expanded_adv_sources[i, ..., -1])
                for i, k in enumerate(weight_deltas)
            }
            adv_sources_components = expanded_adv_sources[..., :-1]

    mask_infos = make_mask_infos(
        component_masks=_interpolate_component_mask(ci, adv_sources_components),
        weight_deltas_and_masks=weight_deltas_and_masks,
        routing_masks=routing_masks,
    )
    out = model(batch, mask_infos=mask_infos)

    flat_logits = einops.rearrange(out, "b seq_len vocab -> (b seq_len) vocab")
    ce_loss = F.cross_entropy(
        flat_logits[:-1], flat_masked_batch[1:], ignore_index=-100, reduction="sum"
    )

    return ce_loss


def _get_pgd_init_tensor(
    init: Literal["random", "ones", "zeroes"],
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
    adv_sources_components: Float[Tensor, "n_layers *batch_dims C"],
) -> dict[str, Float[Tensor, "*batch_dims C"]]:
    """Set the mask value to ci + (1 - ci) * adv_sources_components."""
    assert torch.all(adv_sources_components <= 1.0) and torch.all(adv_sources_components >= 0.0)
    assert adv_sources_components.shape[0] == len(ci)
    assert all(ci[k].shape[-1] == adv_sources_components.shape[-1] for k in ci)
    component_masks: dict[str, Float[Tensor, "*batch_dims C"]] = {}
    for i, module_name in enumerate(ci):
        scaled_noise_to_add = (1 - ci[module_name]) * adv_sources_components[i]
        component_masks[module_name] = ci[module_name] + scaled_noise_to_add
    return component_masks


class PGDCEDiff(Metric):
    """CE difference metric using adversarially-optimized PGD masks.

    This metric uses PGD to find masks that maximize cross-entropy loss against true labels,
    then reports the CE difference from the target model.
    """

    metric_section: ClassVar[str] = "ce_kl"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        pgd_config: PGDConfig,
        use_delta_component: bool,
    ) -> None:
        self.model = model
        self.pgd_config: PGDConfig = pgd_config
        self.use_delta_component: bool = use_delta_component
        self.sum_ce_diff = torch.tensor(0.0, device=device)
        self.n_positions = torch.tensor(0, device=device)

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: CIOutputs,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
        **_: Any,
    ) -> None:
        ce_diff, n_positions = pgd_ce_diff_loss_update(
            model=self.model,
            batch=batch,
            ci=ci.lower_leaky,
            weight_deltas=weight_deltas if self.use_delta_component else None,
            target_out=target_out,
            routing="all",
            pgd_config=self.pgd_config,
        )
        self.sum_ce_diff += ce_diff
        self.n_positions += n_positions

    @override
    def compute(self) -> dict[str, float]:
        sum_ce_diff = all_reduce(self.sum_ce_diff, op=ReduceOp.SUM)
        n_positions = all_reduce(self.n_positions, op=ReduceOp.SUM)
        return {"ce_difference_pgd_masked": (sum_ce_diff / n_positions).item()}
