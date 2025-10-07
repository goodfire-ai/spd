from typing import Any, Literal, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import (
    MaskingConfigType,
    RoutingConfigType,
    StochasticConfig,
    SubsetSelectorConfig,
)
from spd.metrics.base import Metric
from spd.metrics.layer_selector import AllSelector, LayerSelector, LayerwiseSelector, SubsetSelector
from spd.metrics.masker import CIMasker, Masker, StochasticMaskSampler
from spd.models.component_model import ComponentModel
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import calc_sum_recon_loss_lm


class ReconstructionLoss(Metric):
    def __init__(
        self,
        model: ComponentModel,
        output_loss_type: Literal["mse", "kl"],
        layer_selector: LayerSelector,
        masker: Masker,
        device: str,
    ):
        self.model = model
        self.output_loss_type: Literal["mse", "kl"] = output_loss_type
        self.layer_selector = layer_selector
        self.masker = masker
        self.sum_loss = torch.tensor(0.0, device=device)
        self.n_examples = torch.tensor(0, device=device)

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, "... C"]],
        **_: Any,
    ) -> None:
        sum_loss, n_examples = _reconstruction_loss_update(
            model=self.model,
            batch=batch,
            target_out=target_out,
            ci=ci,
            weight_deltas=weight_deltas,
            output_loss_type=self.output_loss_type,
            layer_selector=self.layer_selector,
            masker=self.masker,
        )
        self.sum_loss += sum_loss
        self.n_examples += n_examples

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return sum_loss / n_examples


def _reconstruction_loss_update(
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "... C"]],
    output_loss_type: Literal["mse", "kl"],
    layer_selector: LayerSelector,
    masker: Masker,
):
    sum_loss = torch.tensor(0.0, device=batch.device)
    n_examples = 0

    for layer_set_ci in layer_selector.iterate_layer_sets(ci, weight_deltas):
        for mask_infos in masker.sample_mask_infos(
            model=model,
            batch=batch,
            ci=layer_set_ci,
            weight_deltas=weight_deltas,
            target_out=target_out,
            routing=layer_selector.get_routing(),
        ):
            out = model(batch, mask_infos=mask_infos)
            loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=output_loss_type)
            sum_loss += loss
            n_examples = out.shape.numel() if output_loss_type == "mse" else out.shape[:-1].numel()
            n_examples += n_examples

    return sum_loss, n_examples


def create_reconstruction_loss_components(
    routing_cfg: RoutingConfigType,
    masking_cfg: MaskingConfigType,
    use_delta_component: bool,
    sampling: Literal["continuous", "binomial"],
):
    match masking_cfg:
        case StochasticConfig():
            masker = StochasticMaskSampler(
                use_delta_component=use_delta_component,
                sampling=sampling,
                n_mask_samples=masking_cfg.n_mask_samples,
            )
        case "ci":
            masker = CIMasker()
    match routing_cfg:
        case "all":
            layer_selector = AllSelector()
        case SubsetSelectorConfig():
            layer_selector = SubsetSelector(n_subsets=routing_cfg.n_subsets_samples)
        case "layerwise":
            layer_selector = LayerwiseSelector()
    return layer_selector, masker


def reconstruction_loss(
    routing_cfg: RoutingConfigType,
    masking_cfg: MaskingConfigType,
    output_loss_type: Literal["mse", "kl"],
    use_delta_component: bool,
    sampling: Literal["continuous", "binomial"],
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
):
    match masking_cfg:
        case StochasticConfig():
            masker = StochasticMaskSampler(
                use_delta_component=use_delta_component,
                sampling=sampling,
                n_mask_samples=masking_cfg.n_mask_samples,
            )
        case "ci":
            masker = CIMasker()

    match routing_cfg:
        case "all":
            layer_selector = AllSelector()
        case SubsetSelectorConfig():
            layer_selector = SubsetSelector(n_subsets=routing_cfg.n_subsets_samples)
        case "layerwise":
            layer_selector = LayerwiseSelector()

    sum_loss, n_examples = _reconstruction_loss_update(
        model=model,
        batch=batch,
        target_out=target_out,
        ci=ci,
        weight_deltas=weight_deltas,
        output_loss_type=output_loss_type,
        layer_selector=layer_selector,
        masker=masker,
    )

    return sum_loss / n_examples
