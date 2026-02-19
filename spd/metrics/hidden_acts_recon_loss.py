from typing import Any, ClassVar, override

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import PGDConfig, SamplingType
from spd.metrics.base import Metric
from spd.metrics.pgd_utils import pgd_hidden_acts_recon_loss_update
from spd.models.component_model import CIOutputs, ComponentModel
from spd.models.components import ComponentsMaskInfo, make_mask_infos
from spd.routing import AllLayersRouter
from spd.utils.component_utils import calc_stochastic_component_mask_info
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import get_obj_device


def _calc_hidden_acts_mse(
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    mask_infos: dict[str, ComponentsMaskInfo],
    pre_weight_acts: dict[str, Float[Tensor, "..."]],
) -> tuple[Float[Tensor, ""], int]:
    """Forward with mask_infos and compute MSE against target hidden activations."""
    comp_cache = model(batch, mask_infos=mask_infos, cache_type="input").cache
    device = next(iter(pre_weight_acts.values())).device
    sum_mse = torch.tensor(0.0, device=device)
    n_examples = 0
    for layer_name, target_acts in pre_weight_acts.items():
        assert layer_name in comp_cache, f"{layer_name} not in comp_cache"
        sum_mse = sum_mse + F.mse_loss(comp_cache[layer_name], target_acts, reduction="sum")
        n_examples += target_acts.numel()
    return sum_mse, n_examples


def _stochastic_hidden_acts_recon_loss_update(
    model: ComponentModel,
    sampling: SamplingType,
    n_mask_samples: int,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    pre_weight_acts: dict[str, Float[Tensor, "..."]],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
) -> tuple[Float[Tensor, ""], int]:
    assert ci, "Empty ci"
    assert pre_weight_acts, "Empty pre_weight_acts"
    device = get_obj_device(ci)
    sum_mse = torch.tensor(0.0, device=device)
    n_examples = 0

    stoch_mask_infos_list = [
        calc_stochastic_component_mask_info(
            causal_importances=ci,
            component_mask_sampling=sampling,
            weight_deltas=weight_deltas,
            router=AllLayersRouter(),
        )
        for _ in range(n_mask_samples)
    ]
    for stoch_mask_infos in stoch_mask_infos_list:
        batch_mse, batch_n = _calc_hidden_acts_mse(
            model=model,
            batch=batch,
            mask_infos=stoch_mask_infos,
            pre_weight_acts=pre_weight_acts,
        )
        sum_mse = sum_mse + batch_mse
        n_examples += batch_n

    return sum_mse, n_examples


def _hidden_acts_recon_loss_compute(
    sum_mse: Float[Tensor, ""], n_examples: Int[Tensor, ""] | int
) -> Float[Tensor, ""]:
    return sum_mse / n_examples


def stochastic_hidden_acts_recon_loss(
    model: ComponentModel,
    sampling: SamplingType,
    n_mask_samples: int,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    pre_weight_acts: dict[str, Float[Tensor, "..."]],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
) -> Float[Tensor, ""]:
    sum_mse, n_examples = _stochastic_hidden_acts_recon_loss_update(
        model,
        sampling,
        n_mask_samples,
        batch,
        pre_weight_acts,
        ci,
        weight_deltas,
    )
    return _hidden_acts_recon_loss_compute(sum_mse, n_examples)


class StochasticHiddenActsReconLoss(Metric):
    """Reconstruction loss between target and stochastic hidden activations when sampling with stochastic masks."""

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        sampling: SamplingType,
        use_delta_component: bool,
        n_mask_samples: int,
    ) -> None:
        self.model = model
        self.sampling: SamplingType = sampling
        self.use_delta_component: bool = use_delta_component
        self.n_mask_samples: int = n_mask_samples
        self.sum_mse = torch.tensor(0.0, device=device)
        self.n_examples = torch.tensor(0, device=device)

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        pre_weight_acts: dict[str, Float[Tensor, "..."]],
        ci: CIOutputs,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
        **_: Any,
    ) -> None:
        sum_mse, n_examples = _stochastic_hidden_acts_recon_loss_update(
            model=self.model,
            sampling=self.sampling,
            n_mask_samples=self.n_mask_samples,
            batch=batch,
            pre_weight_acts=pre_weight_acts,
            ci=ci.lower_leaky,
            weight_deltas=weight_deltas if self.use_delta_component else None,
        )
        self.sum_mse += sum_mse
        self.n_examples += n_examples

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_mse = all_reduce(self.sum_mse, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return _hidden_acts_recon_loss_compute(sum_mse, n_examples)


class CIHiddenActsReconLoss(Metric):
    """Reconstruction loss between target and component hidden activations when masking with CI values."""

    metric_section: ClassVar[str] = "loss"

    def __init__(self, model: ComponentModel, device: str) -> None:
        self.model = model
        self.sum_mse = torch.tensor(0.0, device=device)
        self.n_examples = torch.tensor(0, device=device)

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        pre_weight_acts: dict[str, Float[Tensor, "..."]],
        ci: CIOutputs,
        **_: Any,
    ) -> None:
        mask_infos = make_mask_infos(ci.lower_leaky, weight_deltas_and_masks=None)
        sum_mse, n_examples = _calc_hidden_acts_mse(
            model=self.model,
            batch=batch,
            mask_infos=mask_infos,
            pre_weight_acts=pre_weight_acts,
        )
        self.sum_mse += sum_mse
        self.n_examples += n_examples

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_mse = all_reduce(self.sum_mse, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return _hidden_acts_recon_loss_compute(sum_mse, n_examples)


class PGDHiddenActsReconLoss(Metric):
    """Reconstruction loss between target and component hidden activations with PGD-optimized masks."""

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        use_delta_component: bool,
        pgd_config: PGDConfig,
    ) -> None:
        self.model = model
        self.pgd_config: PGDConfig = pgd_config
        self.use_delta_component: bool = use_delta_component
        self.sum_mse = torch.tensor(0.0, device=device)
        self.n_examples = torch.tensor(0, device=device)

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        pre_weight_acts: dict[str, Float[Tensor, "..."]],
        ci: CIOutputs,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
        **_: Any,
    ) -> None:
        sum_mse, n_examples = pgd_hidden_acts_recon_loss_update(
            model=self.model,
            batch=batch,
            ci=ci.lower_leaky,
            weight_deltas=weight_deltas if self.use_delta_component else None,
            pre_weight_acts=pre_weight_acts,
            pgd_config=self.pgd_config,
        )
        self.sum_mse += sum_mse
        self.n_examples += n_examples

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_mse = all_reduce(self.sum_mse, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return _hidden_acts_recon_loss_compute(sum_mse, n_examples)
