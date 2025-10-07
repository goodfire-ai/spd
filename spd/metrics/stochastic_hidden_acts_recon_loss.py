from typing import Any, Literal, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.metrics.base import Metric
from spd.models.component_model import ComponentModel
from spd.utils.component_utils import calc_stochastic_component_mask_info
from spd.utils.distributed_utils import all_reduce


def _stochastic_hidden_acts_recon_loss_update(
    model: ComponentModel,
    sampling: Literal["continuous", "binomial"],
    use_delta_component: bool,
    n_mask_samples: int,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    pre_weight_acts: dict[str, Float[Tensor, "..."]],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, " d_out d_in"]],
) -> tuple[Float[Tensor, ""], int]:
    assert ci, "Empty ci"
    assert weight_deltas, "Empty weight deltas"
    assert pre_weight_acts, "Empty pre_weight_acts"
    device = next(iter(ci.values())).device
    sum_mse = torch.tensor(0.0, device=device)
    n_examples = 0

    stoch_mask_infos_list = [
        calc_stochastic_component_mask_info(
            causal_importances=ci,
            component_mask_sampling=sampling,
            weight_deltas_and_mask_sampling=(weight_deltas, "continuous")
            if use_delta_component
            else None,
            routing="all",
        )
        for _ in range(n_mask_samples)
    ]
    for stoch_mask_infos in stoch_mask_infos_list:
        comp_pre_weight_acts = model(batch, mask_infos=stoch_mask_infos, cache_type="input").cache

        # Calculate MSE between pre_weight_acts with and without components
        for layer_name, target_acts in pre_weight_acts.items():
            assert layer_name in comp_pre_weight_acts, f"{layer_name} not in comp_pre_weight_acts"
            mse = torch.nn.functional.mse_loss(
                comp_pre_weight_acts[layer_name], target_acts, reduction="sum"
            )
            sum_mse += mse
            n_examples += target_acts.numel()

    return sum_mse, n_examples


def _stochastic_hidden_acts_recon_loss_compute(
    sum_mse: Float[Tensor, ""], n_examples: Int[Tensor, ""] | int
) -> Float[Tensor, ""]:
    return sum_mse / n_examples


def stochastic_hidden_acts_recon_loss(
    model: ComponentModel,
    sampling: Literal["continuous", "binomial"],
    use_delta_component: bool,
    n_mask_samples: int,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    pre_weight_acts: dict[str, Float[Tensor, "..."]],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, " d_out d_in"]],
) -> Float[Tensor, ""]:
    sum_mse, n_examples = _stochastic_hidden_acts_recon_loss_update(
        model,
        sampling,
        use_delta_component,
        n_mask_samples,
        batch,
        pre_weight_acts,
        ci,
        weight_deltas,
    )
    return _stochastic_hidden_acts_recon_loss_compute(sum_mse, n_examples)


class StochasticHiddenActsReconLoss(Metric):
    """Reconstruction loss between target and stochastic hidden activations when sampling with stochastic masks."""

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        sampling: Literal["continuous", "binomial"],
        use_delta_component: bool,
        n_mask_samples: int,
    ) -> None:
        self.model = model
        self.sampling: Literal["continuous", "binomial"] = sampling
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
        ci: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, " d_out d_in"]],
        **_: Any,
    ) -> None:
        sum_mse, n_examples = _stochastic_hidden_acts_recon_loss_update(
            model=self.model,
            sampling=self.sampling,
            use_delta_component=self.use_delta_component,
            n_mask_samples=self.n_mask_samples,
            batch=batch,
            pre_weight_acts=pre_weight_acts,
            ci=ci,
            weight_deltas=weight_deltas,
        )
        self.sum_mse += sum_mse
        self.n_examples += n_examples

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_mse = all_reduce(self.sum_mse, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return _stochastic_hidden_acts_recon_loss_compute(sum_mse, n_examples)
