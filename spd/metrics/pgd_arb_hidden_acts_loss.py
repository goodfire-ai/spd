from typing import Any, ClassVar, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import PGDConfig
from spd.metrics.base import Metric
from spd.metrics.pgd_utils import pgd_masked_hidden_acts_recon_loss_update
from spd.models.component_model import CIOutputs, ComponentModel
from spd.utils.distributed_utils import all_reduce


def _pgd_arb_hidden_acts_loss_update(
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, " d_out d_in"]] | None,
    post_target_module_path: str,
    pgd_config: PGDConfig,
) -> tuple[Float[Tensor, ""], int]:
    return pgd_masked_hidden_acts_recon_loss_update(
        model=model,
        batch=batch,
        ci=ci,
        weight_deltas=weight_deltas,
        post_target_module_path=post_target_module_path,
        pgd_config=pgd_config,
    )


def _pgd_arb_hidden_acts_loss_compute(
    sum_mse: Float[Tensor, ""], n_examples: int
) -> Float[Tensor, ""]:
    return sum_mse / n_examples


def pgd_arb_hidden_acts_recon_loss(
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    ci: dict[str, Float[Tensor, "... C"]],
    post_target_module_path: str,
    weight_deltas: dict[str, Float[Tensor, " d_out d_in"]] | None,
    pgd_config: PGDConfig,
) -> Float[Tensor, ""]:
    sum_mse, n_examples = _pgd_arb_hidden_acts_loss_update(
        model=model,
        batch=batch,
        ci=ci,
        weight_deltas=weight_deltas,
        post_target_module_path=post_target_module_path,
        pgd_config=pgd_config,
    )
    return _pgd_arb_hidden_acts_loss_compute(sum_mse, n_examples)


class PGDArbHiddenActsReconLoss(Metric):
    """Arbitrary hidden acts reconstruction loss when masking with adversarially-optimized values."""

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        use_delta_component: bool,
        post_target_module_path: str,
        pgd_config: PGDConfig,
    ) -> None:
        self.model = model
        self.use_delta_component: bool = use_delta_component
        self.post_target_module_path: str = post_target_module_path
        self.pgd_config: PGDConfig = pgd_config

        self.sum_mse = torch.tensor(0.0, device=device)
        self.n_examples = torch.tensor(0, device=device, dtype=torch.long)

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        ci: CIOutputs,
        weight_deltas: dict[str, Float[Tensor, " d_out d_in"]],
        **_: Any,
    ) -> None:
        sum_mse, n_examples = _pgd_arb_hidden_acts_loss_update(
            model=self.model,
            batch=batch,
            ci=ci.lower_leaky,
            weight_deltas=weight_deltas if self.use_delta_component else None,
            post_target_module_path=self.post_target_module_path,
            pgd_config=self.pgd_config,
        )
        self.sum_mse += sum_mse
        self.n_examples += n_examples

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_mse = all_reduce(self.sum_mse, op=ReduceOp.SUM)
        n_examples = int(all_reduce(self.n_examples, op=ReduceOp.SUM).item())
        return _pgd_arb_hidden_acts_loss_compute(sum_mse, n_examples)
