from collections import defaultdict
from typing import Any, ClassVar, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import SamplingType
from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.utils.component_utils import calc_stochastic_component_mask_info
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import get_obj_device, zip_dicts
from spd.utils.module_utils import get_target_module_paths


def _stochastic_arb_hidden_acts_recon_loss_update(
    model: ComponentModel,
    sampling: SamplingType,
    n_mask_samples: int,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, " d_out d_in"]] | None,
    output_target_module_paths: list[str],
) -> dict[str, tuple[Float[Tensor, ""], int]]:
    assert ci, "Empty ci"
    device = get_obj_device(ci)

    target_output_acts = model(
        batch,
        cache_type="output",
        output_target_module_paths=output_target_module_paths,
    ).cache

    stoch_mask_infos_list = [
        calc_stochastic_component_mask_info(
            causal_importances=ci,
            component_mask_sampling=sampling,
            weight_deltas=weight_deltas,
            routing="all",
        )
        for _ in range(n_mask_samples)
    ]

    sum_mse = defaultdict[str, Float[Tensor, ""]](lambda: torch.tensor(0.0, device=device))
    n_examples = defaultdict[str, int](int)

    for stoch_mask_infos in stoch_mask_infos_list:
        stoch_output_acts = model(
            batch,
            cache_type="output",
            mask_infos=stoch_mask_infos,
            output_target_module_paths=output_target_module_paths,
        ).cache

        assert (
            set(stoch_output_acts) == set(output_target_module_paths) == set(target_output_acts)
        ), (
            f"{set(stoch_output_acts)=}\n{set(output_target_module_paths)=}\n{set(target_output_acts)}"
        )

        for module_name, (target_act, stoch_act) in zip_dicts(
            target_output_acts, stoch_output_acts
        ).items():
            mse = torch.nn.functional.mse_loss(target_act, stoch_act, reduction="sum")
            assert target_act.shape == stoch_act.shape

            sum_mse[module_name] += mse
            n_examples[module_name] += target_act.numel()

    return zip_dicts(sum_mse, n_examples)


def _stochastic_arb_hidden_acts_recon_loss_compute(
    results: dict[str, tuple[Float[Tensor, ""], int]],
) -> dict[str, Float[Tensor, ""]]:
    return {mod: (sum / n) for mod, (sum, n) in results.items()}


def stochastic_arb_hidden_acts_recon_loss(
    model: ComponentModel,
    sampling: SamplingType,
    n_mask_samples: int,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    ci: dict[str, Float[Tensor, "... C"]],
    output_target_module_patterns: list[str],
    weight_deltas: dict[str, Float[Tensor, " d_out d_in"]] | None,
) -> dict[str, Float[Tensor, ""]]:
    results = _stochastic_arb_hidden_acts_recon_loss_update(
        model=model,
        sampling=sampling,
        n_mask_samples=n_mask_samples,
        batch=batch,
        ci=ci,
        output_target_module_paths=get_target_module_paths(model.target_model, output_target_module_patterns),
        weight_deltas=weight_deltas,
    )
    return _stochastic_arb_hidden_acts_recon_loss_compute(results)


class StochasticArbHiddenActsReconLoss(Metric):
    """Reconstruction loss between target and stochastic hidden activations when sampling with stochastic masks."""

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        sampling: SamplingType,
        use_delta_component: bool,
        n_mask_samples: int,
        output_target_module_patterns: list[str],
    ) -> None:
        self.model = model
        self.sampling: SamplingType = sampling
        self.use_delta_component: bool = use_delta_component
        self.n_mask_samples: int = n_mask_samples
        self.output_target_module_paths: list[str] = get_target_module_paths(model.target_model, output_target_module_patterns)

        self.sum_mse = defaultdict[str, Float[Tensor, ""]](lambda: torch.tensor(0.0, device=device))
        self.n_examples = defaultdict[str, Int[Tensor, ""]](
            lambda: torch.tensor(0, device=device, dtype=torch.long)
        )

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        pre_weight_acts: dict[str, Float[Tensor, "..."]],
        ci: CIOutputs,
        weight_deltas: dict[str, Float[Tensor, " d_out d_in"]],
        **_: Any,
    ) -> None:
        results = _stochastic_arb_hidden_acts_recon_loss_update(
            model=self.model,
            sampling=self.sampling,
            n_mask_samples=self.n_mask_samples,
            batch=batch,
            ci=ci.lower_leaky,
            weight_deltas=weight_deltas if self.use_delta_component else None,
            output_target_module_paths=self.output_target_module_paths,
        )
        for module_name, (sum_mse, n_examples) in results.items():
            self.sum_mse[module_name] += sum_mse
            self.n_examples[module_name] += n_examples

    @override
    def compute(self) -> dict[str, Float[Tensor, ""]]:
        reduced = {}
        for module_name, (sum_mse, n_examples) in zip_dicts(self.sum_mse, self.n_examples).items():
            sum_mse = all_reduce(sum_mse, op=ReduceOp.SUM)
            n_examples = int(all_reduce(n_examples, op=ReduceOp.SUM).item())
            reduced[module_name] = (sum_mse, n_examples)

        return _stochastic_arb_hidden_acts_recon_loss_compute(reduced)
