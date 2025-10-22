from collections import defaultdict
from typing import Any, ClassVar, Literal, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import PGDConfig, SamplingType
from spd.metrics.base import Metric
from spd.metrics.pgd_utils import pgd_masked_hidden_acts_recon_loss_update
from spd.models.component_model import CIOutputs, ComponentModel
from spd.utils.component_utils import calc_stochastic_component_mask_info
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import get_obj_device, zip_dicts
from spd.utils.module_utils import get_target_module_paths

CachePoint = tuple[str, Literal["pre", "post"]]


def _pgd_arb_hidden_acts_loss_update(
    model: ComponentModel,
    sampling: SamplingType,
    n_mask_samples: int,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, " d_out d_in"]] | None,
    post_target_module_paths: list[str],
    pre_target_module_paths: list[str],
    pgd_config: PGDConfig
) -> dict[CachePoint, tuple[Float[Tensor, ""], int]]:
    return pgd_masked_hidden_acts_recon_loss_update(
        model=model,
        batch=batch,
        ci=ci,
        weight_deltas=weight_deltas,
        target_out=target_out,
        output_loss_type=output_loss_type,
        pre_target_module_paths=pre_target_module_paths,
        post_target_module_paths=post_target_module_paths,
    )

    assert pre_target_module_paths or post_target_module_paths

    device = get_obj_device(ci)

    target_pre_acts = model(
        batch,
        cache_type="input",
        cache_points=pre_target_module_paths,
    ).cache

    target_post_acts = model(
        batch,
        cache_type="output",
        cache_points=post_target_module_paths,
    ).cache

    sum_mse = defaultdict[CachePoint, Float[Tensor, ""]](lambda: torch.tensor(0.0, device=device))
    n_examples = defaultdict[CachePoint, int](int)

    for _ in range(n_mask_samples):
        stoch_mask_infos = calc_stochastic_component_mask_info(
            causal_importances=ci,
            component_mask_sampling=sampling,
            weight_deltas=weight_deltas,
            routing="all",
        )

        # Pre
        stoch_pre_acts = model(
            batch,
            cache_type="input",
            mask_infos=stoch_mask_infos,
            cache_points=pre_target_module_paths,
        ).cache

        assert set(stoch_pre_acts) == set(pre_target_module_paths) == set(target_pre_acts), (
            f"{set(stoch_pre_acts)=}\n{set(pre_target_module_paths)=}\n{set(target_pre_acts)}"
        )

        for module_name, (target_act, stoch_act) in zip_dicts(
            target_pre_acts, stoch_pre_acts
        ).items():
            mse = torch.nn.functional.mse_loss(target_act, stoch_act, reduction="sum")
            assert target_act.shape == stoch_act.shape

            sum_mse[(module_name, "pre")] += mse
            n_examples[(module_name, "pre")] += target_act.numel()

        # Post
        stoch_post_acts = model(
            batch,
            cache_type="output",
            mask_infos=stoch_mask_infos,
            cache_points=post_target_module_paths,
        ).cache

        assert set(stoch_post_acts) == set(post_target_module_paths) == set(target_post_acts), (
            f"{set(stoch_post_acts)=}\n{set(post_target_module_paths)=}\n{set(target_post_acts)}"
        )

        for module_name, (target_act, stoch_act) in zip_dicts(
            target_post_acts, stoch_post_acts
        ).items():
            mse = torch.nn.functional.mse_loss(target_act, stoch_act, reduction="sum")
            assert target_act.shape == stoch_act.shape

            sum_mse[(module_name, "post")] += mse
            n_examples[(module_name, "post")] += target_act.numel()

    return zip_dicts(sum_mse, n_examples)


def _stochastic_arb_hidden_acts_recon_loss_compute(
    results: dict[CachePoint, tuple[Float[Tensor, ""], int]],
) -> dict[str, Float[Tensor, ""]]:
    return {
        f"recon_{layer_name}_{point}": (sum / n)
        for (layer_name, point), (sum, n) in results.items()
    }


def stochastic_arb_hidden_acts_recon_loss(
    model: ComponentModel,
    sampling: SamplingType,
    n_mask_samples: int,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    ci: dict[str, Float[Tensor, "... C"]],
    post_target_module_patterns: list[str] | None,
    pre_target_module_patterns: list[str] | None,
    weight_deltas: dict[str, Float[Tensor, " d_out d_in"]] | None,
) -> dict[str, Float[Tensor, ""]]:
    results = _stochastic_arb_hidden_acts_recon_loss_update(
        model=model,
        sampling=sampling,
        n_mask_samples=n_mask_samples,
        batch=batch,
        ci=ci,
        post_target_module_paths=get_target_module_paths(
            model.target_model, post_target_module_patterns or []
        ),
        pre_target_module_paths=get_target_module_paths(
            model.target_model, pre_target_module_patterns or []
        ),
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
        pre_target_module_patterns: list[str] | None,
        post_target_module_patterns: list[str] | None,
    ) -> None:
        self.model = model
        self.sampling: SamplingType = sampling
        self.use_delta_component: bool = use_delta_component
        self.n_mask_samples: int = n_mask_samples
        self.pre_target_module_paths: list[str] = get_target_module_paths(
            model.target_model, pre_target_module_patterns or []
        )
        self.post_target_module_paths: list[str] = get_target_module_paths(
            model.target_model, post_target_module_patterns or []
        )

        self.sum_mse = defaultdict[CachePoint, Float[Tensor, ""]](
            lambda: torch.tensor(0.0, device=device)
        )
        self.n_examples = defaultdict[CachePoint, Int[Tensor, ""]](
            lambda: torch.tensor(0, device=device, dtype=torch.long)
        )

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
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
            pre_target_module_paths=self.pre_target_module_paths,
            post_target_module_paths=self.post_target_module_paths,
        )
        for cache_point, (sum_mse, n_examples) in results.items():
            self.sum_mse[cache_point] += sum_mse
            self.n_examples[cache_point] += n_examples

    @override
    def compute(self) -> dict[str, Float[Tensor, ""]]:
        reduced = {}
        for cache_point, (sum_mse, n_examples) in zip_dicts(self.sum_mse, self.n_examples).items():
            sum_mse = all_reduce(sum_mse, op=ReduceOp.SUM)
            n_examples = int(all_reduce(n_examples, op=ReduceOp.SUM).item())
            reduced[cache_point] = (sum_mse, n_examples)

        return _stochastic_arb_hidden_acts_recon_loss_compute(reduced)
