from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Literal, override

from jaxtyping import Float, Int
from torch import Tensor

from spd.metrics.pgd_utils import PGDInitStrategy, create_pgd_masks
from spd.models.component_model import ComponentModel
from spd.models.components import ComponentsMaskInfo, make_mask_infos
from spd.utils.component_utils import (
    calc_stochastic_component_mask_info,
    sample_uniform_k_subset_routing_masks,
)


class Masker(ABC):
    @abstractmethod
    def sample_mask_infos(
        self,
        model: ComponentModel,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        ci: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, "..."]],
        target_out: Float[Tensor, "... vocab"],
        routing: Literal["all", "uniform_k-stochastic"],
    ) -> Iterable[dict[str, ComponentsMaskInfo]]: ...


class CIMasker(Masker):
    @override
    def sample_mask_infos(
        self,
        model: ComponentModel,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        ci: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, "..."]],
        target_out: Float[Tensor, "... vocab"],
        routing: Literal["all", "uniform_k-stochastic"],
        # routing_mask perhaps?
    ) -> Iterable[dict[str, ComponentsMaskInfo]]:
        if routing == "uniform_k-stochastic":
            routing_masks = sample_uniform_k_subset_routing_masks(
                mask_shape=next(iter(ci.values())).shape[:-1],
                module_names=list(ci.keys()),
                device=batch.device,
            )
        else:
            routing_masks = "all"

        yield make_mask_infos(ci, weight_deltas_and_masks=None, routing_masks=routing_masks)


class StochasticMaskSampler(Masker):
    def __init__(
        self,
        use_delta_component: bool,
        sampling: Literal["continuous", "binomial"],
        n_mask_samples: int,
    ):
        self.use_delta_component: bool = use_delta_component
        self.sampling: Literal["continuous", "binomial"] = sampling
        self.n_mask_samples: int = n_mask_samples

    @override
    def sample_mask_infos(
        self,
        model: ComponentModel,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        ci: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, "..."]],
        target_out: Float[Tensor, "... vocab"],
        routing: Literal["all", "uniform_k-stochastic"],
    ) -> Iterable[dict[str, ComponentsMaskInfo]]:
        for _ in range(self.n_mask_samples):
            weight_deltas_and_mask_sampling = (
                (weight_deltas, "continuous") if self.use_delta_component else None
            )
            yield calc_stochastic_component_mask_info(
                causal_importances=ci,
                component_mask_sampling=self.sampling,
                weight_deltas_and_mask_sampling=weight_deltas_and_mask_sampling,
                routing=routing,
            )


class PGDMasker(Masker):
    def __init__(
        self,
        init: PGDInitStrategy,
        step_size: float,
        n_steps: int,
        output_loss_type: Literal["mse", "kl"],
        n_mask_samples: int,
    ):
        self.init: PGDInitStrategy = init
        self.step_size: float = step_size
        self.n_steps: int = n_steps
        self.output_loss_type: Literal["mse", "kl"] = output_loss_type
        self.n_mask_samples: int = n_mask_samples

    @override
    def sample_mask_infos(
        self,
        model: ComponentModel,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        ci: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, "..."]],
        target_out: Float[Tensor, "... vocab"],
        routing: Literal["all", "uniform_k-stochastic"],
    ) -> Iterable[dict[str, ComponentsMaskInfo]]:
        for _ in range(self.n_mask_samples):
            component_masks, weight_deltas_and_mask_sampling = create_pgd_masks(
                model=model,
                batch=batch,
                init=self.init,
                step_size=self.step_size,
                n_steps=self.n_steps,
                ci=ci,
                weight_deltas={k: v for k, v in weight_deltas.items() if k in ci},  # HACK
                target_out=target_out,
                output_loss_type=self.output_loss_type,
                routing=routing,
            )

            mask_infos = calc_stochastic_component_mask_info(
                causal_importances=ci,
                component_mask_sampling=("given", component_masks),
                weight_deltas_and_mask_sampling=weight_deltas_and_mask_sampling,
                routing=routing,
            )

            yield mask_infos
