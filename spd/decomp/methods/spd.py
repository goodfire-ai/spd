"""SPD-specific activation functions for the generic decomposition pipeline."""

from collections.abc import Callable

import torch
from torch import Tensor

from spd.configs import SamplingType
from spd.harvest.schemas import HarvestBatch
from spd.models.component_model import ComponentModel


def make_spd_harvest_fn(
    model: ComponentModel,
    sampling: SamplingType,
    ci_threshold: float = 0.0,
) -> Callable[[Tensor], HarvestBatch]:
    def _harvest_fn(batch: Tensor) -> HarvestBatch:
        out = model(batch, cache_type="input")
        ci_dict = model.calc_causal_importances(
            pre_weight_acts=out.cache,
            detach_inputs=True,
            sampling=sampling,
        ).lower_leaky

        firings = {layer: ci > ci_threshold for layer, ci in ci_dict.items()}
        activations = {layer: {"causal_importance": ci_dict[layer]} for layer in ci_dict}

        return HarvestBatch(
            tokens=batch,
            firings=firings,
            activations=activations,
            output_probs=torch.softmax(out.output, dim=-1),
        )

    return _harvest_fn
