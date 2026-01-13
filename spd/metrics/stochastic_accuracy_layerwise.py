"""Accuracy metric when stochastically ablating each layer.

Similar to StochasticReconLayerwiseLoss but computes classification accuracy
instead of reconstruction loss. Useful for classification tasks like MNIST.
"""

from typing import Any, ClassVar, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import SamplingType
from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.routing import AllLayersRouter
from spd.utils.component_utils import calc_stochastic_component_mask_info
from spd.utils.distributed_utils import all_reduce


class StochasticAccuracyLayerwise(Metric):
    """Accuracy when sampling with stochastic masks one layer at a time.

    For each layer, stochastically ablates components using the causal importance
    and computes the classification accuracy. Reports accuracy for each layer separately.
    """

    slow: ClassVar[bool] = True
    metric_section: ClassVar[str] = "accuracy"

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

        # Track correct predictions and total samples per layer
        self.correct_per_layer: dict[str, torch.Tensor] = {}
        self.total_per_layer: dict[str, torch.Tensor] = {}
        self.device = device

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: CIOutputs,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
        **_: Any,
    ) -> None:
        # Get target predictions (from the original model output)
        target_preds = target_out.argmax(dim=-1)  # Shape: (batch,)

        stochastic_mask_infos_list = [
            calc_stochastic_component_mask_info(
                causal_importances=ci.lower_leaky,
                component_mask_sampling=self.sampling,
                weight_deltas=weight_deltas if self.use_delta_component else None,
                router=AllLayersRouter(),
            )
            for _ in range(self.n_mask_samples)
        ]

        for stochastic_mask_infos in stochastic_mask_infos_list:
            for module_name, mask_info in stochastic_mask_infos.items():
                # Forward pass with this layer ablated
                out = self.model(batch, mask_infos={module_name: mask_info})
                preds = out.argmax(dim=-1)

                # Count correct predictions
                correct = (preds == target_preds).sum()

                # Initialize counters for this layer if needed
                if module_name not in self.correct_per_layer:
                    self.correct_per_layer[module_name] = torch.tensor(
                        0, device=self.device, dtype=torch.long
                    )
                    self.total_per_layer[module_name] = torch.tensor(
                        0, device=self.device, dtype=torch.long
                    )

                self.correct_per_layer[module_name] += correct
                self.total_per_layer[module_name] += target_preds.numel()

    @override
    def compute(self) -> dict[str, Float[Tensor, ""]]:
        results: dict[str, Float[Tensor, ""]] = {}

        for module_name in self.correct_per_layer:
            correct = all_reduce(self.correct_per_layer[module_name], op=ReduceOp.SUM)
            total = all_reduce(self.total_per_layer[module_name], op=ReduceOp.SUM)
            accuracy = correct.float() / total.float()
            results[f"stochastic_ablate_{module_name}"] = accuracy

        return results
