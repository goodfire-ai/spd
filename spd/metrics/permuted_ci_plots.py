from typing import Any, ClassVar, override

from PIL import Image
from torch import Tensor

from spd.configs import SamplingType
from spd.metrics.base import Metric
from spd.models.component_model import ComponentModel
from spd.plotting import plot_causal_importance_vals
from spd.utils.distributed_utils import is_main_process


class PermutedCIPlots(Metric):
    slow: ClassVar[bool] = True
    input_magnitude: ClassVar[float] = 0.75

    metric_section: ClassVar[str] = "figures"

    def __init__(
        self,
        model: ComponentModel,
        sampling: SamplingType,
        identity_patterns: list[str] | None = None,
        dense_patterns: list[str] | None = None,
    ) -> None:
        self.model = model
        self.sampling: SamplingType = sampling
        self.identity_patterns = identity_patterns
        self.dense_patterns = dense_patterns

        self.batch_shape: tuple[int, ...] | None = None

    @override
    def update(self, *, batch: Tensor, **_: Any) -> None:
        if self.batch_shape is None:
            self.batch_shape = tuple(batch.shape)

    @override
    def compute(self) -> dict[str, Image.Image]:
        assert self.batch_shape is not None, "haven't seen any inputs yet"

        if not is_main_process():
            return {}

        figures = plot_causal_importance_vals(
            model=self.model,
            batch_shape=self.batch_shape,
            input_magnitude=self.input_magnitude,
            identity_patterns=self.identity_patterns,
            dense_patterns=self.dense_patterns,
            sampling=self.sampling,
        )[0]

        return {k: v for k, v in figures.items()}
