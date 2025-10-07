from typing import Any, ClassVar, override

from PIL import Image
from torch import Tensor

from spd.metrics.base import Metric
from spd.models.component_model import ComponentModel
from spd.models.sigmoids import SigmoidTypes
from spd.plotting import plot_causal_importance_vals


class PermutedCIPlots(Metric):
    slow: ClassVar[bool] = True
    input_magnitude: ClassVar[float] = 0.75

    def __init__(
        self,
        model: ComponentModel,
        do_binomial_fuzz: bool,
        sigmoid_type: SigmoidTypes,
        identity_patterns: list[str] | None = None,
        dense_patterns: list[str] | None = None,
    ) -> None:
        self.model = model
        self.do_binomial_fuzz: bool = do_binomial_fuzz
        self.sigmoid_type: SigmoidTypes = sigmoid_type
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

        figures = plot_causal_importance_vals(
            model=self.model,
            batch_shape=self.batch_shape,
            input_magnitude=self.input_magnitude,
            sigmoid_type=self.sigmoid_type,
            identity_patterns=self.identity_patterns,
            dense_patterns=self.dense_patterns,
            do_binomial_fuzz=self.do_binomial_fuzz,
        )[0]

        return {f"figures/{k}": v for k, v in figures.items()}
