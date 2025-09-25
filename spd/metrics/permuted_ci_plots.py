from typing import Any, override

from PIL import Image
from torch import Tensor
from torchmetrics import Metric

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.plotting import plot_causal_importance_vals


class PermutedCIPlots(Metric):
    slow = True
    is_differentiable: bool | None = False
    full_state_update: bool | None = False  # Avoid double update calls

    input_magnitude: float = 0.75

    def __init__(
        self,
        model: ComponentModel,
        config: Config,
        identity_patterns: list[str] | None = None,
        dense_patterns: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.config = config
        self.model_device = next(iter(model.parameters())).device
        self.identity_patterns = identity_patterns
        self.dense_patterns = dense_patterns

        self.batch_shape: tuple[int, ...] | None = None

        assert config.task_config.task_name != "lm", (
            "PermutedCIPlots currently only works with models that take float inputs (not lms). "
        )

    @override
    def update(self, batch: Tensor, **kwargs: Any) -> None:
        if self.batch_shape is None:
            self.batch_shape = tuple(batch.shape)

    @override
    def compute(self) -> dict[str, Image.Image]:
        assert self.batch_shape is not None, "haven't seen any inputs yet"

        figures = plot_causal_importance_vals(
            model=self.model,
            batch_shape=self.batch_shape,
            input_magnitude=self.input_magnitude,
            sigmoid_type=self.config.sigmoid_type,
            identity_patterns=self.identity_patterns,
            dense_patterns=self.dense_patterns,
            sampling=self.config.sampling,
        )[0]

        return {f"figures/{k}": v for k, v in figures.items()}
