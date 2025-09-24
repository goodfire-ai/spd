from typing import Any, override

from PIL import Image
from torch import Tensor
from torchmetrics import Metric

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.plotting import plot_causal_importance_vals, plot_UV_matrices


class UVPlots(Metric):
    slow = True
    is_differentiable: bool | None = False

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
        self.identity_patterns = identity_patterns
        self.dense_patterns = dense_patterns

        self.batch_shape: tuple[int, ...] | None = None

        assert config.task_config.task_name != "lm", (
            "UVPlots currently only works with models that take float inputs (not lms). "
        )

    @override
    def update(self, batch: Tensor, **kwargs: Any) -> None:
        if self.batch_shape is None:
            self.batch_shape = tuple(batch.shape)

    @override
    def compute(self) -> dict[str, Image.Image]:
        assert self.batch_shape is not None, "haven't seen any inputs yet"

        all_perm_indices = plot_causal_importance_vals(
            model=self.model,
            batch_shape=self.batch_shape,
            input_magnitude=0.75,
            sigmoid_type=self.config.sigmoid_type,
            identity_patterns=self.identity_patterns,
            dense_patterns=self.dense_patterns,
            sampling=self.config.sampling,
        )[1]

        uv_matrices = plot_UV_matrices(
            components=self.model.components, all_perm_indices=all_perm_indices
        )

        return {"figures/uv_matrices": uv_matrices}
