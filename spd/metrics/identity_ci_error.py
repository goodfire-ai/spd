from typing import Any, override

from torch import Tensor
from torchmetrics import Metric

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.plotting import get_single_feature_causal_importances
from spd.utils.target_ci_solutions import compute_target_metrics, make_target_ci_solution


class IdentityCIError(Metric):
    """Error between the CI values and an Identity or Dense CI pattern."""

    slow = True
    is_differentiable: bool | None = False
    full_state_update: bool | None = False  # Avoid double update calls

    def __init__(
        self,
        model: ComponentModel,
        config: Config,
        identity_ci: list[dict[str, str | int]] | None = None,
        dense_ci: list[dict[str, str | int]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.config = config
        self.identity_ci = identity_ci
        self.dense_ci = dense_ci

        self.batch_shape: tuple[int, ...] | None = None

        assert config.task_config.task_name != "lm", (
            "IdentityCIError currently only works with models that take float inputs (not lms). "
        )

    @override
    def update(self, batch: Tensor, **kwargs: Any) -> None:
        if self.batch_shape is None:
            self.batch_shape = tuple(batch.shape)

    @override
    def compute(self) -> dict[str, float]:
        assert self.batch_shape is not None, "haven't seen any inputs yet"

        target_solution = make_target_ci_solution(
            identity_ci=self.identity_ci, dense_ci=self.dense_ci
        )
        if target_solution is None:
            return {}

        ci_arrays, _ = get_single_feature_causal_importances(
            model=self.model,
            batch_shape=self.batch_shape,
            input_magnitude=0.75,
            sampling=self.config.sampling,
            sigmoid_type=self.config.sigmoid_type,
        )

        target_metrics = compute_target_metrics(
            causal_importances=ci_arrays, target_solution=target_solution
        )
        return target_metrics
