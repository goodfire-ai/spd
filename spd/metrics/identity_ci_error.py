from typing import Literal, override

from torch import Tensor

from spd.metrics.base import MetricInterface
from spd.models.component_model import ComponentModel
from spd.models.sigmoids import SigmoidTypes
from spd.plotting import get_single_feature_causal_importances
from spd.utils.target_ci_solutions import compute_target_metrics, make_target_ci_solution


class IdentityCIError(MetricInterface):
    """Error between the CI values and an Identity or Dense CI pattern."""

    input_magnitude: float = 0.75

    def __init__(
        self,
        model: ComponentModel,
        sampling: Literal["continuous", "binomial"],
        sigmoid_type: SigmoidTypes,
        identity_ci: list[dict[str, str | int]] | None = None,
        dense_ci: list[dict[str, str | int]] | None = None,
    ) -> None:
        self.model = model
        self.sampling: Literal["continuous", "binomial"] = sampling
        self.sigmoid_type: SigmoidTypes = sigmoid_type
        self.identity_ci = identity_ci
        self.dense_ci = dense_ci

        self.batch_shape: tuple[int, ...] | None = None

    @override
    def update(
        self,
        batch: Tensor,
        target_out: Tensor,
        ci: dict[str, Tensor],
        current_frac_of_training: float,
        ci_upper_leaky: dict[str, Tensor],
        weight_deltas: dict[str, Tensor],
    ) -> None:
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
            input_magnitude=self.input_magnitude,
            sampling=self.sampling,
            sigmoid_type=self.sigmoid_type,
        )

        target_metrics = compute_target_metrics(
            causal_importances=ci_arrays, target_solution=target_solution
        )
        return target_metrics
