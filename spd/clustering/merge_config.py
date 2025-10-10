import functools
import hashlib
from typing import Any, Literal

from jaxtyping import Float
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
)

from spd.clustering.consts import ClusterCoactivationShaped, MergePair
from torch import Tensor

from spd.clustering.math.merge_pair_samplers import (
    MERGE_PAIR_SAMPLERS,
    MergePairSampler,
    MergePairSamplerKey,
)
from spd.clustering.util import ModuleFilterFunc, ModuleFilterSource
from spd.spd_types import Probability

MergeConfigKey = Literal[
    "activation_threshold",
    "alpha",
    "iters",
    "merge_pair_sampling_method",
    "merge_pair_sampling_kwargs",
    "pop_component_prob",
    "filter_dead_threshold",
]


def _to_module_filter(
    filter_modules: ModuleFilterSource,
) -> ModuleFilterFunc:
    """Convert the filter_modules argument to a callable."""
    if filter_modules is None:
        return lambda _: True
    elif isinstance(filter_modules, str):
        return lambda module_name: module_name.startswith(filter_modules)
    elif isinstance(filter_modules, set):
        return lambda module_name: module_name in filter_modules
    elif callable(filter_modules):
        return filter_modules
    else:
        raise TypeError(f"filter_modules must be str, set, or callable, got {type(filter_modules)}")  # pyright: ignore[reportUnreachable]


class MergeConfig(BaseModel):
    activation_threshold: Probability | None = Field(
        default=0.01,
        description="Threshold for considering a component active in a group. If None, use raw scalar causal importances",
    )
    alpha: float = Field(
        default=1.0,
        description="rank weight factor. Higher values mean a higher penalty on 'sending' the component weights",
    )
    iters: PositiveInt | None = Field(
        default=100,
        description="max number of iterations to run the merge algorithm for. If `None`, set to number of components (after filtering) minus one.",
    )
    merge_pair_sampling_method: MergePairSamplerKey = Field(
        default="range",
        description="Method for sampling merge pairs. Options: 'range', 'mcmc'.",
    )
    merge_pair_sampling_kwargs: dict[str, Any] = Field(
        default_factory=lambda: {"threshold": 0.05},
        description="Keyword arguments for the merge pair sampling method.",
    )
    pop_component_prob: Probability = Field(
        default=0,
        description="Probability of popping a component in each iteration. If 0, no components are popped.",
    )
    filter_dead_threshold: float = Field(
        default=0.001,
        description="Threshold for filtering out dead components. If a component's activation is below this threshold, it is considered dead and not included in the merge.",
    )
    module_name_filter: ModuleFilterSource = Field(
        default=None,
        description="Filter for module names. Can be a string prefix, a set of names, or a callable that returns True for modules to include.",
    )

    @property
    def merge_pair_sample_func(self) -> MergePairSampler:
        return functools.partial(
            MERGE_PAIR_SAMPLERS[self.merge_pair_sampling_method],
            **self.merge_pair_sampling_kwargs,
        )

    def merge_pair_sample(
        self,
        costs: ClusterCoactivationShaped,
    ) -> MergePair:
        """do merge sampling based on the configured method and kwargs

        has signature `MergePairSampler = Callable[[ClusterCoactivationShaped], MergePair]`
        """
        return self.merge_pair_sample_func(costs=costs)

    @property
    def filter_modules(self) -> ModuleFilterFunc:
        """Get the module filter function based on the provided source."""
        return _to_module_filter(self.module_name_filter)

<<<<<<< HEAD
    def get_num_iters(self, n_components: int) -> PositiveInt:
        """Get the number of iterations to run the merge algorithm for.

        Args:
            n_components: Number of components (after filtering)

        Returns:
            Number of iterations to run
        """
        if self.iters is None:
            return n_components - 1
        else:
            return self.iters

    @property
    def stable_hash(self) -> str:
        return hashlib.md5(self.model_dump_json().encode()).hexdigest()[:6]
=======
    @property
    def stable_hash(self) -> str:
        return hashlib.md5(self.model_dump_json().encode()).hexdigest()[:6]

    @property
    def config_identifier(self) -> str:
        """Unique identifier for this specific config on this specific model.

        Format: model_abc123-a0.1-i1k-b64-n10-h_12ab
        Allows filtering in WandB for all runs with this exact config and model.
        """
        return f"a{self.alpha:g}-h_{self.stable_hash}"
>>>>>>> chinyemba/feature/clustering-sjcs
