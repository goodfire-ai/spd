from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Literal, override

from jaxtyping import Float
from torch import Tensor


class LayerSelector(ABC):
    @abstractmethod
    def iterate_layer_sets(
        self, ci: dict[str, Float[Tensor, "... C"]], weight_deltas: dict[str, Float[Tensor, "..."]]
    ) -> Iterable[dict[str, Float[Tensor, "... C"]]]: ...

    @abstractmethod
    def get_routing(
        self,
    ) -> Literal["all", "uniform_k-stochastic"]: ...


class LayerwiseSelector(LayerSelector):
    @override
    def iterate_layer_sets(
        self, ci: dict[str, Float[Tensor, "... C"]], weight_deltas: dict[str, Float[Tensor, "..."]]
    ):
        for layer in ci:
            yield {layer: ci[layer]}

    @override
    def get_routing(self):
        return "all"


class SubsetSelector(LayerSelector):
    def __init__(self, n_subsets: int):
        self.n_subsets: int = n_subsets

    @override
    def iterate_layer_sets(
        self,
        ci: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, "..."]],
    ):
        return [ci]

    @override
    def get_routing(self):
        return "uniform_k-stochastic"


class AllSelector(LayerSelector):
    @override
    def iterate_layer_sets(
        self, ci: dict[str, Float[Tensor, "... C"]], weight_deltas: dict[str, Float[Tensor, "..."]]
    ):
        return [ci]

    @override
    def get_routing(self):
        return "all"
