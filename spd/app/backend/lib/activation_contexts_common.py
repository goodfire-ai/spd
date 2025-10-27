import heapq
from collections.abc import Generator, Iterable
from dataclasses import dataclass

import torch
from jaxtyping import Float

from spd.configs import Config
from spd.models.component_model import ComponentModel


@dataclass
class SubcomponentExample:
    window_token_ids: list[int]
    """Windowed tokens around the firing position"""
    pos: int
    """Absolute position within the original sequence"""
    active_pos_in_window: int
    """Position within window_token_ids corresponding to pos"""
    token_ci_values: list[float]
    """CI values aligned to window_token_ids"""
    last_tok_importance: float
    """CI value at the firing position"""


class _TopKExamples:
    def __init__(self, k: int):
        self.k = k
        # Min-heap of tuples (importance, counter, example)
        self.heap: list[tuple[float, int, SubcomponentExample]] = []
        self._counter: int = 0

    def maybe_add(self, example: SubcomponentExample) -> None:
        key = (example.last_tok_importance, self._counter, example)
        self._counter += 1
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, key)
            return
        # Heap full: replace min if better
        if self.heap[0][0] < example.last_tok_importance:
            heapq.heapreplace(self.heap, key)

    def to_sorted_list_desc(self) -> list[SubcomponentExample]:
        # Return examples sorted by importance descending
        return [ex for _, _, ex in sorted(self.heap, key=lambda t: t[0], reverse=True)]


def roll_batch_size_1_into_x(
    singleton_batches: Iterable[torch.Tensor],
    batch_size: int,
) -> Generator[torch.Tensor]:
    examples = []
    for batch in singleton_batches:
        assert batch.shape[0] == 1, "Batch size must be 1"
        examples.append(batch[0])
        if len(examples) == batch_size:
            yield torch.stack(examples)
            examples = []
    if examples:
        yield torch.stack(examples)


@dataclass
class ActivationsData:
    examples: dict[str, dict[int, _TopKExamples]]
    component_token_densities: dict[str, dict[int, list[tuple[str, float]]]]
    component_mean_cis: dict[str, Float[torch.Tensor, " C"]]


TOPK_EXAMPLES = 100


def _get_importances_by_module(
    cm: ComponentModel, batch: torch.Tensor, config: Config
) -> dict[str, Float[torch.Tensor, "B S C"]]:
    """returns a dictionary of module names to causal importances, where the shape is (B, S, C)"""

    with torch.no_grad():
        pre_weight_acts = cm(
            batch,
            cache_type="input",
            module_names=list(cm.components.keys()),
        ).cache
        importances_by_module = cm.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            detach_inputs=True,
            sampling=config.sampling,
        ).lower_leaky
    return importances_by_module
