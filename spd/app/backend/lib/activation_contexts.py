import heapq
import sys
from collections import Counter, defaultdict
from collections.abc import Generator, Iterable
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.app.backend.api import (
    ActivationContext,
    ModelActivationContexts,
    SubcomponentActivationContexts,
    TokenDensity,
)
from spd.app.backend.services.run_context_service import TrainRunContext
from spd.configs import Config
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import extract_batch_data

DEVICE = get_device()


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


def get_topk_by_subcomponent(
    run_context: TrainRunContext,
    importance_threshold: float,
    max_examples_per_subcomponent: int,
    n_batches: int,
    n_tokens_either_side: int,
    batch_size: int,
) -> tuple[dict[tuple[str, int], _TopKExamples], dict[str, Int[torch.Tensor, " C"]]]:
    # Tracks top-k examples per (module_name, component_idx)
    topk_by_subcomponent = defaultdict[tuple[str, int], _TopKExamples](
        lambda: _TopKExamples(max_examples_per_subcomponent)
    )

    C = run_context.cm.C

    n_toks_seen = 0
    component_sum_cis = defaultdict[str, Float[torch.Tensor, " C"]](lambda: torch.zeros(C).float())

    batches = roll_batch_size_1_into_x(
        singleton_batches=(extract_batch_data(b).to(DEVICE) for b in run_context.train_loader),
        batch_size=batch_size,
    )

    for _ in tqdm(range(n_batches), desc="Harvesting activation contexts", file=sys.stderr):
        batch: Int[torch.Tensor, "B S"] = next(batches)
        assert not batch.requires_grad, "Batch tensors with requires_grad are not supported"
        assert isinstance(batch, torch.Tensor)
        assert batch.ndim == 2, "Expected batch tensor of shape (B, S)"
        B, S = batch.shape

        n_toks_seen += B * S

        importances_by_module = _get_importances_by_module(
            run_context.cm, batch, run_context.config
        )

        for module_name, causal_importances in importances_by_module.items():
            assert causal_importances.shape == (B, S, C), "Expected (B,S,C) per module"

            # Thresholding to find "firings"
            mask = causal_importances > importance_threshold
            if not mask.any():
                continue

            component_sum_cis[module_name] += causal_importances.sum(dim=(0, 1))

            # (K,) indices of all firings
            batch_idx, seq_idx, comp_idx = torch.where(mask)
            (K,) = batch_idx.shape

            # TODO: try me when we're sure this works
            # Sort for cache-friendly iteration
            # order = torch.argsort(batch_idx * S + seq_idx)
            # batch_idx = batch_idx[order]
            # seq_idx = seq_idx[order]
            # comp_idx = comp_idx[order]

            # Iterate across K firings
            for j in range(K):
                b = int(batch_idx[j].item())
                s = int(seq_idx[j].item())
                m = int(comp_idx[j].item())
                key = (module_name, m)

                importance_val = float(causal_importances[b, s, m].item())

                # Build window around the firing position
                start_idx = max(0, s - n_tokens_either_side)
                end_idx = min(S, s + n_tokens_either_side + 1)

                window_token_ids: list[int] = batch[b, start_idx:end_idx].cpu().tolist()
                active_pos_in_window = s - start_idx

                token_ci_values: list[float] = (
                    causal_importances[b, start_idx:end_idx, m].cpu().tolist()
                )

                ex = SubcomponentExample(
                    window_token_ids=window_token_ids,
                    pos=s,
                    active_pos_in_window=active_pos_in_window,
                    token_ci_values=token_ci_values,
                    last_tok_importance=importance_val,
                )

                topk_by_subcomponent[key].maybe_add(ex)
    
    component_mean_cis = {
        module_name: component_sum_cis[module_name] / n_toks_seen
        for module_name in component_sum_cis
    }

    return topk_by_subcomponent, component_mean_cis


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


@dataclass
class ComponentSummary:
    module_name: str
    component_idx: int
    density: float
    examples: list[SubcomponentExample]


def get_subcomponents_activation_contexts(
    run_context: TrainRunContext,
    importance_threshold: float,
    max_examples_per_subcomponent: int,
    n_batches: int,
    n_tokens_either_side: int,
    batch_size: int,
) -> ModelActivationContexts:
    logger.info("Getting activation contexts")

    topk_by_subcomponent, component_mean_cis = get_topk_by_subcomponent(
        run_context,
        importance_threshold,
        max_examples_per_subcomponent,
        n_batches,
        n_tokens_either_side,
        batch_size,
    )

    return map_to_model_ctxs(run_context, topk_by_subcomponent, component_mean_cis)


def map_to_model_ctxs(
    run_context: TrainRunContext,
    topk_by_subcomponent: dict[tuple[str, int], _TopKExamples],
    component_mean_cis: dict[str, Float[torch.Tensor, " C"]],
) -> ModelActivationContexts:
    # use dict of dicts to achieve ≈O(1) access
    # convert to ModelActivationContexts which uses tuple form for persistence
    subcomponent_activation_contexts = defaultdict[str, dict[int, SubcomponentActivationContexts]](dict)
    for module_name, densities in component_mean_cis.items():
        for subcomponent_idx, density in enumerate(densities):
            subcomponent_activation_contexts[module_name][subcomponent_idx] = SubcomponentActivationContexts(
                subcomponent_idx=subcomponent_idx,
                examples=[],
                token_densities=[],
                mean_ci=0.0,
            )

    for (module_name, subcomponent_idx), heap_obj in topk_by_subcomponent.items():
        for ex in heap_obj.to_sorted_list_desc():
            activation_context = ActivationContext(
                token_strings=run_context.tokenizer.convert_ids_to_tokens(ex.window_token_ids),  # pyright: ignore[reportAttributeAccessIssue]
                token_ci_values=ex.token_ci_values,
                active_position=ex.active_pos_in_window,
                ci_value=ex.last_tok_importance,
            )
            subcomponent_activation_contexts[module_name][subcomponent_idx]
            .examples.append(activation_context)

    layers = defaultdict[str, defaultdict[int, list[ActivationContext]]](lambda: defaultdict(list))
    for module_name, densities in component_mean_cis.items():
        for subcomponent_idx, density in enumerate(densities):
            activation_context = ActivationContext(
                token_strings=run_context.tokenizer.convert_ids_to_tokens(ex.window_token_ids),  # pyright: ignore[reportAttributeAccessIssue]
                token_ci_values=ex.token_ci_values,
                active_position=ex.active_pos_in_window,
                ci_value=ex.last_tok_importance,
            )

            layers[module_name][subcomponent_idx].append(activation_context)

    return ModelActivationContexts(layers=layers)