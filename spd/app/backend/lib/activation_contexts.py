import heapq
from collections import defaultdict
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int
from transformers import PreTrainedTokenizer

from spd.app.backend.schemas import (
    ActivationContext,
    ModelActivationContexts,
    SubcomponentActivationContexts,
    TokenPR,
)
from spd.app.backend.services.run_context_service import TrainRunContext
from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.utils.general_utils import extract_batch_data


@dataclass
class SubcomponentExample:
    window_token_ids: list[int]
    """Windowed tokens around the firing position"""
    active_pos_in_window: int
    """Position within window_token_ids corresponding to pos"""
    token_ci_values: list[float]
    """CI values aligned to window_token_ids"""
    active_pos_importance: float
    """CI value at the firing position"""


class _TopKExamples:
    def __init__(self, k: int):
        self.k = k
        # Min-heap of tuples (importance, counter, example)
        self.heap: list[tuple[float, int, SubcomponentExample]] = []
        self._counter: int = 0

    def maybe_add(self, example: SubcomponentExample) -> None:
        key = (example.active_pos_importance, self._counter, example)
        self._counter += 1
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, key)
            return
        # Heap full: replace min if better
        if self.heap[0][0] < example.active_pos_importance:
            heapq.heapreplace(self.heap, key)

    def as_activation_contexts(self, tok: PreTrainedTokenizer) -> list[ActivationContext]:
        return [
            ActivationContext(
                token_strings=tok.convert_ids_to_tokens(ex.window_token_ids),  # pyright: ignore[reportAttributeAccessIssue]
                token_ci_values=ex.token_ci_values,
                active_position=ex.active_pos_in_window,
                ci_value=ex.active_pos_importance,
            )
            for _, _, ex in sorted(self.heap, key=lambda t: t[0], reverse=True)
        ]


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


PAD_VALUE = 0


def get_activations_data_streaming(
    run_context: TrainRunContext,
    importance_threshold: float,
    n_batches: int,
    n_tokens_either_side: int,
    batch_size: int,
    topk_examples: int,
) -> Generator[
    tuple[Literal["progress"], int] | tuple[Literal["complete"], ModelActivationContexts],
]:
    device = next(run_context.cm.parameters()).device

    # for each (module_name, component_idx), track:
    # - the top-k activating examples
    examples = defaultdict[str, defaultdict[int, _TopKExamples]](
        lambda: defaultdict(lambda: _TopKExamples(k=topk_examples))
    )
    # - the number of activations for each component
    component_activation_counts = defaultdict[str, defaultdict[int, int]](lambda: defaultdict(int))
    # - and the number of times each token activates for each component
    component_activation_tokens = defaultdict[str, defaultdict[int, dict[int, int]]](
        lambda: defaultdict(lambda: defaultdict(int))
    )
    # - the sum of causal importances
    C = run_context.cm.C
    component_sum_cis = defaultdict[str, Float[torch.Tensor, " C"]](
        lambda: torch.zeros(C, device=device, dtype=torch.float)
    )

    # also track total occurrences of each token across all batches
    total_token_counts = defaultdict[int, int](int)
    n_toks_seen = 0

    pad_token_id = int(getattr(run_context.tokenizer, "pad_token_id", 0))

    batches = roll_batch_size_1_into_x(
        singleton_batches=(extract_batch_data(b).to(device) for b in run_context.train_loader),
        batch_size=batch_size,
    )

    for i in range(n_batches):
        batch: Int[torch.Tensor, "B S"] = next(batches)
        assert not batch.requires_grad, "Batch tensors with requires_grad are not supported"
        assert isinstance(batch, torch.Tensor)
        assert batch.ndim == 2, "Expected batch tensor of shape (B, S)"
        B, S = batch.shape

        n_toks_seen += B * S

        # Count all token occurrences in batch for precision calculation
        token_ids_in_batch, counts = batch.flatten().unique(return_counts=True)
        for token_id, count in zip(token_ids_in_batch.tolist(), counts.tolist(), strict=True):
            total_token_counts[token_id] += count

        importances_by_module = _get_importances_by_module(
            run_context.cm, batch, run_context.config
        )

        for module_name, ci in importances_by_module.items():
            assert ci.shape == (B, S, C), "Expected (B,S,C) per module"

            component_sum_cis[module_name] += ci.sum(dim=(0, 1))

            # Thresholding to find "firings"
            mask = ci > importance_threshold
            if not mask.any():
                continue

            batch_idx: Int[torch.Tensor, " n_firings"]
            seq_idx: Int[torch.Tensor, " n_firings"]
            comp_idx: Int[torch.Tensor, " n_firings"]
            batch_idx, seq_idx, comp_idx = torch.where(mask)
            (n_firings,) = batch_idx.shape

            # Pad batch and ci to avoid boundary issues
            batch_padded = torch.nn.functional.pad(
                batch, (n_tokens_either_side, n_tokens_either_side), value=pad_token_id
            )
            assert batch_padded.shape == (B, S + 2 * n_tokens_either_side), (
                "Expected (B,S+2*n_tokens_either_side)"
            )

            ci_padded = torch.nn.functional.pad(
                ci, (0, 0, n_tokens_either_side, n_tokens_either_side), value=0.0
            )
            assert ci_padded.shape == (B, S + 2 * n_tokens_either_side, C), (
                "Expected (B,S+2*n_tokens_either_side,C) per module"
            )

            # Adjust sequence indices for padding
            seq_idx_padded = seq_idx + n_tokens_either_side

            # Extract importance values and token IDs for all firings at once
            importance_vals = ci[batch_idx, seq_idx, comp_idx]
            token_ids = batch[batch_idx, seq_idx]

            # Vectorized window extraction
            window_size = 2 * n_tokens_either_side + 1
            window_offsets = torch.arange(
                -n_tokens_either_side,
                n_tokens_either_side + 1,
                device=batch.device,
                dtype=torch.long,
            )

            # Extract windows for tokens using advanced indexing
            batch_idx_expanded = batch_idx.unsqueeze(1).expand(-1, window_size)
            assert batch_idx_expanded.shape == (n_firings, window_size)

            # This indexes into batch_padded[batch_idx_expanded] to get the windowed tokens
            # a row might be eg. [5, 6, 7, 8]
            window_indices = seq_idx_padded.unsqueeze(1) + window_offsets.unsqueeze(0)
            assert window_indices.shape == (n_firings, window_size)

            # Extract windows for causal importances
            comp_idx_expanded = comp_idx.unsqueeze(1).expand(-1, window_size)
            assert comp_idx_expanded.shape == (n_firings, window_size)

            window_token_ids: Int[torch.Tensor, "n_firings W"] = batch_padded[
                batch_idx_expanded, window_indices
            ]
            window_ci_values: Float[torch.Tensor, "n_firings W"] = ci_padded[
                batch_idx_expanded, window_indices, comp_idx_expanded
            ]

            # Move everything to CPU for final processing
            comp_idx_list = comp_idx.tolist()
            token_ids_list = token_ids.tolist()
            importance_vals_list = importance_vals.tolist()
            window_token_ids_list = window_token_ids.tolist()
            window_ci_values_list = window_ci_values.tolist()

            # Now iterate to create examples
            for firing_idx in range(n_firings):
                ex = SubcomponentExample(
                    window_token_ids=window_token_ids_list[firing_idx],
                    # Active position is always at the center of the window
                    active_pos_in_window=n_tokens_either_side,
                    token_ci_values=window_ci_values_list[firing_idx],
                    active_pos_importance=importance_vals_list[firing_idx],
                )

                m = comp_idx_list[firing_idx]
                token_id = token_ids_list[firing_idx]

                examples[module_name][m].maybe_add(ex)
                component_activation_counts[module_name][m] += 1
                component_activation_tokens[module_name][m][token_id] += 1

        # Yield progress update
        yield ("progress", i)

    model_ctxs: dict[str, list[SubcomponentActivationContexts]] = {}
    for module_name in component_activation_tokens:
        module_acts = component_activation_tokens[module_name]
        module_examples = examples[module_name]
        module_activation_counts = component_activation_counts[module_name]
        module_mean_cis = (component_sum_cis[module_name] / n_toks_seen).tolist()
        module_subcomponent_ctxs: list[SubcomponentActivationContexts] = []
        for component_idx in module_acts:
            component_token_prs = _get_component_token_pr(
                component_token_acts=module_acts[component_idx],
                total_token_counts=total_token_counts,
                run_context=run_context,
                component_activation_count=module_activation_counts[component_idx],
            )
            activation_contexts = module_examples[component_idx].as_activation_contexts(
                run_context.tokenizer
            )
            subcomponent_ctx = SubcomponentActivationContexts(
                subcomponent_idx=component_idx,
                examples=activation_contexts,
                token_prs=component_token_prs,
                mean_ci=module_mean_cis[component_idx],
            )
            module_subcomponent_ctxs.append(subcomponent_ctx)
        module_subcomponent_ctxs.sort(key=lambda x: x.mean_ci, reverse=True)
        model_ctxs[module_name] = module_subcomponent_ctxs

    yield ("complete", ModelActivationContexts(layers=model_ctxs))


def _get_importances_by_module(
    cm: ComponentModel, batch: torch.Tensor, config: Config
) -> dict[str, Float[torch.Tensor, "B S C"]]:
    """returns a dictionary of module names to causal importances, where the shape is (B, S, C)"""

    with torch.no_grad():
        pre_weight_acts = cm(batch, cache_type="input").cache
        importances_by_module = cm.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            detach_inputs=True,
            sampling=config.sampling,
        ).lower_leaky
    return importances_by_module


def _get_component_token_pr(
    component_token_acts: dict[int, int],
    total_token_counts: dict[int, int],
    run_context: TrainRunContext,
    component_activation_count: int,
):
    token_prs: list[TokenPR] = []
    for token_id in component_token_acts:
        # recall: P(token | firing)
        recall = component_token_acts[token_id] / component_activation_count
        # precision: P(firing | token)
        precision = component_token_acts[token_id] / total_token_counts[token_id]
        tok_str = run_context.tokenizer.convert_ids_to_tokens(token_id)  # pyright: ignore[reportAttributeAccessIssue]
        assert isinstance(tok_str, str), "Token id should convert to string"
        pr = TokenPR(token=tok_str, recall=recall, precision=precision)
        token_prs.append(pr)
    # sort by recall descending
    token_prs.sort(key=lambda x: x.recall, reverse=True)
    return token_prs
