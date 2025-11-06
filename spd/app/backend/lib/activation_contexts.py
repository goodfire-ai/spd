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
from spd.log import logger
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

    @property
    def active_pos_ci(self) -> float:
        return self.token_ci_values[self.active_pos_in_window]


class _TopKExamples:
    def __init__(self, k: int):
        self.k = k
        # Min-heap of tuples (importance, counter, example)
        self.heap: list[tuple[float, int, SubcomponentExample]] = []
        self._counter: int = 0

    def maybe_add(self, example: SubcomponentExample) -> None:
        key = (example.active_pos_ci, self._counter, example)
        self._counter += 1
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, key)
            return
        # Heap full: replace min if better
        if self.heap[0][0] < example.active_pos_ci:
            heapq.heapreplace(self.heap, key)

    def as_activation_contexts(self, tok: PreTrainedTokenizer) -> list[ActivationContext]:
        return [
            ActivationContext(
                token_strings=tok.convert_ids_to_tokens(ex.window_token_ids),  # pyright: ignore[reportAttributeAccessIssue]
                token_ci_values=ex.token_ci_values,
                active_position=ex.active_pos_in_window,
                ci_value=ex.active_pos_ci,
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


DEFAULT_PAD_TOKEN_ID = 0


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
    # - and the number of times each next token appears after a component activation
    component_next_token_activations = defaultdict[str, defaultdict[int, dict[int, int]]](
        lambda: defaultdict(lambda: defaultdict(int))
    )
    # - the sum of causal importances
    C = run_context.cm.C
    component_sum_cis = defaultdict[str, Float[torch.Tensor, " C"]](
        lambda: torch.zeros(C, device=device, dtype=torch.float)
    )

    # also track total occurrences of each token across all batches
    total_token_counts = defaultdict[int, int](int)
    # track total occurrences of each token as a "next token" (at position i+1 when i exists)
    total_next_token_counts = defaultdict[int, int](int)
    n_toks_seen = 0

    pad_token_id = int(getattr(run_context.tokenizer, "pad_token_id", None) or DEFAULT_PAD_TOKEN_ID)

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

        # Count tokens that appear as "next tokens" (at position i+1 when position i exists and is not padding)
        # This excludes the first token of each sequence and padding tokens
        for b_idx in range(B):
            for s_idx in range(S - 1):  # S-1 because we're looking at next tokens
                current_token_id = int(batch[b_idx, s_idx].item())
                if current_token_id != pad_token_id:  # Current position is not padding
                    next_token_id = int(batch[b_idx, s_idx + 1].item())
                    if next_token_id != pad_token_id:  # Next token is not padding
                        total_next_token_counts[next_token_id] += 1

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
            comp_idx_list: list[int] = comp_idx.tolist()
            window_token_ids_list: list[list[int]] = window_token_ids.tolist()
            window_ci_values_list: list[list[float]] = window_ci_values.tolist()

            # Now iterate to create examples
            for firing_idx in range(n_firings):
                # Get the actual start and end indices of the window, ignoring padding
                start_idx, end_idx = get_pad_indices(
                    window_token_ids_list[firing_idx], pad_token_id
                )
                this_window_token_ids = window_token_ids_list[firing_idx]
                this_window_ci_vals = window_ci_values_list[firing_idx]

                ex = SubcomponentExample(
                    active_pos_in_window=n_tokens_either_side - start_idx,
                    window_token_ids=this_window_token_ids[start_idx:end_idx],
                    token_ci_values=this_window_ci_vals[start_idx:end_idx],
                )

                component_idx = comp_idx_list[firing_idx]
                token_id = this_window_token_ids[n_tokens_either_side]

                examples[module_name][component_idx].maybe_add(ex)
                component_activation_counts[module_name][component_idx] += 1
                component_activation_tokens[module_name][component_idx][token_id] += 1

                # Track next token if it exists (not at end of sequence and not padding)
                seq_pos = int(seq_idx[firing_idx].item())
                if seq_pos + 1 < S:  # Check if next position is within sequence
                    next_token_id = int(
                        batch[int(batch_idx[firing_idx].item()), seq_pos + 1].item()
                    )
                    if next_token_id != pad_token_id:  # Ignore padding tokens
                        component_next_token_activations[module_name][component_idx][
                            next_token_id
                        ] += 1

        # Yield progress update
        logger.info(f"Processed batch {i + 1}/{n_batches}")
        yield ("progress", i)

    model_ctxs: dict[str, list[SubcomponentActivationContexts]] = {}
    for module_name in component_activation_tokens:
        module_acts = component_activation_tokens[module_name]
        module_next_token_acts = component_next_token_activations[module_name]
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
            # Compute next token PRs using total_next_token_counts for precision
            component_next_token_prs = _get_component_token_pr(
                component_token_acts=module_next_token_acts.get(component_idx, {}),
                total_token_counts=total_next_token_counts,
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
                next_token_prs=component_next_token_prs,
                mean_ci=module_mean_cis[component_idx],
            )
            module_subcomponent_ctxs.append(subcomponent_ctx)
        module_subcomponent_ctxs.sort(key=lambda x: x.mean_ci, reverse=True)
        model_ctxs[module_name] = module_subcomponent_ctxs

    logger.info("Completed streaming activation contexts")
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


def get_pad_indices(lst: list[int], pad_val: int) -> tuple[int, int]:
    start_idx = 0
    end_idx = len(lst) - 1
    while start_idx <= end_idx and lst[start_idx] == pad_val:
        start_idx += 1
    while start_idx <= end_idx and lst[end_idx] == pad_val:
        end_idx -= 1
    return start_idx, end_idx + 1
