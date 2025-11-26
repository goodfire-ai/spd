import heapq
import itertools
import time
from collections import defaultdict
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import tqdm
from jaxtyping import Float, Int
from numpy.typing import NDArray

from spd.app.backend.schemas import (
    ModelActivationContexts,
    SubcomponentActivationContexts,
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
    """Maintains top-k examples by CI value using a min-heap."""

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

    def add_batch(
        self,
        window_token_ids: NDArray[np.int64],
        window_ci_values: NDArray[np.float32],
        active_positions: NDArray[np.int64],
        ci_at_active: NDArray[np.float32],
        pad_token_id: int,
    ) -> None:
        """Add a batch of examples, keeping only top-k overall.

        This is more efficient than calling maybe_add() repeatedly because:
        1. We pre-sort by CI value and only process candidates that could make top-k
        2. We use numpy arrays directly instead of Python lists where possible
        """
        n_examples = len(ci_at_active)
        if n_examples == 0:
            return

        # Sort by CI descending - we only need to consider examples that could make top-k
        sorted_indices = np.argsort(ci_at_active)[::-1]

        # Early termination threshold: if heap is full, we can skip examples below min
        min_threshold = self.heap[0][0] if len(self.heap) >= self.k else float("-inf")

        for idx in sorted_indices:
            ci_val = float(ci_at_active[idx])

            # Early termination: since sorted descending, if we're below threshold, stop
            if len(self.heap) >= self.k and ci_val <= min_threshold:
                break

            # Trim padding from this example
            tokens = window_token_ids[idx]
            ci_vals = window_ci_values[idx]
            active_pos = int(active_positions[idx])

            start_idx, end_idx = _get_pad_indices_numpy(tokens, pad_token_id)

            ex = SubcomponentExample(
                active_pos_in_window=active_pos - start_idx,
                window_token_ids=tokens[start_idx:end_idx].tolist(),
                token_ci_values=ci_vals[start_idx:end_idx].tolist(),
            )

            key = (ci_val, self._counter, ex)
            self._counter += 1

            if len(self.heap) < self.k:
                heapq.heappush(self.heap, key)
                min_threshold = self.heap[0][0]
            elif ci_val > self.heap[0][0]:
                heapq.heapreplace(self.heap, key)
                min_threshold = self.heap[0][0]

    def as_columnar(
        self,
        token_strings: dict[int, str],
    ) -> tuple[list[list[str]], list[list[float]], list[int], list[float]]:
        """Return columnar data: (example_tokens, example_ci, active_pos, active_ci)"""
        sorted_examples = [ex for _, _, ex in sorted(self.heap, key=lambda t: t[0], reverse=True)]

        example_tokens = []
        example_ci = []
        active_pos = []
        active_ci = []
        for ex in sorted_examples:
            example_tokens.append(_token_ids_to_strings(ex.window_token_ids, token_strings))
            example_ci.append([round(v, 3) for v in ex.token_ci_values])
            active_pos.append(ex.active_pos_in_window)
            active_ci.append(round(ex.active_pos_ci, 3))

        return example_tokens, example_ci, active_pos, active_ci


def _token_ids_to_strings(token_ids: list[int], token_strings: dict[int, str]) -> list[str]:
    """Convert token IDs to strings, stripping leading space from first token."""
    if not token_ids:
        return []
    result = [token_strings[tid] for tid in token_ids]
    # Strip leading space from first token (wordpiece adds space prefix to non-## tokens)
    if result and result[0].startswith(" "):
        result[0] = result[0][1:]
    return result


def _get_pad_indices_numpy(arr: NDArray[np.int64], pad_val: int) -> tuple[int, int]:
    """Find start/end indices excluding padding. Vectorized version."""
    non_pad = arr != pad_val
    if not non_pad.any():
        return 0, 0
    non_pad_indices = np.where(non_pad)[0]
    return int(non_pad_indices[0]), int(non_pad_indices[-1]) + 1


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

# Minimum interval between progress updates (seconds)
PROGRESS_THROTTLE_INTERVAL = 0.1


def get_activations_data_streaming(
    run_context: TrainRunContext,
    importance_threshold: float,
    n_batches: int,
    n_tokens_either_side: int,
    batch_size: int,
    topk_examples: int,
) -> Generator[
    tuple[Literal["progress"], float] | tuple[Literal["complete"], ModelActivationContexts],
]:
    logger.info(
        f"Getting activations data: {n_batches=}, {importance_threshold=}, "
        f"{n_tokens_either_side=}, {batch_size=}, {topk_examples=}"
    )

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

    pad_token_id = int(getattr(run_context.tokenizer, "pad_token_id", None) or DEFAULT_PAD_TOKEN_ID)

    batches = roll_batch_size_1_into_x(
        singleton_batches=(extract_batch_data(b).to(device) for b in run_context.train_loader),
        batch_size=batch_size,
    )

    last_progress_time = 0.0
    n_modules = len(run_context.cm.target_module_paths)

    pbar = tqdm.tqdm(total=n_batches * n_modules, desc="Processing batches", unit="batch,layer")

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

        for module_idx, (module_name, ci) in enumerate(importances_by_module.items()):
            pbar.update(1)
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

            # Get CI values at active position (center of window) for sorting
            ci_at_active = window_ci_values[:, n_tokens_either_side]

            # Move to CPU/numpy once (faster than .tolist())
            comp_idx_np = comp_idx.cpu().numpy()
            window_token_ids_np = window_token_ids.cpu().numpy()
            window_ci_values_np = window_ci_values.cpu().numpy()
            ci_at_active_np = ci_at_active.cpu().numpy()

            # Get token IDs at active position for token counting
            active_token_ids = window_token_ids_np[:, n_tokens_either_side]

            # Process by component - group firings and use batch add
            unique_components = np.unique(comp_idx_np)
            for c_idx in unique_components:
                c_idx_int = int(c_idx)
                mask_c = comp_idx_np == c_idx

                # Update activation counts
                n_firings_for_component = int(mask_c.sum())
                component_activation_counts[module_name][c_idx_int] += n_firings_for_component

                # Update token counts for this component
                tokens_for_component = active_token_ids[mask_c]
                unique_tokens, token_counts = np.unique(tokens_for_component, return_counts=True)
                for tok_id, count in zip(unique_tokens, token_counts, strict=True):
                    component_activation_tokens[module_name][c_idx_int][int(tok_id)] += int(count)

                # Add examples in batch (with early termination optimization)
                examples[module_name][c_idx_int].add_batch(
                    window_token_ids=window_token_ids_np[mask_c],
                    window_ci_values=window_ci_values_np[mask_c],
                    active_positions=np.full(n_firings_for_component, n_tokens_either_side),
                    ci_at_active=ci_at_active_np[mask_c],
                    pad_token_id=pad_token_id,
                )

            # Yield progress update within batch (throttled)
            current_time = time.monotonic()
            if current_time - last_progress_time >= PROGRESS_THROTTLE_INTERVAL:
                # Progress: batch progress + fractional module progress within batch
                progress = (i + (module_idx + 1) / n_modules) / n_batches
                yield ("progress", progress)
                last_progress_time = current_time

    model_ctxs: dict[str, list[SubcomponentActivationContexts]] = {}
    for module_name in component_activation_tokens:
        module_acts = component_activation_tokens[module_name]
        module_examples = examples[module_name]
        module_activation_counts = component_activation_counts[module_name]
        module_mean_cis = (component_sum_cis[module_name] / n_toks_seen).tolist()
        module_subcomponent_ctxs: list[SubcomponentActivationContexts] = []
        for component_idx in module_acts:
            pr_tokens, pr_recalls, pr_precisions = _get_component_token_pr(
                component_token_acts=module_acts[component_idx],
                total_token_counts=total_token_counts,
                token_strings=run_context.token_strings,
                component_activation_count=module_activation_counts[component_idx],
            )
            example_tokens, example_ci, example_active_pos, example_active_ci = module_examples[
                component_idx
            ].as_columnar(run_context.token_strings)
            subcomponent_ctx = SubcomponentActivationContexts(
                subcomponent_idx=component_idx,
                mean_ci=round(module_mean_cis[component_idx], 3),
                example_tokens=example_tokens,
                example_ci=example_ci,
                example_active_pos=example_active_pos,
                example_active_ci=example_active_ci,
                pr_tokens=pr_tokens,
                pr_recalls=pr_recalls,
                pr_precisions=pr_precisions,
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
    token_strings: dict[int, str],
    component_activation_count: int,
) -> tuple[list[str], list[float], list[float]]:
    """Return columnar data: (tokens, recalls, precisions) sorted by recall descending."""
    # Build parallel arrays
    tokens: list[str] = []
    recalls: list[float] = []
    precisions: list[float] = []

    for token_id in component_token_acts:
        # recall: P(token | firing)
        recall = round(component_token_acts[token_id] / component_activation_count, 3)
        # precision: P(firing | token)
        precision = round(component_token_acts[token_id] / total_token_counts[token_id], 3)
        tokens.append(token_strings[token_id])
        recalls.append(recall)
        precisions.append(precision)

    # Sort by recall descending
    sorted_indices = sorted(range(len(recalls)), key=lambda i: recalls[i], reverse=True)
    tokens = [tokens[i] for i in sorted_indices]
    recalls = [recalls[i] for i in sorted_indices]
    precisions = [precisions[i] for i in sorted_indices]

    return tokens, recalls, precisions
