"""Optimized version of activation_contexts.py.

Key optimizations:
1. Pre-filter candidates with numpy mask before sorting
2. Use argpartition instead of full sort for large batches
3. Vectorize padding trimming for all candidates at once
4. Store numpy arrays in heap, delay .tolist() conversion
5. Batch processing improvements
"""

import heapq
import time
from collections import defaultdict
from collections.abc import Generator, Iterable
from typing import Literal

import numpy as np
import torch
import tqdm
from jaxtyping import Float, Int
from numpy.typing import NDArray
from torch import Tensor

from spd.app.backend.schemas import (
    ModelActivationContexts,
    SubcomponentActivationContexts,
)
from spd.app.backend.services.run_context_service import TrainRunContext
from spd.configs import Config
from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.utils.general_utils import extract_batch_data


def _get_pad_bounds_vectorized(
    arr: NDArray[np.int64], pad_val: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Compute start/end indices for all rows at once, excluding padding."""
    non_pad = arr != pad_val
    n_rows, n_cols = arr.shape

    # Find first non-pad index for each row
    # argmax returns 0 if all False, so we need to handle all-padding rows
    start_indices = np.argmax(non_pad, axis=1)

    # Find last non-pad index for each row (search from right)
    end_indices = n_cols - np.argmax(non_pad[:, ::-1], axis=1)

    # Handle rows that are all padding
    all_pad_rows = ~non_pad.any(axis=1)
    start_indices[all_pad_rows] = 0
    end_indices[all_pad_rows] = 0

    return start_indices, end_indices


class _TopKExamplesV2:
    """Optimized top-k examples tracker.

    Key optimizations:
    - Pre-filter candidates before sorting
    - Use argpartition for large batches
    - Store numpy arrays directly, convert only at the end
    - Vectorized padding trimming
    """

    def __init__(self, k: int):
        self.k = k
        # Store (ci_val, counter, token_ids_array, ci_array, active_pos_in_trimmed)
        self.heap: list[tuple[float, int, NDArray[np.int64], NDArray[np.float32], int]] = []
        self._counter: int = 0

    def add_batch(
        self,
        window_token_ids: NDArray[np.int64],
        window_ci_values: NDArray[np.float32],
        active_positions: NDArray[np.int64],
        ci_at_active: NDArray[np.float32],
        pad_token_id: int,
    ) -> None:
        """Add a batch of examples, keeping only top-k overall."""
        n_examples = len(ci_at_active)
        if n_examples == 0:
            return

        min_threshold = self.heap[0][0] if len(self.heap) >= self.k else float("-inf")

        # Optimization 1: Pre-filter candidates that can't make it
        if len(self.heap) >= self.k:
            candidate_mask = ci_at_active > min_threshold
            n_candidates = candidate_mask.sum()
            if n_candidates == 0:
                return
            # Filter to only candidates
            window_token_ids = window_token_ids[candidate_mask]
            window_ci_values = window_ci_values[candidate_mask]
            active_positions = active_positions[candidate_mask]
            ci_at_active = ci_at_active[candidate_mask]
            n_examples = int(n_candidates)

        # Optimization 2: Use argpartition for large batches
        # Only need to consider top ~2k candidates
        if n_examples > self.k * 2:
            n_to_consider = self.k * 2
            # argpartition is O(n) vs O(n log n) for full sort
            top_indices = np.argpartition(ci_at_active, -n_to_consider)[-n_to_consider:]
            sorted_order = np.argsort(ci_at_active[top_indices])[::-1]
            sorted_indices = top_indices[sorted_order]
        else:
            sorted_indices = np.argsort(ci_at_active)[::-1]

        # Optimization 3: Vectorized padding trimming for all candidates
        start_indices, end_indices = _get_pad_bounds_vectorized(window_token_ids, pad_token_id)

        # Process in sorted order with early termination
        for idx in sorted_indices:
            ci_val = float(ci_at_active[idx])

            # Early termination since sorted descending
            if len(self.heap) >= self.k and ci_val <= min_threshold:
                break

            start = int(start_indices[idx])
            end = int(end_indices[idx])
            active_pos = int(active_positions[idx])

            # Store trimmed arrays - use slice for tokens (no modification needed)
            # Pre-round CI values here to avoid rounding overhead in as_columnar
            trimmed_tokens = window_token_ids[idx, start:end].copy()
            trimmed_ci = np.round(window_ci_values[idx, start:end], 3)
            active_pos_trimmed = active_pos - start

            key = (ci_val, self._counter, trimmed_tokens, trimmed_ci, active_pos_trimmed)
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
        # Sort by CI descending
        sorted_items = sorted(self.heap, key=lambda t: t[0], reverse=True)

        example_tokens: list[list[str]] = []
        example_ci: list[list[float]] = []
        active_pos: list[int] = []
        active_ci: list[float] = []

        for ci_val, _counter, token_ids, ci_vals, pos in sorted_items:
            # Convert token IDs to strings
            example_tokens.append([token_strings[int(tid)] for tid in token_ids])
            # CI values already rounded in add_batch
            example_ci.append(ci_vals.tolist())
            active_pos.append(pos)
            active_ci.append(round(ci_val, 3))

        return example_tokens, example_ci, active_pos, active_ci


def roll_batch_size_1_into_x(
    singleton_batches: Iterable[Tensor],
    batch_size: int,
) -> Generator[Tensor]:
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
PROGRESS_THROTTLE_INTERVAL_SECONDS = 0.1


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

    # Use optimized TopK tracker
    examples = defaultdict[str, defaultdict[int, _TopKExamplesV2]](
        lambda: defaultdict(lambda: _TopKExamplesV2(k=topk_examples))
    )
    component_activation_counts = defaultdict[str, defaultdict[int, int]](lambda: defaultdict(int))
    component_activation_tokens = defaultdict[str, defaultdict[int, dict[int, int]]](
        lambda: defaultdict(lambda: defaultdict(int))
    )
    C = run_context.cm.C
    component_sum_cis = defaultdict[str, Float[Tensor, " C"]](
        lambda: torch.zeros(C, device=device, dtype=torch.float)
    )

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
        batch: Int[Tensor, "B S"] = next(batches)
        assert not batch.requires_grad, "Batch tensors with requires_grad are not supported"
        assert isinstance(batch, Tensor)
        assert batch.ndim == 2, "Expected batch tensor of shape (B, S)"
        B, S = batch.shape

        n_toks_seen += B * S

        # Count all token occurrences in batch for precision calculation
        token_ids_in_batch: Tensor
        counts: Tensor
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

            batch_idx: Int[Tensor, " n_firings"]
            seq_idx: Int[Tensor, " n_firings"]
            comp_idx: Int[Tensor, " n_firings"]
            batch_idx, seq_idx, comp_idx = torch.where(mask)
            (n_firings,) = batch_idx.shape

            # Pad batch and ci to avoid boundary issues
            batch_padded = torch.nn.functional.pad(
                batch, (n_tokens_either_side, n_tokens_either_side), value=pad_token_id
            )
            ci_padded = torch.nn.functional.pad(
                ci, (0, 0, n_tokens_either_side, n_tokens_either_side), value=0.0
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

            batch_idx_expanded = batch_idx.unsqueeze(1).expand(-1, window_size)
            window_indices = seq_idx_padded.unsqueeze(1) + window_offsets.unsqueeze(0)
            comp_idx_expanded = comp_idx.unsqueeze(1).expand(-1, window_size)

            window_token_ids: Int[Tensor, "n_firings W"] = batch_padded[
                batch_idx_expanded, window_indices
            ]
            window_ci_values: Float[Tensor, "n_firings W"] = ci_padded[
                batch_idx_expanded, window_indices, comp_idx_expanded
            ]

            ci_at_active = window_ci_values[:, n_tokens_either_side]

            # Move to CPU/numpy once
            comp_idx_np = comp_idx.cpu().numpy()
            window_token_ids_np = window_token_ids.cpu().numpy()
            window_ci_values_np = window_ci_values.cpu().numpy()
            ci_at_active_np = ci_at_active.cpu().numpy()

            active_token_ids = window_token_ids_np[:, n_tokens_either_side]

            # Process by component
            unique_components = np.unique(comp_idx_np)
            for c_idx in unique_components:
                c_idx_int = int(c_idx)
                mask_c = comp_idx_np == c_idx

                n_firings_for_component = int(mask_c.sum())
                component_activation_counts[module_name][c_idx_int] += n_firings_for_component

                tokens_for_component = active_token_ids[mask_c]
                unique_tokens, token_counts = np.unique(tokens_for_component, return_counts=True)
                for tok_id, count in zip(unique_tokens, token_counts, strict=True):
                    component_activation_tokens[module_name][c_idx_int][int(tok_id)] += int(count)

                examples[module_name][c_idx_int].add_batch(
                    window_token_ids=window_token_ids_np[mask_c],
                    window_ci_values=window_ci_values_np[mask_c],
                    active_positions=np.full(n_firings_for_component, n_tokens_either_side),
                    ci_at_active=ci_at_active_np[mask_c],
                    pad_token_id=pad_token_id,
                )

            current_time = time.monotonic()
            if current_time - last_progress_time >= PROGRESS_THROTTLE_INTERVAL_SECONDS:
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
    cm: ComponentModel, batch: Tensor, config: Config
) -> dict[str, Float[Tensor, "B S C"]]:
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
    tokens: list[str] = []
    recalls: list[float] = []
    precisions: list[float] = []

    for token_id in component_token_acts:
        recall = round(component_token_acts[token_id] / component_activation_count, 3)
        precision = round(component_token_acts[token_id] / total_token_counts[token_id], 3)
        tokens.append(token_strings[token_id])
        recalls.append(recall)
        precisions.append(precision)

    sorted_indices = sorted(range(len(recalls)), key=lambda i: recalls[i], reverse=True)
    tokens = [tokens[i] for i in sorted_indices]
    recalls = [recalls[i] for i in sorted_indices]
    precisions = [precisions[i] for i in sorted_indices]

    return tokens, recalls, precisions
