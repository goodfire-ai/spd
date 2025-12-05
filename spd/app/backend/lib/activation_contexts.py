import heapq
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import tqdm
from jaxtyping import Float, Int
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.app.backend.schemas import (
    ModelActivationContexts,
    SubcomponentActivationContexts,
)
from spd.configs import Config
from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.utils.general_utils import extract_batch_data

DEFAULT_PAD_TOKEN_ID = 0

# Minimum interval between progress updates
PROGRESS_THROTTLE_INTERVAL_SECONDS = 0.1


def get_activations_data(
    config: Config,
    cm: ComponentModel,
    tokenizer: PreTrainedTokenizerBase,
    train_loader: DataLoader[Int[Tensor, "B S"]],
    token_strings: dict[int, str],
    importance_threshold: float,
    n_batches: int,
    n_tokens_either_side: int,
    topk_examples: int,
    separation_tokens: int = 0,
    onprogress: Callable[[float], None] | None = None,
) -> ModelActivationContexts:
    logger.info(
        f"Getting activations data: {n_batches=}, {importance_threshold=}, "
        f"{n_tokens_either_side=}, {topk_examples=}, {separation_tokens=}"
    )
    if separation_tokens > 0:
        print(f"[activation_contexts] Position separation enabled: {separation_tokens} tokens")
        assert separation_tokens <= n_tokens_either_side, (
            "separation_tokens must be less than or equal to n_tokens_either_side"
        )

    device = next(cm.parameters()).device

    # for each (module_name, component_idx), track:
    # - the top-k activating examples
    examples = defaultdict[str, defaultdict[int, _TopKExamples]](
        lambda: defaultdict(lambda: _TopKExamples(k=topk_examples))
    )
    # - the number of activations for each component
    component_activation_counts = defaultdict[str, defaultdict[int, int]](lambda: defaultdict(int))
    # - and the number of times each token activates for each component (input token)
    component_activation_tokens = defaultdict[str, defaultdict[int, dict[int, int]]](
        lambda: defaultdict(lambda: defaultdict(int))
    )
    # - the number of times each token is predicted when component fires
    component_predicted_tokens = defaultdict[str, defaultdict[int, dict[int, int]]](
        lambda: defaultdict(lambda: defaultdict(int))
    )
    # - the sum of causal importances
    C = cm.C
    component_sum_cis = defaultdict[str, Float[Tensor, " C"]](
        lambda: torch.zeros(C, device=device, dtype=torch.float)
    )

    # also track total occurrences of each token across all batches
    total_token_counts = defaultdict[int, int](int)
    n_toks_seen = 0

    pad_token_id = int(getattr(tokenizer, "pad_token_id", None) or DEFAULT_PAD_TOKEN_ID)

    last_progress_time = 0.0
    n_modules = len(cm.target_module_paths)

    pbar = tqdm.tqdm(total=n_batches * n_modules, desc="Processing batches", unit="batch,layer")

    train_iter = iter(train_loader)
    for i in range(n_batches):
        batch: Int[Tensor, "B S"] = extract_batch_data(next(train_iter)).to(device)
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

        with torch.no_grad():
            output_with_cache = cm(batch, cache_type="input")
            logits = output_with_cache.output
            ci_vals = cm.calc_causal_importances(
                pre_weight_acts=output_with_cache.cache,
                detach_inputs=True,
                sampling=config.sampling,
            ).lower_leaky

        # Get predicted tokens (argmax of logits at each position)
        predicted_token_ids: Int[Tensor, "B S"] = logits.argmax(dim=-1)

        for module_idx, (module_name, ci_val) in enumerate(ci_vals.items()):
            pbar.update(1)
            assert ci_val.shape == (B, S, C), "Expected (B,S,C) per module"

            component_sum_cis[module_name] += ci_val.sum(dim=(0, 1))

            # Thresholding to find "firings"
            mask = ci_val > importance_threshold
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
            assert batch_padded.shape == (B, S + 2 * n_tokens_either_side), (
                "Expected (B,S+2*n_tokens_either_side)"
            )

            ci_padded = torch.nn.functional.pad(
                ci_val, (0, 0, n_tokens_either_side, n_tokens_either_side), value=0.0
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

            window_token_ids: Int[Tensor, "n_firings W"] = batch_padded[
                batch_idx_expanded, window_indices
            ]
            window_ci_values: Float[Tensor, "n_firings W"] = ci_padded[
                batch_idx_expanded, window_indices, comp_idx_expanded
            ]

            # Get CI values at active position (center of window) for sorting
            ci_at_active = window_ci_values[:, n_tokens_either_side]

            # Move to CPU/numpy once (faster than .tolist())
            batch_idx_np = batch_idx.cpu().numpy()
            seq_idx_np = seq_idx.cpu().numpy()
            comp_idx_np = comp_idx.cpu().numpy()
            window_token_ids_np = window_token_ids.cpu().numpy()
            window_ci_values_np = window_ci_values.cpu().numpy()
            ci_at_active_np = ci_at_active.cpu().numpy()

            # Get token IDs at active position for token counting
            active_token_ids = window_token_ids_np[:, n_tokens_either_side]

            # Get predicted tokens at each firing position
            firing_predicted_tokens = predicted_token_ids[batch_idx, seq_idx].cpu().numpy()

            # Process by component - group firings and use batch add
            unique_components = np.unique(comp_idx_np)
            for c_idx in unique_components:
                c_idx_int = int(c_idx)
                mask_c = comp_idx_np == c_idx

                # Update activation counts (all firings)
                n_firings_for_component = int(mask_c.sum())
                component_activation_counts[module_name][c_idx_int] += n_firings_for_component

                # Update token counts for this component (input tokens)
                tokens_for_component = active_token_ids[mask_c]
                unique_tokens, token_counts = np.unique(tokens_for_component, return_counts=True)
                for tok_id, count in zip(unique_tokens, token_counts, strict=True):
                    component_activation_tokens[module_name][c_idx_int][int(tok_id)] += int(count)

                # Update predicted token counts for this component
                predicted_for_component = firing_predicted_tokens[mask_c]
                unique_predicted, predicted_counts = np.unique(
                    predicted_for_component, return_counts=True
                )
                for tok_id, count in zip(unique_predicted, predicted_counts, strict=True):
                    component_predicted_tokens[module_name][c_idx_int][int(tok_id)] += int(count)

                # Apply position separation filter for example diversity only
                if separation_tokens > 0:
                    component_indices = np.where(mask_c)[0]
                    sep_keep = _apply_position_separation(
                        seq_positions=seq_idx_np[component_indices],
                        batch_indices=batch_idx_np[component_indices],
                        min_separation=separation_tokens,
                    )
                    mask_c_examples = np.zeros_like(mask_c)
                    mask_c_examples[component_indices[sep_keep]] = True
                else:
                    mask_c_examples = mask_c

                # Add examples in batch (with early termination optimization)
                n_examples_to_add = int(mask_c_examples.sum())
                examples[module_name][c_idx_int].add_batch(
                    window_token_ids=window_token_ids_np[mask_c_examples],
                    window_ci_values=window_ci_values_np[mask_c_examples],
                    active_positions=np.full(n_examples_to_add, n_tokens_either_side),
                    ci_at_active=ci_at_active_np[mask_c_examples],
                    pad_token_id=pad_token_id,
                )

            current_time = time.monotonic()
            if current_time - last_progress_time >= PROGRESS_THROTTLE_INTERVAL_SECONDS:
                progress = (i + (module_idx + 1) / n_modules) / n_batches
                if onprogress:
                    onprogress(progress)
                last_progress_time = current_time

    model_ctxs: dict[str, list[SubcomponentActivationContexts]] = {}
    for module_name in component_activation_tokens:
        module_acts = component_activation_tokens[module_name]
        module_predicted = component_predicted_tokens[module_name]
        module_examples = examples[module_name]
        module_activation_counts = component_activation_counts[module_name]
        module_mean_cis = (component_sum_cis[module_name] / n_toks_seen).tolist()
        module_subcomponent_ctxs: list[SubcomponentActivationContexts] = []
        for component_idx in module_acts:
            pr_tokens, pr_recalls, pr_precisions = _get_component_token_pr(
                component_token_acts=module_acts[component_idx],
                total_token_counts=total_token_counts,
                token_strings=token_strings,
                component_activation_count=module_activation_counts[component_idx],
            )
            predicted_tokens, predicted_probs = _get_component_predicted_tokens(
                component_predicted_counts=module_predicted[component_idx],
                token_strings=token_strings,
                component_activation_count=module_activation_counts[component_idx],
            )
            example_tokens, example_ci, example_active_pos, example_active_ci = module_examples[
                component_idx
            ].as_columnar(token_strings)
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
                predicted_tokens=predicted_tokens,
                predicted_probs=predicted_probs,
            )
            module_subcomponent_ctxs.append(subcomponent_ctx)
        module_subcomponent_ctxs.sort(key=lambda x: x.mean_ci, reverse=True)
        model_ctxs[module_name] = module_subcomponent_ctxs

    logger.info("Completed streaming activation contexts")
    return ModelActivationContexts(layers=model_ctxs)


def _apply_position_separation(
    seq_positions: NDArray[np.int64],
    batch_indices: NDArray[np.int64],
    min_separation: int,
) -> NDArray[np.bool_]:
    """Filter firings to ensure minimum token separation within same batch example.

    Since torch.where returns indices in lexicographic order, positions within
    each batch are already sorted ascending. We just track the last kept position
    per batch and reject any firing too close to it.

    Args:
        seq_positions: Sequence position for each firing
        batch_indices: Batch index for each firing
        min_separation: Minimum token separation between kept firings

    Returns:
        Boolean mask indicating which firings to keep
    """
    n = len(seq_positions)
    if n == 0 or min_separation <= 0:
        return np.ones(n, dtype=bool)

    keep = np.ones(n, dtype=bool)
    last_kept: dict[int, int] = {}  # batch_idx -> last kept position

    for i in range(n):
        b = int(batch_indices[i])
        pos = int(seq_positions[i])

        if b in last_kept and pos < last_kept[b] + min_separation:
            keep[i] = False
        else:
            last_kept[b] = pos

    return keep


@dataclass
class _SubcomponentExample:
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
        self.heap: list[tuple[float, int, _SubcomponentExample]] = []
        self._counter: int = 0

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

            ex = _SubcomponentExample(
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
            example_tokens.append([token_strings[tid] for tid in ex.window_token_ids])
            example_ci.append([round(v, 3) for v in ex.token_ci_values])
            active_pos.append(ex.active_pos_in_window)
            active_ci.append(round(ex.active_pos_ci, 3))

        return example_tokens, example_ci, active_pos, active_ci


def _get_pad_indices_numpy(arr: NDArray[np.int64], pad_val: int) -> tuple[int, int]:
    """Find start/end indices excluding padding. Vectorized version."""
    non_pad = arr != pad_val
    if not non_pad.any():
        return 0, 0
    non_pad_indices = np.where(non_pad)[0]
    return int(non_pad_indices[0]), int(non_pad_indices[-1]) + 1


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


def _get_component_predicted_tokens(
    component_predicted_counts: dict[int, int],
    token_strings: dict[int, str],
    component_activation_count: int,
) -> tuple[list[str], list[float]]:
    """Return columnar data: (tokens, probs) sorted by probability descending.

    prob = P(predicted_token = X | component fires)
    """
    tokens: list[str] = []
    probs: list[float] = []

    for token_id, count in component_predicted_counts.items():
        prob = round(count / component_activation_count, 3)
        tokens.append(token_strings[token_id])
        probs.append(prob)

    # Sort by probability descending
    sorted_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    tokens = [tokens[i] for i in sorted_indices]
    probs = [probs[i] for i in sorted_indices]

    return tokens, probs
