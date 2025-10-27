"""V4: Remove sorting, optimize tolist conversions."""

from collections import defaultdict

import torch
from jaxtyping import Float, Int
from tqdm import tqdm

from spd.app.backend.lib.activation_contexts_common import (
    TOPK_EXAMPLES,
    ActivationsData,
    SubcomponentExample,
    _TopKExamples,
    _get_importances_by_module,
    roll_batch_size_1_into_x,
)
from spd.app.backend.services.run_context_service import TrainRunContext
from spd.utils.general_utils import extract_batch_data


def get_topk_by_subcomponent_v4(
    run_context: TrainRunContext,
    importance_threshold: float,
    n_batches: int,
    n_tokens_either_side: int,
    batch_size: int,
    device: str,
) -> ActivationsData:
    # for each (module_name, component_idx), track the top-k activations
    examples = defaultdict[str, defaultdict[int, _TopKExamples]](
        lambda: defaultdict(lambda: _TopKExamples(k=TOPK_EXAMPLES))
    )

    # for each (module_name, component_idx):
    # the number of tokens seen
    component_activation_counts = defaultdict[str, defaultdict[int, int]](lambda: defaultdict(int))
    # and the number of activations for each token
    component_activation_tokens = defaultdict[str, defaultdict[int, dict[int, int]]](
        lambda: defaultdict(lambda: defaultdict(int))
    )

    C = run_context.cm.C

    n_toks_seen = 0
    component_sum_cis = defaultdict[str, Float[torch.Tensor, " C"]](
        lambda: torch.zeros(C, device=device, dtype=torch.float)
    )

    batches = roll_batch_size_1_into_x(
        singleton_batches=(extract_batch_data(b).to(device) for b in run_context.train_loader),
        batch_size=batch_size,
    )

    for _ in tqdm(range(n_batches), desc="Harvesting activation contexts"):
        batch: Int[torch.Tensor, "B S"] = next(batches)
        assert not batch.requires_grad, "Batch tensors with requires_grad are not supported"
        assert isinstance(batch, torch.Tensor)
        assert batch.ndim == 2, "Expected batch tensor of shape (B, S)"
        B, S = batch.shape

        n_toks_seen += B * S

        importances_by_module = _get_importances_by_module(
            run_context.cm, batch, run_context.config
        )

        # Pad batch once for all modules
        batch_padded = torch.nn.functional.pad(
            batch, (n_tokens_either_side, n_tokens_either_side), value=0
        )

        # Create window offsets once for all modules
        window_size = 2 * n_tokens_either_side + 1
        window_offsets = torch.arange(
            -n_tokens_either_side, n_tokens_either_side + 1, device=batch.device
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

            # Skip sorting - it may not help and adds overhead

            # Pad causal_importances for this module
            causal_importances_padded = torch.nn.functional.pad(
                causal_importances, (0, 0, n_tokens_either_side, n_tokens_either_side), value=0.0
            )

            # Adjust sequence indices for padding
            seq_idx_padded = seq_idx + n_tokens_either_side

            # Extract importance values and token IDs for all firings at once
            importance_vals = causal_importances[batch_idx, seq_idx, comp_idx]
            token_ids = batch[batch_idx, seq_idx]

            # Vectorized window extraction using unfold-like indexing
            # Broadcast to get all window indices: (K, window_size)
            window_indices = seq_idx_padded.unsqueeze(1) + window_offsets.unsqueeze(0)

            # Extract windows for tokens using advanced indexing
            batch_idx_expanded = batch_idx.unsqueeze(1).expand(-1, window_size)
            window_token_ids_tensor = batch_padded[batch_idx_expanded, window_indices]

            # Extract windows for causal importances
            comp_idx_expanded = comp_idx.unsqueeze(1).expand(-1, window_size)
            window_ci_values_tensor = causal_importances_padded[
                batch_idx_expanded, window_indices, comp_idx_expanded
            ]

            # Move everything to CPU for final processing
            window_token_ids_cpu = window_token_ids_tensor.cpu()
            window_ci_values_cpu = window_ci_values_tensor.cpu()
            seq_idx_cpu = seq_idx.cpu()
            comp_idx_cpu = comp_idx.cpu()
            token_ids_cpu = token_ids.cpu()
            importance_vals_cpu = importance_vals.cpu()

            # Active position is always at the center of the window
            active_pos_in_window = n_tokens_either_side

            # Convert entire tensors to Python lists once (faster than row-by-row)
            window_token_ids_list = window_token_ids_cpu.tolist()
            window_ci_values_list = window_ci_values_cpu.tolist()
            seq_idx_list = seq_idx_cpu.tolist()
            comp_idx_list = comp_idx_cpu.tolist()
            token_ids_list = token_ids_cpu.tolist()
            importance_vals_list = importance_vals_cpu.tolist()

            # Now iterate to create examples
            for j in range(K):
                ex = SubcomponentExample(
                    window_token_ids=window_token_ids_list[j],
                    pos=seq_idx_list[j],
                    active_pos_in_window=active_pos_in_window,
                    token_ci_values=window_ci_values_list[j],
                    last_tok_importance=importance_vals_list[j],
                )

                m = comp_idx_list[j]
                token_id = token_ids_list[j]

                examples[module_name][m].maybe_add(ex)
                component_activation_counts[module_name][m] += 1
                component_activation_tokens[module_name][m][token_id] += 1

    component_mean_cis = {
        module_name: component_sum_cis[module_name] / n_toks_seen
        for module_name in component_sum_cis
    }

    component_token_densities = defaultdict[str, dict[int, list[tuple[str, float]]]](dict)
    for module_name, tokens_by_components in component_activation_tokens.items():
        for component_idx, component_token_acts in tokens_by_components.items():
            component_densities: list[tuple[str, float]] = []
            for token_id, count in component_token_acts.items():
                density = count / component_activation_counts[module_name][component_idx]
                tok_str = run_context.tokenizer.convert_ids_to_tokens(token_id)  # pyright: ignore[reportAttributeAccessIssue]
                assert isinstance(tok_str, str), "Token id should convert to string"
                component_densities.append((tok_str, density))

            # sort by density descending
            component_densities.sort(key=lambda x: x[1], reverse=True)
            component_token_densities[module_name][component_idx] = component_densities

    return ActivationsData(
        examples=examples,
        component_token_densities=component_token_densities,
        component_mean_cis=component_mean_cis,
    )
