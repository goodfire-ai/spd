"""V0: Original implementation from HEAD^ - element-by-element loop with .item() calls."""

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


def get_topk_by_subcomponent_v0(
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

        for module_name, causal_importances in tqdm(
            importances_by_module.items(), desc="Processing importances"
        ):
            assert causal_importances.shape == (B, S, C), "Expected (B,S,C) per module"

            # Thresholding to find "firings"
            mask = causal_importances > importance_threshold
            if not mask.any():
                continue

            component_sum_cis[module_name] += causal_importances.sum(dim=(0, 1))

            # (K,) indices of all firings
            batch_idx, seq_idx, comp_idx = torch.where(mask)
            (K,) = batch_idx.shape

            # Iterate across K firings
            for j in range(K):
                b = int(batch_idx[j].item())
                s = int(seq_idx[j].item())
                m = int(comp_idx[j].item())
                token_id = int(batch[b, s].item())
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
