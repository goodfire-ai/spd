import heapq
from collections import defaultdict
from collections.abc import Generator, Iterable, Mapping
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from tqdm import tqdm

from spd.app.backend.schemas import (
    ActivationContext,
    ModelActivationContexts,
    SubcomponentActivationContexts,
    TokenDensity,
)
from spd.app.backend.services.run_context_service import TrainRunContext
from spd.configs import Config
from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.utils.general_utils import extract_batch_data


def get_subcomponents_activation_contexts(
    run_context: TrainRunContext,
    importance_threshold: float,
    n_batches: int,
    n_tokens_either_side: int,
    batch_size: int,
    device: str,
) -> ModelActivationContexts:
    logger.info("Getting activation contexts")

    activations_data = get_topk_by_subcomponent(
        run_context,
        importance_threshold,
        n_batches,
        n_tokens_either_side,
        batch_size,
        device,
    )

    return map_to_model_ctxs(run_context, activations_data)


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
    examples: Mapping[str, Mapping[int, _TopKExamples]]
    component_token_densities: dict[str, dict[int, list[tuple[str, float]]]]
    component_mean_cis: dict[str, Float[torch.Tensor, " C"]]


TOPK_EXAMPLES = 100


def get_topk_by_subcomponent(
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

            # No sorting - just use the indices as-is

            # Pad batch and causal_importances to avoid boundary issues
            # Pad left and right with zeros
            batch_padded = torch.nn.functional.pad(
                batch, (n_tokens_either_side, n_tokens_either_side), value=0
            )
            causal_importances_padded = torch.nn.functional.pad(
                causal_importances, (0, 0, n_tokens_either_side, n_tokens_either_side), value=0.0
            )

            # Adjust sequence indices for padding
            seq_idx_padded = seq_idx + n_tokens_either_side

            # Extract importance values and token IDs for all firings at once
            importance_vals = causal_importances[batch_idx, seq_idx, comp_idx]
            token_ids = batch[batch_idx, seq_idx]

            # Vectorized window extraction using unfold-like indexing
            window_size = 2 * n_tokens_either_side + 1

            # Create index offsets for the window (e.g., [-2, -1, 0, 1, 2] for n_tokens_either_side=2)
            window_offsets = torch.arange(
                -n_tokens_either_side, n_tokens_either_side + 1, device=batch.device
            )

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

            # Move everything to CPU/numpy for final processing
            window_token_ids_np = window_token_ids_tensor.cpu().numpy()
            window_ci_values_np = window_ci_values_tensor.cpu().numpy()
            seq_idx_np = seq_idx.cpu().numpy()
            comp_idx_np = comp_idx.cpu().numpy()
            token_ids_np = token_ids.cpu().numpy()
            importance_vals_np = importance_vals.cpu().numpy()

            # Active position is always at the center of the window
            active_pos_in_window = n_tokens_either_side

            # Now iterate to create examples (just data structure creation, no heavy computation)
            for j in range(K):
                window_token_ids: list[int] = window_token_ids_np[j].tolist()
                token_ci_values: list[float] = window_ci_values_np[j].tolist()

                ex = SubcomponentExample(
                    window_token_ids=window_token_ids,
                    pos=int(seq_idx_np[j]),
                    active_pos_in_window=active_pos_in_window,
                    token_ci_values=token_ci_values,
                    last_tok_importance=float(importance_vals_np[j]),
                )

                m = int(comp_idx_np[j])
                token_id = int(token_ids_np[j])

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


def map_to_model_ctxs(
    run_context: TrainRunContext, activations_data: ActivationsData
) -> ModelActivationContexts:
    model_ctxs: dict[str, list[SubcomponentActivationContexts]] = {}

    for module_name, module_examples_by_component in activations_data.examples.items():
        module_subcomponent_ctxs = []
        module_mean_cis = activations_data.component_mean_cis[module_name]

        for subcomponent_idx, subcomponent_examples in module_examples_by_component.items():
            activation_contexts = []
            for example in subcomponent_examples.to_sorted_list_desc():
                activation_context = ActivationContext(
                    token_strings=run_context.tokenizer.convert_ids_to_tokens(  # pyright: ignore[reportAttributeAccessIssue]
                        example.window_token_ids
                    ),
                    token_ci_values=example.token_ci_values,
                    active_position=example.active_pos_in_window,
                    ci_value=example.last_tok_importance,
                )
                activation_contexts.append(activation_context)

            token_densities = [
                TokenDensity(token=token, density=density)
                for token, density in activations_data.component_token_densities[module_name][
                    subcomponent_idx
                ]
            ]

            mean_ci = float(module_mean_cis[subcomponent_idx].item())

            subcomponent_ctx = SubcomponentActivationContexts(
                subcomponent_idx=subcomponent_idx,
                examples=activation_contexts,
                token_densities=token_densities,
                mean_ci=mean_ci,
            )

            module_subcomponent_ctxs.append(subcomponent_ctx)

        module_subcomponent_ctxs.sort(key=lambda x: x.mean_ci, reverse=True)
        model_ctxs[module_name] = module_subcomponent_ctxs

    return ModelActivationContexts(layers=model_ctxs)
