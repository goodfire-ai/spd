import heapq
from collections import defaultdict
from collections.abc import Generator, Iterable, Mapping
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from tqdm import tqdm

from spd.app.backend.api import (
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

            # Sort for cache-friendly iteration
            order = torch.argsort(batch_idx * S + seq_idx)
            batch_idx = batch_idx[order]
            seq_idx = seq_idx[order]
            comp_idx = comp_idx[order]

            # Move all indices to CPU once
            batch_idx_cpu = batch_idx.cpu().numpy()
            seq_idx_cpu = seq_idx.cpu().numpy()
            comp_idx_cpu = comp_idx.cpu().numpy()

            # Extract importance values and token IDs for all firings at once
            importance_vals = causal_importances[batch_idx, seq_idx, comp_idx].cpu().numpy()
            token_ids = batch[batch_idx, seq_idx].cpu().numpy()

            # Move batch and causal_importances to CPU once for window extraction
            batch_cpu = batch.cpu()
            causal_importances_cpu = causal_importances.cpu()

            # Iterate across K firings
            for j in range(K):
                b = int(batch_idx_cpu[j])
                s = int(seq_idx_cpu[j])
                m = int(comp_idx_cpu[j])
                token_id = int(token_ids[j])
                importance_val = float(importance_vals[j])

                # Build window around the firing position
                start_idx = max(0, s - n_tokens_either_side)
                end_idx = min(S, s + n_tokens_either_side + 1)

                window_token_ids: list[int] = batch_cpu[b, start_idx:end_idx].tolist()
                active_pos_in_window = s - start_idx

                token_ci_values: list[float] = causal_importances_cpu[b, start_idx:end_idx, m].tolist()

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
