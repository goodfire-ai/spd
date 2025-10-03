from __future__ import annotations

from collections import defaultdict
import heapq
from dataclasses import dataclass

import torch
from pydantic import BaseModel
from transformers import PreTrainedTokenizer

from spd.app.backend.services.run_context_service import RunContext, RunContextService
from spd.configs import Config
from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import extract_batch_data

DEVICE = get_device()


@dataclass
class SubcomponentExample:
    window_tokens: list[int]
    """Windowed tokens around the firing position"""
    pos: int
    """Absolute position within the original sequence"""
    active_pos_in_window: int
    """Position within window_tokens corresponding to pos"""
    token_ci_values: list[float]
    """CI values aligned to window_tokens"""
    last_tok_importance: float
    """CI value at the firing position"""


@dataclass
class FastComponentSummary:
    module_name: str
    component_idx: int
    density: float
    examples: list[SubcomponentExample]


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


@dataclass
class WorkerArgs:
    wandb_id: str
    importance_threshold: float
    separation_threshold_tokens: int
    max_examples_per_subcomponent: int
    n_steps: int
    n_tokens_either_side: int


def main(args: WorkerArgs) -> ModelActivationContexts:
    logger.info("activation contexts")

    rcs = RunContextService()
    rcs.load_run_from_wandb_id(args.wandb_id)
    assert (run_context := rcs.run_context) is not None, "Run context not found"

    run_context.cm.to(DEVICE)

    topk_by_subcomponent = _get_topk_by_subcomponent(
        run_context,
        args.importance_threshold,
        args.separation_threshold_tokens,
        args.max_examples_per_subcomponent,
        args.n_steps,
        args.n_tokens_either_side,
    )

    return map_to_model_ctxs(run_context, topk_by_subcomponent)


def _get_topk_by_subcomponent(
    run_context: RunContext,
    importance_threshold: float,
    separation_threshold_tokens: int,
    max_examples_per_subcomponent: int,
    n_steps: int,
    n_tokens_either_side: int,
) -> dict[tuple[str, int], _TopKExamples]:
    # Tracking structures
    topk_by_subcomponent: dict[tuple[str, int], _TopKExamples] = {}
    last_pos_in_seq: dict[tuple[str, int, int], int] = {}
    firing_counts: dict[tuple[str, int], int] = {}
    tokens_seen_total: int = 0

    C = run_context.cm.C

    # Iterate limited steps over data
    data_iter = iter(run_context.train_loader)
    for _ in range(n_steps):
        batch = extract_batch_data(next(data_iter)).to(DEVICE)
        assert isinstance(batch, torch.Tensor)
        assert batch.ndim == 2, "Expected batch tensor of shape (B, S)"

        B, S = batch.shape

        importances_by_module = _get_importances_by_module(
            run_context.cm, batch, run_context.config
        )

        # Process each module tensor: threshold -> where -> counts -> gather examples
        for module_name, causal_importances in importances_by_module.items():
            assert causal_importances.shape == (B, S, C), "Expected (B,S,C) per module"

            mask = causal_importances > importance_threshold
            if not mask.any():
                continue

            # (K,) indices of all firings
            b_idx, s_idx, m_idx = torch.where(mask)
            K = b_idx.numel()

            # Update density numerators via bincount on component index
            comp_counts = torch.bincount(m_idx, minlength=C)
            for comp_i in range(C):
                c = int(comp_counts[comp_i].item())
                if c:
                    firing_counts[(module_name, comp_i)] = (
                        firing_counts.get((module_name, comp_i), 0) + c
                    )

            # Sort to iterate in cache-friendly order
            order = torch.argsort(b_idx * S + s_idx)
            b_idx = b_idx[order]
            s_idx = s_idx[order]
            m_idx = m_idx[order]

            # Iterate once across the K firings, enforcing intra-sequence separation and top-k cap
            for j in range(K):
                b = int(b_idx[j].item())
                s = int(s_idx[j].item())
                m = int(m_idx[j].item())
                key = (module_name, m)

                # Separation within this sequence only
                lp_key = (module_name, m, b)
                last = last_pos_in_seq.get(lp_key, -separation_threshold_tokens - 1)
                if s < last + separation_threshold_tokens:
                    continue

                importance_val = float(causal_importances[b, s, m].item())

                # Enforce component-local top-k with a bounded heap
                if key not in topk_by_subcomponent:
                    topk_by_subcomponent[key] = _TopKExamples(max_examples_per_subcomponent)

                heap = topk_by_subcomponent[key].heap
                if len(heap) == max_examples_per_subcomponent and importance_val <= heap[0][0]:
                    # still advance separation to avoid clustering
                    last_pos_in_seq[lp_key] = s
                    continue

                # Materialize a window of tokens around the firing position
                start_idx = max(0, s - n_tokens_either_side)
                end_idx = min(S, s + n_tokens_either_side + 1)

                window_tokens = batch[b, start_idx:end_idx].detach().clone().to("cpu").tolist()
                active_pos_in_window = s - start_idx

                # Build token_ci_values aligned with the window
                token_ci_values: list[float] = []
                for k in range(len(window_tokens)):
                    orig_idx = start_idx + k
                    if orig_idx < S and bool(mask[b, orig_idx, m].item()):
                        token_ci_values.append(float(causal_importances[b, orig_idx, m].item()))
                    else:
                        token_ci_values.append(0.0)
                # Ensure active token uses the exact firing value
                token_ci_values[active_pos_in_window] = importance_val

                ex = SubcomponentExample(
                    window_tokens=window_tokens,
                    pos=s,
                    active_pos_in_window=active_pos_in_window,
                    token_ci_values=token_ci_values,
                    last_tok_importance=importance_val,
                )
                topk_by_subcomponent[key].maybe_add(ex)
                last_pos_in_seq[lp_key] = s

        tokens_seen_total += B * S

    return topk_by_subcomponent


def _get_importances_by_module(
    cm: ComponentModel, batch: torch.Tensor, config: Config
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        _, pre_weight_acts = cm(
            batch,
            mode="input_cache",
            module_names=list(cm.components.keys()),
        )
        importances_by_module, _ = cm.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            sigmoid_type=config.sigmoid_type,
            detach_inputs=True,
            sampling=config.sampling,
        )
    return importances_by_module


class ActivationContext(BaseModel):
    raw_text: str
    offset_mapping: list[tuple[int, int]]
    token_ci_values: list[float]
    active_position: int
    ci_value: float


class SubcomponentActivationContexts(BaseModel):
    subcomponent_idx: int
    examples: list[ActivationContext]


class ModelActivationContexts(BaseModel):
    layers: dict[str, list[SubcomponentActivationContexts]]


def example_to_activation_context(
    ex: SubcomponentExample,
    tokenizer: PreTrainedTokenizer,
) -> ActivationContext:
    raw_text = tokenizer.decode(ex.window_tokens, add_special_tokens=False)  # pyright: ignore[reportAttributeAccessIssue]
    tokenized = tokenizer.encode_plus(  # pyright: ignore[reportAttributeAccessIssue]
        raw_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=False,
        padding=False,
        add_special_tokens=False,
    )
    offset_mapping = tokenized["offset_mapping"][0].tolist()
    return ActivationContext(
        raw_text=raw_text,
        offset_mapping=offset_mapping,
        token_ci_values=ex.token_ci_values,
        active_position=ex.active_pos_in_window,
        ci_value=ex.last_tok_importance,
    )


def map_to_model_ctxs(
    run_context: RunContext,
    topk_by_subcomponent: dict[tuple[str, int], _TopKExamples],
) -> ModelActivationContexts:
    # use dict of dicts to achieve ≈O(1) access
    layers = defaultdict[str, defaultdict[int, list[ActivationContext]]](lambda: defaultdict(list))
    for (module_name, subcomponent_idx), heap_obj in topk_by_subcomponent.items():
        for ex in heap_obj.to_sorted_list_desc():
            activation_context = example_to_activation_context(ex, run_context.tokenizer)
            layers[module_name][subcomponent_idx].append(activation_context)

    # convert to ModelActivationContexts which uses tuple form for persistence
    return ModelActivationContexts(
        layers={
            layer: [
                SubcomponentActivationContexts(subcomponent_idx=subcomponent_idx, examples=examples)
                for subcomponent_idx, examples in subcomponents.items()
            ]
            for layer, subcomponents in layers.items()
        }
    )
