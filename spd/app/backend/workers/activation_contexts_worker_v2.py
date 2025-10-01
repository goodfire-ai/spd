from __future__ import annotations

import heapq
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from spd.app.backend.services.run_context_service import RunContextService
from spd.configs import Config
from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.settings import SPD_CACHE_DIR
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import extract_batch_data

DEVICE = get_device()


@dataclass
class FastComponentExample:
    # Windowed tokens around the firing position
    window_tokens: list[int]
    # Absolute position within the original sequence
    pos: int
    # Position within window_tokens corresponding to pos
    active_pos_in_window: int
    # CI values aligned to window_tokens
    token_ci_values: list[float]
    # CI value at the firing position
    last_tok_importance: float


@dataclass
class FastComponentSummary:
    module_name: str
    component_idx: int
    density: float
    examples: list[FastComponentExample]


class _TopKExamples:
    def __init__(self, k: int):
        self.k = k
        # Min-heap of tuples (importance, counter, example)
        self.heap: list[tuple[float, int, FastComponentExample]] = []
        self._counter: int = 0

    def maybe_add(self, example: FastComponentExample) -> None:
        key = (example.last_tok_importance, self._counter, example)
        self._counter += 1
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, key)
            return
        # Heap full: replace min if better
        if self.heap[0][0] < example.last_tok_importance:
            heapq.heapreplace(self.heap, key)

    def to_sorted_list_desc(self) -> list[FastComponentExample]:
        # Return examples sorted by importance descending
        return [ex for _, _, ex in sorted(self.heap, key=lambda t: t[0], reverse=True)]


def _default_output_path(wandb_id: str) -> Path:
    run_dir: Path = SPD_CACHE_DIR / "runs" / f"spd-{wandb_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Emit in v1-compatible filename
    return run_dir / "component_activation_contexts.json"


def _write_json_atomic(path: Path, payload: Any) -> None:
    tmp_path: Path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


def main(
    wandb_id: str,
    out: Path | None,
    importance_threshold: float,
    separation_threshold_tokens: int,
    max_examples_per_component: int,
    n_steps: int,
    n_tokens_either_side: int,
) -> Path | None:
    out_path: Path = out or _default_output_path(wandb_id)
    lock_path: Path = out_path.with_suffix(out_path.suffix + ".lock")

    # Try to obtain a simple lock to avoid duplicate computation.
    try:
        with open(lock_path, "x") as f:
            f.write(str(os.getpid()))
    except FileExistsError:
        logger.info(f"Lock exists at {lock_path}, another worker may be running. Exiting.")
        return None

    try:
        activations_json = _main(
            wandb_id,
            importance_threshold,
            separation_threshold_tokens,
            max_examples_per_component,
            n_steps,
            n_tokens_either_side,
        )
        _write_json_atomic(out_path, activations_json)
        logger.info(f"Wrote v1-compatible activation contexts to {out_path}")
        return out_path
    except Exception as e:  # pylint: disable=broad-except
        logger.warning(f"Activation contexts worker v2 failed: {e}")
        return None
    finally:
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            pass


def _main(
    wandb_id: str,
    importance_threshold: float,
    separation_threshold_tokens: int,
    max_examples_per_component: int,
    n_steps: int,
    n_tokens_either_side: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    logger.info("Computing v2 activation contexts in v1 format")

    rcs = RunContextService()
    rcs.load_run_from_wandb_id(wandb_id)
    assert (rc := rcs.run_context) is not None, "Run context not found"

    cm = rc.cm
    cm.to(DEVICE)
    dataloader = rc.train_loader

    # Tracking structures
    topk_by_component: dict[tuple[str, int], _TopKExamples] = {}
    last_pos_in_seq: dict[tuple[str, int, int], int] = {}
    firing_counts: dict[tuple[str, int], int] = {}
    tokens_seen_total: int = 0

    # Iterate limited steps over data
    data_iter = iter(dataloader)
    for _ in range(n_steps):
        batch = extract_batch_data(next(data_iter)).to(DEVICE)
        assert isinstance(batch, torch.Tensor)
        assert batch.ndim == 2, "Expected batch tensor of shape (B, S)"

        B, S = batch.shape

        importances_by_module = _get_importances_by_module(cm, batch, rc.config)

        # Process each module tensor: threshold -> where -> counts -> gather examples
        for module_name, causal_importances in importances_by_module.items():
            assert causal_importances.shape == (B, S, cm.C), "Expected (B,S,C) per module"

            mask = causal_importances > importance_threshold
            if not mask.any():
                continue

            # (K,) indices of all firings
            b_idx, s_idx, m_idx = torch.where(mask)
            K = b_idx.numel()

            # Update density numerators via bincount on component index
            comp_counts = torch.bincount(m_idx, minlength=cm.C)
            for comp_i in range(cm.C):
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
                if key not in topk_by_component:
                    topk_by_component[key] = _TopKExamples(max_examples_per_component)

                heap = topk_by_component[key].heap
                if len(heap) == max_examples_per_component and importance_val <= heap[0][0]:
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

                ex = FastComponentExample(
                    window_tokens=window_tokens,
                    pos=s,
                    active_pos_in_window=active_pos_in_window,
                    token_ci_values=token_ci_values,
                    last_tok_importance=importance_val,
                )
                topk_by_component[key].maybe_add(ex)
                last_pos_in_seq[lp_key] = s

        tokens_seen_total += B * S

    # Build v1-shaped ActivationContextsByModule JSON (componentIdx keys as strings)
    activations_json: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for (module_name, comp), heap_obj in topk_by_component.items():
        examples_sorted = heap_obj.to_sorted_list_desc()
        layer_entry = activations_json.setdefault(module_name, {})
        comp_key = str(comp)
        comp_examples = layer_entry.setdefault(comp_key, [])
        for ex in examples_sorted:
            raw_text = rc.tokenizer.decode(ex.window_tokens, add_special_tokens=False)  # pyright: ignore[reportAttributeAccessIssue]
            tokenized = rc.tokenizer.encode_plus(  # pyright: ignore[reportAttributeAccessIssue]
                raw_text,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=False,
                padding=False,
                add_special_tokens=False,
            )
            offset_mapping = tokenized["offset_mapping"][0].tolist()
            comp_examples.append(
                {
                    "raw_text": raw_text,
                    "offset_mapping": offset_mapping,
                    "token_ci_values": ex.token_ci_values,
                    "active_position": ex.active_pos_in_window,
                    "ci_value": ex.last_tok_importance,
                }
            )

    return activations_json


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
