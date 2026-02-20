"""Edge computation between neural network units.

Provides two context-independent connectivity methods:

1. **Output projections** — logit effects via ``down_proj`` column @ ``lm_head``
2. **Weight-based edges** — upstream connections via ``up_proj``/``gate_proj``/``down_proj``
   weight products (SwiGLU architecture)

Both methods work with standard HuggingFace LlamaForCausalLM / Qwen2ForCausalLM models.

Usage::

    from circuits.connectivity import compute_connectivity, ConnectivityConfig

    edges, enriched_units = compute_connectivity(model, tokenizer, units, ConnectivityConfig())
"""

from __future__ import annotations

import gc
import heapq
from dataclasses import dataclass

import torch

from .schemas import ConnectivityMethod, Edge, TokenProjection, Unit

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ConnectivityConfig:
    """Settings for connectivity computation."""

    method: ConnectivityMethod = ConnectivityMethod.WEIGHT_GRAPH
    top_k_per_neuron: int = 100
    include_output_projections: bool = True
    include_input_projections: bool = True
    chunk_size: int = 4000  # target neurons per matmul chunk
    min_upstream_layer: int = 0


# ---------------------------------------------------------------------------
# Output projections
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_output_projections(
    model,
    tokenizer,
    layer: int,
    neuron: int,
    top_k: int = 50,
) -> dict[str, list[TokenProjection]]:
    """Compute down_proj column @ lm_head to get promoted/suppressed tokens.

    For a given MLP neuron, its output direction is the corresponding column
    of ``down_proj``.  Projecting that direction through ``lm_head`` gives the
    logit contribution for every token in the vocabulary.

    Returns ``{"promotes": [...], "suppresses": [...]}`` with the top-K
    tokens in each direction, sorted by magnitude descending.
    """
    mlp = model.model.layers[layer].mlp
    lm_head = model.lm_head.weight  # [vocab_size, hidden_dim]

    # down_proj: [hidden_dim, intermediate_size] — column *neuron* is the
    # output direction for this neuron.
    output_dir = mlp.down_proj.weight[:, neuron].float()  # [hidden_dim]

    # Project through lm_head: [vocab_size, hidden_dim] @ [hidden_dim] = [vocab_size]
    logits = torch.matmul(lm_head.float(), output_dir)  # [vocab_size]

    # Top-K promoted (highest logit contribution)
    top_pos_vals, top_pos_ids = torch.topk(logits, top_k)
    promotes = []
    for val, tid in zip(top_pos_vals, top_pos_ids):
        token_id = tid.item()
        promotes.append(TokenProjection(
            token=tokenizer.decode([token_id]),
            token_id=token_id,
            weight=round(val.item(), 4),
        ))

    # Top-K suppressed (most negative logit contribution)
    top_neg_vals, top_neg_ids = torch.topk(-logits, top_k)
    suppresses = []
    for val, tid in zip(top_neg_vals, top_neg_ids):
        token_id = tid.item()
        suppresses.append(TokenProjection(
            token=tokenizer.decode([token_id]),
            token_id=token_id,
            weight=round(-val.item(), 4),
        ))

    return {"promotes": promotes, "suppresses": suppresses}


# ---------------------------------------------------------------------------
# Weight-based edges
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_weight_edges(
    model,
    target_layer: int,
    config: ConnectivityConfig,
) -> list[Edge]:
    """Compute weight-based upstream connections for all neurons in a layer.

    For each target neuron *N* at ``target_layer``, for each upstream layer::

        c_up[m]   = up_proj[N, :] · down_proj[:, m]
        c_gate[m] = gate_proj[N, :] · down_proj[:, m]

    Polarity is determined by sign agreement:
    - both positive → excitatory
    - both negative → inhibitory

    Strength is ``|c_up| + |c_gate|``.  The top-K connections per polarity
    are kept across *all* upstream layers using a min-heap, so memory stays
    bounded.

    Returns a list of :class:`Edge` objects with
    ``method=ConnectivityMethod.WEIGHT_GRAPH``.
    """
    target_mlp = model.model.layers[target_layer].mlp
    n_intermediate = target_mlp.up_proj.weight.shape[0]

    # Target weights — kept in native dtype, converted per chunk
    e_up_full = target_mlp.up_proj.weight      # [intermediate, hidden]
    e_gate_full = target_mlp.gate_proj.weight   # [intermediate, hidden]

    top_k = config.top_k_per_neuron
    chunk_size = config.chunk_size
    min_upstream = config.min_upstream_layer

    # Per-neuron heaps: (strength, upstream_layer, upstream_neuron)
    # Min-heap so smallest element is evicted when a stronger one arrives.
    heaps_exc: list[list] = [[] for _ in range(n_intermediate)]
    heaps_inh: list[list] = [[] for _ in range(n_intermediate)]

    for up_layer in range(min_upstream, target_layer):
        up_down_proj = model.model.layers[up_layer].mlp.down_proj.weight
        # [hidden, intermediate_upstream]

        for chunk_start in range(0, n_intermediate, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_intermediate)

            e_up_chunk = e_up_full[chunk_start:chunk_end].float()
            e_gate_chunk = e_gate_full[chunk_start:chunk_end].float()
            up_down_float = up_down_proj.float()

            # [chunk, hidden] @ [hidden, intermediate_upstream] = [chunk, intermediate_upstream]
            c_up_matrix = torch.matmul(e_up_chunk, up_down_float)
            c_gate_matrix = torch.matmul(e_gate_chunk, up_down_float)

            for i in range(chunk_end - chunk_start):
                target_idx = chunk_start + i
                c_up = c_up_matrix[i]
                c_gate = c_gate_matrix[i]

                both_pos = (c_up > 0) & (c_gate > 0)
                both_neg = (c_up < 0) & (c_gate < 0)
                c_combined = torch.abs(c_up) + torch.abs(c_gate)

                # --- excitatory ---
                exc_vals = c_combined.clone()
                exc_vals[~both_pos] = 0
                if exc_vals.max() > 0:
                    k = min(top_k, int((exc_vals > 0).sum().item()))
                    if k > 0:
                        topk = torch.topk(exc_vals, k)
                        heap = heaps_exc[target_idx]
                        for j in range(k):
                            strength = topk.values[j].item()
                            nidx = topk.indices[j].item()
                            if len(heap) < top_k:
                                heapq.heappush(heap, (strength, up_layer, nidx))
                            elif strength > heap[0][0]:
                                heapq.heapreplace(heap, (strength, up_layer, nidx))

                # --- inhibitory ---
                inh_vals = c_combined.clone()
                inh_vals[~both_neg] = 0
                if inh_vals.max() > 0:
                    k = min(top_k, int((inh_vals > 0).sum().item()))
                    if k > 0:
                        topk = torch.topk(inh_vals, k)
                        heap = heaps_inh[target_idx]
                        for j in range(k):
                            strength = topk.values[j].item()
                            nidx = topk.indices[j].item()
                            if len(heap) < top_k:
                                heapq.heappush(heap, (strength, up_layer, nidx))
                            elif strength > heap[0][0]:
                                heapq.heapreplace(heap, (strength, up_layer, nidx))

            del c_up_matrix, c_gate_matrix, e_up_chunk, e_gate_chunk, up_down_float
            torch.cuda.empty_cache()

    # Assemble Edge objects
    edges: list[Edge] = []
    for target_idx in range(n_intermediate):
        for strength, up_layer, up_neuron in heaps_exc[target_idx]:
            edges.append(Edge(
                src_layer=up_layer,
                src_index=up_neuron,
                tgt_layer=target_layer,
                tgt_index=target_idx,
                weight=round(strength, 6),
                method=ConnectivityMethod.WEIGHT_GRAPH,
            ))
        for strength, up_layer, up_neuron in heaps_inh[target_idx]:
            edges.append(Edge(
                src_layer=up_layer,
                src_index=up_neuron,
                tgt_layer=target_layer,
                tgt_index=target_idx,
                weight=round(-strength, 6),  # negative weight for inhibitory
                method=ConnectivityMethod.WEIGHT_GRAPH,
            ))

    del heaps_exc, heaps_inh
    gc.collect()

    return edges


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_connectivity(
    model,
    tokenizer,
    units: list[Unit],
    config: ConnectivityConfig,
) -> tuple[list[Edge], list[Unit]]:
    """Compute edges and enrich units with output projections.

    Dispatches to the configured connectivity method for edge computation,
    and optionally computes output projections for each unit.

    Returns ``(edges, enriched_units)`` where *enriched_units* is a copy of
    the input list with ``output_projections`` populated.
    """
    edges: list[Edge] = []

    # Compute weight-based edges if requested
    if config.method == ConnectivityMethod.WEIGHT_GRAPH:
        # Find unique target layers from the provided units
        target_layers = sorted({u.layer for u in units})
        for layer in target_layers:
            layer_edges = compute_weight_edges(model, layer, config)
            edges.extend(layer_edges)

    # Enrich units with output projections
    enriched_units = list(units)
    if config.include_output_projections:
        for unit in enriched_units:
            unit.output_projections = compute_output_projections(
                model, tokenizer, unit.layer, unit.index,
                top_k=config.top_k_per_neuron,
            )

    return edges, enriched_units
