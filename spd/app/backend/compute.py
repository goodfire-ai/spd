"""Core attribution computation functions.

Copied and cleaned up from spd/scripts/calc_prompt_attributions.py and calc_dataset_attributions.py
to avoid importing script files with global execution.
"""

import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, override

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.app.backend.optim_cis import OptimCIConfig, OptimizationMetrics, optimize_ci_values
from spd.configs import SamplingType
from spd.log import logger
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.models.components import make_mask_infos
from spd.topology import CanonicalWeight, Embed, TransformerTopology
from spd.utils.general_utils import bf16_autocast


@dataclass
class LayerAliveInfo:
    """Info about alive components for a layer."""

    alive_mask: Bool[Tensor, "s C"]  # Which (pos, component) pairs are alive
    alive_c_idxs: list[int]  # Components alive at any position


MAX_OUTPUT_NODES_PER_POS = 15


def compute_layer_alive_info(
    layer_name: str,
    ci_lower_leaky: dict[str, Tensor],
    ci_masked_out_probs: Float[Tensor, "1 seq vocab"] | None,
    output_prob_threshold: float,
    n_seq: int,
    device: str,
    topology: TransformerTopology,
) -> LayerAliveInfo:
    """Compute alive info for a layer. Handles regular, embedding, and unembed layers.

    For CI layers, all components with CI > 0 are considered alive.
    Filtering by CI threshold is done at display time, not computation time.

    For unembed layer, caps at MAX_OUTPUT_NODES_PER_POS per position to keep
    edge computation tractable with large vocabularies.
    """
    embed_path = topology.path_schema.embedding_path
    unembed_path = topology.path_schema.unembed_path

    if layer_name == embed_path:
        alive_mask = torch.ones(n_seq, 1, device=device, dtype=torch.bool)
        alive_c_idxs = [0]
    elif layer_name == unembed_path:
        assert ci_masked_out_probs is not None
        assert ci_masked_out_probs.shape[0] == 1
        probs = ci_masked_out_probs[0]  # [seq, vocab]
        alive_mask = probs > output_prob_threshold
        # Cap per position: keep only top-k per seq pos
        for s in range(n_seq):
            pos_alive = torch.where(alive_mask[s])[0]
            if len(pos_alive) > MAX_OUTPUT_NODES_PER_POS:
                pos_probs = probs[s, pos_alive]
                _, keep_local = torch.topk(pos_probs, MAX_OUTPUT_NODES_PER_POS)
                keep_idxs = pos_alive[keep_local]
                alive_mask[s] = False
                alive_mask[s, keep_idxs] = True
        alive_c_idxs = torch.where(alive_mask.any(dim=0))[0].tolist()
    else:
        ci = ci_lower_leaky[layer_name]
        assert ci.shape[0] == 1
        alive_mask = ci[0] > 0.0
        alive_c_idxs = torch.where(alive_mask.any(dim=0))[0].tolist()

    return LayerAliveInfo(alive_mask, alive_c_idxs)


@dataclass
class Node:
    layer: CanonicalWeight
    seq_pos: int
    component_idx: int

    @override
    def __str__(self) -> str:
        return f"{self.layer.canonical_str()}:{self.seq_pos}:{self.component_idx}"


def _get_seq_pos(node_key: str) -> int:
    """Extract sequence position from node key format 'layer:seq:cIdx'."""
    parts = node_key.split(":")
    assert len(parts) == 3, f"Invalid node key format: {node_key}"
    return int(parts[1])


@dataclass
class Edge:
    """Edge in the attribution graph."""

    source: Node
    target: Node
    strength: float
    is_cross_seq: bool


@dataclass
class PromptAttributionResult:
    """Result of computing prompt attributions for a prompt."""

    edges: list[Edge]
    ci_masked_out_probs: Float[Tensor, "seq vocab"]  # CI-masked (SPD model) softmax probabilities
    ci_masked_out_logits: Float[Tensor, "seq vocab"]  # CI-masked (SPD model) raw logits
    target_out_probs: Float[Tensor, "seq vocab"]  # Target model softmax probabilities
    target_out_logits: Float[Tensor, "seq vocab"]  # Target model raw logits
    node_ci_vals: dict[str, float]  # layer:seq:c_idx -> ci_val
    node_subcomp_acts: dict[str, float]  # layer:seq:c_idx -> subcomponent activation (v_i^T @ a)


@dataclass
class OptimizedPromptAttributionResult:
    """Result of computing prompt attributions with optimized CI values."""

    edges: list[Edge]
    ci_masked_out_probs: Float[Tensor, "seq vocab"]  # CI-masked (SPD model) softmax probabilities
    ci_masked_out_logits: Float[Tensor, "seq vocab"]  # CI-masked (SPD model) raw logits
    target_out_probs: Float[Tensor, "seq vocab"]  # Target model softmax probabilities
    target_out_logits: Float[Tensor, "seq vocab"]  # Target model raw logits
    node_ci_vals: dict[str, float]  # layer:seq:c_idx -> ci_val
    node_subcomp_acts: dict[str, float]  # layer:seq:c_idx -> subcomponent activation (v_i^T @ a)
    metrics: OptimizationMetrics  # Final loss metrics from optimization


def get_sources_by_target(
    model: ComponentModel,
    topology: TransformerTopology,
    device: str,
    sampling: SamplingType,
) -> dict[str, list[str]]:
    """Find valid gradient connections grouped by target layer.

    Includes embedding as a source and unembed as a target, using the topology's
    actual module paths (not pseudo-names).

    Returns:
        Dict mapping out_layer -> list of in_layers that have gradient flow to it.
    """
    # Use a small dummy batch - we only need to trace gradient connections
    batch: Float[Tensor, "batch seq"] = torch.zeros(2, 3, dtype=torch.long, device=device)

    with torch.no_grad(), bf16_autocast():
        output_with_cache: OutputWithCache = model(batch, cache_type="input")

        ci = model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache,
            sampling=sampling,
            detach_inputs=False,
        )

    # Create masks so we can use all components
    mask_infos = make_mask_infos(
        component_masks={k: torch.ones_like(v) for k, v in ci.lower_leaky.items()},
        routing_masks="all",
    )

    embed_path = topology.path_schema.embedding_path
    unembed_path = topology.path_schema.unembed_path

    # Hook to capture embedding output with gradients
    embed_cache: dict[str, Tensor] = {}

    def embed_hook(
        _module: nn.Module, _args: tuple[Any, ...], _kwargs: dict[Any, Any], output: Tensor
    ) -> Any:
        output.requires_grad_(True)
        embed_cache[f"{embed_path}_post_detach"] = output
        return output

    embed_handle = topology.embedding_module.register_forward_hook(embed_hook, with_kwargs=True)

    with torch.enable_grad(), bf16_autocast():
        comp_output_with_cache: OutputWithCache = model(
            batch,
            mask_infos=mask_infos,
            cache_type="component_acts",
        )

    embed_handle.remove()

    cache = comp_output_with_cache.cache
    cache[f"{embed_path}_post_detach"] = embed_cache[f"{embed_path}_post_detach"]
    cache[f"{unembed_path}_pre_detach"] = comp_output_with_cache.output

    layers = [embed_path, *model.target_module_paths, unembed_path]

    # Test all ordered pairs for gradient flow
    test_pairs = []
    for in_layer in layers[:-1]:  # Don't include "output" as source
        for out_layer in layers[1:]:  # Don't include embed as target
            if layers.index(in_layer) < layers.index(out_layer):
                test_pairs.append((in_layer, out_layer))

    sources_by_target: dict[str, list[str]] = defaultdict(list)
    for in_layer, out_layer in test_pairs:
        out_pre_detach = cache[f"{out_layer}_pre_detach"]
        in_post_detach = cache[f"{in_layer}_post_detach"]
        out_value = out_pre_detach[0, 0, 0]
        grads = torch.autograd.grad(
            outputs=out_value,
            inputs=in_post_detach,
            retain_graph=True,
            allow_unused=True,
        )
        assert len(grads) == 1
        grad = grads[0]
        if grad is not None:  # pyright: ignore[reportUnnecessaryComparison]
            sources_by_target[out_layer].append(in_layer)
    return dict(sources_by_target)


ProgressCallback = Callable[[int, int, str], None]  # (current, total, stage)


def _setup_embed_hook() -> tuple[Callable[..., Any], list[Tensor]]:
    """Create hook to capture embedding output with gradients.

    Returns the hook function and a mutable container for the cached output.
    The container is a list to allow mutation from the hook closure.
    """
    embed_cache: list[Tensor] = []

    def embed_hook(
        _module: nn.Module, _args: tuple[Any, ...], _kwargs: dict[Any, Any], output: Tensor
    ) -> Any:
        output.requires_grad_(True)
        assert len(embed_cache) == 0, "embedding output should be cached only once"
        embed_cache.append(output)
        return output

    return embed_hook, embed_cache


def _compute_edges_for_target(
    target: str,
    sources: list[str],
    target_info: LayerAliveInfo,
    source_infos: list[LayerAliveInfo],
    cache: dict[str, Tensor],
    loss_seq_pos: int,
    topology: TransformerTopology,
) -> list[Edge]:
    """Compute all edges flowing into a single target layer.

    For each alive (s_out, c_out) in the target layer, computes gradient-based
    attribution strengths from all alive source components.

    Args:
        loss_seq_pos: Maximum sequence position to include (inclusive).
                      Only compute edges for target positions <= loss_seq_pos.
    """
    edges: list[Edge] = []
    out_pre_detach: Float[Tensor, "1 s C"] = cache[f"{target}_pre_detach"]
    in_post_detaches: list[Float[Tensor, "1 s C"]] = [
        cache[f"{source}_post_detach"] for source in sources
    ]

    for s_out in range(loss_seq_pos + 1):
        s_out_alive_c = [c for c in target_info.alive_c_idxs if target_info.alive_mask[s_out, c]]
        if not s_out_alive_c:
            continue

        for c_out in s_out_alive_c:
            grads = torch.autograd.grad(
                outputs=out_pre_detach[0, s_out, c_out],
                inputs=in_post_detaches,
                retain_graph=True,
            )
            with torch.no_grad():
                canonical_target = topology.get_canonical_weight(target)
                for source, source_info, grad, in_post_detach in zip(
                    sources, source_infos, grads, in_post_detaches, strict=True
                ):
                    canonical_source = topology.get_canonical_weight(source)
                    is_cross_seq = topology.is_cross_seq_pair(canonical_source, canonical_target)
                    weighted: Float[Tensor, "s C"] = (grad * in_post_detach)[0]
                    if isinstance(canonical_source, Embed):
                        weighted = weighted.sum(dim=1, keepdim=True)

                    s_in_range = range(s_out + 1) if is_cross_seq else [s_out]
                    for s_in in s_in_range:
                        for c_in in source_info.alive_c_idxs:
                            if not source_info.alive_mask[s_in, c_in]:
                                continue
                            edges.append(
                                Edge(
                                    source=Node(
                                        layer=canonical_source, seq_pos=s_in, component_idx=c_in
                                    ),
                                    target=Node(
                                        layer=canonical_target, seq_pos=s_out, component_idx=c_out
                                    ),
                                    strength=weighted[s_in, c_in].item(),
                                    is_cross_seq=is_cross_seq,
                                )
                            )
    return edges


def compute_edges_from_ci(
    model: ComponentModel,
    topology: TransformerTopology,
    tokens: Float[Tensor, "1 seq"],
    ci_lower_leaky: dict[str, Float[Tensor, "1 seq C"]],
    pre_weight_acts: dict[str, Float[Tensor, "1 seq d_in"]],
    sources_by_target: dict[str, list[str]],
    target_out_probs: Float[Tensor, "1 seq vocab"],
    target_out_logits: Float[Tensor, "1 seq vocab"],
    output_prob_threshold: float,
    device: str,
    on_progress: ProgressCallback | None = None,
    loss_seq_pos: int | None = None,
) -> PromptAttributionResult:
    """Core edge computation from pre-computed CI values.

    Computes gradient-based attribution edges between components using the
    provided CI values for masking. All components with CI > 0 are included;
    filtering by CI threshold is done at display time.

    For the attribution computation, we use:
    1. unmasked components. This is because we don't want to have to rely on the CI masks being
        very accurate, we still want to pick up all the attribution information we can.
    2. unmasked weight deltas. After all, these are conceptually the same as components that our
        model is using to approximate the target model.

    We compute CI-masked output probs separately (for display) before running the unmasked
    forward pass used for gradient computation.

    Args:
        loss_seq_pos: Maximum sequence position to include (inclusive).
                      If None, includes all positions (default behavior).
    """
    t_start = time.perf_counter()
    n_seq = tokens.shape[1]
    if loss_seq_pos is None:
        loss_seq_pos = n_seq - 1

    # Compute CI-masked output probs (for display) before the gradient computation
    t0 = time.perf_counter()
    with torch.no_grad(), bf16_autocast():
        ci_masks = make_mask_infos(component_masks=ci_lower_leaky)
        ci_masked_logits: Tensor = model(tokens, mask_infos=ci_masks)
        ci_masked_out_probs = torch.softmax(ci_masked_logits, dim=-1)
    logger.info(f"[perf] CI-masked forward: {time.perf_counter() - t0:.2f}s")

    embed_path = topology.path_schema.embedding_path
    unembed_path = topology.path_schema.unembed_path

    # Setup embedding hook and run forward pass for gradient computation
    t0 = time.perf_counter()
    embed_hook, embed_cache = _setup_embed_hook()
    embed_handle = topology.embedding_module.register_forward_hook(embed_hook, with_kwargs=True)

    weight_deltas = model.calc_weight_deltas()
    weight_deltas_and_masks = {
        k: (v, torch.ones(tokens.shape, device=device)) for k, v in weight_deltas.items()
    }
    unmasked_masks = make_mask_infos(
        component_masks={k: torch.ones_like(v) for k, v in ci_lower_leaky.items()},
        weight_deltas_and_masks=weight_deltas_and_masks,
    )
    with torch.enable_grad(), bf16_autocast():
        comp_output_with_cache: OutputWithCache = model(
            tokens, mask_infos=unmasked_masks, cache_type="component_acts"
        )

    embed_handle.remove()
    assert len(embed_cache) == 1, "embedding output should be cached"

    cache = comp_output_with_cache.cache
    cache[f"{embed_path}_post_detach"] = embed_cache[0]
    cache[f"{unembed_path}_pre_detach"] = comp_output_with_cache.output
    logger.info(f"[perf] Gradient forward pass: {time.perf_counter() - t0:.2f}s")

    # Compute alive info for all layers upfront
    t0 = time.perf_counter()
    all_layers: set[str] = set(sources_by_target.keys())
    for sources in sources_by_target.values():
        all_layers.update(sources)

    alive_info: dict[str, LayerAliveInfo] = {
        layer: compute_layer_alive_info(
            layer_name=layer,
            ci_lower_leaky=ci_lower_leaky,
            ci_masked_out_probs=ci_masked_out_probs,
            output_prob_threshold=output_prob_threshold,
            n_seq=n_seq,
            device=device,
            topology=topology,
        )
        for layer in all_layers
    }
    total_alive = sum(len(info.alive_c_idxs) for info in alive_info.values())
    unembed_alive = len(alive_info.get(unembed_path, LayerAliveInfo(torch.tensor([]), [])).alive_c_idxs)
    logger.info(
        f"[perf] Alive info: {time.perf_counter() - t0:.2f}s "
        f"({total_alive} alive components, {unembed_alive} output nodes)"
    )

    # Compute edges for each target layer
    t0 = time.perf_counter()
    edges: list[Edge] = []
    total_source_layers = sum(len(sources) for sources in sources_by_target.values())
    progress_count = 0

    for target, sources in sources_by_target.items():
        t_target = time.perf_counter()
        target_edges = _compute_edges_for_target(
            target=target,
            sources=sources,
            target_info=alive_info[target],
            source_infos=[alive_info[source] for source in sources],
            cache=cache,
            loss_seq_pos=loss_seq_pos,
            topology=topology,
        )
        edges.extend(target_edges)
        canonical_target = topology.get_canonical_weight(target).canonical_str()
        logger.info(
            f"[perf]   {canonical_target}: {time.perf_counter() - t_target:.2f}s, "
            f"{len(target_edges)} edges"
        )

        progress_count += len(sources)
        if on_progress is not None:
            on_progress(progress_count, total_source_layers, target)

    logger.info(f"[perf] Edge computation total: {time.perf_counter() - t0:.2f}s ({len(edges)} edges)")

    t0 = time.perf_counter()
    node_ci_vals = extract_node_ci_vals(ci_lower_leaky, topology)
    component_acts = model.get_all_component_acts(pre_weight_acts)
    node_subcomp_acts = extract_node_subcomp_acts(
        component_acts, ci_threshold=0.0, ci_lower_leaky=ci_lower_leaky, topology=topology
    )
    logger.info(f"[perf] Node CI/subcomp extraction: {time.perf_counter() - t0:.2f}s")

    # Filter nodes and output tensors to only include positions <= loss_seq_pos
    node_ci_vals = {k: v for k, v in node_ci_vals.items() if _get_seq_pos(k) <= loss_seq_pos}
    node_subcomp_acts = {
        k: v for k, v in node_subcomp_acts.items() if _get_seq_pos(k) <= loss_seq_pos
    }

    return PromptAttributionResult(
        edges=edges,
        ci_masked_out_probs=ci_masked_out_probs[0, : loss_seq_pos + 1],
        ci_masked_out_logits=ci_masked_logits[0, : loss_seq_pos + 1],
        target_out_probs=target_out_probs[0, : loss_seq_pos + 1],
        target_out_logits=target_out_logits[0, : loss_seq_pos + 1],
        node_ci_vals=node_ci_vals,
        node_subcomp_acts=node_subcomp_acts,
    )


def filter_ci_to_included_nodes(
    ci_lower_leaky: dict[str, Float[Tensor, "1 seq C"]],
    included_nodes: set[str],
) -> dict[str, Float[Tensor, "1 seq C"]]:
    """Zero out CI values for nodes not in included_nodes.

    This causes compute_layer_alive_info() to mark them as not alive,
    so they're skipped during edge computation (more efficient than
    filtering edges after computation).

    Uses batch tensor operations for efficiency with large node sets.

    Args:
        ci_lower_leaky: Dict mapping layer name to CI tensor [1, seq, C].
        included_nodes: Set of node keys to include (format: "layer:seq:cIdx").

    Returns:
        New dict with CI values zeroed for non-included nodes.

    Raises:
        AssertionError: If any node has invalid format or references invalid layer.
    """
    # Pre-group nodes by layer: layer -> list of (seq_pos, c_idx)
    nodes_by_layer: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for node_key in included_nodes:
        parts = node_key.split(":")
        assert len(parts) == 3, f"Invalid node key format: {node_key}"
        layer, seq_str, c_str = parts
        nodes_by_layer[layer].append((int(seq_str), int(c_str)))

    # Validate all layers exist (Issue 8: fail fast on invalid nodes)
    valid_layers = set(ci_lower_leaky.keys())
    invalid_layers = set(nodes_by_layer.keys()) - valid_layers
    assert not invalid_layers, f"Nodes reference invalid layers: {invalid_layers}"

    filtered = {}
    for layer_name, ci_tensor in ci_lower_leaky.items():
        new_ci = torch.zeros_like(ci_tensor)
        n_seq, n_components = ci_tensor.shape[1], ci_tensor.shape[2]

        coords = nodes_by_layer.get(layer_name, [])
        if coords:
            # Validate bounds
            for seq_pos, c_idx in coords:
                assert 0 <= seq_pos < n_seq, f"seq_pos {seq_pos} out of bounds [0, {n_seq})"
                assert 0 <= c_idx < n_components, f"c_idx {c_idx} out of bounds [0, {n_components})"

            # Batch assignment using advanced indexing (more efficient for large node sets)
            seq_indices = torch.tensor([c[0] for c in coords], device=ci_tensor.device)
            c_indices = torch.tensor([c[1] for c in coords], device=ci_tensor.device)
            new_ci[0, seq_indices, c_indices] = ci_tensor[0, seq_indices, c_indices]

        filtered[layer_name] = new_ci

    return filtered


def compute_prompt_attributions(
    model: ComponentModel,
    topology: TransformerTopology,
    tokens: Float[Tensor, "1 seq"],
    sources_by_target: dict[str, list[str]],
    output_prob_threshold: float,
    sampling: SamplingType,
    device: str,
    on_progress: ProgressCallback | None = None,
    included_nodes: set[str] | None = None,
    loss_seq_pos: int | None = None,
) -> PromptAttributionResult:
    """Compute prompt attributions using the model's natural CI values.

    Computes CI via forward pass, then delegates to compute_edges_from_ci().
    For optimized sparse CI values, use compute_prompt_attributions_optimized().

    If included_nodes is provided, CI values for non-included nodes are zeroed out
    before edge computation. This efficiently filters to only compute edges between
    the specified nodes (useful for generating graphs from a selection).

    Args:
        loss_seq_pos: Maximum sequence position to include (inclusive).
                      If None, includes all positions (default behavior).
    """
    t0 = time.perf_counter()
    with torch.no_grad(), bf16_autocast():
        output_with_cache = model(tokens, cache_type="input")
        pre_weight_acts = output_with_cache.cache
        target_out_logits = output_with_cache.output
        target_out_probs = torch.softmax(target_out_logits, dim=-1)
        ci = model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            sampling=sampling,
            detach_inputs=False,
        )
    logger.info(f"[perf] CI forward pass: {time.perf_counter() - t0:.2f}s")

    ci_lower_leaky = ci.lower_leaky
    if included_nodes is not None:
        ci_lower_leaky = filter_ci_to_included_nodes(ci_lower_leaky, included_nodes)

    return compute_edges_from_ci(
        model=model,
        topology=topology,
        tokens=tokens,
        ci_lower_leaky=ci_lower_leaky,
        pre_weight_acts=pre_weight_acts,
        sources_by_target=sources_by_target,
        target_out_probs=target_out_probs,
        target_out_logits=target_out_logits,
        output_prob_threshold=output_prob_threshold,
        device=device,
        on_progress=on_progress,
        loss_seq_pos=loss_seq_pos,
    )


def compute_prompt_attributions_optimized(
    model: ComponentModel,
    topology: TransformerTopology,
    tokens: Float[Tensor, "1 seq"],
    sources_by_target: dict[str, list[str]],
    optim_config: OptimCIConfig,
    output_prob_threshold: float,
    device: str,
    on_progress: ProgressCallback | None = None,
) -> OptimizedPromptAttributionResult:
    """Compute prompt attributions using optimized sparse CI values.

    Runs CI optimization to find a minimal sparse mask that preserves
    the model's prediction, then computes edges.

    L0 stats are computed dynamically at display time from node_ci_vals,
    not here at computation time.
    """
    # Compute target model output probs (unmasked forward pass)
    with torch.no_grad(), bf16_autocast():
        target_logits = model(tokens)
        target_out_probs = torch.softmax(target_logits, dim=-1)

    optim_result = optimize_ci_values(
        model=model,
        tokens=tokens,
        config=optim_config,
        device=device,
        on_progress=on_progress,
    )
    ci_outputs = optim_result.params.create_ci_outputs(model, device)

    # Signal transition to graph computation stage
    if on_progress is not None:
        on_progress(0, 1, "graph")

    # Get pre_weight_acts for subcomponent activation computation
    with torch.no_grad(), bf16_autocast():
        pre_weight_acts = model(tokens, cache_type="input").cache

    # Extract loss_seq_pos from optimization config
    loss_seq_pos = optim_config.loss_config.position

    result = compute_edges_from_ci(
        model=model,
        topology=topology,
        tokens=tokens,
        ci_lower_leaky=ci_outputs.lower_leaky,
        pre_weight_acts=pre_weight_acts,
        sources_by_target=sources_by_target,
        target_out_probs=target_out_probs,
        target_out_logits=target_logits,
        output_prob_threshold=output_prob_threshold,
        device=device,
        on_progress=on_progress,
        loss_seq_pos=loss_seq_pos,
    )

    return OptimizedPromptAttributionResult(
        edges=result.edges,
        ci_masked_out_probs=result.ci_masked_out_probs,
        ci_masked_out_logits=result.ci_masked_out_logits,
        target_out_probs=result.target_out_probs,
        target_out_logits=result.target_out_logits,
        node_ci_vals=result.node_ci_vals,
        node_subcomp_acts=result.node_subcomp_acts,
        metrics=optim_result.metrics,
    )


@dataclass
class CIOnlyResult:
    """Result of computing CI values only (no attribution graph)."""

    ci_lower_leaky: dict[str, Float[Tensor, "1 seq n_components"]]
    target_out_probs: Float[Tensor, "1 seq vocab"]  # Target model (unmasked) softmax probs
    pre_weight_acts: dict[str, Float[Tensor, "1 seq d_in"]]
    component_acts: dict[str, Float[Tensor, "1 seq C"]]


def compute_ci_only(
    model: ComponentModel,
    tokens: Float[Tensor, "1 seq"],
    sampling: SamplingType,
) -> CIOnlyResult:
    """Fast CI computation without full attribution graph.

    This is much faster than compute_prompt_attributions() because it only
    requires a forward pass and CI computation (no gradient loop).

    Args:
        model: The ComponentModel to analyze.
        tokens: Tokenized prompt of shape [1, seq_len].
        sampling: Sampling type to use for causal importances.

    Returns:
        CIOnlyResult containing CI values per layer, target model output probabilities, pre-weight activations, and component activations.
    """
    with torch.no_grad(), bf16_autocast():
        output_with_cache: OutputWithCache = model(tokens, cache_type="input")
        ci = model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache,
            sampling=sampling,
            detach_inputs=False,
        )
        target_out_probs = torch.softmax(output_with_cache.output, dim=-1)
        component_acts = model.get_all_component_acts(output_with_cache.cache)

    return CIOnlyResult(
        ci_lower_leaky=ci.lower_leaky,
        target_out_probs=target_out_probs,
        pre_weight_acts=output_with_cache.cache,
        component_acts=component_acts,
    )


def extract_node_ci_vals(
    ci_lower_leaky: dict[str, Float[Tensor, "1 seq n_components"]],
    topology: TransformerTopology,
) -> dict[str, float]:
    """Extract per-node CI values from CI tensors.

    Returns dict mapping canonical node key to CI value.
    """
    node_ci_vals: dict[str, float] = {}
    for layer_name, ci_tensor in ci_lower_leaky.items():
        canonical = topology.get_canonical_weight(layer_name).canonical_str()
        n_seq = ci_tensor.shape[1]
        n_components = ci_tensor.shape[2]
        for seq_pos in range(n_seq):
            for c_idx in range(n_components):
                key = f"{canonical}:{seq_pos}:{c_idx}"
                node_ci_vals[key] = float(ci_tensor[0, seq_pos, c_idx].item())
    return node_ci_vals


def extract_node_subcomp_acts(
    component_acts: dict[str, Float[Tensor, "1 seq C"]],
    ci_threshold: float,
    ci_lower_leaky: dict[str, Float[Tensor, "1 seq C"]],
    topology: TransformerTopology,
) -> dict[str, float]:
    """Extract per-node subcomponent activations from pre-computed component acts.

    Returns dict mapping canonical node key to subcomponent activation value.
    """
    node_subcomp_acts: dict[str, float] = {}
    for layer_name, subcomp_acts in component_acts.items():
        canonical = topology.get_canonical_weight(layer_name).canonical_str()
        ci = ci_lower_leaky[layer_name]
        alive_mask = ci[0] > ci_threshold  # [seq, C]
        alive_seq_indices, alive_c_indices = torch.where(alive_mask)
        for seq_pos, c_idx in zip(
            alive_seq_indices.tolist(), alive_c_indices.tolist(), strict=True
        ):
            key = f"{canonical}:{seq_pos}:{c_idx}"
            node_subcomp_acts[key] = float(subcomp_acts[0, seq_pos, c_idx].item())

    return node_subcomp_acts


@dataclass
class InterventionResult:
    """Result of intervention forward pass."""

    input_tokens: list[str]
    predictions_per_position: list[
        list[tuple[str, int, float, float, float, float]]
    ]  # [(token, id, spd_prob, logit, target_prob, target_logit)]


def compute_intervention_forward(
    model: ComponentModel,
    tokens: Float[Tensor, "1 seq"],
    active_nodes: list[tuple[str, int, int]],  # [(layer, seq_pos, component_idx)]
    top_k: int,
    tokenizer: AppTokenizer,
) -> InterventionResult:
    """Forward pass with only specified nodes active.

    Args:
        model: ComponentModel to run intervention on.
        tokens: Input tokens of shape [1, seq].
        active_nodes: List of (layer, seq_pos, component_idx) tuples specifying which nodes to activate.
        top_k: Number of top predictions to return per position.
        tokenizer: Tokenizer for decoding tokens.

    Returns:
        InterventionResult with input tokens and top-k predictions per position.
    """

    seq_len = tokens.shape[1]
    device = tokens.device

    # Build component masks: all zeros, then set 1s for active nodes
    component_masks: dict[str, Float[Tensor, "1 seq C"]] = {}
    for layer_name, C in model.module_to_c.items():
        component_masks[layer_name] = torch.zeros(1, seq_len, C, device=device)

    for layer, seq_pos, c_idx in active_nodes:
        assert layer in component_masks, f"Layer {layer} not in model"
        assert 0 <= seq_pos < seq_len, f"seq_pos {seq_pos} out of bounds [0, {seq_len})"
        assert 0 <= c_idx < model.module_to_c[layer], (
            f"component_idx {c_idx} out of bounds [0, {model.module_to_c[layer]})"
        )
        component_masks[layer][0, seq_pos, c_idx] = 1.0

    mask_infos = make_mask_infos(component_masks, routing_masks="all")

    with torch.no_grad(), bf16_autocast():
        # SPD model forward pass (with component masks)
        spd_logits: Float[Tensor, "1 seq vocab"] = model(tokens, mask_infos=mask_infos)
        spd_probs: Float[Tensor, "1 seq vocab"] = torch.softmax(spd_logits, dim=-1)

        # Target model forward pass (no masks)
        target_logits: Float[Tensor, "1 seq vocab"] = model(tokens)
        target_out_probs: Float[Tensor, "1 seq vocab"] = torch.softmax(target_logits, dim=-1)

    # Get top-k predictions per position (based on SPD model's top-k)
    predictions_per_position: list[list[tuple[str, int, float, float, float, float]]] = []
    for pos in range(seq_len):
        pos_spd_probs = spd_probs[0, pos]
        pos_spd_logits = spd_logits[0, pos]
        pos_target_out_probs = target_out_probs[0, pos]
        pos_target_logits = target_logits[0, pos]
        top_probs, top_ids = torch.topk(pos_spd_probs, top_k)

        pos_predictions: list[tuple[str, int, float, float, float, float]] = []
        for spd_prob, token_id in zip(top_probs, top_ids, strict=True):
            tid = int(token_id.item())
            token_str = tokenizer.get_tok_display(tid)
            target_prob = float(pos_target_out_probs[tid].item())
            target_logit = float(pos_target_logits[tid].item())
            pos_predictions.append(
                (
                    token_str,
                    tid,
                    float(spd_prob.item()),
                    float(pos_spd_logits[tid].item()),
                    target_prob,
                    target_logit,
                )
            )
        predictions_per_position.append(pos_predictions)

    # Decode input tokens
    input_tokens = tokenizer.get_spans([int(t.item()) for t in tokens[0]])

    return InterventionResult(
        input_tokens=input_tokens,
        predictions_per_position=predictions_per_position,
    )
