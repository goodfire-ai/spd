"""Core attribution computation functions.

Copied and cleaned up from spd/scripts/calc_local_attributions.py and calc_global_attributions.py
to avoid importing script files with global execution.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, override

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn
from tqdm.auto import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.app.backend.optim_cis import OptimCIConfig, compute_label_prob, optimize_ci_values
from spd.configs import SamplingType
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.models.components import make_mask_infos


@dataclass
class LayerAliveInfo:
    """Info about alive components for a layer."""

    alive_mask: Bool[Tensor, "s C"]  # Which (pos, component) pairs are alive
    alive_c_idxs: list[int]  # Components alive at any position


def compute_layer_alive_info(
    layer_name: str,
    ci_lower_leaky: dict[str, Tensor],
    ci_masked_out_probs: Float[Tensor, "1 seq vocab"] | None,
    output_prob_threshold: float,
    n_seq: int,
    device: str,
    ci_threshold: float,
) -> LayerAliveInfo:
    """Compute alive info for a layer. Handles regular, wte, and output layers.

    Args:
        layer_name: Name of the layer ("wte", "output", or a component layer name).
        ci_lower_leaky: Dict mapping layer name to CI tensor [1, seq, C].
        ci_masked_out_probs: CI-masked output probabilities [1, seq, vocab]. Required for output layer.
        output_prob_threshold: Threshold for output token activation.
        n_seq: Sequence length.
        device: Device string for tensor allocation.
        ci_threshold: Threshold for component activation based on CI value.
    """
    if layer_name == "wte":
        # WTE: single pseudo-component, always alive at all positions
        alive_mask = torch.ones(n_seq, 1, device=device, dtype=torch.bool)
        alive_c_idxs = [0]
    elif layer_name == "output":
        assert ci_masked_out_probs is not None
        assert ci_masked_out_probs.shape[0] == 1
        alive_mask = ci_masked_out_probs[0] > output_prob_threshold
        alive_c_idxs = torch.where(alive_mask.any(dim=0))[0].tolist()
    else:
        ci = ci_lower_leaky[layer_name]
        assert ci.shape[0] == 1
        alive_mask = ci[0] > ci_threshold
        alive_c_idxs = torch.where(alive_mask.any(dim=0))[0].tolist()

    return LayerAliveInfo(alive_mask, alive_c_idxs)


@dataclass
class Node:
    layer: str
    seq_pos: int
    component_idx: int

    @override
    def __str__(self) -> str:
        return f"{self.layer}:{self.seq_pos}:{self.component_idx}"


@dataclass
class Edge:
    """Edge in the attribution graph."""

    source: Node
    target: Node
    strength: float


@dataclass
class LocalAttributionResult:
    """Result of computing local attributions for a prompt."""

    edges: list[Edge]
    ci_masked_out_probs: Float[Tensor, "seq vocab"]  # CI-masked (SPD model) softmax probabilities
    ci_masked_out_logits: Float[Tensor, "seq vocab"]  # CI-masked (SPD model) raw logits
    target_out_probs: Float[Tensor, "seq vocab"]  # Target model softmax probabilities
    target_out_logits: Float[Tensor, "seq vocab"]  # Target model raw logits
    node_ci_vals: dict[str, float]  # layer:seq:c_idx -> ci_val
    node_subcomp_acts: dict[str, float]  # layer:seq:c_idx -> subcomponent activation (v_i^T @ a)
    label_prob: float | None = None  # P(label_token) with optimized CI mask, only for optimized


def is_kv_to_o_pair(in_layer: str, out_layer: str) -> bool:
    """Check if pair requires cross-sequence gradient computation.

    For k/v → o_proj within the same attention block, output at s_out
    has gradients w.r.t. inputs at all s_in ≤ s_out (causal attention).
    """
    in_is_kv = any(x in in_layer for x in ["k_proj", "v_proj"])
    out_is_o = "o_proj" in out_layer
    if not (in_is_kv and out_is_o):
        return False

    # Check same attention block: "h.{idx}.attn.{proj}"
    in_block = in_layer.split(".")[1]
    out_block = out_layer.split(".")[1]
    return in_block == out_block


def get_sources_by_target(
    model: ComponentModel,
    device: str,
    sampling: SamplingType,
) -> dict[str, list[str]]:
    """Find valid gradient connections grouped by target layer.

    Includes wte (input embeddings) as a source and output (logits) as a target.

    Returns:
        Dict mapping out_layer -> list of in_layers that have gradient flow to it.
    """
    # Use a small dummy batch - we only need to trace gradient connections
    batch: Float[Tensor, "batch seq"] = torch.zeros(2, 3, dtype=torch.long, device=device)

    with torch.no_grad():
        output_with_cache: OutputWithCache = model(batch, cache_type="input")

    with torch.no_grad():
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

    # Hook to capture wte output with gradients
    wte_cache: dict[str, Tensor] = {}

    def wte_hook(
        _module: nn.Module, _args: tuple[Any, ...], _kwargs: dict[Any, Any], output: Tensor
    ) -> Any:
        output.requires_grad_(True)
        wte_cache["wte_post_detach"] = output
        return output

    assert isinstance(model.target_model.wte, nn.Module), "wte is not a module"
    wte_handle = model.target_model.wte.register_forward_hook(wte_hook, with_kwargs=True)

    with torch.enable_grad():
        comp_output_with_cache: OutputWithCache = model(
            batch,
            mask_infos=mask_infos,
            cache_type="component_acts",
        )

    wte_handle.remove()

    cache = comp_output_with_cache.cache
    cache["wte_post_detach"] = wte_cache["wte_post_detach"]
    cache["output_pre_detach"] = comp_output_with_cache.output

    # Build layer list: wte first, component layers, output last
    layers = ["wte"]
    component_layer_names = [
        "attn.q_proj",
        "attn.k_proj",
        "attn.v_proj",
        "attn.o_proj",
        "mlp.c_fc",
        "mlp.down_proj",
    ]
    n_blocks = get_model_n_blocks(model.target_model)
    for i in range(n_blocks):
        layers.extend([f"h.{i}.{layer_name}" for layer_name in component_layer_names])

    # Add lm_head if it exists in target_module_paths (unembedding matrix)
    if "lm_head" in model.target_module_paths:
        layers.append("lm_head")

    layers.append("output")

    # Test all pairs: wte can feed into anything, anything can feed into output
    test_pairs = []
    for in_layer in layers[:-1]:  # Don't include "output" as source
        for out_layer in layers[1:]:  # Don't include "wte" as target
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


def _setup_wte_hook() -> tuple[Callable[..., Any], list[Tensor]]:
    """Create hook to capture wte output with gradients.

    Returns the hook function and a mutable container for the cached output.
    The container is a list to allow mutation from the hook closure.
    """
    wte_cache: list[Tensor] = []

    def wte_hook(
        _module: nn.Module, _args: tuple[Any, ...], _kwargs: dict[Any, Any], output: Tensor
    ) -> Any:
        output.requires_grad_(True)
        assert len(wte_cache) == 0, "wte output should be cached only once"
        wte_cache.append(output)
        return output

    return wte_hook, wte_cache


def _compute_edges_for_target(
    target: str,
    sources: list[str],
    target_info: LayerAliveInfo,
    source_infos: list[LayerAliveInfo],
    cache: dict[str, Tensor],
    n_seq: int,
    source_cache_suffix: str = "post_detach",
    target_seq_positions: list[int] | None = None,
) -> list[Edge]:
    """Compute all edges flowing into a single target layer.

    For each alive (s_out, c_out) in the target layer, computes gradient-based
    attribution strengths from all alive source components.

    Args:
        target: Target layer name (e.g., "h.0.mlp.c_fc" or "output")
        sources: List of source layer names
        target_info: Alive info for target layer
        source_infos: Alive info for each source layer (same order as sources)
        cache: Dict containing cached tensors ({layer}_pre_detach, {layer}_post_detach)
        n_seq: Sequence length
        source_cache_suffix: Cache key suffix for source tensors ("post_detach" or "pre_detach")
        target_seq_positions: Which target seq positions to compute. None = all positions.
    """
    edges: list[Edge] = []
    out_pre_detach = cache[f"{target}_pre_detach"]
    in_sources = [cache[f"{source}_{source_cache_suffix}"] for source in sources]

    seq_positions = target_seq_positions if target_seq_positions is not None else range(n_seq)

    for s_out in seq_positions:
        s_out_alive_c = [c for c in target_info.alive_c_idxs if target_info.alive_mask[s_out, c]]
        if not s_out_alive_c:
            continue

        for c_out in s_out_alive_c:
            grads = torch.autograd.grad(
                outputs=out_pre_detach[0, s_out, c_out],
                inputs=in_sources,
                retain_graph=True,
            )
            with torch.no_grad():
                for source, source_info, grad, in_source in zip(
                    sources, source_infos, grads, in_sources, strict=True
                ):
                    # For k/v → o_proj or any → output, consider all source positions up to s_out
                    causal_positions = is_kv_to_o_pair(source, target) or target == "output"

                    weighted = (grad * in_source)[0]
                    if source == "wte":
                        weighted = weighted.sum(dim=1, keepdim=True)

                    s_in_range = range(s_out + 1) if causal_positions else [s_out]
                    for s_in in s_in_range:
                        for c_in in source_info.alive_c_idxs:
                            if not source_info.alive_mask[s_in, c_in]:
                                continue
                            edges.append(
                                Edge(
                                    source=Node(layer=source, seq_pos=s_in, component_idx=c_in),
                                    target=Node(layer=target, seq_pos=s_out, component_idx=c_out),
                                    strength=weighted[s_in, c_in].item(),
                                )
                            )
    return edges


def compute_attributions(
    model: ComponentModel,
    tokens: Float[Tensor, "1 seq"],
    ci_lower_leaky: dict[str, Float[Tensor, "1 seq C"]],
    pre_weight_acts: dict[str, Float[Tensor, "1 seq d_in"]],
    sources_by_target: dict[str, list[str]],
    target_out_probs: Float[Tensor, "1 seq vocab"],
    target_out_logits: Float[Tensor, "1 seq vocab"],
    output_prob_threshold: float,
    device: str,
    show_progress: bool,
    on_progress: ProgressCallback | None = None,
) -> LocalAttributionResult:
    """Core edge computation

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
    """
    n_seq = tokens.shape[1]

    # Compute CI-masked output probs (for display) before the gradient computation
    with torch.no_grad():
        ci_masks = make_mask_infos(component_masks=ci_lower_leaky)
        ci_masked_logits: Tensor = model(tokens, mask_infos=ci_masks)
        ci_masked_out_probs = torch.softmax(ci_masked_logits, dim=-1)

    # Setup wte hook and run forward pass for gradient computation
    wte_hook, wte_cache = _setup_wte_hook()
    assert isinstance(model.target_model.wte, nn.Module), "wte is not a module"
    wte_handle = model.target_model.wte.register_forward_hook(wte_hook, with_kwargs=True)

    weight_deltas = model.calc_weight_deltas()
    weight_deltas_and_masks = {
        k: (v, torch.ones(tokens.shape, device=device)) for k, v in weight_deltas.items()
    }
    unmasked_masks = make_mask_infos(
        component_masks={k: torch.ones_like(v) for k, v in ci_lower_leaky.items()},
        weight_deltas_and_masks=weight_deltas_and_masks,
    )
    with torch.enable_grad():
        comp_output_with_cache: OutputWithCache = model(
            tokens,
            mask_infos=unmasked_masks,
            cache_type="component_acts",
            detach_component_acts=True,
        )

    wte_handle.remove()
    assert len(wte_cache) == 1, "wte output should be cached"

    cache = comp_output_with_cache.cache
    cache["wte_post_detach"] = wte_cache[0]
    cache["output_pre_detach"] = comp_output_with_cache.output

    # Compute alive info for all layers upfront
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
            ci_threshold=0.0,
        )
        for layer in all_layers
    }

    # Compute edges for each target layer
    edges: list[Edge] = []
    total_source_layers = sum(len(sources) for sources in sources_by_target.values())
    progress_count = 0
    pbar = (
        tqdm(total=total_source_layers, desc="Source layers by target", leave=True)
        if show_progress
        else None
    )

    for target, sources in sources_by_target.items():
        if pbar is not None:
            pbar.set_description(f"Source layers by target: {target}")

        target_edges = _compute_edges_for_target(
            target=target,
            sources=sources,
            target_info=alive_info[target],
            source_infos=[alive_info[source] for source in sources],
            cache=cache,
            n_seq=n_seq,
        )
        edges.extend(target_edges)

        progress_count += len(sources)
        if pbar is not None:
            pbar.update(len(sources))
        if on_progress is not None:
            on_progress(progress_count, total_source_layers, target)

    if pbar is not None:
        pbar.close()

    node_ci_vals = extract_node_ci_vals(ci_lower_leaky)
    component_acts = model.get_all_component_acts(pre_weight_acts)
    node_subcomp_acts = extract_node_subcomp_acts(
        component_acts, ci_threshold=0.0, ci_lower_leaky=ci_lower_leaky
    )

    return LocalAttributionResult(
        edges=edges,
        ci_masked_out_probs=ci_masked_out_probs[0],
        ci_masked_out_logits=comp_output_with_cache.output[0],
        target_out_probs=target_out_probs[0],
        target_out_logits=target_out_logits[0],
        node_ci_vals=node_ci_vals,
        node_subcomp_acts=node_subcomp_acts,
    )


def compute_local_attributions(
    model: ComponentModel,
    tokens: Float[Tensor, "1 seq"],
    sources_by_target: dict[str, list[str]],
    output_prob_threshold: float,
    sampling: SamplingType,
    device: str,
    show_progress: bool,
    on_progress: ProgressCallback | None = None,
) -> LocalAttributionResult:
    """Compute local attributions using the model's natural CI values.

    Computes CI via forward pass, then delegates to compute_edges_from_ci().
    For optimized sparse CI values, use compute_local_attributions_optimized().
    """
    with torch.no_grad():
        output_with_cache = model(tokens, cache_type="input")
        pre_weight_acts = output_with_cache.cache
        target_out_logits = output_with_cache.output
        target_out_probs = torch.softmax(target_out_logits, dim=-1)
        ci = model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            sampling=sampling,
            detach_inputs=False,
        )

    return compute_attributions(
        model=model,
        tokens=tokens,
        ci_lower_leaky=ci.lower_leaky,
        pre_weight_acts=pre_weight_acts,
        sources_by_target=sources_by_target,
        target_out_probs=target_out_probs,
        target_out_logits=target_out_logits,
        output_prob_threshold=output_prob_threshold,
        device=device,
        show_progress=show_progress,
        on_progress=on_progress,
    )


def compute_local_attributions_optimized(
    model: ComponentModel,
    tokens: Float[Tensor, "1 seq"],
    sources_by_target: dict[str, list[str]],
    optim_config: OptimCIConfig,
    output_prob_threshold: float,
    device: str,
    show_progress: bool,
    on_progress: ProgressCallback | None = None,
) -> LocalAttributionResult:
    """Compute local attributions using optimized sparse CI values.

    Runs CI optimization to find a minimal sparse mask that preserves
    the model's prediction, then computes edges.

    L0 stats are computed dynamically at display time from node_ci_vals,
    not here at computation time.
    """
    # Compute target model output probs (unmasked forward pass)
    with torch.no_grad():
        target_logits = model(tokens)
        target_out_probs = torch.softmax(target_logits, dim=-1)

    ci_params = optimize_ci_values(
        model=model,
        tokens=tokens,
        config=optim_config,
        device=device,
        on_progress=on_progress,
    )
    ci_outputs = ci_params.create_ci_outputs(model, device)

    # Get label probability with optimized CI mask (if CE loss is used)
    label_prob: float | None = None
    if optim_config.ce_loss_config is not None:
        with torch.no_grad():
            label_prob = compute_label_prob(
                model, tokens, ci_outputs.lower_leaky, optim_config.ce_loss_config.label_token
            )

    # Signal transition to graph computation stage
    if on_progress is not None:
        on_progress(0, 1, "graph")

    # Get pre_weight_acts for subcomponent activation computation
    with torch.no_grad():
        pre_weight_acts = model(tokens, cache_type="input").cache

    result = compute_attributions(
        model=model,
        tokens=tokens,
        ci_lower_leaky=ci_outputs.lower_leaky,
        pre_weight_acts=pre_weight_acts,
        sources_by_target=sources_by_target,
        target_out_probs=target_out_probs,
        target_out_logits=target_logits,
        output_prob_threshold=output_prob_threshold,
        device=device,
        show_progress=show_progress,
        on_progress=on_progress,
    )

    result.label_prob = label_prob
    return result


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

    This is much faster than compute_local_attributions() because it only
    requires a forward pass and CI computation (no gradient loop).

    Args:
        model: The ComponentModel to analyze.
        tokens: Tokenized prompt of shape [1, seq_len].
        sampling: Sampling type to use for causal importances.

    Returns:
        CIOnlyResult containing CI values per layer, target model output probabilities, pre-weight activations, and component activations.
    """
    with torch.no_grad():
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
) -> dict[str, float]:
    """Extract per-node CI values from CI tensors.

    Args:
        ci_lower_leaky: Dict mapping layer name to CI tensor [1, seq, n_components].

    Returns:
        Dict mapping "layer:seq:c_idx" to CI value.
    """
    node_ci_vals: dict[str, float] = {}
    for layer_name, ci_tensor in ci_lower_leaky.items():
        n_seq = ci_tensor.shape[1]
        n_components = ci_tensor.shape[2]
        for seq_pos in range(n_seq):
            for c_idx in range(n_components):
                key = f"{layer_name}:{seq_pos}:{c_idx}"
                node_ci_vals[key] = float(ci_tensor[0, seq_pos, c_idx].item())
    return node_ci_vals


def extract_node_subcomp_acts(
    component_acts: dict[str, Float[Tensor, "1 seq C"]],
    ci_threshold: float,
    ci_lower_leaky: dict[str, Float[Tensor, "1 seq C"]],
) -> dict[str, float]:
    """Extract per-node subcomponent activations from pre-computed component acts.

    Args:
        component_acts: Dict mapping layer name to component activations [1, seq, C].
        ci_threshold: Threshold for filtering nodes by CI value.
        ci_lower_leaky: Dict mapping layer name to CI tensor [1, seq, C].

    Returns:
        Dict mapping "layer:seq:c_idx" to subcomponent activation value.
    """
    node_subcomp_acts: dict[str, float] = {}
    for layer_name, subcomp_acts in component_acts.items():
        ci = ci_lower_leaky[layer_name]
        alive_mask = ci[0] > ci_threshold  # [seq, C]
        alive_seq_indices, alive_c_indices = torch.where(alive_mask)
        for seq_pos, c_idx in zip(
            alive_seq_indices.tolist(), alive_c_indices.tolist(), strict=True
        ):
            key = f"{layer_name}:{seq_pos}:{c_idx}"
            node_subcomp_acts[key] = float(subcomp_acts[0, seq_pos, c_idx].item())

    return node_subcomp_acts


def extract_active_from_ci(
    ci_lower_leaky: dict[str, Float[Tensor, "1 seq n_components"]],
    target_out_probs: Float[Tensor, "1 seq vocab"],
    ci_threshold: float,
    output_prob_threshold: float,  # TODO change me to topP (cumulative probability threshold)
    n_seq: int,
) -> dict[str, tuple[float, list[int]]]:
    """Build inverted index data directly from CI values.

    For regular component layers, a component is active at positions where CI > threshold.
    For the output layer, a token is active at positions where prob > threshold.
    For wte, a single pseudo-component (idx 0) is always active at all positions.

    Args:
        ci_lower_leaky: Dict mapping layer name to CI tensor [1, seq, n_components].
        target_out_probs: Target model output probability tensor [1, seq, vocab].
        ci_threshold: Threshold for component activation.
        output_prob_threshold: Threshold for output token activation.
        n_seq: Sequence length.

    Returns:
        Dict mapping component_key ("layer:c_idx") to (max_ci, positions).
    """
    active: dict[str, tuple[float, list[int]]] = {}

    # Regular component layers
    for layer, ci_tensor in ci_lower_leaky.items():
        n_components = ci_tensor.shape[-1]
        for c_idx in range(n_components):
            ci_per_pos = ci_tensor[0, :, c_idx]
            positions = torch.where(ci_per_pos > ci_threshold)[0].tolist()
            if positions:
                key = f"{layer}:{c_idx}"
                max_ci = float(ci_per_pos.max().item())
                active[key] = (max_ci, positions)

    # Output layer - use probability threshold
    for c_idx in range(target_out_probs.shape[-1]):
        prob_per_pos = target_out_probs[0, :, c_idx]
        positions = torch.where(prob_per_pos > output_prob_threshold)[0].tolist()
        if positions:
            key = f"output:{c_idx}"
            max_prob = float(prob_per_pos.max().item())
            active[key] = (max_prob, positions)

    # WTE - single pseudo-component always active at all positions
    active["wte:0"] = (1.0, list(range(n_seq)))

    return active


def get_model_n_blocks(model: nn.Module) -> int:
    """Get the number of blocks in the model."""
    from simple_stories_train.models.gpt2_simple import GPT2Simple
    from simple_stories_train.models.llama_simple import LlamaSimple
    from simple_stories_train.models.llama_simple_mlp import LlamaSimpleMLP
    from transformers.models.gpt2 import GPT2LMHeadModel

    match model:
        case GPT2LMHeadModel():
            return len(model.transformer.h)
        case GPT2Simple() | LlamaSimple() | LlamaSimpleMLP():
            return len(model.h)
        case _:
            raise ValueError(f"Unsupported model: {type(model)}")


@dataclass
class InterventionResult:
    """Result of intervention forward pass."""

    input_tokens: list[str]
    predictions_per_position: list[
        list[tuple[str, int, float, float, float, float]]
    ]  # [(token, id, spd_prob, logit, target_prob, target_logit)]


def parse_node_key(key: str) -> Node:
    """Parse a node key string like 'layer:seq:cIdx' back to a Node dataclass."""
    parts = key.split(":")
    assert len(parts) == 3, f"Invalid node key format: {key}"
    layer, seq_pos, component_idx = parts
    return Node(layer=layer, seq_pos=int(seq_pos), component_idx=int(component_idx))


def compute_direct_output_attributions(
    model: ComponentModel,
    tokens: Float[Tensor, "1 seq"],
    output_prob_threshold: float,
    sampling: SamplingType,
    device: str,
) -> LocalAttributionResult:
    """Compute attribution from each component directly to output logits via autograd.

    Uses the chain rule: attribution = ∂logit/∂component_output × component_output

    This computes the mathematically correct total attribution from any component
    to any output token, properly accounting for all paths through the network.
    Unlike edge chaining, this doesn't multiply in intermediate activation factors.

    For the attribution computation, we use unmasked components and weight deltas
    (same as compute_attributions) because we want to capture all attribution paths
    through the network, not rely on CI masks being accurate.

    All components with CI > 0 are included; filtering by CI threshold is done at
    display time (matching compute_attributions behavior).

    Args:
        model: The ComponentModel to analyze.
        tokens: Input tokens [1, seq].
        output_prob_threshold: Output tokens with prob > threshold are included.
        sampling: Sampling type for CI computation.
        device: Device string for tensor allocation.

    Returns:
        LocalAttributionResult with edges, output probs, CI values, and activations.
    """
    n_seq = tokens.shape[1]
    final_seq_pos = n_seq - 1

    # Compute CI values and target model output
    with torch.no_grad():
        target_output = model(tokens, cache_type="input")
        pre_weight_acts = target_output.cache
        target_out_probs = torch.softmax(target_output.output, dim=-1)

        ci = model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            sampling=sampling,
            detach_inputs=False,
        )

    ci_lower_leaky = ci.lower_leaky

    # Compute CI-masked output probs (for display and determining alive outputs)
    with torch.no_grad():
        ci_masks = make_mask_infos(component_masks=ci_lower_leaky)
        ci_masked_logits: Tensor = model(tokens, mask_infos=ci_masks)
        ci_masked_out_probs = torch.softmax(ci_masked_logits, dim=-1)

    # Setup wte hook to capture embedding output with gradients
    wte_hook, wte_cache = _setup_wte_hook()
    assert isinstance(model.target_model.wte, nn.Module), "wte is not a module"
    wte_handle = model.target_model.wte.register_forward_hook(wte_hook, with_kwargs=True)

    # Forward pass with gradients enabled for attribution computation
    # Use unmasked components and weight deltas to capture all attribution paths
    # Use detach_component_acts=False to preserve gradient flow for end-to-end attribution
    weight_deltas = model.calc_weight_deltas()
    weight_deltas_and_masks = {
        k: (v, torch.ones(tokens.shape, device=device)) for k, v in weight_deltas.items()
    }
    unmasked_masks = make_mask_infos(
        component_masks={k: torch.ones_like(v) for k, v in ci_lower_leaky.items()},
        weight_deltas_and_masks=weight_deltas_and_masks,
    )
    with torch.enable_grad():
        unmasked_output: OutputWithCache = model(
            tokens,
            mask_infos=unmasked_masks,
            cache_type="component_acts",
            detach_component_acts=False,
        )

    wte_handle.remove()
    assert len(wte_cache) == 1, "wte output should be cached"

    logits = unmasked_output.output  # [1, seq, vocab]
    cache = unmasked_output.cache
    cache["wte_pre_detach"] = wte_cache[0]
    cache["output_pre_detach"] = logits

    # Extract node CI values and activations (for return)
    node_ci_vals = extract_node_ci_vals(ci_lower_leaky)
    component_acts = model.get_all_component_acts(pre_weight_acts)
    node_subcomp_acts = extract_node_subcomp_acts(
        component_acts, ci_threshold=0.0, ci_lower_leaky=ci_lower_leaky
    )

    # Build alive info for all sources and output
    sources = ["wte"] + list(ci_lower_leaky.keys())
    all_layers = sources + ["output"]
    alive_info: dict[str, LayerAliveInfo] = {}
    for layer in all_layers:
        alive_info[layer] = compute_layer_alive_info(
            layer_name=layer,
            ci_lower_leaky=ci_lower_leaky,
            ci_masked_out_probs=ci_masked_out_probs,
            output_prob_threshold=output_prob_threshold,
            n_seq=n_seq,
            device=device,
            ci_threshold=0.0,
        )

    # Compute edges using shared function
    edges = _compute_edges_for_target(
        target="output",
        sources=sources,
        target_info=alive_info["output"],
        source_infos=[alive_info[s] for s in sources],
        cache=cache,
        n_seq=n_seq,
        source_cache_suffix="pre_detach",
        target_seq_positions=[final_seq_pos],
    )

    return LocalAttributionResult(
        edges=edges,
        ci_masked_out_probs=ci_masked_out_probs[0],
        ci_masked_out_logits=ci_masked_logits[0],
        target_out_probs=target_out_probs[0],
        target_out_logits=target_output.output[0],
        node_ci_vals=node_ci_vals,
        node_subcomp_acts=node_subcomp_acts,
    )


def compute_intervention_forward(
    model: ComponentModel,
    tokens: Float[Tensor, "1 seq"],
    active_nodes: list[tuple[str, int, int]],  # [(layer, seq_pos, component_idx)]
    top_k: int,
    tokenizer: PreTrainedTokenizerBase,
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

    with torch.no_grad():
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
            token_str = tokenizer.decode([tid])
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
    input_tokens = [tokenizer.decode([int(t.item())]) for t in tokens[0]]

    return InterventionResult(
        input_tokens=input_tokens,
        predictions_per_position=predictions_per_position,
    )
