"""Core attribution computation functions.

Copied and cleaned up from spd/scripts/calc_local_attributions.py and calc_global_attributions.py
to avoid importing script files with global execution.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn
from tqdm.auto import tqdm

from spd.app.backend.optim_cis.run_optim_cis import OptimCIConfig, optimize_ci_values
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
    output_probs: Float[Tensor, "1 seq vocab"] | None,
    ci_threshold: float,
    output_prob_threshold: float,
    n_seq: int,
    device: str,
) -> LayerAliveInfo:
    """Compute alive info for a layer. Handles regular, wte, and output layers."""
    if layer_name == "wte":
        # WTE: single pseudo-component, always alive at all positions
        alive_mask = torch.ones(n_seq, 1, device=device, dtype=torch.bool)
        alive_c_idxs = [0]
    elif layer_name == "output":
        assert output_probs is not None
        assert output_probs.shape[0] == 1
        alive_mask = output_probs[0] >= output_prob_threshold
        alive_c_idxs = torch.where(alive_mask.any(dim=0))[0].tolist()
    else:
        ci = ci_lower_leaky[layer_name]
        assert ci.shape[0] == 1
        alive_mask = ci[0] >= ci_threshold
        alive_c_idxs = torch.where(alive_mask.any(dim=0))[0].tolist()

    return LayerAliveInfo(alive_mask, alive_c_idxs)


# tuple for speed
Edge = tuple[str, str, int, int, int, int, float, bool]
"""(source, target, c_in, c_out, s_in, s_out, strength, is_cross_seq)"""


@dataclass
class LocalAttributionResult:
    """Result of computing local attributions for a prompt."""

    edges: list[Edge]
    output_probs: Float[Tensor, "seq vocab"]  # Softmax probabilities for output logits


@dataclass
class OptimizationStats:
    """Statistics from CI optimization."""

    label_prob: float  # P(label_token) with optimized CI mask
    l0_total: float  # Total L0 across all layers
    l0_per_layer: dict[str, float]  # L0 per layer


@dataclass
class OptimizedLocalAttributionResult:
    """Result of computing local attributions with optimized CI values."""

    edges: list[Edge]
    output_probs: Float[Tensor, "seq vocab"]
    stats: OptimizationStats


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


def compute_edges_from_ci(
    model: ComponentModel,
    tokens: Float[Tensor, "1 seq"],
    ci_lower_leaky: dict[str, Float[Tensor, "1 seq C"]],
    sources_by_target: dict[str, list[str]],
    ci_threshold: float,
    output_prob_threshold: float,
    device: str,
    show_progress: bool,
    on_progress: ProgressCallback | None = None,
) -> LocalAttributionResult:
    """Core edge computation from pre-computed CI values.

    Computes gradient-based attribution edges between components using the
    provided CI values for masking and thresholding.

    Use compute_local_attributions() for automatic CI computation, or
    compute_local_attributions_optimized() for optimized sparse CI values.
    """
    # Add break for debugger to attach here
    n_seq = tokens.shape[1]

    ci_masked_infos = make_mask_infos(component_masks=ci_lower_leaky, routing_masks="all")

    # Hook to capture wte output with gradients
    # this is gross but basedpyright reports unreachable if we make this a `Tensor | None`
    wte_cached_output: Tensor = torch.tensor([])

    def wte_hook(
        _module: nn.Module, _args: tuple[Any, ...], _kwargs: dict[Any, Any], output: Tensor
    ) -> Any:
        nonlocal wte_cached_output
        output.requires_grad_(True)
        assert wte_cached_output.numel() == 0, "wte output should be cached only once"
        wte_cached_output = output
        return output

    assert isinstance(model.target_model.wte, nn.Module), "wte is not a module"
    wte_handle = model.target_model.wte.register_forward_hook(wte_hook, with_kwargs=True)

    with torch.enable_grad():
        comp_output_with_cache: OutputWithCache = model(
            tokens, mask_infos=ci_masked_infos, cache_type="component_acts"
        )

    wte_handle.remove()
    assert wte_cached_output.numel() > 0, "wte output should be cached"

    cache = comp_output_with_cache.cache
    cache["wte_post_detach"] = wte_cached_output
    cache["output_pre_detach"] = comp_output_with_cache.output

    # Compute output probabilities for thresholding
    output_probs = torch.softmax(comp_output_with_cache.output, dim=-1)

    # Compute alive info for all layers upfront
    all_layers: set[str] = set(sources_by_target.keys())
    for sources in sources_by_target.values():
        all_layers.update(sources)

    alive_info: dict[str, LayerAliveInfo] = {}
    for layer in all_layers:
        alive_info[layer] = compute_layer_alive_info(
            layer, ci_lower_leaky, output_probs, ci_threshold, output_prob_threshold, n_seq, device
        )

    edges: list[Edge] = []

    target_iter = sources_by_target.items()
    total_source_layers = sum(len(sources) for sources in sources_by_target.values())
    progress_count = 0

    if show_progress:
        pbar = tqdm(
            total=total_source_layers,
            desc="Source layers by target",
            leave=True,
        )
    else:
        pbar = None

    for target, sources in list(target_iter):
        if pbar is not None:
            pbar.set_description(f"Source layers by target: {target}")

        target_info = alive_info[target]
        out_pre_detach: Float[Tensor, "1 s C"] = cache[f"{target}_pre_detach"]

        source_infos = [alive_info[source] for source in sources]
        in_post_detaches: list[Float[Tensor, "1 s C"]] = [
            cache[f"{source}_post_detach"] for source in sources
        ]

        for s_out in range(n_seq):
            s_out_alive_c: list[int] = [
                c for c in target_info.alive_c_idxs if target_info.alive_mask[s_out, c]
            ]
            if not s_out_alive_c:
                continue

            for c_out in s_out_alive_c:
                in_post_detach_grads = torch.autograd.grad(
                    outputs=out_pre_detach[0, s_out, c_out],
                    inputs=in_post_detaches,
                    retain_graph=True,
                )
                with torch.no_grad():
                    for source, source_info, grad, in_post_detach in zip(
                        sources,
                        source_infos,
                        in_post_detach_grads,
                        in_post_detaches,
                        strict=True,
                    ):
                        is_kv_to_o_pair_flag = is_kv_to_o_pair(source, target)
                        weighted: Float[Tensor, "s C"] = (grad * in_post_detach)[0]
                        if source == "wte":
                            # Sum over embedding_dim to get single pseudo-component
                            # assert weighted.shape == (n_seq, model.C)
                            weighted = weighted.sum(dim=1, keepdim=True)

                        s_in_range = range(s_out + 1) if is_kv_to_o_pair_flag else [s_out]

                        for s_in in s_in_range:
                            for c_in in source_info.alive_c_idxs:
                                if not source_info.alive_mask[s_in, c_in]:
                                    continue
                                strength = weighted[s_in, c_in].item()
                                edge: Edge = (
                                    source,
                                    target,
                                    c_in,
                                    c_out,
                                    s_in,
                                    s_out,
                                    strength,
                                    is_kv_to_o_pair_flag,
                                )
                                edges.append(edge)
        progress_count += len(sources)
        if pbar is not None:
            # different targets have different number of sources so this makes the progress bar more accurate
            pbar.update(len(sources))
        if on_progress is not None:
            on_progress(progress_count, total_source_layers, target)

    if pbar is not None:
        pbar.close()

    return LocalAttributionResult(edges=edges, output_probs=output_probs)


def compute_local_attributions(
    model: ComponentModel,
    tokens: Float[Tensor, "1 seq"],
    sources_by_target: dict[str, list[str]],
    ci_threshold: float,
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
        pre_weight_acts = model(tokens, cache_type="input").cache
        ci = model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            sampling=sampling,
            detach_inputs=False,
        )

    return compute_edges_from_ci(
        model=model,
        tokens=tokens,
        ci_lower_leaky=ci.lower_leaky,
        sources_by_target=sources_by_target,
        ci_threshold=ci_threshold,
        output_prob_threshold=output_prob_threshold,
        device=device,
        show_progress=show_progress,
        on_progress=on_progress,
    )


def compute_local_attributions_optimized(
    model: ComponentModel,
    tokens: Float[Tensor, "1 seq"],
    label_token: int,
    sources_by_target: dict[str, list[str]],
    optim_config: OptimCIConfig,
    ci_threshold: float,
    output_prob_threshold: float,
    device: str,
    show_progress: bool,
    on_progress: ProgressCallback | None = None,
) -> OptimizedLocalAttributionResult:
    """Compute local attributions using optimized sparse CI values.

    Runs CI optimization to find a minimal sparse mask that preserves
    the model's prediction of label_token, then computes edges.
    """
    ci_params = optimize_ci_values(
        model=model,
        tokens=tokens,
        label_token=label_token,
        config=optim_config,
        device=device,
        on_progress=on_progress,
    )
    ci_outputs = ci_params.create_ci_outputs(model, device)

    # Compute optimization stats
    l0_per_layer: dict[str, float] = {}
    for layer_name, ci_tensor in ci_outputs.lower_leaky.items():
        # L0 = count of components with CI > threshold, averaged over sequence
        l0_per_layer[layer_name] = float((ci_tensor > ci_threshold).float().sum().item())
    l0_total = sum(l0_per_layer.values())

    # Get label probability with optimized CI mask
    with torch.no_grad():
        mask_infos = make_mask_infos(ci_outputs.lower_leaky, routing_masks="all")
        logits = model(tokens, mask_infos=mask_infos)
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        label_prob = float(probs[label_token].item())

    stats = OptimizationStats(
        label_prob=label_prob,
        l0_total=l0_total,
        l0_per_layer=l0_per_layer,
    )

    result = compute_edges_from_ci(
        model=model,
        tokens=tokens,
        ci_lower_leaky=ci_outputs.lower_leaky,
        sources_by_target=sources_by_target,
        ci_threshold=ci_threshold,
        output_prob_threshold=output_prob_threshold,
        device=device,
        show_progress=show_progress,
    )

    return OptimizedLocalAttributionResult(
        edges=result.edges,
        output_probs=result.output_probs,
        stats=stats,
    )


@dataclass
class CIOnlyResult:
    """Result of computing CI values only (no attribution graph)."""

    ci_lower_leaky: dict[str, Float[Tensor, "1 seq n_components"]]
    output_probs: Float[Tensor, "1 seq vocab"]


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
        CIOnlyResult containing CI values per layer and output probabilities.
    """
    with torch.no_grad():
        output_with_cache: OutputWithCache = model(tokens, cache_type="input")
        ci = model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache,
            sampling=sampling,
            detach_inputs=False,
        )
        output_probs = torch.softmax(output_with_cache.output, dim=-1)

    return CIOnlyResult(ci_lower_leaky=ci.lower_leaky, output_probs=output_probs)


def extract_active_from_ci(
    ci_lower_leaky: dict[str, Float[Tensor, "1 seq n_components"]],
    output_probs: Float[Tensor, "1 seq vocab"],
    ci_threshold: float,
    output_prob_threshold: float,  # TODO change me to topP (cumulative probability threshold)
    n_seq: int,
) -> dict[str, tuple[float, list[int]]]:
    """Build inverted index data directly from CI values.

    For regular component layers, a component is active at positions where CI >= threshold.
    For the output layer, a token is active at positions where prob >= threshold.
    For wte, a single pseudo-component (idx 0) is always active at all positions.

    Args:
        ci_lower_leaky: Dict mapping layer name to CI tensor [1, seq, n_components].
        output_probs: Output probability tensor [1, seq, vocab].
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
            positions = torch.where(ci_per_pos >= ci_threshold)[0].tolist()
            if positions:
                key = f"{layer}:{c_idx}"
                max_ci = float(ci_per_pos.max().item())
                active[key] = (max_ci, positions)

    # Output layer - use probability threshold
    for c_idx in range(output_probs.shape[-1]):
        prob_per_pos = output_probs[0, :, c_idx]
        positions = torch.where(prob_per_pos >= output_prob_threshold)[0].tolist()
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
    from transformers.models.gpt2 import GPT2LMHeadModel

    match model:
        case GPT2LMHeadModel():
            return len(model.transformer.h)
        case GPT2Simple() | LlamaSimple():
            return len(model.h)
        case _:
            raise ValueError(f"Unsupported model: {type(model)}")
