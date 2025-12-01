"""Core attribution computation functions.

Copied and cleaned up from spd/scripts/calc_local_attributions.py and calc_global_attributions.py
to avoid importing script files with global execution.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from jaxtyping import Bool, Float
from torch import Tensor, nn
from tqdm.auto import tqdm

from spd.configs import SamplingType
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.models.components import make_mask_infos


@dataclass
class LayerAliveInfo:
    """Info about alive components for a layer."""

    alive_mask: Bool[Tensor, "1 s dim"]  # Which (pos, component) pairs are alive
    alive_c_idxs: list[int]  # Components alive at any position
    c_to_trimmed: dict[int, int]  # original idx -> trimmed idx


def compute_layer_alive_info(
    layer_name: str,
    ci_lower_leaky: dict[str, Tensor],
    output_probs: Float[Tensor, "1 s vocab"] | None,
    ci_threshold: float,
    output_prob_threshold: float,
    n_seq: int,
    device: str,
) -> LayerAliveInfo:
    """Compute alive info for a layer. Handles regular, wte, and output layers."""
    if layer_name == "wte":
        # WTE: single pseudo-component, always alive at all positions
        alive_mask = torch.ones(1, n_seq, 1, device=device, dtype=torch.bool)
        alive_c_idxs = [0]
    elif layer_name == "output":
        assert output_probs is not None
        alive_mask = output_probs >= output_prob_threshold
        alive_c_idxs = torch.where(alive_mask[0].any(dim=0))[0].tolist()
    else:
        ci = ci_lower_leaky[layer_name]
        alive_mask = ci >= ci_threshold
        alive_c_idxs = torch.where(alive_mask[0].any(dim=0))[0].tolist()

    c_to_trimmed = {c: i for i, c in enumerate(alive_c_idxs)}
    return LayerAliveInfo(alive_mask, alive_c_idxs, c_to_trimmed)


@dataclass
class PairAttribution:
    """Attribution between a source and target layer for a single prompt."""

    source: str
    target: str
    attribution: Float[Tensor, "s_in trimmed_c_in s_out trimmed_c_out"]
    trimmed_c_in_idxs: list[int]
    trimmed_c_out_idxs: list[int]
    is_kv_to_o_pair: bool


@dataclass
class LocalAttributionResult:
    """Result of computing local attributions for a prompt."""

    pairs: list[PairAttribution]
    output_probs: Float[Tensor, "1 seq vocab"]  # Softmax probabilities for output logits


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
    n_blocks: int,
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


def compute_local_attributions(
    model: ComponentModel,
    tokens: Float[Tensor, "1 seq"],
    sources_by_target: dict[str, list[str]],
    ci_threshold: float,
    output_prob_threshold: float,
    sampling: SamplingType,
    device: str,
    show_progress: bool = True,
) -> LocalAttributionResult:
    """Compute local attributions for a single prompt.

    For each valid layer pair (in_layer, out_layer), computes the gradient-based
    attribution of output component activations with respect to input component
    activations, preserving sequence position information.

    Supports three layer types:
    - wte: Input embeddings (single pseudo-component per position)
    - Regular component layers (h.{i}.attn.*, h.{i}.mlp.*)
    - output: Output logits (vocab tokens as components)

    Args:
        model: The ComponentModel to analyze.
        tokens: Tokenized prompt of shape [1, seq_len].
        sources_by_target: Dict mapping out_layer -> list of in_layers.
        ci_threshold: Threshold for considering a component alive at a position.
        output_prob_threshold: Threshold for considering an output logit alive (on softmax probs).
        sampling: Sampling type to use for causal importances.
        device: Device to run on.
        show_progress: Whether to show progress bars.

    Returns:
        LocalAttributionResult containing pairs and output_probs.
    """
    n_seq = tokens.shape[1]

    with torch.no_grad():
        output_with_cache: OutputWithCache = model(tokens, cache_type="input")

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
            tokens, mask_infos=mask_infos, cache_type="component_acts"
        )

    wte_handle.remove()

    cache = comp_output_with_cache.cache
    cache["wte_post_detach"] = wte_cache["wte_post_detach"]
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
            layer, ci.lower_leaky, output_probs, ci_threshold, output_prob_threshold, n_seq, device
        )

    local_attributions: list[PairAttribution] = []

    target_iter = sources_by_target.items()
    if show_progress:
        target_iter = tqdm(list(target_iter), desc="Target layers", leave=False)

    for target, sources in target_iter:
        target_info = alive_info[target]
        out_pre_detach: Float[Tensor, "1 s dim"] = cache[f"{target}_pre_detach"]

        source_infos = [alive_info[source] for source in sources]
        in_post_detaches: list[Float[Tensor, "1 s dim"]] = [
            cache[f"{source}_post_detach"] for source in sources
        ]

        # Initialize attribution tensors at final trimmed size
        attributions: list[Float[Tensor, "s_in n_c_in s_out n_c_out"]] = [
            torch.zeros(
                n_seq,
                len(source_info.alive_c_idxs),
                n_seq,
                len(target_info.alive_c_idxs),
                device=device,
            )
            for source_info in source_infos
        ]

        is_attention_output = any(is_kv_to_o_pair(source, target) for source in sources)

        for s_out in range(n_seq):
            # Get alive output components at this position
            s_out_alive_c: list[int] = [
                c for c in target_info.alive_c_idxs if target_info.alive_mask[0, s_out, c]
            ]
            if not s_out_alive_c:
                continue

            for c_out in s_out_alive_c:
                in_post_detach_grads = torch.autograd.grad(
                    outputs=out_pre_detach[0, s_out, c_out],
                    inputs=in_post_detaches,
                    retain_graph=True,
                )
                # Handle causal attention mask
                s_in_range = range(s_out + 1) if is_attention_output else range(s_out, s_out + 1)
                trimmed_c_out = target_info.c_to_trimmed[c_out]

                with torch.no_grad():
                    for source, source_info, grad, in_post_detach, attr in zip(
                        sources,
                        source_infos,
                        in_post_detach_grads,
                        in_post_detaches,
                        attributions,
                        strict=True,
                    ):
                        weighted: Float[Tensor, "s dim"] = (grad * in_post_detach)[0]
                        if source == "wte":
                            # Sum over embedding_dim to get single pseudo-component
                            weighted = weighted.sum(dim=1, keepdim=True)

                        for s_in in s_in_range:
                            alive_c_in = [
                                c
                                for c in source_info.alive_c_idxs
                                if source_info.alive_mask[0, s_in, c]
                            ]
                            for c_in in alive_c_in:
                                trimmed_c_in = source_info.c_to_trimmed[c_in]
                                attr[s_in, trimmed_c_in, s_out, trimmed_c_out] = weighted[s_in, c_in]

        for source, source_info, attr in zip(sources, source_infos, attributions, strict=True):
            local_attributions.append(
                PairAttribution(
                    source=source,
                    target=target,
                    attribution=attr,
                    trimmed_c_in_idxs=source_info.alive_c_idxs,
                    trimmed_c_out_idxs=target_info.alive_c_idxs,
                    is_kv_to_o_pair=is_kv_to_o_pair(source, target),
                )
            )

    return LocalAttributionResult(pairs=local_attributions, output_probs=output_probs)
