"""Core attribution computation functions.

Copied and cleaned up from spd/scripts/calc_local_attributions.py and calc_global_attributions.py
to avoid importing script files with global execution.
"""

from collections import defaultdict
from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor
from tqdm.auto import tqdm

from spd.models.component_model import ComponentModel, OutputWithCache
from spd.models.components import make_mask_infos


@dataclass
class PairAttribution:
    """Attribution between a source and target layer for a single prompt."""

    source: str
    target: str
    attribution: Float[Tensor, "s_in trimmed_c_in s_out trimmed_c_out"]
    trimmed_c_in_idxs: list[int]
    trimmed_c_out_idxs: list[int]
    is_kv_to_o_pair: bool


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
    sampling: str,
    n_blocks: int,
) -> dict[str, list[str]]:
    """Find valid gradient connections grouped by target layer.

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

    component_masks = ci.lower_leaky
    mask_infos = make_mask_infos(
        component_masks=component_masks,
        routing_masks="all",
    )
    with torch.enable_grad():
        comp_output_with_cache: OutputWithCache = model(
            batch,
            mask_infos=mask_infos,
            cache_type="component_acts",
        )

    cache = comp_output_with_cache.cache
    layer_names = [
        "attn.q_proj",
        "attn.k_proj",
        "attn.v_proj",
        "attn.o_proj",
        "mlp.c_fc",
        "mlp.down_proj",
    ]
    layers = []
    for i in range(n_blocks):
        layers.extend([f"h.{i}.{layer_name}" for layer_name in layer_names])

    test_pairs = []
    for in_layer in layers:
        for out_layer in layers:
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
        if grad is not None:
            sources_by_target[out_layer].append(in_layer)
    return dict(sources_by_target)


def compute_local_attributions(
    model: ComponentModel,
    tokens: Float[Tensor, "1 seq"],
    sources_by_target: dict[str, list[str]],
    ci_threshold: float,
    sampling: str,
    device: str,
    show_progress: bool = True,
) -> list[PairAttribution]:
    """Compute local attributions for a single prompt.

    For each valid layer pair (in_layer, out_layer), computes the gradient-based
    attribution of output component activations with respect to input component
    activations, preserving sequence position information.

    Args:
        model: The ComponentModel to analyze.
        tokens: Tokenized prompt of shape [1, seq_len].
        sources_by_target: Dict mapping out_layer -> list of in_layers.
        ci_threshold: Threshold for considering a component alive at a position.
        sampling: Sampling type to use for causal importances.
        device: Device to run on.
        show_progress: Whether to show progress bars.

    Returns:
        List of PairAttribution objects.
    """
    n_seq = tokens.shape[1]
    C = model.C

    with torch.no_grad():
        output_with_cache: OutputWithCache = model(tokens, cache_type="input")

    with torch.no_grad():
        ci = model.calc_causal_importances(
            pre_weight_acts=output_with_cache.cache,
            sampling=sampling,
            detach_inputs=False,
        )

    component_masks = ci.lower_leaky
    mask_infos = make_mask_infos(component_masks=component_masks, routing_masks="all")

    with torch.enable_grad():
        comp_output_with_cache: OutputWithCache = model(
            tokens, mask_infos=mask_infos, cache_type="component_acts"
        )

    cache = comp_output_with_cache.cache
    local_attributions: list[PairAttribution] = []

    target_iter = sources_by_target.items()
    if show_progress:
        target_iter = tqdm(target_iter, desc="Target layers", leave=False)

    for out_layer, in_layers in target_iter:
        out_pre_detach: Float[Tensor, "1 s C"] = cache[f"{out_layer}_pre_detach"]
        ci_out: Float[Tensor, "1 s C"] = ci.lower_leaky[out_layer]

        in_post_detaches: list[Float[Tensor, "1 s C"]] = [
            cache[f"{in_layer}_post_detach"] for in_layer in in_layers
        ]
        ci_ins: list[Float[Tensor, "1 s C"]] = [ci.lower_leaky[in_layer] for in_layer in in_layers]

        attributions: list[Float[Tensor, "s_in C s_out C"]] = [
            torch.zeros(n_seq, C, n_seq, C, device=device) for _ in in_layers
        ]

        is_attention_output = any(is_kv_to_o_pair(in_layer, out_layer) for in_layer in in_layers)

        alive_out_mask: Float[Tensor, "1 s C"] = ci_out >= ci_threshold
        alive_out_c_idxs: list[int] = torch.where(alive_out_mask[0].any(dim=0))[0].tolist()

        alive_in_masks: list[Float[Tensor, "1 s C"]] = [ci_in >= ci_threshold for ci_in in ci_ins]
        alive_in_c_idxs: list[list[int]] = [
            torch.where(alive_in_mask[0].any(dim=0))[0].tolist() for alive_in_mask in alive_in_masks
        ]

        for s_out in range(n_seq):
            s_out_alive_c_idxs: list[int] = torch.where(alive_out_mask[0, s_out])[0].tolist()
            if len(s_out_alive_c_idxs) == 0:
                continue

            for c_out in s_out_alive_c_idxs:
                in_post_detach_grads = torch.autograd.grad(
                    outputs=out_pre_detach[0, s_out, c_out],
                    inputs=in_post_detaches,
                    retain_graph=True,
                )
                s_in_range = range(s_out + 1) if is_attention_output else range(s_out, s_out + 1)

                with torch.no_grad():
                    for in_post_detach_grad, in_post_detach, alive_in_mask, attribution in zip(
                        in_post_detach_grads,
                        in_post_detaches,
                        alive_in_masks,
                        attributions,
                        strict=True,
                    ):
                        weighted: Float[Tensor, "s C"] = (in_post_detach_grad * in_post_detach)[0]
                        for s_in in s_in_range:
                            alive_c_in: list[int] = torch.where(alive_in_mask[0, s_in])[0].tolist()
                            for c_in in alive_c_in:
                                attribution[s_in, c_in, s_out, c_out] = weighted[s_in, c_in]

        for in_layer, attribution, layer_alive_in_c_idxs in zip(
            in_layers, attributions, alive_in_c_idxs, strict=True
        ):
            trimmed_attribution = attribution[:, layer_alive_in_c_idxs][:, :, :, alive_out_c_idxs]
            local_attributions.append(
                PairAttribution(
                    source=in_layer,
                    target=out_layer,
                    attribution=trimmed_attribution,
                    trimmed_c_in_idxs=layer_alive_in_c_idxs,
                    trimmed_c_out_idxs=alive_out_c_idxs,
                    is_kv_to_o_pair=is_kv_to_o_pair(in_layer, out_layer),
                )
            )

    return local_attributions
