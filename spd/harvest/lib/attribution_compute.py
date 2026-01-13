"""Gradient-based attribution computation for harvest pipeline.

Computes attribution values (grad * activation) between component pairs for
accumulation into global attributions. Uses unmasked components + weight deltas
for gradient computation (per commit b76e4d9e).
"""

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos


def compute_batch_attributions(
    model: ComponentModel,
    batch: Int[Tensor, "B S"],
    ci_dict: dict[str, Float[Tensor, "B S C"]],
    ci_threshold: float,
    layer_names: list[str],
    layer_offsets: dict[str, int],
    device: torch.device,
) -> tuple[Int[Tensor, " n_edges"], Int[Tensor, " n_edges"], Float[Tensor, " n_edges"]]:
    """Compute gradient-based attributions for one batch.

    Uses unmasked components + weight deltas for gradient computation.
    Only computes attributions for components with CI > threshold.

    Args:
        model: ComponentModel with target model
        batch: Token IDs [B, S]
        ci_dict: CI values per layer {layer: [B, S, C]}
        ci_threshold: Threshold for considering a component "alive"
        layer_names: Ordered list of layer names
        layer_offsets: Maps layer name -> flat index offset
        device: Device for computation

    Returns:
        Tuple of (source_flat_indices, target_flat_indices, attribution_values)
        All tensors have shape [n_edges] where n_edges is the number of
        (source, target) pairs with non-zero attribution.
    """
    # Concatenate CI values into flat tensor for alive detection
    ci_flat: Float[Tensor, "B S n_comp"] = torch.cat(
        [ci_dict[layer] for layer in layer_names], dim=2
    )
    alive_mask: Bool[Tensor, "B S n_comp"] = ci_flat > ci_threshold

    # Early exit if no alive components
    if not alive_mask.any():
        empty = torch.empty(0, dtype=torch.long, device=device)
        empty_float = torch.empty(0, dtype=torch.float, device=device)
        return empty, empty, empty_float

    # Setup for unmasked forward pass with weight deltas
    weight_deltas = model.calc_weight_deltas()
    weight_deltas_and_masks = {
        k: (v, torch.ones(batch.shape, device=device)) for k, v in weight_deltas.items()
    }
    unmasked_masks = make_mask_infos(
        component_masks={k: torch.ones_like(v) for k, v in ci_dict.items()},
        weight_deltas_and_masks=weight_deltas_and_masks,
    )

    # Forward pass with gradients enabled
    with torch.enable_grad():
        comp_output = model(batch, mask_infos=unmasked_masks, cache_type="component_acts")

    cache = comp_output.cache

    # Source layers are all component layers (no wte pseudo-component)
    source_layers = layer_names
    in_post_detaches = [cache[f"{layer}_post_detach"] for layer in source_layers]

    # Collect all edges
    all_source_indices: list[Tensor] = []
    all_target_indices: list[Tensor] = []
    all_values: list[Tensor] = []

    # Count total alive targets to know when to release the graph
    total_alive = int(alive_mask.sum().item())
    grad_count = 0

    # For each target layer
    for target_layer in layer_names:
        target_offset = layer_offsets[target_layer]
        n_components_target = model.module_to_c[target_layer]
        out_pre_detach: Float[Tensor, "B S C"] = cache[f"{target_layer}_pre_detach"]

        # Find alive (batch, seq, component) for this target layer
        layer_idx = layer_names.index(target_layer)
        layer_start = sum(model.module_to_c[ln] for ln in layer_names[:layer_idx])
        layer_end = layer_start + n_components_target
        target_alive = alive_mask[:, :, layer_start:layer_end]

        alive_b, alive_s, alive_c = torch.where(target_alive)
        if len(alive_b) == 0:
            continue

        # Process each alive target in chunks to manage memory
        CHUNK_SIZE = 100  # Process up to 100 targets at once
        for chunk_start in range(0, len(alive_b), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(alive_b))
            chunk_b = alive_b[chunk_start:chunk_end]
            chunk_s = alive_s[chunk_start:chunk_end]
            chunk_c = alive_c[chunk_start:chunk_end]

            # Compute gradients for this chunk
            targets = out_pre_detach[chunk_b, chunk_s, chunk_c]

            for i, (b, s, c) in enumerate(
                zip(chunk_b.tolist(), chunk_s.tolist(), chunk_c.tolist(), strict=True)
            ):
                grad_count += 1
                # Only retain graph if more gradients need to be computed
                retain = grad_count < total_alive

                grads = torch.autograd.grad(
                    outputs=targets[i],
                    inputs=in_post_detaches,
                    retain_graph=retain,
                    allow_unused=True,  # Some source layers may not connect to target
                )

                target_flat_idx = target_offset + c

                with torch.no_grad():
                    for src_layer, grad, in_post_detach in zip(
                        source_layers, grads, in_post_detaches, strict=True
                    ):
                        # Skip if no gradient (source layer not connected to target)
                        # Note: allow_unused=True makes grad optional but types don't reflect this
                        if grad is None:  # pyright: ignore[reportUnnecessaryComparison]
                            continue

                        # Compute attribution: grad * activation
                        weighted: Float[Tensor, "S C"] = (grad * in_post_detach)[b]

                        src_offset = layer_offsets[src_layer]
                        n_src_components = model.module_to_c[src_layer]

                        # Get sources at same sequence position
                        src_values = weighted[s, :n_src_components]  # [C]

                        # Find non-zero attributions
                        nonzero = src_values != 0
                        if not nonzero.any():
                            continue

                        nonzero_indices = torch.where(nonzero)[0]
                        nonzero_values = src_values[nonzero]

                        src_flat_indices = (src_offset + nonzero_indices).to(torch.long)
                        tgt_flat_indices = torch.full_like(
                            src_flat_indices, target_flat_idx, dtype=torch.long
                        )

                        all_source_indices.append(src_flat_indices)
                        all_target_indices.append(tgt_flat_indices)
                        all_values.append(nonzero_values)

    # Concatenate all edges
    if not all_source_indices:
        empty = torch.empty(0, dtype=torch.long, device=device)
        empty_float = torch.empty(0, dtype=torch.float, device=device)
        return empty, empty, empty_float

    source_indices = torch.cat(all_source_indices)
    target_indices = torch.cat(all_target_indices)
    values = torch.cat(all_values)

    return source_indices, target_indices, values
