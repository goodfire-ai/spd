"""Core attribution harvester for computing dataset-aggregated attributions.

Computes component-to-component attribution strengths aggregated over the full
training dataset using gradient x activation formula, summed over all positions
and batches.

Matrix structure is (n_sources, n_targets) where:
- Sources: wte tokens [0, vocab_size) + component layers [vocab_size, ...)
- Targets: component layers [0, n_components) + output tokens [n_components, ...)
"""

from typing import Any

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from tqdm.auto import tqdm

from spd.configs import SamplingType
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.models.components import make_mask_infos


def _setup_wte_hook() -> tuple[Any, list[Tensor]]:
    """Create hook to capture wte output with gradients.

    Returns the hook function and a mutable container for the cached output.
    """
    wte_cache: list[Tensor] = []

    def wte_hook(
        _module: nn.Module, _args: tuple[Any, ...], _kwargs: dict[Any, Any], output: Tensor
    ) -> Any:
        output.requires_grad_(True)
        wte_cache.clear()
        wte_cache.append(output)
        return output

    return wte_hook, wte_cache


class AttributionHarvester:
    """Accumulates attribution strengths across batches.

    The attribution formula is:
        attribution[src, tgt] = Σ_batch Σ_pos (∂out[pos, tgt] / ∂in[pos, src]) × in_act[pos, src]

    Key optimization: We sum outputs over positions BEFORE computing gradients,
    reducing backward passes from O(positions × components) to O(components).

    For output targets, we compute attribution to each vocab token individually.
    For wte sources, we compute attribution from each input token individually using scatter_add_.

    Index structure matches DatasetAttributionStorage:
        - Sources: wte tokens [0, vocab_size) + component layers [vocab_size, ...)
        - Targets: component layers [0, n_components) + output tokens [n_components, ...)
    """

    sampling: SamplingType

    def __init__(
        self,
        model: ComponentModel,
        sources_by_target: dict[str, list[str]],
        n_components: int,
        vocab_size: int,
        source_alive: Bool[Tensor, " n_sources"],
        target_alive: Bool[Tensor, " n_targets"],
        sampling: SamplingType,
        device: torch.device,
        show_progress: bool = False,
    ):
        self.model = model
        self.sources_by_target = sources_by_target
        self.n_components = n_components
        self.vocab_size = vocab_size
        self.source_alive = source_alive
        self.target_alive = target_alive
        self.sampling = sampling
        self.device = device
        self.show_progress = show_progress

        # Matrix shape matches storage
        n_sources = vocab_size + n_components  # wte tokens + component layers
        n_targets = n_components + vocab_size  # component layers + output tokens
        self.accumulator = torch.zeros(n_sources, n_targets, device=device)
        self.n_batches = 0
        self.n_tokens = 0

        # Build per-layer index ranges for sources and targets
        self.component_layer_names = list(model.target_module_paths)
        self.source_layer_to_idx_range = self._build_source_layer_index_ranges()
        self.target_layer_to_idx_range = self._build_target_layer_index_ranges()

        # Pre-compute alive indices per layer for efficiency
        self.alive_source_idxs_per_layer = self._build_alive_source_indices_per_layer()
        self.alive_target_idxs_per_layer = self._build_alive_target_indices_per_layer()

    def _build_source_layer_index_ranges(self) -> dict[str, tuple[int, int]]:
        """Build mapping from layer name to (start_idx, end_idx) in source index space.

        Source order: wte tokens [0, vocab_size), then component layers [vocab_size, ...).
        """
        layer_to_idx_range: dict[str, tuple[int, int]] = {}

        # wte covers vocab_size entries (one per token)
        layer_to_idx_range["wte"] = (0, self.vocab_size)
        idx = self.vocab_size

        for layer in self.component_layer_names:
            n_layer_components = self.model.module_to_c[layer]
            layer_to_idx_range[layer] = (idx, idx + n_layer_components)
            idx += n_layer_components

        return layer_to_idx_range

    def _build_target_layer_index_ranges(self) -> dict[str, tuple[int, int]]:
        """Build mapping from layer name to (start_idx, end_idx) in target index space.

        Target order: component layers [0, n_components), then output tokens [n_components, ...).
        """
        layer_to_idx_range: dict[str, tuple[int, int]] = {}
        idx = 0

        for layer in self.component_layer_names:
            n_layer_components = self.model.module_to_c[layer]
            layer_to_idx_range[layer] = (idx, idx + n_layer_components)
            idx += n_layer_components

        # output covers vocab_size entries (one per token)
        layer_to_idx_range["output"] = (idx, idx + self.vocab_size)

        return layer_to_idx_range

    def _build_alive_source_indices_per_layer(self) -> dict[str, list[int]]:
        """Get list of alive source component local indices for each layer."""
        alive_per_layer: dict[str, list[int]] = {}

        for layer in self.source_layer_to_idx_range:
            start_idx, end_idx = self.source_layer_to_idx_range[layer]
            layer_alive = self.source_alive[start_idx:end_idx]
            alive_per_layer[layer] = torch.where(layer_alive)[0].tolist()

        return alive_per_layer

    def _build_alive_target_indices_per_layer(self) -> dict[str, list[int]]:
        """Get list of alive target component local indices for each layer."""
        alive_per_layer: dict[str, list[int]] = {}

        for layer in self.target_layer_to_idx_range:
            start_idx, end_idx = self.target_layer_to_idx_range[layer]
            layer_alive = self.target_alive[start_idx:end_idx]
            alive_per_layer[layer] = torch.where(layer_alive)[0].tolist()

        return alive_per_layer

    def process_batch(self, tokens: Int[Tensor, "batch seq"]) -> None:
        """Accumulate attributions from one batch.

        Uses the key optimization of summing outputs before backward pass.
        """
        batch_size, seq_len = tokens.shape
        self.n_batches += 1
        self.n_tokens += batch_size * seq_len

        # Setup wte hook and create unmasked masks for all components
        wte_hook_fn, wte_cache = _setup_wte_hook()
        assert isinstance(self.model.target_model.wte, nn.Module)
        wte_handle = self.model.target_model.wte.register_forward_hook(
            wte_hook_fn, with_kwargs=True
        )

        # Create masks with all components active
        with torch.no_grad():
            out = self.model(tokens, cache_type="input")
            ci = self.model.calc_causal_importances(
                pre_weight_acts=out.cache,
                sampling=self.sampling,
                detach_inputs=False,
            )

        mask_infos = make_mask_infos(
            component_masks={k: torch.ones_like(v) for k, v in ci.lower_leaky.items()},
            routing_masks="all",
        )

        # Forward pass with gradient-enabled caching
        with torch.enable_grad():
            comp_output: OutputWithCache = self.model(
                tokens,
                mask_infos=mask_infos,
                cache_type="component_acts",
            )

        wte_handle.remove()
        assert len(wte_cache) == 1

        cache = comp_output.cache
        cache["wte_post_detach"] = wte_cache[0]
        cache["output_pre_detach"] = comp_output.output  # Logits [B, S, vocab]
        cache["tokens"] = tokens  # Store tokens for per-token wte attributions

        # Process each target layer
        pbar = tqdm(
            self.sources_by_target.items(),
            desc="Target layers",
            disable=not self.show_progress,
            leave=False,
        )

        target_layers_list = list(self.sources_by_target.items())
        for i, (target_layer, source_layers) in enumerate(pbar):
            if self.show_progress:
                pbar.set_description(f"Target: {target_layer}")

            is_last_layer = i == len(target_layers_list) - 1
            self._process_target_layer(
                target_layer=target_layer,
                source_layers=source_layers,
                cache=cache,
                is_last_layer=is_last_layer,
            )

    def _process_target_layer(
        self,
        target_layer: str,
        source_layers: list[str],
        cache: dict[str, Tensor],
        is_last_layer: bool,
    ) -> None:
        """Process attributions for a single target layer."""
        target_start, _ = self.target_layer_to_idx_range[target_layer]
        alive_target_c_idxs = self.alive_target_idxs_per_layer[target_layer]

        if not alive_target_c_idxs:
            return

        # Get target activations
        out_pre_detach = cache[f"{target_layer}_pre_detach"]

        # Handle output as a special case: one target per vocab token
        is_output_target = target_layer == "output"
        if is_output_target:
            # output: [B, S, vocab] -> sum over batch and sequence, keep vocab dim
            # This gives us per-token logit sums that we can differentiate individually
            total_out_per_token: Float[Tensor, " vocab"] = out_pre_detach.sum(dim=(0, 1))
            # alive_target_c_idxs for output contains indices 0..vocab_size-1
            target_components = [(t, total_out_per_token[t]) for t in alive_target_c_idxs]
        else:
            # Regular component layer: [B, S, C]
            # Sum over batch and sequence BEFORE backward (key optimization)
            total_out: Float[Tensor, " C"] = out_pre_detach.sum(dim=(0, 1))
            target_components = [(c, total_out[c]) for c in alive_target_c_idxs]

        # Gather source post_detach activations
        in_post_detaches: list[Tensor] = []
        for source_layer in source_layers:
            in_post_detaches.append(cache[f"{source_layer}_post_detach"])

        # Process each alive target component
        n_targets = len(target_components)
        for idx, (c_out_local, out_value) in enumerate(target_components):
            c_out_global = target_start + c_out_local

            # Release graph on last backward pass to free memory
            is_last_component = is_last_layer and idx == n_targets - 1
            grads = torch.autograd.grad(
                outputs=out_value,
                inputs=in_post_detaches,
                retain_graph=not is_last_component,
            )

            # Compute attributions for each source layer
            with torch.no_grad():
                for source_layer, grad, in_post_detach in zip(
                    source_layers, grads, in_post_detaches, strict=True
                ):
                    alive_source_c_idxs = self.alive_source_idxs_per_layer[source_layer]

                    if not alive_source_c_idxs:
                        continue

                    # Handle wte as a special case: per-token attributions
                    is_wte_source = source_layer == "wte"
                    if is_wte_source:
                        # wte: [B, S, d_model] -> per-token attributions
                        # Attribution per position = sum of grad * activation over d_model
                        tokens: Int[Tensor, "batch seq"] = cache["tokens"]
                        attr_per_position: Float[Tensor, "batch seq"] = (grad * in_post_detach).sum(
                            dim=-1
                        )
                        # Use scatter_add_ to accumulate by token id
                        flat_tokens = tokens.flatten()  # [B*S]
                        flat_attr = attr_per_position.flatten()  # [B*S]
                        # Source indices for wte are [0, vocab_size), which is the token id
                        self.accumulator[:, c_out_global].scatter_add_(0, flat_tokens, flat_attr)
                    else:
                        # Regular component layer: [B, S, C_in]
                        # Attribution = grad * activation, summed over batch and sequence
                        source_start, _ = self.source_layer_to_idx_range[source_layer]
                        attr_per_component: Float[Tensor, " C_in"] = (grad * in_post_detach).sum(
                            dim=(0, 1)
                        )

                        # Accumulate only for alive source components
                        for c_in_local in alive_source_c_idxs:
                            c_in_global = source_start + c_in_local
                            self.accumulator[c_in_global, c_out_global] += attr_per_component[
                                c_in_local
                            ]
