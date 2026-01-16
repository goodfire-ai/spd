"""Core attribution harvester for computing dataset-aggregated attributions.

Computes component-to-component attribution strengths aggregated over the full
training dataset using gradient × activation formula, summed over all positions
and batches.
"""

from dataclasses import dataclass
from typing import Any

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from tqdm.auto import tqdm

from spd.configs import SamplingType
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.models.components import make_mask_infos


@dataclass
class AttributionHarvesterState:
    """Serializable state for potential future parallel merging."""

    component_keys: list[str]
    accumulator: Float[Tensor, "n_components n_components"]
    n_batches: int
    n_tokens: int

    @staticmethod
    def merge(states: list["AttributionHarvesterState"]) -> "AttributionHarvesterState":
        """Merge states from parallel workers (simple sum)."""
        assert len(states) > 0
        assert all(s.component_keys == states[0].component_keys for s in states)
        merged_accumulator = states[0].accumulator.clone()
        for s in states[1:]:
            merged_accumulator += s.accumulator
        return AttributionHarvesterState(
            component_keys=states[0].component_keys,
            accumulator=merged_accumulator,
            n_batches=sum(s.n_batches for s in states),
            n_tokens=sum(s.n_tokens for s in states),
        )


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
        attribution[c_in, c_out] = Σ_batch Σ_pos (∂out[pos, c_out] / ∂in[pos, c_in]) × in_act[pos, c_in]

    Key optimization: We sum outputs over positions BEFORE computing gradients,
    reducing backward passes from O(positions × components) to O(components).
    """

    sampling: SamplingType

    def __init__(
        self,
        model: ComponentModel,
        sources_by_target: dict[str, list[str]],
        component_keys: list[str],
        alive_mask: Bool[Tensor, " n_components"],
        sampling: SamplingType,
        device: torch.device,
        show_progress: bool = False,
    ):
        self.model = model
        self.sources_by_target = sources_by_target
        self.component_keys = component_keys
        self.alive_mask = alive_mask
        self.sampling = sampling
        self.device = device
        self.show_progress = show_progress

        n = len(component_keys)
        self.accumulator = torch.zeros(n, n, device=device)
        self.n_batches = 0
        self.n_tokens = 0

        self.key_to_idx = {k: i for i, k in enumerate(component_keys)}

        # Build per-layer index ranges
        self.layer_names = list(model.target_module_paths)
        self.layer_to_idx_range = self._build_layer_index_ranges()

        # Pre-compute alive indices per layer for efficiency
        self.alive_c_idxs_per_layer = self._build_alive_indices_per_layer()

    def _build_layer_index_ranges(self) -> dict[str, tuple[int, int]]:
        """Build mapping from layer name to (start_idx, end_idx) in flat component space."""
        layer_to_idx_range: dict[str, tuple[int, int]] = {}
        idx = 0
        for layer in self.layer_names:
            n_components = self.model.module_to_c[layer]
            layer_to_idx_range[layer] = (idx, idx + n_components)
            idx += n_components
        return layer_to_idx_range

    def _build_alive_indices_per_layer(self) -> dict[str, list[int]]:
        """Get list of alive component indices for each layer."""
        alive_per_layer: dict[str, list[int]] = {}
        for layer in self.layer_names:
            start_idx, end_idx = self.layer_to_idx_range[layer]
            layer_alive = self.alive_mask[start_idx:end_idx]
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

        # Process each target layer
        pbar = tqdm(
            self.sources_by_target.items(),
            desc="Target layers",
            disable=not self.show_progress,
            leave=False,
        )

        for target_layer, source_layers in pbar:
            if self.show_progress:
                pbar.set_description(f"Target: {target_layer}")

            self._process_target_layer(
                target_layer=target_layer,
                source_layers=source_layers,
                cache=cache,
            )

    def _process_target_layer(
        self,
        target_layer: str,
        source_layers: list[str],
        cache: dict[str, Tensor],
    ) -> None:
        """Process attributions for a single target layer."""
        # Get target indices
        target_start, _ = self.layer_to_idx_range[target_layer]
        alive_target_c_idxs = self.alive_c_idxs_per_layer[target_layer]

        if not alive_target_c_idxs:
            return

        # Get target activations (pre_detach for gradient computation)
        out_pre_detach: Float[Tensor, "B S C"] = cache[f"{target_layer}_pre_detach"]

        # Sum over batch and sequence BEFORE backward (key optimization)
        # This reduces backward passes from O(B*S*C_out) to O(C_out)
        total_out: Float[Tensor, " C"] = out_pre_detach.sum(dim=(0, 1))

        # Gather source post_detach activations
        in_post_detaches: list[Float[Tensor, "B S C"]] = []
        for source_layer in source_layers:
            in_post_detaches.append(cache[f"{source_layer}_post_detach"])

        # Process each alive target component
        for c_out_local in alive_target_c_idxs:
            c_out_global = target_start + c_out_local

            # Single backward pass per target component
            grads = torch.autograd.grad(
                outputs=total_out[c_out_local],
                inputs=in_post_detaches,
                retain_graph=True,
            )

            # Compute attributions for each source layer
            with torch.no_grad():
                for source_layer, grad, in_post_detach in zip(
                    source_layers, grads, in_post_detaches, strict=True
                ):
                    source_start, _ = self.layer_to_idx_range[source_layer]
                    alive_source_c_idxs = self.alive_c_idxs_per_layer[source_layer]

                    if not alive_source_c_idxs:
                        continue

                    # Attribution = grad * activation, summed over batch and sequence
                    # grad: [B, S, C_in], in_post_detach: [B, S, C_in]
                    attr_per_component: Float[Tensor, " C_in"] = (grad * in_post_detach).sum(
                        dim=(0, 1)
                    )

                    # Accumulate only for alive source components
                    for c_in_local in alive_source_c_idxs:
                        c_in_global = source_start + c_in_local
                        self.accumulator[c_in_global, c_out_global] += attr_per_component[
                            c_in_local
                        ]

    def get_state(self) -> AttributionHarvesterState:
        """Extract state for potential parallel merging."""
        return AttributionHarvesterState(
            component_keys=self.component_keys,
            accumulator=self.accumulator.cpu(),
            n_batches=self.n_batches,
            n_tokens=self.n_tokens,
        )

    @classmethod
    def from_state(
        cls,
        state: AttributionHarvesterState,
        model: ComponentModel,
        sources_by_target: dict[str, list[str]],
        alive_mask: Bool[Tensor, " n_components"],
        sampling: SamplingType,
        device: torch.device,
    ) -> "AttributionHarvester":
        """Reconstruct harvester from state (for merging)."""
        harvester = cls(
            model=model,
            sources_by_target=sources_by_target,
            component_keys=state.component_keys,
            alive_mask=alive_mask,
            sampling=sampling,
            device=device,
        )
        harvester.accumulator = state.accumulator.to(device)
        harvester.n_batches = state.n_batches
        harvester.n_tokens = state.n_tokens
        return harvester
