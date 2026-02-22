"""Core attribution harvester for computing dataset-aggregated attributions.

Computes component-to-component attribution strengths aggregated over the full
training dataset using gradient x activation formula, summed over all positions
and batches.

Uses residual-based storage for scalability:
- Component targets with C > d_input: accumulated as attributions to the layer's input
  residual (d_input dimensions), projected through V at query time.
- Component targets with C <= d_input: accumulated directly (fewer backwards that way).
- Output targets: accumulated as attributions to output residual stream (source_to_out_residual)
  Output attributions computed on-the-fly at query time via w_unembed
"""

from typing import Any

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from tqdm.auto import tqdm

from spd.configs import SamplingType
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.models.components import LinearComponents, make_mask_infos
from spd.utils.general_utils import bf16_autocast


class AttributionHarvester:
    """Accumulates attribution strengths across batches.

    The attribution formula is:
        attribution[src, tgt] = Σ_batch Σ_pos (∂out[pos, tgt] / ∂in[pos, src]) × in_act[pos, src]

    Key optimizations:
    1. Sum outputs over positions BEFORE computing gradients, reducing backward
       passes from O(positions × components) to O(components).
    2. For output targets, store attributions to the pre-unembed residual
       (d_model dimensions) instead of vocab tokens. This eliminates the expensive
       O((V+C) × d_model × V) matmul during harvesting and reduces storage.
    3. Input-residual trick: for target layers where C > d_input, backprop from
       the layer input (d_input dims) instead of per-component (C dims), then
       recover component attributions at query time via V matrix multiplication.

    Index structure:
        - Sources: wte tokens [0, vocab_size) + component layers [vocab_size, ...)
        - Component targets: [0, n_components) in comp_accumulator
        - Output targets: via out_residual_accumulator (computed on-the-fly at query time)
    """

    sampling: SamplingType

    def __init__(
        self,
        model: ComponentModel,
        sources_by_target: dict[str, list[str]],
        n_components: int,
        vocab_size: int,
        source_alive: Bool[Tensor, " n_sources"],
        target_alive: Bool[Tensor, " n_components"],
        sampling: SamplingType,
        embedding_module: nn.Embedding,
        unembed_module: nn.Linear,
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
        self.embedding_module = embedding_module
        self.unembed_module = unembed_module
        self.device = device
        self.show_progress = show_progress

        self.n_sources = vocab_size + n_components
        self.n_batches = 0
        self.n_tokens = 0

        # Split accumulators for component and output targets
        self.comp_accumulator = torch.zeros(self.n_sources, n_components, device=device)

        # For output targets: store attributions to output residual dimensions
        self.d_model = unembed_module.in_features
        self.out_residual_accumulator = torch.zeros(self.n_sources, self.d_model, device=device)

        # Per-layer input-residual accumulators for layers using the residual trick
        # Maps layer_name -> (n_sources, d_input) accumulator
        self.input_residual_accumulators: dict[str, Tensor] = {}
        self.residual_trick_layers: set[str] = set()
        self._setup_residual_trick_layers()

        # Build per-layer index ranges for sources
        self.component_layer_names = list(model.target_module_paths)
        self.source_layer_to_idx_range = self._build_source_layer_index_ranges()
        self.target_layer_to_idx_range = self._build_target_layer_index_ranges()

        # Pre-compute alive indices per layer
        self.alive_source_idxs_per_layer = self._build_alive_indices(
            self.source_layer_to_idx_range, source_alive
        )
        self.alive_target_idxs_per_layer = self._build_alive_indices(
            self.target_layer_to_idx_range, target_alive
        )

    def _setup_residual_trick_layers(self) -> None:
        """Determine which layers benefit from the input-residual trick.

        For layers where d_input < C (number of components), it's cheaper to backprop
        from the d_input-dimensional layer input and recover per-component attributions
        via V matrix multiplication at query time.
        """
        for layer_name in self.model.target_module_paths:
            comp = self.model.components[layer_name]
            if not isinstance(comp, LinearComponents):
                continue
            C = self.model.module_to_c[layer_name]
            d_input = comp.d_in
            if d_input < C:
                self.residual_trick_layers.add(layer_name)
                self.input_residual_accumulators[layer_name] = torch.zeros(
                    self.n_sources, d_input, device=self.device
                )

    def _build_source_layer_index_ranges(self) -> dict[str, tuple[int, int]]:
        """Source order: wte tokens [0, vocab_size), then component layers."""
        ranges: dict[str, tuple[int, int]] = {"wte": (0, self.vocab_size)}
        idx = self.vocab_size
        for layer in self.component_layer_names:
            n = self.model.module_to_c[layer]
            ranges[layer] = (idx, idx + n)
            idx += n
        return ranges

    def _build_target_layer_index_ranges(self) -> dict[str, tuple[int, int]]:
        """Target order: component layers [0, n_components). Output handled separately."""
        ranges: dict[str, tuple[int, int]] = {}
        idx = 0
        for layer in self.component_layer_names:
            n = self.model.module_to_c[layer]
            ranges[layer] = (idx, idx + n)
            idx += n
        return ranges

    def _build_alive_indices(
        self, layer_ranges: dict[str, tuple[int, int]], alive_mask: Bool[Tensor, " n"]
    ) -> dict[str, Tensor]:
        """Get alive local indices for each layer as device tensors."""
        return {
            layer: torch.where(alive_mask[start:end])[0].to(self.device)
            for layer, (start, end) in layer_ranges.items()
        }

    def process_batch(self, tokens: Int[Tensor, "batch seq"]) -> None:
        """Accumulate attributions from one batch."""
        self.n_batches += 1
        self.n_tokens += tokens.numel()

        # Setup hooks to capture wte output and pre-unembed residual
        wte_out: list[Tensor] = []
        pre_unembed: list[Tensor] = []

        def wte_hook(_mod: nn.Module, _args: Any, _kwargs: Any, out: Tensor) -> Tensor:
            out.requires_grad_(True)
            wte_out.clear()
            wte_out.append(out)
            return out

        def pre_unembed_hook(_mod: nn.Module, args: tuple[Any, ...], _kwargs: Any) -> None:
            args[0].requires_grad_(True)
            pre_unembed.clear()
            pre_unembed.append(args[0])

        # Setup hooks to capture inputs to target layers using the residual trick
        layer_inputs: dict[str, Tensor] = {}

        def make_input_capture_hook(layer_name: str):
            def hook(_mod: nn.Module, args: tuple[Any, ...], _kwargs: Any) -> None:
                x = args[0]
                assert isinstance(x, Tensor)
                x.requires_grad_(True)
                layer_inputs[layer_name] = x

            return hook

        h1 = self.embedding_module.register_forward_hook(wte_hook, with_kwargs=True)
        h2 = self.unembed_module.register_forward_pre_hook(pre_unembed_hook, with_kwargs=True)

        input_hooks = []
        for layer_name in self.residual_trick_layers:
            target_module = dict(self.model.target_model.named_modules())[layer_name]
            h = target_module.register_forward_pre_hook(
                make_input_capture_hook(layer_name), with_kwargs=True
            )
            input_hooks.append(h)

        # Get masks with all components active
        with torch.no_grad(), bf16_autocast():
            out = self.model(tokens, cache_type="input")
            ci = self.model.calc_causal_importances(
                pre_weight_acts=out.cache, sampling=self.sampling, detach_inputs=False
            )
        mask_infos = make_mask_infos(
            component_masks={k: torch.ones_like(v) for k, v in ci.lower_leaky.items()},
            routing_masks="all",
        )

        # Forward pass with gradients
        with torch.enable_grad(), bf16_autocast():
            comp_output: OutputWithCache = self.model(
                tokens, mask_infos=mask_infos, cache_type="component_acts"
            )

        h1.remove()
        h2.remove()
        for h in input_hooks:
            h.remove()

        cache = comp_output.cache
        cache["wte_post_detach"] = wte_out[0]
        cache["pre_unembed"] = pre_unembed[0]
        cache["tokens"] = tokens

        # Process each target layer
        layers = list(self.sources_by_target.items())
        pbar = tqdm(layers, desc="Targets", disable=not self.show_progress, leave=False)
        for target_layer, source_layers in pbar:
            if target_layer == "output":
                self._process_output_targets(source_layers, cache)
            elif target_layer in self.residual_trick_layers:
                assert target_layer in layer_inputs, f"Missing input capture for {target_layer}"
                self._process_component_targets_residual(
                    target_layer, source_layers, cache, layer_inputs[target_layer]
                )
            else:
                self._process_component_targets_direct(target_layer, source_layers, cache)

    def _process_component_targets_direct(
        self,
        target_layer: str,
        source_layers: list[str],
        cache: dict[str, Tensor],
    ) -> None:
        """Process attributions to a component layer directly (one backward per component)."""
        target_start, _ = self.target_layer_to_idx_range[target_layer]
        alive_targets = self.alive_target_idxs_per_layer[target_layer]
        if len(alive_targets) == 0:
            return

        target_acts = cache[f"{target_layer}_pre_detach"].sum(dim=(0, 1))
        source_acts = [cache[f"{s}_post_detach"] for s in source_layers]

        for t_idx in alive_targets.tolist():
            grads = torch.autograd.grad(target_acts[t_idx], source_acts, retain_graph=True)
            self._accumulate_attributions(
                self.comp_accumulator[:, target_start + t_idx],
                source_layers,
                grads,
                source_acts,
                cache["tokens"],
            )

    def _process_component_targets_residual(
        self,
        target_layer: str,
        source_layers: list[str],
        cache: dict[str, Tensor],
        layer_input: Tensor,
    ) -> None:
        """Process attributions using the input-residual trick.

        Instead of one backward per component (C passes), backprop from each dimension
        of the layer's input (d_input passes). Per-component attributions are recovered
        at query time via: attr[src, k] = input_residual_attr[src, :] @ V[:, k]
        """
        accumulator = self.input_residual_accumulators[target_layer]
        source_acts = [cache[f"{s}_post_detach"] for s in source_layers]

        # Sum layer input over batch and sequence -> [d_input]
        layer_input_sum = layer_input.sum(dim=(0, 1))
        d_input = layer_input_sum.shape[0]

        for d_idx in range(d_input):
            grads = torch.autograd.grad(layer_input_sum[d_idx], source_acts, retain_graph=True)
            self._accumulate_attributions(
                accumulator[:, d_idx],
                source_layers,
                grads,
                source_acts,
                cache["tokens"],
            )

    def _process_output_targets(
        self,
        source_layers: list[str],
        cache: dict[str, Tensor],
    ) -> None:
        """Process output attributions via output-residual-space storage.

        Instead of computing and storing attributions to vocab tokens directly,
        we store attributions to output residual dimensions. Output attributions are
        computed on-the-fly at query time via: attr[src, token] = out_residual[src] @ w_unembed[:, token]
        """
        out_residual = cache["pre_unembed"].sum(dim=(0, 1))
        source_acts = [cache[f"{s}_post_detach"] for s in source_layers]

        for d_idx in range(self.d_model):
            grads = torch.autograd.grad(out_residual[d_idx], source_acts, retain_graph=True)
            self._accumulate_attributions(
                self.out_residual_accumulator[:, d_idx],
                source_layers,
                grads,
                source_acts,
                cache["tokens"],
            )

    def _accumulate_attributions(
        self,
        target_col: Float[Tensor, " n_sources"],
        source_layers: list[str],
        grads: tuple[Tensor, ...],
        source_acts: list[Tensor],
        tokens: Int[Tensor, "batch seq"],
    ) -> None:
        """Accumulate grad*act attributions from sources to a target column."""
        with torch.no_grad():
            for layer, grad, act in zip(source_layers, grads, source_acts, strict=True):
                alive = self.alive_source_idxs_per_layer[layer]
                if len(alive) == 0:
                    continue

                if layer == "wte":
                    attr = (grad * act).sum(dim=-1).flatten().to(target_col.dtype)
                    target_col.scatter_add_(0, tokens.flatten(), attr)
                else:
                    start, _ = self.source_layer_to_idx_range[layer]
                    attr = (grad * act).sum(dim=(0, 1)).to(target_col.dtype)
                    target_col.scatter_add_(0, start + alive, attr[alive])

    def finalize_comp_accumulator(self) -> Float[Tensor, "n_sources n_components"]:
        """Return comp_accumulator with input-residual layers projected through V.

        For layers using the residual trick, their columns are computed as:
            result[:, target_start:target_start+C] = input_residual_accumulator @ V

        Safe to call multiple times — does not mutate comp_accumulator.
        """
        result = self.comp_accumulator.clone()
        for layer_name, accumulator in self.input_residual_accumulators.items():
            target_start, _ = self.target_layer_to_idx_range[layer_name]
            C = self.model.module_to_c[layer_name]
            V = self.model.components[layer_name].V  # [d_input, C]
            result[:, target_start : target_start + C] = accumulator @ V
        return result
