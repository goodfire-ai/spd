"""Core attribution harvester for computing dataset-aggregated attributions.

Computes component-to-component attribution strengths aggregated over the full
training dataset using gradient x activation formula, summed over all positions
and batches.

Storage strategy (minimizing backward passes per batch):
- For each target layer, we backprop from min(d_input, C) dimensions:
  - If d_input < C: backprop from layer input dims, recover components via @ V
  - If d_input >= C: backprop from component activations directly
- Output targets: backprop from d_model output residual dims, recover via @ w_unembed
"""

from dataclasses import dataclass
from typing import Any

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from tqdm.auto import tqdm

from spd.configs import SamplingType
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.models.components import LinearComponents, make_mask_infos
from spd.utils.general_utils import bf16_autocast


@dataclass
class _TargetLayerInfo:
    """Pre-computed info for processing a target layer."""

    target_start: int
    n_components: int
    use_input_residual: bool
    # Only set when use_input_residual is True
    d_input: int | None
    V: Tensor | None  # [d_input, C] — for projecting residual attributions to components


class AttributionHarvester:
    """Accumulates attribution strengths across batches.

    The attribution formula is:
        attribution[src, tgt] = Σ_batch Σ_pos (∂out[pos, tgt] / ∂in[pos, src]) × in_act[pos, src]

    Key optimizations:
    1. Sum outputs over positions BEFORE computing gradients, reducing backward
       passes from O(positions × components) to O(components).
    2. For each target layer, backprop from min(d_input, C) dimensions. When
       d_input < C, store attributions to the layer input and recover per-component
       attributions via V matmul.
    3. For output targets, store attributions to the pre-unembed residual
       (d_model dimensions) instead of vocab tokens.
    4. Vectorized accumulation via scatter_add_.
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
        self.sampling = sampling
        self.device = device
        self.show_progress = show_progress

        self.n_sources = vocab_size + n_components
        self.n_batches = 0
        self.n_tokens = 0
        self.d_model = unembed_module.in_features

        # Accumulators (private — access via get_comp_attributions / out_residual_accumulator)
        self._comp_accumulator = torch.zeros(self.n_sources, n_components, device=device)
        self._input_residual_accumulators: dict[str, Tensor] = {}
        self.out_residual_accumulator = torch.zeros(self.n_sources, self.d_model, device=device)

        # Build index ranges
        self.component_layer_names = list(model.target_module_paths)
        self._source_layer_to_idx_range = self._build_source_layer_index_ranges()
        self._target_layer_to_idx_range = self._build_target_layer_index_ranges()

        # Pre-compute alive indices as device tensors
        self._alive_source_idxs = self._build_alive_indices(
            self._source_layer_to_idx_range, source_alive
        )
        alive_target_idxs = self._build_alive_indices(self._target_layer_to_idx_range, target_alive)

        # Pre-compute per-target-layer info and decide strategy
        self._target_layer_infos: dict[str, _TargetLayerInfo] = {}
        for layer_name in self.component_layer_names:
            target_start, _ = self._target_layer_to_idx_range[layer_name]
            C = model.module_to_c[layer_name]
            comp = model.components[layer_name]
            use_residual = isinstance(comp, LinearComponents) and comp.d_in < C
            d_in = comp.d_in if isinstance(comp, LinearComponents) else None
            info = _TargetLayerInfo(
                target_start=target_start,
                n_components=C,
                use_input_residual=use_residual,
                d_input=d_in if use_residual else None,
                V=comp.V if use_residual else None,
            )
            self._target_layer_infos[layer_name] = info
            if use_residual:
                assert d_in is not None
                self._input_residual_accumulators[layer_name] = torch.zeros(
                    self.n_sources, d_in, device=device
                )

        # Persistent hooks (capture is gated by self._capturing flag)
        self._capturing = False
        self._wte_out: Tensor | None = None
        self._pre_unembed: Tensor | None = None
        self._layer_inputs: dict[str, Tensor] = {}

        self._hooks: list[Any] = []
        self._register_hooks(embedding_module, unembed_module)

        # Pre-compute alive target indices as lists (for the backward loop)
        self._alive_targets_by_layer: dict[str, list[int]] = {
            layer: idxs.tolist() for layer, idxs in alive_target_idxs.items()
        }

    def _register_hooks(self, embedding_module: nn.Embedding, unembed_module: nn.Linear) -> None:
        """Register persistent hooks that capture activations when self._capturing is True."""

        def wte_hook(_mod: nn.Module, _args: Any, _kwargs: Any, out: Tensor) -> Tensor:
            if self._capturing:
                out.requires_grad_(True)
                self._wte_out = out
            return out

        def pre_unembed_hook(_mod: nn.Module, args: tuple[Any, ...], _kwargs: Any) -> None:
            if self._capturing:
                args[0].requires_grad_(True)
                self._pre_unembed = args[0]

        def make_input_hook(layer_name: str):
            def hook(_mod: nn.Module, args: tuple[Any, ...], _kwargs: Any) -> None:
                if self._capturing:
                    x = args[0]
                    assert isinstance(x, Tensor)
                    x.requires_grad_(True)
                    self._layer_inputs[layer_name] = x

            return hook

        self._hooks.append(embedding_module.register_forward_hook(wte_hook, with_kwargs=True))
        self._hooks.append(
            unembed_module.register_forward_pre_hook(pre_unembed_hook, with_kwargs=True)
        )
        target_modules = dict(self.model.target_model.named_modules())
        for layer_name in self._input_residual_accumulators:
            self._hooks.append(
                target_modules[layer_name].register_forward_pre_hook(
                    make_input_hook(layer_name), with_kwargs=True
                )
            )

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def process_batch(self, tokens: Int[Tensor, "batch seq"]) -> None:
        """Accumulate attributions from one batch."""
        self.n_batches += 1
        self.n_tokens += tokens.numel()

        # First forward: get CI values (no gradient needed)
        with torch.no_grad(), bf16_autocast():
            out = self.model(tokens, cache_type="input")
            ci = self.model.calc_causal_importances(
                pre_weight_acts=out.cache, sampling=self.sampling, detach_inputs=False
            )
        mask_infos = make_mask_infos(
            component_masks={k: torch.ones_like(v) for k, v in ci.lower_leaky.items()},
            routing_masks="all",
        )

        # Second forward: with gradients, hooks capture activations
        self._capturing = True
        with torch.enable_grad(), bf16_autocast():
            comp_output: OutputWithCache = self.model(
                tokens, mask_infos=mask_infos, cache_type="component_acts"
            )
        self._capturing = False

        assert self._wte_out is not None
        assert self._pre_unembed is not None

        cache = comp_output.cache
        cache["wte_post_detach"] = self._wte_out
        cache["pre_unembed"] = self._pre_unembed
        cache["tokens"] = tokens

        # Process each target layer
        layers = list(self.sources_by_target.items())
        pbar = tqdm(layers, desc="Targets", disable=not self.show_progress, leave=False)
        for target_layer, source_layers in pbar:
            if target_layer == "output":
                self._process_targets(
                    cache["pre_unembed"].sum(dim=(0, 1)),
                    self.out_residual_accumulator,
                    source_layers,
                    cache,
                )
            else:
                info = self._target_layer_infos[target_layer]
                if info.use_input_residual:
                    assert target_layer in self._layer_inputs
                    self._process_targets(
                        self._layer_inputs[target_layer].sum(dim=(0, 1)),
                        self._input_residual_accumulators[target_layer],
                        source_layers,
                        cache,
                    )
                else:
                    self._process_targets(
                        cache[f"{target_layer}_pre_detach"].sum(dim=(0, 1)),
                        self._comp_accumulator[
                            :, info.target_start : info.target_start + info.n_components
                        ],
                        source_layers,
                        cache,
                        alive_indices=self._alive_targets_by_layer[target_layer],
                    )

        # Clear per-batch captures
        self._wte_out = None
        self._pre_unembed = None
        self._layer_inputs.clear()

    def _process_targets(
        self,
        target_vector: Float[Tensor, " n_dims"],
        accumulator: Float[Tensor, "n_sources n_dims"],
        source_layers: list[str],
        cache: dict[str, Tensor],
        alive_indices: list[int] | None = None,
    ) -> None:
        """Backprop from each dimension of target_vector, accumulate grad*act attributions.

        Args:
            target_vector: Sum-over-positions target activations [n_dims].
            accumulator: Where to accumulate [n_sources, n_dims].
            source_layers: Which source layers to compute attributions from.
            cache: Forward pass cache with source activations and tokens.
            alive_indices: If provided, only backprop from these indices of target_vector.
                If None, backprop from all dimensions.
        """
        source_acts = [cache[f"{s}_post_detach"] for s in source_layers]
        dims = alive_indices if alive_indices is not None else range(target_vector.shape[0])

        for d_idx in dims:
            grads = torch.autograd.grad(target_vector[d_idx], source_acts, retain_graph=True)
            self._accumulate_attributions(
                accumulator[:, d_idx],
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
                alive = self._alive_source_idxs[layer]
                if len(alive) == 0:
                    continue

                if layer == "wte":
                    attr = (grad * act).sum(dim=-1).flatten().to(target_col.dtype)
                    target_col.scatter_add_(0, tokens.flatten(), attr)
                else:
                    start, _ = self._source_layer_to_idx_range[layer]
                    attr = (grad * act).sum(dim=(0, 1)).to(target_col.dtype)
                    target_col.scatter_add_(0, start + alive, attr[alive])

    def get_comp_attributions(self) -> Float[Tensor, "n_sources n_components"]:
        """Return the full source-to-component attribution matrix.

        Projects input-residual accumulators through V for layers using that trick.
        Safe to call multiple times.
        """
        result = self._comp_accumulator.clone()
        for layer_name, accumulator in self._input_residual_accumulators.items():
            info = self._target_layer_infos[layer_name]
            assert info.V is not None
            result[:, info.target_start : info.target_start + info.n_components] = (
                accumulator @ info.V
            )
        return result

    def _build_source_layer_index_ranges(self) -> dict[str, tuple[int, int]]:
        ranges: dict[str, tuple[int, int]] = {"wte": (0, self.vocab_size)}
        idx = self.vocab_size
        for layer in self.component_layer_names:
            n = self.model.module_to_c[layer]
            ranges[layer] = (idx, idx + n)
            idx += n
        return ranges

    def _build_target_layer_index_ranges(self) -> dict[str, tuple[int, int]]:
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
        return {
            layer: torch.where(alive_mask[start:end])[0].to(self.device)
            for layer, (start, end) in layer_ranges.items()
        }
