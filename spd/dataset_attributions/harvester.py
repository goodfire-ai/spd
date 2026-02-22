"""Core attribution harvester for computing dataset-aggregated attributions.

Computes component-to-component attribution strengths aggregated over the full
training dataset using gradient x activation formula, summed over all positions
and batches.

Storage strategy (minimizing backward passes per batch):
- For each target layer, backprop from d_input dimensions of the layer's input,
  then recover per-component attributions via @ V at query time.
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

    layer_name: str
    target_start: int
    n_components: int
    d_input: int


class AttributionHarvester:
    """Accumulates attribution strengths across batches.

    The attribution formula is:
        attribution[src, tgt] = Σ_batch Σ_pos (∂out[pos, tgt] / ∂in[pos, src]) × in_act[pos, src]

    Key optimizations:
    1. Sum outputs over positions BEFORE computing gradients, reducing backward
       passes from O(positions × components) to O(components).
    2. Input-residual trick: backprop from d_input dimensions of each target layer's
       input, then recover per-component attributions via V matmul at query time.
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

        # Per-layer input-residual accumulators: [n_sources, d_input]
        # Component attributions recovered at query time via @ V
        self._input_residual_accumulators: dict[str, Tensor] = {}
        # Output residual accumulator: [n_sources, d_model]
        self._out_residual_accumulator = torch.zeros(self.n_sources, self.d_model, device=device)

        # Build index ranges
        self.component_layer_names = list(model.target_module_paths)
        self._source_layer_to_idx_range = self._build_source_layer_index_ranges()
        self._target_layer_to_idx_range = self._build_target_layer_index_ranges()

        # Pre-compute alive indices as device tensors
        self._alive_source_idxs = self._build_alive_indices(
            self._source_layer_to_idx_range, source_alive
        )

        # Pre-compute per-target-layer info
        self._target_layer_infos: dict[str, _TargetLayerInfo] = {}
        for layer_name in self.component_layer_names:
            target_start, _ = self._target_layer_to_idx_range[layer_name]
            C = model.module_to_c[layer_name]
            comp = model.components[layer_name]
            assert isinstance(comp, LinearComponents)
            info = _TargetLayerInfo(
                layer_name=layer_name,
                target_start=target_start,
                n_components=C,
                d_input=comp.d_in,
            )
            self._target_layer_infos[layer_name] = info
            self._input_residual_accumulators[layer_name] = torch.zeros(
                self.n_sources, comp.d_in, device=device
            )

        # Persistent hooks (capture is gated by self._capturing flag)
        self._capturing = False
        self._wte_out: Tensor | None = None
        self._pre_unembed: Tensor | None = None
        self._layer_inputs: dict[str, Tensor] = {}

        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks(embedding_module, unembed_module)

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
            h.remove()  # type: ignore[union-attr]
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
                    self._out_residual_accumulator,
                    source_layers,
                    cache,
                )
            else:
                assert target_layer in self._layer_inputs
                self._process_targets(
                    self._layer_inputs[target_layer].sum(dim=(0, 1)),
                    self._input_residual_accumulators[target_layer],
                    source_layers,
                    cache,
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
    ) -> None:
        """Backprop from each dimension of target_vector, accumulate grad*act attributions."""
        source_acts = [cache[f"{s}_post_detach"] for s in source_layers]

        for d_idx in range(target_vector.shape[0]):
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

        Projects each layer's input-residual accumulator through V.
        Safe to call multiple times.
        """
        result = torch.zeros(self.n_sources, self.n_components, device=self.device)
        for layer_name, accumulator in self._input_residual_accumulators.items():
            info = self._target_layer_infos[layer_name]
            V = self.model.components[layer_name].V  # [d_input, C]
            result[:, info.target_start : info.target_start + info.n_components] = accumulator @ V
        return result

    def get_out_residual_attributions(self) -> Float[Tensor, "n_sources d_model"]:
        """Return the source-to-output-residual attribution matrix."""
        return self._out_residual_accumulator.clone()

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
