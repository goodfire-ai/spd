"""Core attribution harvester for computing dataset-aggregated attributions.

Computes component-to-component attribution strengths aggregated over the full
training dataset using gradient x activation formula, summed over all positions
and batches.

Uses residual-based storage for scalability:
- Component targets: accumulated directly to comp_accumulator
- Output targets: accumulated as attributions to output residual stream (source_to_out_residual)
  Output attributions computed on-the-fly at query time via w_unembed
"""

from typing import Any

import torch
from jaxtyping import Bool, Int
from torch import Tensor, nn

from spd.configs import SamplingType
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.models.components import make_mask_infos
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
        vocab_size: int,
        component_alive: dict[str, Bool[Tensor, " n_components"]],
        sampling: SamplingType,
        embedding_module: nn.Embedding,
        unembed_module: nn.Linear,
        device: torch.device,
    ):
        self.model = model
        self.sources_by_target = sources_by_target
        self.vocab_size = vocab_size
        self.component_alive = component_alive
        self.sampling = sampling
        self.embedding_module = embedding_module
        self.unembed_module = unembed_module
        self.device = device

        self.n_batches = 0
        self.n_tokens = 0
        self.output_d_model = unembed_module.in_features

        # Split accumulators for component and output targets
        self.component_attr_accumulator = self._get_component_attr_accumulator(
            sources_by_target,
            component_alive,
            unembed_module,
            vocab_size,
            device,
        )

    def _get_component_attr_accumulator(
        self,
        sources_by_target: dict[str, list[str]],
        component_alive: dict[str, Bool[Tensor, " n_components"]],
        unembed_module: nn.Linear,
        vocab_size: int,
        device: torch.device,
    ) -> dict[str, dict[str, Tensor]]:
        component_attr_accumulator: dict[str, dict[str, Tensor]] = {}

        for target_layer, source_layers in sources_by_target.items():
            if target_layer == "output":
                target_d = unembed_module.in_features
            else:
                (target_c,) = component_alive[target_layer].shape
                target_d = target_c

            source_attr_accumulator: dict[str, Tensor] = {}
            for source_layer in source_layers:
                if source_layer == "wte":
                    source_d = vocab_size
                else:
                    (source_c,) = component_alive[source_layer].shape
                    source_d = source_c

                source_attr_accumulator[source_layer] = torch.zeros(
                    (target_d, source_d), device=device
                )

            component_attr_accumulator[target_layer] = source_attr_accumulator

        return component_attr_accumulator

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

        h1 = self.embedding_module.register_forward_hook(wte_hook, with_kwargs=True)
        h2 = self.unembed_module.register_forward_pre_hook(pre_unembed_hook, with_kwargs=True)

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

        cache = comp_output.cache
        cache["wte_post_detach"] = wte_out[0]
        cache["pre_unembed"] = pre_unembed[0]
        # cache["tokens"] = tokens

        # Process each target layer
        for target_layer in self.sources_by_target:
            if target_layer == "output":
                self._process_output_targets(cache, ci.lower_leaky, tokens)
            else:
                self._process_component_targets(target_layer, ci.lower_leaky, cache, tokens)

    def _process_output_targets(
        self,
        cache: dict[str, Tensor],
        ci: dict[str, Tensor],
        tokens: Int[Tensor, "batch seq"],
    ) -> None:
        """Process output attributions via output-residual-space storage.

        Instead of computing and storing attributions to vocab tokens directly,
        we store attributions to output residual dimensions. Output attributions are
        computed on-the-fly at query time via: attr[src, token] = out_residual[src] @ w_unembed[:, token]
        """
        # Sum output residual over batch and sequence -> [d_model]
        out_residual = cache["pre_unembed"].sum(dim=(0, 1))

        source_layers = self.sources_by_target["output"]
        source_acts = [cache[f"{s}_post_detach"] for s in source_layers]

        for d_idx in range(self.output_d_model):
            grads = torch.autograd.grad(out_residual[d_idx], source_acts, retain_graph=True)
            source_acts_grads = list(zip(source_layers, source_acts, grads, strict=True))

            self._accumulate_attributions(
                "output",
                d_idx,
                source_acts_grads,
                ci,
                tokens,
            )

    def _process_component_targets(
        self,
        target_layer: str,
        ci: dict[str, Tensor],
        cache: dict[str, Tensor],
        tokens: Int[Tensor, "batch seq"],
    ) -> None:
        """Process attributions to a component layer."""
        alive_targets = self.component_alive[target_layer]
        if not alive_targets.any():
            return

        # Sum over batch and sequence

        target_acts_raw = cache[f"{target_layer}_pre_detach"]
        ci_weighted_target_acts = (target_acts_raw * ci[target_layer]).sum(dim=(0, 1))

        source_layers = self.sources_by_target[target_layer]
        source_acts = [cache[f"{s}_post_detach"] for s in source_layers]

        for t_idx in alive_targets.tolist():
            grads = torch.autograd.grad(
                ci_weighted_target_acts[t_idx], source_acts, retain_graph=True
            )

            source_acts_grads = list(zip(source_layers, source_acts, grads, strict=True))

            self._accumulate_attributions(
                target_layer,
                t_idx,
                source_acts_grads,
                ci,
                tokens,
            )

    @torch.no_grad()
    def _accumulate_attributions(
        self,
        target_layer: str,
        target_idx: int,
        source_acts_grads: list[tuple[str, Tensor, Tensor]],
        ci: dict[str, Tensor],
        tokens: Int[Tensor, "batch seq"],
    ) -> None:
        """Accumulate grad*act attributions from sources to a target column."""
        target_accs = self.component_attr_accumulator[target_layer]

        for source_layer, act, grad in source_acts_grads:
            attr_accumulator = target_accs[source_layer][target_idx]

            ci_weighted_attr = grad * act * ci[source_layer]

            if source_layer == "wte":
                # Per-token: sum grad*act*ci over d_model, scatter by token id
                # TODO(oli): figure out why this works
                attr = ci_weighted_attr.sum(dim=-1).flatten()
                attr_accumulator.scatter_add_(0, tokens.flatten(), attr)
            else:
                # Per-component: sum grad*act*ci over batch and sequence
                attr = ci_weighted_attr.sum(dim=(0, 1))
                attr_accumulator.add_(attr)
