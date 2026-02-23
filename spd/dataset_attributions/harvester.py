"""Core attribution harvester for computing dataset-aggregated attributions.

Computes component-to-component attribution strengths aggregated over the full
training dataset using gradient x activation formula, summed over all positions
and batches.

Three metrics are accumulated:
- attr:         E[∂y/∂x · x]           (signed mean attribution)
- attr_abs:     E[∂|y|/∂x · x]         (attribution to absolute value of target)
- squared_attr: E[(∂y/∂x · x)²]        (mean squared attribution, for RMS)

Naming convention: modifier before "attr" applies to the target (e.g. attr_abs =
attribution to |target|), modifier after applies to the attribution itself
(e.g. squared_attr = squared attribution).

Uses residual-based storage for scalability:
- Component targets: accumulated directly
- Output targets: accumulated as attributions to output residual stream,
  computed on-the-fly at query time via w_unembed

All layer keys are concrete module paths (e.g. "wte", "h.0.attn.q_proj", "lm_head").
Translation to canonical names happens at the storage boundary in harvest.py.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from jaxtyping import Bool, Int
from torch import Tensor, nn

from spd.configs import SamplingType
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.models.components import make_mask_infos
from spd.utils.general_utils import bf16_autocast


class AttributionHarvester:
    """Accumulates attribution strengths across batches using concrete module paths.

    The attribution formula is:
        attribution[src, tgt] = Σ_batch Σ_pos (∂out[pos, tgt] / ∂in[pos, src]) × in_act[pos, src]

    Key optimizations:
    1. Sum outputs over positions BEFORE computing gradients, reducing backward
       passes from O(positions × components) to O(components).
    2. For output targets, store attributions to the pre-unembed residual
       (d_model dimensions) instead of vocab tokens. This eliminates the expensive
       O((V+C) × d_model × V) matmul during harvesting and reduces storage.
    """

    sampling: SamplingType

    def __init__(
        self,
        model: ComponentModel,
        sources_by_target: dict[str, list[str]],
        vocab_size: int,
        component_alive: dict[str, Bool[Tensor, " n_components"]],
        sampling: SamplingType,
        embed_path: str,
        embedding_module: nn.Embedding,
        unembed_path: str,
        unembed_module: nn.Linear,
        device: torch.device,
    ):
        self.model = model
        self.sources_by_target = sources_by_target
        self.vocab_size = vocab_size
        self.component_alive = component_alive
        self.sampling = sampling
        self.embed_path = embed_path
        self.embedding_module = embedding_module
        self.unembed_path = unembed_path
        self.unembed_module = unembed_module
        self.device = device

        self.n_batches = 0
        self.n_tokens = 0
        self.output_d_model = unembed_module.in_features

        self.attr_accumulator = self._build_accumulator(
            sources_by_target, component_alive, unembed_module, vocab_size, device
        )
        self.attr_abs_accumulator = self._build_accumulator(
            sources_by_target, component_alive, unembed_module, vocab_size, device
        )
        self.square_attr_accumulator = self._build_accumulator(
            sources_by_target, component_alive, unembed_module, vocab_size, device
        )

    def _build_accumulator(
        self,
        sources_by_target: dict[str, list[str]],
        component_alive: dict[str, Bool[Tensor, " n_components"]],
        unembed_module: nn.Linear,
        vocab_size: int,
        device: torch.device,
    ) -> dict[str, dict[str, Tensor]]:
        accumulator: dict[str, dict[str, Tensor]] = {}

        for target_layer, source_layers in sources_by_target.items():
            if target_layer == self.unembed_path:
                target_d = unembed_module.in_features
            else:
                (target_c,) = component_alive[target_layer].shape
                target_d = target_c

            source_acc: dict[str, Tensor] = {}
            for source_layer in source_layers:
                if source_layer == self.embed_path:
                    source_d = vocab_size
                else:
                    (source_c,) = component_alive[source_layer].shape
                    source_d = source_c

                source_acc[source_layer] = torch.zeros((target_d, source_d), device=device)

            accumulator[target_layer] = source_acc

        return accumulator

    @dataclass
    class NormalizedAttrs:
        attr: dict[str, dict[str, Tensor]]
        attr_abs: dict[str, dict[str, Tensor]]
        mean_squared_attr: dict[str, dict[str, Tensor]]

    def normalized_attrs(self) -> NormalizedAttrs:
        """Return the accumulated attributions normalized by n_tokens.

        mean_squared_attr is pre-sqrt so it can be merged across workers.
        """
        attr = defaultdict[str, dict[str, Tensor]](dict)
        attr_abs = defaultdict[str, dict[str, Tensor]](dict)
        mean_squared_attr = defaultdict[str, dict[str, Tensor]](dict)

        for target in self.attr_accumulator:
            for source in self.sources_by_target[target]:
                attr[target][source] = self.attr_accumulator[target][source] / self.n_tokens
                attr_abs[target][source] = self.attr_abs_accumulator[target][source] / self.n_tokens
                mean_squared_attr[target][source] = (
                    self.square_attr_accumulator[target][source] / self.n_tokens
                )

        return self.NormalizedAttrs(
            attr=attr,
            attr_abs=attr_abs,
            mean_squared_attr=mean_squared_attr,
        )

    def process_batch(self, tokens: Int[Tensor, "batch seq"]) -> None:
        """Accumulate attributions from one batch."""
        self.n_batches += 1
        self.n_tokens += tokens.numel()

        # Setup hooks to capture embedding output and pre-unembed residual
        embed_out: list[Tensor] = []
        pre_unembed: list[Tensor] = []

        def embed_hook(_mod: nn.Module, _args: Any, _kwargs: Any, out: Tensor) -> Tensor:
            out.requires_grad_(True)
            embed_out.clear()
            embed_out.append(out)
            return out

        def pre_unembed_hook(_mod: nn.Module, args: tuple[Any, ...], _kwargs: Any) -> None:
            args[0].requires_grad_(True)
            pre_unembed.clear()
            pre_unembed.append(args[0])

        h1 = self.embedding_module.register_forward_hook(embed_hook, with_kwargs=True)
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
        cache[f"{self.embed_path}_post_detach"] = embed_out[0]
        cache[f"{self.unembed_path}_pre_detach"] = pre_unembed[0]

        for target_layer in self.sources_by_target:
            if target_layer == self.unembed_path:
                self._process_output_targets(cache, ci.lower_leaky, tokens)
            else:
                self._process_component_targets(target_layer, ci.lower_leaky, cache, tokens)

    def _process_output_targets(
        self,
        cache: dict[str, Tensor],
        ci: dict[str, Tensor],
        tokens: Int[Tensor, "batch seq"],
    ) -> None:
        """Process output attributions via output-residual-space storage."""
        out_residual = cache[f"{self.unembed_path}_pre_detach"]

        out_residual_sum = out_residual.sum(dim=(0, 1))
        out_residual_sum_abs = out_residual.abs().sum(dim=(0, 1))

        source_layers = self.sources_by_target[self.unembed_path]
        source_acts = [cache[f"{s}_post_detach"] for s in source_layers]

        for d_idx in range(self.output_d_model):
            grads = torch.autograd.grad(out_residual_sum[d_idx], source_acts, retain_graph=True)
            abs_grads = torch.autograd.grad(
                out_residual_sum_abs[d_idx], source_acts, retain_graph=True
            )

            self._accumulate_attributions(
                self.unembed_path,
                d_idx,
                source_layers,
                source_acts,
                list(grads),
                list(abs_grads),
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

        target_acts_raw = cache[f"{target_layer}_pre_detach"]

        ci_weighted_target_acts = (target_acts_raw * ci[target_layer]).sum(dim=(0, 1))
        ci_weighted_target_acts_abs = (target_acts_raw.abs() * ci[target_layer]).sum(dim=(0, 1))

        source_layers = self.sources_by_target[target_layer]
        source_acts = [cache[f"{s}_post_detach"] for s in source_layers]

        for t_idx in torch.where(alive_targets)[0].tolist():
            grads = torch.autograd.grad(
                ci_weighted_target_acts[t_idx], source_acts, retain_graph=True
            )

            abs_grads = torch.autograd.grad(
                ci_weighted_target_acts_abs[t_idx], source_acts, retain_graph=True
            )

            self._accumulate_attributions(
                target_layer,
                t_idx,
                source_layers,
                source_acts,
                list(grads),
                list(abs_grads),
                ci,
                tokens,
            )

    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
    def _accumulate_attributions(
        self,
        target_layer: str,
        target_idx: int,
        source_layers: list[str],
        source_acts: list[Tensor],
        source_grads: list[Tensor],
        source_abs_grads: list[Tensor],
        ci: dict[str, Tensor],
        tokens: Int[Tensor, "batch seq"],
    ) -> None:
        """Accumulate grad*act attributions from sources to a target column."""

        attr_accumulator = self.attr_accumulator[target_layer]
        attr_abs_accumulator = self.attr_abs_accumulator[target_layer]
        square_attr_accumulator = self.square_attr_accumulator[target_layer]

        for source_layer, act, grad, abs_grad in zip(
            source_layers, source_acts, source_grads, source_abs_grads, strict=True
        ):
            attr_acc = attr_accumulator[source_layer][target_idx]
            attr_abs_acc = attr_abs_accumulator[source_layer][target_idx]
            square_attr_acc = square_attr_accumulator[source_layer][target_idx]

            # Embed has no CI (all tokens always active)
            source_ci = ci[source_layer] if source_layer != self.embed_path else 1.0

            ci_weighted_attr = grad * act * source_ci
            ci_weighted_attr_abs = abs_grad * act * source_ci
            ci_weighted_squared_attr = ci_weighted_attr.square()

            if source_layer == self.embed_path:
                # Per-token: sum grad*act*ci over d_model, scatter by token id
                # TODO(oli): figure out why this works
                attr = ci_weighted_attr.sum(dim=-1).flatten()
                attr_abs = ci_weighted_attr_abs.sum(dim=-1).flatten()
                attr_squared = ci_weighted_squared_attr.sum(dim=-1).flatten()

                attr_acc.scatter_add_(0, tokens.flatten(), attr)
                attr_abs_acc.scatter_add_(0, tokens.flatten(), attr_abs)
                square_attr_acc.scatter_add_(0, tokens.flatten(), attr_squared)
            else:
                # Per-component: sum grad*act*ci over batch and sequence
                attr_acc.add_(ci_weighted_attr.sum(dim=(0, 1)))
                attr_abs_acc.add_(ci_weighted_attr_abs.sum(dim=(0, 1)))
                square_attr_acc.add_(ci_weighted_squared_attr.sum(dim=(0, 1)))
