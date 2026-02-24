"""Core attribution harvester for computing dataset-aggregated attributions.

Computes component-to-component attribution strengths aggregated over the full
training dataset using gradient x activation formula, summed over all positions
and batches.

Three metrics are accumulated:
- attr:         E[∂y/∂x · x]           (signed mean attribution)
- attr_abs:     E[∂|y|/∂x · x]         (attribution to absolute value of target)

Output (pseudo-) component attributions are handled differently: We accumulate attributions
to the output residual stream, then later project this into token space.

All layer keys are concrete module paths (e.g. "wte", "h.0.attn.q_proj", "lm_head").
Translation to canonical names happens at the storage boundary in harvest.py.
"""

from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from spd.configs import SamplingType
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.models.components import make_mask_infos
from spd.utils.general_utils import bf16_autocast


class AttributionAccumulator:
    def __init__(
        self,
        regular_layers: dict[str, int],
        sources_by_target: dict[str, list[str]],
        component_alive: dict[str, Bool[Tensor, " n_components"]],
        unembed_path: str,
        unembed_module: nn.Linear,
        embed_path: str,
        embedding_module: nn.Embedding,
        device: torch.device,
    ):
        self._regular_layers = regular_layers
        self._embedding_module = embedding_module


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

        #     self._attr_val_accumulator = self._build_attr_accumulator(sources_by_target)
        #     self._attr_abs_accumulator = self._build_attr_accumulator(sources_by_target)

        #     self._square_act_accumulator = {
        #         layer: torch.zeros(c, device=device) for layer, c in self.model.module_to_c.items()
        #     }

        #     self._ci_sum_accumulator = {
        #         layer: torch.zeros(c, device=device) for layer, c in self.model.module_to_c.items()
        #     }

        # def _build_attr_accumulator(
        #     self,
        #     sources_by_target: dict[str, list[str]],
        # ) -> dict[str, dict[str, Tensor]]:
        #     accumulator: dict[str, dict[str, Tensor]] = {}

        #     for target_layer, source_layers in sources_by_target.items():
        #         if target_layer == self.unembed_path:
        #             target_d = self.unembed_module.in_features
        #         else:
        #             target_d = self.model.module_to_c[target_layer]

        #         source_acc: dict[str, Tensor] = {}
        #         for source_layer in source_layers:
        #             if source_layer == self.embed_path:
        #                 source_d = self.embedding_module.num_embeddings
        #             else:
        #                 source_d = self.model.module_to_c[source_layer]

        #             source_acc[source_layer] = torch.zeros((target_d, source_d), device=self.device)

        #         accumulator[target_layer] = source_acc

        #     return accumulator

        sources_by_regular_target = self.sources_by_target.copy()

        unembed_sources = sources_by_regular_target[self.unembed_path].copy()
        del sources_by_regular_target[self.unembed_path]

        # we store attributions to the embedding *output*
        embed_tgts_acc: dict[str, Tensor] = {}
        n_emb = self.embedding_module.num_embeddings
        for target, sources in sources_by_regular_target.items():
            if self.embed_path in sources:
                tgt_c = self.model.module_to_c[target]
                embed_tgts_acc[target] = torch.zeros((tgt_c, n_emb), device=self.device)
                continue
            sources.remove(self.embed_path)

        # we use d_model here because we store attributions to the pre-unembed residual
        unembed_srcs_acc: dict[str, Tensor] = {}
        d_model = self.unembed_module.in_features
        for source in unembed_sources:
            src_c = self.model.module_to_c[source]
            unembed_srcs_acc[source] = torch.zeros((d_model, src_c), device=self.device)

        # for normal components, we just go C <-> C
        acc: dict[str, dict[str, Tensor]] = {}
        for target_layer, source_layers in sources_by_regular_target.items():
            acc[target_layer] = {}
            for source_layer in source_layers:
                tgt_c = self.model.module_to_c[target_layer]
                src_c = self.model.module_to_c[source_layer]
                acc[target_layer][source_layer] = torch.zeros(
                    (tgt_c, src_c), device=self.device
                )

        self._acc = acc
        self._embed_tgts_acc = embed_tgts_acc
        self._unembed_srcs_acc = unembed_srcs_acc

    # def add_embed_attr_(
    #     self,
    #     target_layer: str,
    #     target_idx: int,
    #     tokens: Int[Tensor, "batch seq"],
    #     ci_weighted_attr_val: Float[Tensor, " c"],
    # ) -> None:

    def add_unembed_attr_(
        self,
        target_idx: int,
        source_layer: str,
        ci_weighted_attr_val: Float[Tensor, " c"],
    ) -> None:
        self._unembed_targets[target_layer][target_idx].add_(ci_weighted_attr_val)
        # if source_layer == self.embed_path:
        #     # Per-token: sum grad*act*ci over d_model, scatter by token id
        #     attr_val = ci_weighted_attr_val.sum(dim=-1).flatten()

        #     attr_acc.scatter_add_(0, tokens.flatten(), attr_val)
        # else:
        #     # Per-component: sum grad*act*ci over batch and sequence
        #     attr_acc.add_(ci_weighted_attr_val.sum(dim=(0, 1)))

    @dataclass
    class NormalizedAttrs:
        attr: dict[str, dict[str, Float[Tensor, "c_target c_source"]]]
        attr_abs: dict[str, dict[str, Float[Tensor, "c_target c_source"]]]

    def normalized_attrs(self) -> NormalizedAttrs:
        """Return the accumulated attributions normalized by n_tokens.

        mean_squared_attr is pre-sqrt so it can be merged across workers.
        """
        normed_attr_val = defaultdict[str, dict[str, Float[Tensor, "c_target c_source"]]](dict)
        normed_attr_abs = defaultdict[str, dict[str, Float[Tensor, "c_target c_source"]]](dict)

        for target in self._attr_val_accumulator:
            mean_squared_act = self._square_act_accumulator[target] / self.n_tokens
            mean_target_act_l2 = mean_squared_act.sqrt()  # (C_target,)

            for source in self.sources_by_target[target]:
                mean_attr_val = self._attr_val_accumulator[target][source]  # (C_target, C_source)
                mean_attr_abs = self._attr_abs_accumulator[target][source]  # (C_target, C_source)

                source_ci_sum = (
                    self._ci_sum_accumulator[source] if source != self.embed_path else 1.0
                )  # (C_source,)

                ci_weighted_mean_attr_val = mean_attr_val / source_ci_sum  # (C_target, C_source)
                ci_weighted_mean_attr_abs = mean_attr_abs / source_ci_sum  # (C_target, C_source)

                normed_attr_val[target][source] = (
                    ci_weighted_mean_attr_val / mean_target_act_l2[..., None]
                )
                normed_attr_abs[target][source] = (
                    ci_weighted_mean_attr_abs / mean_target_act_l2[..., None]
                )

        return self.NormalizedAttrs(
            attr=normed_attr_val,
            attr_abs=normed_attr_abs,
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

        for real_layer, ci_vals in ci.lower_leaky.items():
            self._ci_sum_accumulator[real_layer].add_(ci_vals.sum(dim=(0, 1)))

        for target_layer in self.sources_by_target:
            # I think this will error because there's no output components hook, in fact, there are no
            # output components
            target_acts_raw = cache[f"{target_layer}_post_detach"]
            self._square_act_accumulator[target_layer].add_(
                target_acts_raw.square().sum(dim=(0, 1))
            )

            if target_layer == self.unembed_path:
                self._process_output_targets(cache, tokens, ci.lower_leaky)
            else:
                self._process_component_targets(cache, tokens, ci.lower_leaky, target_layer)

    def _process_output_targets(
        self,
        cache: dict[str, Tensor],
        tokens: Int[Tensor, "batch seq"],
        ci: dict[str, Tensor],
    ) -> None:
        """Process output attributions via output-residual-space storage."""
        out_residual = cache[f"{self.unembed_path}_pre_detach"]

        out_residual_sum = out_residual.sum(dim=(0, 1))

        source_layers = self.sources_by_target[self.unembed_path]
        source_acts = [cache[f"{s}_post_detach"] for s in source_layers]

        for d_idx in range(self.output_d_model):
            grads = torch.autograd.grad(out_residual_sum[d_idx], source_acts, retain_graph=True)

            self._accumulate_attributions(
                attr_accumulator=self._attr_val_accumulator[self.unembed_path],
                target_idx=d_idx,
                source_layers=source_layers,
                source_acts=source_acts,
                source_grads=list(grads),
                ci=ci,
                tokens=tokens,
            )

    def _process_component_targets(
        self,
        cache: dict[str, Tensor],
        tokens: Int[Tensor, "batch seq"],
        ci: dict[str, Tensor],
        target_layer: str,
    ) -> None:
        """Process attributions to a component layer."""
        alive_targets = self.component_alive[target_layer]
        if not alive_targets.any():
            return

        target_acts_raw = cache[f"{target_layer}_pre_detach"]

        target_acts = target_acts_raw.sum(dim=(0, 1))
        target_acts_abs = target_acts_raw.abs().sum(dim=(0, 1))

        source_layers = self.sources_by_target[target_layer]
        source_acts = [cache[f"{s}_post_detach"] for s in source_layers]

        for t_idx in torch.where(alive_targets)[0].tolist():
            attr_for_target = partial(
                self._accumulate_attributions,
                target_idx=t_idx,
                source_layers=source_layers,
                source_acts=source_acts,
                ci=ci,
                tokens=tokens,
            )

            val_grads = torch.autograd.grad(target_acts[t_idx], source_acts, retain_graph=True)
            attr_for_target(
                attr_accumulator=self._attr_val_accumulator[target_layer],
                source_grads=list(val_grads),
            )

            abs_grads = torch.autograd.grad(target_acts_abs[t_idx], source_acts, retain_graph=True)
            attr_for_target(
                attr_accumulator=self._attr_abs_accumulator[target_layer],
                source_grads=list(abs_grads),
            )

    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
    def _accumulate_attributions(
        self,
        attr_accumulator: dict[str, Float[Tensor, "target_c source_c"]],
        target_idx: int,
        source_layers: list[str],
        source_acts: list[Float[Tensor, "batch seq c"]],
        source_grads: list[Float[Tensor, "batch seq c"]],
        ci: dict[str, Float[Tensor, "batch seq c"]],
        tokens: Int[Tensor, "batch seq"],
    ) -> None:
        """Accumulate grad*act attributions from sources to a target column."""
        for source_layer, act, grad in zip(source_layers, source_acts, source_grads, strict=True):
            attr_acc = attr_accumulator[source_layer][target_idx]  # (C_source,)

            # Embed has no CI (all tokens always active)
            source_ci = ci[source_layer] if source_layer != self.embed_path else 1.0

            ci_weighted_attr_val = grad * act * source_ci  # (B S C)

            if source_layer == self.embed_path:
                # Per-token: sum grad*act*ci over d_model, scatter by token id
                attr_val = ci_weighted_attr_val.sum(dim=-1).flatten()

                attr_acc.scatter_add_(0, tokens.flatten(), attr_val)
            else:
                # Per-component: sum grad*act*ci over batch and sequence
                attr_acc.add_(ci_weighted_attr_val.sum(dim=(0, 1)))


# TODO symbolic reified in / out handling
