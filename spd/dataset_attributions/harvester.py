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

        sources_by_regular_target = self.sources_by_target.copy()

        unembed_sources = sources_by_regular_target[self.unembed_path].copy()
        del sources_by_regular_target[self.unembed_path]
        unembed_sources.remove(self.embed_path)

        self._emb_unemb_attr_acc = torch.zeros(
            (self.unembed_module.in_features, self.embedding_module.num_embeddings),
            device=self.device,
        )

        # we store attributions to the embedding *output*
        embed_tgts_acc: dict[str, Tensor] = {}
        embed_tgts_acc_abs: dict[str, Tensor] = {}
        n_emb = self.embedding_module.num_embeddings
        for target, sources in sources_by_regular_target.items():
            if self.embed_path in sources:
                tgt_c = self.model.module_to_c[target]
                embed_tgts_acc[target] = torch.zeros((tgt_c, n_emb), device=self.device)
                embed_tgts_acc_abs[target] = torch.zeros((tgt_c, n_emb), device=self.device)
                sources.remove(self.embed_path)

        # we use d_model here because we store attributions to the pre-unembed residual
        # no abs version here because output is always positive
        unembed_srcs_acc: dict[str, Tensor] = {}
        d_model = self.unembed_module.in_features
        for source in unembed_sources:
            src_c = self.model.module_to_c[source]
            unembed_srcs_acc[source] = torch.zeros((d_model, src_c), device=self.device)

        # for normal components, we just go C <-> C
        acc: dict[str, dict[str, Tensor]] = {}
        acc_abs: dict[str, dict[str, Tensor]] = {}
        for target_layer, source_layers in sources_by_regular_target.items():
            acc[target_layer] = {}
            acc_abs[target_layer] = {}
            for source_layer in source_layers:
                tgt_c = self.model.module_to_c[target_layer]
                src_c = self.model.module_to_c[source_layer]
                acc[target_layer][source_layer] = torch.zeros((tgt_c, src_c), device=self.device)
                acc_abs[target_layer][source_layer] = torch.zeros(
                    (tgt_c, src_c), device=self.device
                )

        self._embed_tgts_acc = embed_tgts_acc
        self._embed_tgts_acc_abs = embed_tgts_acc_abs

        self._regular_layers_acc = acc
        self._regular_layers_acc_abs = acc_abs

        self._unembed_srcs_acc = unembed_srcs_acc

        self._ci_sum_accumulator = {
            layer: torch.zeros((c), device=self.device)
            for layer, c in self.model.module_to_c.items()
        }

        self._square_component_act_accumulator = {
            layer: torch.zeros((c), device=self.device)
            for layer, c in self.model.module_to_c.items()
        }

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
            sum_ci = ci_vals.sum(dim=(0, 1))
            self._ci_sum_accumulator[real_layer].add_(sum_ci)

        for target_layer in self.sources_by_target:
            if target_layer == self.unembed_path:
                self._process_output_targets(cache, tokens, ci.lower_leaky)
            else:
                sum_sq_acts = cache[f"{target_layer}_post_detach"].square().sum(dim=(0, 1))
                self._square_component_act_accumulator[target_layer].add_(sum_sq_acts)
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
        assert self.embed_path in source_layers, "remove me when passed"

        source_acts = [cache[f"{s}_post_detach"] for s in source_layers]

        for d_idx in range(self.output_d_model):
            grads = torch.autograd.grad(out_residual_sum[d_idx], source_acts, retain_graph=True)
            for source_layer, act, grad in zip(source_layers, source_acts, grads, strict=True):
                if source_layer == self.embed_path:
                    # token attribution is just grad * act
                    # because act is just the embedding
                    token_attr = (grad * act).sum(dim=-1)  # (B S)
                    self._emb_unemb_attr_acc[d_idx].scatter_add_(
                        0, tokens.flatten(), token_attr.flatten()
                    )
                else:
                    # Per-component: sum grad*act*ci over batch and sequence
                    ci_weighted_attr = (grad * act * ci[source_layer]).sum(dim=(0, 1))
                    self._unembed_srcs_acc[source_layer][d_idx].add_(ci_weighted_attr)

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
            grads_val = torch.autograd.grad(target_acts[t_idx], source_acts, retain_graph=True)
            grads_abs = torch.autograd.grad(target_acts_abs[t_idx], source_acts, retain_graph=True)

            for source_layer, act, grad_val, grad_abs in zip(
                source_layers, source_acts, grads_val, grads_abs, strict=True
            ):
                if source_layer == self.embed_path:
                    # token attribution is just grad * act
                    # because act is just the embedding
                    tok_embeddings = act

                    token_attr = (grad_val * tok_embeddings).sum(dim=-1)  # (B S)
                    token_attr_abs = (grad_abs * tok_embeddings).sum(dim=-1)  # (B S)

                    acc = self._embed_tgts_acc[source_layer][t_idx]
                    acc_abs = self._embed_tgts_acc_abs[source_layer][t_idx]

                    acc.scatter_add_(0, tokens.flatten(), token_attr.flatten())
                    acc_abs.scatter_add_(0, tokens.flatten(), token_attr_abs.flatten())
                else:
                    ci_weighted_attr_val = grad_val * act * ci[source_layer]  # (B S C)
                    ci_weighted_attr_abs = grad_abs * act * ci[source_layer]  # (B S C)

                    ci_weighted_attr_abs_sum = ci_weighted_attr_abs.sum(dim=(0, 1))  # (C,)
                    ci_weighted_attr_val_sum = ci_weighted_attr_val.sum(dim=(0, 1))  # (C,)

                    attr_acc = self._regular_layers_acc[target_layer][source_layer][t_idx]
                    attr_acc_abs = self._regular_layers_acc_abs[target_layer][source_layer][t_idx]

                    attr_acc.add_(ci_weighted_attr_val_sum)
                    attr_acc_abs.add_(ci_weighted_attr_abs_sum)

    # def normalized_attrs(self) -> NormalizedAttrs:
    #     """Return the accumulated attributions normalized by n_tokens.

    #     mean_squared_attr is pre-sqrt so it can be merged across workers.
    #     """
    #     normed_attr_val = defaultdict[str, dict[str, Float[Tensor, "c_target c_source"]]](dict)
    #     normed_attr_abs = defaultdict[str, dict[str, Float[Tensor, "c_target c_source"]]](dict)

    #     for target in self._attr_val_accumulator:
    #         mean_squared_act = self._square_act_accumulator[target] / self.n_tokens
    #         mean_target_act_l2 = mean_squared_act.sqrt()  # (C_target,)

    #         for source in self.sources_by_target[target]:
    #             mean_attr_val = self._attr_val_accumulator[target][source]  # (C_target, C_source)
    #             mean_attr_abs = self._attr_abs_accumulator[target][source]  # (C_target, C_source)

    #             source_ci_sum = (
    #                 self._ci_sum_accumulator[source] if source != self.embed_path else 1.0
    #             )  # (C_source,)

    #             ci_weighted_mean_attr_val = mean_attr_val / source_ci_sum  # (C_target, C_source)
    #             ci_weighted_mean_attr_abs = mean_attr_abs / source_ci_sum  # (C_target, C_source)

    #             normed_attr_val[target][source] = (
    #                 ci_weighted_mean_attr_val / mean_target_act_l2[..., None]
    #             )
    #             normed_attr_abs[target][source] = (
    #                 ci_weighted_mean_attr_abs / mean_target_act_l2[..., None]
    #             )

    #     return self.NormalizedAttrs(
    #         attr=normed_attr_val,
    #         attr_abs=normed_attr_abs,
    #     )
