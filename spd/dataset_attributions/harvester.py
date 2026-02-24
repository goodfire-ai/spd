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
from spd.dataset_attributions.storage import DatasetAttributionStorage
from spd.models.component_model import ComponentModel, OutputWithCache
from spd.models.components import make_mask_infos
from spd.topology import TransformerTopology
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

        sources_by_regular_target = {k: v.copy() for k, v in self.sources_by_target.items()}

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

        self._logit_sq_sum = torch.zeros(self.vocab_size, device=self.device)

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

        with torch.no_grad():
            for real_layer, ci_vals in ci.lower_leaky.items():
                self._ci_sum_accumulator[real_layer].add_(ci_vals.sum(dim=(0, 1)))
            self._logit_sq_sum.add_(comp_output.output.detach().square().sum(dim=(0, 1)))

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
            with torch.no_grad():
                for source_layer, act, grad in zip(source_layers, source_acts, grads, strict=True):
                    if source_layer == self.embed_path:
                        token_attr = (grad * act).sum(dim=-1)  # (B S)
                        self._emb_unemb_attr_acc[d_idx].scatter_add_(
                            0, tokens.flatten(), token_attr.flatten()
                        )
                    else:
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

        source_layers = self.sources_by_target[target_layer]
        if not source_layers:
            return
        source_acts = [cache[f"{s}_post_detach"] for s in source_layers]

        for t_idx in torch.where(alive_targets)[0].tolist():
            grads_val = torch.autograd.grad(target_acts[t_idx], source_acts, retain_graph=True)
            self._accumulate_grads(
                grads_val,
                source_layers,
                source_acts,
                tokens,
                ci,
                target_layer,
                t_idx,
                self._embed_tgts_acc,
                self._regular_layers_acc,
            )
            del grads_val

            abs_scalar = target_acts_raw[:, :, t_idx].abs().sum()
            grads_abs = torch.autograd.grad(abs_scalar, source_acts, retain_graph=True)
            self._accumulate_grads(
                grads_abs,
                source_layers,
                source_acts,
                tokens,
                ci,
                target_layer,
                t_idx,
                self._embed_tgts_acc_abs,
                self._regular_layers_acc_abs,
            )
            del grads_abs

    def _accumulate_grads(
        self,
        grads: tuple[Tensor, ...],
        source_layers: list[str],
        source_acts: list[Tensor],
        tokens: Int[Tensor, "batch seq"],
        ci: dict[str, Tensor],
        target_layer: str,
        t_idx: int,
        embed_acc: dict[str, Tensor],
        regular_acc: dict[str, dict[str, Tensor]],
    ) -> None:
        with torch.no_grad():
            for source_layer, act, grad in zip(source_layers, source_acts, grads, strict=True):
                if source_layer == self.embed_path:
                    token_attr = (grad * act).sum(dim=-1)  # (B S)
                    embed_acc[target_layer][t_idx].scatter_add_(
                        0, tokens.flatten(), token_attr.flatten()
                    )
                else:
                    ci_weighted = (grad * act * ci[source_layer]).sum(dim=(0, 1))  # (C,)
                    regular_acc[target_layer][source_layer][t_idx].add_(ci_weighted)

    def finalize(
        self, topology: TransformerTopology, ci_threshold: float
    ) -> DatasetAttributionStorage:
        """Package raw accumulators into storage. No normalization — that happens at query time."""
        assert self.n_tokens > 0, "No batches processed"

        to_canon = topology.target_to_canon

        def _canon_nested(acc: dict[str, dict[str, Tensor]]) -> dict[str, dict[str, Tensor]]:
            return {
                to_canon(t): {to_canon(s): v for s, v in srcs.items()} for t, srcs in acc.items()
            }

        def _canon(acc: dict[str, Tensor]) -> dict[str, Tensor]:
            return {to_canon(k): v for k, v in acc.items()}

        return DatasetAttributionStorage(
            regular_attr=_canon_nested(self._regular_layers_acc),
            regular_attr_abs=_canon_nested(self._regular_layers_acc_abs),
            embed_attr=_canon(self._embed_tgts_acc),
            embed_attr_abs=_canon(self._embed_tgts_acc_abs),
            unembed_attr=_canon(self._unembed_srcs_acc),
            embed_unembed_attr=self._emb_unemb_attr_acc,
            w_unembed=topology.get_unembed_weight(),
            ci_sum=_canon(self._ci_sum_accumulator),
            component_act_sq_sum=_canon(self._square_component_act_accumulator),
            logit_sq_sum=self._logit_sq_sum,
            vocab_size=self.vocab_size,
            ci_threshold=ci_threshold,
            n_batches_processed=self.n_batches,
            n_tokens_processed=self.n_tokens,
        )
