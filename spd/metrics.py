"""Metrics and figures for SPD experiments.

This file contains metrics and visualizations that can be logged during SPD optimization.
These can be selected and configured in the Config.
"""

from collections import defaultdict
from collections.abc import Mapping
from fnmatch import fnmatch
from typing import Any, override

import einops
import torch
import torch.nn.functional as F
import wandb
from einops import reduce
from jaxtyping import Float, Int
from PIL import Image
from torch import Tensor
from torchmetrics import Metric

from spd.configs import Config
from spd.mask_info import make_mask_infos
from spd.models.component_model import ComponentModel
from spd.models.components import ComponentsOrModule
from spd.plotting import (
    get_single_feature_causal_importances,
    plot_causal_importance_vals,
    plot_ci_values_histograms,
    plot_component_activation_density,
    plot_mean_component_cis_both_scales,
    plot_UV_matrices,
)
from spd.utils.component_utils import (
    StochasticMasks,
    calc_ci_l_zero,
    calc_stochastic_masks,
)
from spd.utils.general_utils import calc_kl_divergence_lm
from spd.utils.target_ci_solutions import compute_target_metrics, make_target_ci_solution


def calc_weight_deltas(
    model: ComponentModel, device: str | torch.device
) -> dict[str, Float[Tensor, " d_out d_in"]]:
    """Calculate the weight differences between the target model and component weights (V@U) for
    each layer."""
    weight_deltas: dict[str, Float[Tensor, " d_out d_in"]] = {}
    for comp_name, components_or_module in model.components_or_modules.items():
        assert isinstance(components_or_module, ComponentsOrModule)
        if components_or_module.components is not None:
            weight_deltas[comp_name] = (
                components_or_module.original_weight - components_or_module.components.weight
            )
        if components_or_module.identity_components is not None:
            id_name = f"identity_{comp_name}"
            id_mat = components_or_module.identity_components.weight
            weight_deltas[id_name] = (
                torch.eye(id_mat.shape[0], device=device, dtype=id_mat.dtype) - id_mat
            )
    return weight_deltas


def calc_masked_recon_layerwise_loss(
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    mask_infos_list: list[dict[str, Any]],
    target_out: Float[Tensor, "... vocab"],
    loss_type: str,
    device: str,
) -> Float[Tensor, ""]:
    """Calculate layerwise reconstruction loss for masked model outputs."""
    total_loss = torch.tensor(0.0, device=device)
    for mask_infos in mask_infos_list:
        out = model(batch, mode="components", mask_infos=mask_infos)
        if loss_type == "mse":
            loss = ((out - target_out) ** 2).sum()
        else:
            loss = calc_kl_divergence_lm(pred=out, target=target_out, reduce=False).sum()
        total_loss += loss
    return total_loss / len(mask_infos_list)


def calc_masked_recon_loss(
    model: ComponentModel,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    mask_infos_list: list[dict[str, Any]],
    target_out: Float[Tensor, "... vocab"],
    loss_type: str,
    device: str,
) -> Float[Tensor, ""]:
    """Calculate reconstruction loss for masked model outputs."""
    total_loss = torch.tensor(0.0, device=device)
    for mask_infos in mask_infos_list:
        out = model(batch, mode="components", mask_infos=mask_infos)
        if loss_type == "mse":
            loss = ((out - target_out) ** 2).sum()
        else:
            loss = calc_kl_divergence_lm(pred=out, target=target_out, reduce=False).sum()
        total_loss += loss
    return total_loss / len(mask_infos_list)


class CI_L0(Metric):
    slow = False
    is_differentiable: bool | None = False

    def __init__(
        self,
        model: ComponentModel,
        config: Config,
        groups: dict[str, list[str]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.l0_threshold = config.ci_alive_threshold
        self.groups = groups  # Optional: {"layer_0": ["model.layers.0.*"], ...}

        # Use a regular attribute for per-layer running lists
        self.l0s: dict[str, list[float]] = defaultdict(list)

    @override
    def update(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
        **kwargs: Any,
    ) -> None:
        import re

        # Track group sums for this batch
        group_sums = defaultdict(float) if self.groups else {}

        for layer_name, layer_ci in ci.items():
            l0_val = calc_ci_l_zero(layer_ci, self.l0_threshold)
            self.l0s[layer_name].append(l0_val)

            # Accumulate into matching groups
            if self.groups:
                for group_name, patterns in self.groups.items():
                    for pattern in patterns:
                        if re.match(pattern.replace("*", ".*"), layer_name):
                            group_sums[group_name] += l0_val
                            break

        # Append group sums to their lists
        for group_name, group_sum in group_sums.items():
            self.l0s[group_name].append(group_sum)

    @override
    def compute(self) -> Mapping[str, float | Any]:
        out = {}
        table_data = []
        for name, l0s in self.l0s.items():
            avg_l0 = sum(l0s) / len(l0s)
            out[f"l0_{self.l0_threshold}/{name}"] = avg_l0
            table_data.append((name, avg_l0))

        bar_chart = wandb.plot.bar(
            table=wandb.Table(columns=["layer", "l0"], data=table_data),
            label="layer",
            value="l0",
            title=f"L0_{self.l0_threshold}",
        )
        out["l0_bar_chart"] = bar_chart
        return out

    @override
    def reset(self) -> None:
        super().reset()
        self.l0s = defaultdict(list)


class CEandKLLosses(Metric):
    slow = False
    is_differentiable: bool | None = False

    def __init__(
        self, model: ComponentModel, config: Config, rounding_threshold: float, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.config = config
        self.rounding_threshold = rounding_threshold

        # Use a regular attribute for per-key running lists
        self.ce_losses: dict[str, list[float]] = defaultdict(list)

    @override
    def update(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
        **kwargs: Any,
    ) -> None:
        ce_losses = self._calc_ce_and_kl_losses(batch, target_out, ci)
        for key, value in ce_losses.items():
            self.ce_losses[key].append(value)

    def _calc_ce_and_kl_losses(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
    ) -> Mapping[str, float]:
        assert batch.ndim == 2, "Batch must be 2D (batch, seq_len)"

        # make sure labels don't "wrap around": you **can't** predict the first token.
        masked_batch = batch.clone()
        masked_batch[:, 0] = -100
        flat_masked_batch = masked_batch.flatten()

        def ce_vs_labels(logits: Tensor) -> float:
            flat_logits = einops.rearrange(logits, "b seq_len vocab -> (b seq_len) vocab")
            return F.cross_entropy(
                flat_logits[:-1], flat_masked_batch[1:], ignore_index=-100
            ).item()

        def kl_vs_target(logits: Tensor) -> float:
            return calc_kl_divergence_lm(pred=logits, target=target_out).item()

        # CE When...
        # we use the causal importances as a mask
        ci_mask_infos = make_mask_infos(ci)
        ci_masked_logits = self.model(batch, mode="components", mask_infos=ci_mask_infos)
        ci_masked_ce_loss = ce_vs_labels(ci_masked_logits)
        ci_masked_kl_loss = kl_vs_target(ci_masked_logits)

        # we use the regular stochastic masks
        stoch_masks = [
            m.component_masks
            for m in calc_stochastic_masks(ci, n_mask_samples=1, sampling=self.config.sampling)
        ][0]
        stoch_masked_logits = self.model(
            batch, mode="components", mask_infos=make_mask_infos(stoch_masks)
        )
        stoch_masked_ce_loss = ce_vs_labels(stoch_masked_logits)
        stoch_masked_kl_loss = kl_vs_target(stoch_masked_logits)

        # we use all components
        nonmask = {k: torch.ones_like(v) for k, v in ci.items()}
        unmasked_logits = self.model(batch, mode="components", mask_infos=make_mask_infos(nonmask))
        unmasked_ce_loss = ce_vs_labels(unmasked_logits)
        unmasked_kl_loss = kl_vs_target(unmasked_logits)

        # we use completely random masks
        rand_masks = {layer: torch.rand_like(v) for layer, v in ci.items()}
        random_masked_logits = self.model(
            batch, mode="components", mask_infos=make_mask_infos(rand_masks)
        )
        random_masked_ce_loss = ce_vs_labels(random_masked_logits)
        random_masked_kl_loss = kl_vs_target(random_masked_logits)

        # we use rounded causal importances as masks
        rounded_ci = {k: (v > self.rounding_threshold).float() for k, v in ci.items()}
        rounded_masked_logits = self.model(
            batch, mode="components", mask_infos=make_mask_infos(rounded_ci)
        )
        rounded_masked_ce_loss = ce_vs_labels(rounded_masked_logits)
        rounded_masked_kl_loss = kl_vs_target(rounded_masked_logits)

        # we zero all the components
        zero_masks = {k: torch.zeros_like(v) for k, v in ci.items()}
        zero_masked_logits = self.model(
            batch, mode="components", mask_infos=make_mask_infos(zero_masks)
        )
        zero_masked_ce_loss = ce_vs_labels(zero_masked_logits)
        zero_masked_kl_loss = kl_vs_target(zero_masked_logits)

        target_model_ce_loss = ce_vs_labels(target_out)

        def pct_ce_unrecovered(ce: float) -> float:
            """pct of ce loss that is unrecovered, between zero masked and target model ce loss"""
            return (ce - target_model_ce_loss) / (zero_masked_ce_loss - target_model_ce_loss)

        def ce_difference(ce: float) -> float:
            """difference between ce loss and target model ce loss"""
            return ce - target_model_ce_loss

        return {
            "kl/ci_masked": ci_masked_kl_loss,
            "kl/unmasked": unmasked_kl_loss,
            "kl/stoch_masked": stoch_masked_kl_loss,
            "kl/random_masked": random_masked_kl_loss,
            "kl/rounded_masked": rounded_masked_kl_loss,
            "kl/zero_masked": zero_masked_kl_loss,
            "ce_difference/ci_masked": ce_difference(ci_masked_ce_loss),
            "ce_difference/unmasked": ce_difference(unmasked_ce_loss),
            "ce_difference/stoch_masked": ce_difference(stoch_masked_ce_loss),
            "ce_difference/random_masked": ce_difference(random_masked_ce_loss),
            "ce_difference/rounded_masked": ce_difference(rounded_masked_ce_loss),
            "ce_unrecovered/ci_masked": pct_ce_unrecovered(ci_masked_ce_loss),
            "ce_unrecovered/unmasked": pct_ce_unrecovered(unmasked_ce_loss),
            "ce_unrecovered/stoch_masked": pct_ce_unrecovered(stoch_masked_ce_loss),
            "ce_unrecovered/random_masked": pct_ce_unrecovered(random_masked_ce_loss),
            "ce_unrecovered/rounded_masked": pct_ce_unrecovered(rounded_masked_ce_loss),
        }

    @override
    def compute(self) -> Mapping[str, float]:
        return {k: sum(v) / len(v) for k, v in self.ce_losses.items()}

    @override
    def reset(self) -> None:
        super().reset()
        self.ce_losses = defaultdict(list)


class CIHistograms(Metric):
    slow = True
    is_differentiable: bool | None = False

    def __init__(
        self,
        model: ComponentModel,
        config: Config,
        n_batches_accum: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.n_batches_accum = n_batches_accum

        # Use regular attributes for accumulation to avoid type issues with non-Tensor states
        self.causal_importances: dict[str, list[Tensor]] = defaultdict(list)
        self.batches_seen: int = 0

    @override
    def update(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
        **kwargs: Any,
    ) -> None:
        self.batches_seen += 1
        if self.n_batches_accum is not None and self.batches_seen > self.n_batches_accum:
            return
        for k, v in ci.items():
            self.causal_importances[k].append(v.detach().cpu())

    @override
    def compute(self) -> Mapping[str, Image.Image]:
        combined_causal_importances = {k: torch.cat(v) for k, v in self.causal_importances.items()}
        fig = plot_ci_values_histograms(causal_importances=combined_causal_importances)
        return {"figures/causal_importance_values": fig}

    @override
    def reset(self) -> None:
        super().reset()
        self.causal_importances = defaultdict(list)
        self.batches_seen = 0


class ComponentActivationDensity(Metric):
    slow = True
    is_differentiable: bool | None = False

    def __init__(self, model: ComponentModel, config: Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.config = config
        self.model_device = next(iter(model.parameters())).device

        self.add_state("n_tokens", default=torch.tensor(0.0), dist_reduce_fx="sum")

        for module_name in model.components:
            self.add_state(
                f"component_activation_counts_{module_name}",
                default=torch.zeros(model.C, device=self.model_device),
                dist_reduce_fx="sum",
            )

    @override
    def update(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
        **kwargs: Any,
    ) -> None:
        n_tokens = next(iter(ci.values())).shape[:-1].numel()
        self.n_tokens += n_tokens

        for module_name, ci_vals in ci.items():
            active_components = ci_vals > self.config.ci_alive_threshold
            n_activations_per_component = reduce(active_components, "... C -> C", "sum")
            counts = getattr(self, f"component_activation_counts_{module_name}")
            counts += n_activations_per_component

    @override
    def compute(self) -> Mapping[str, Image.Image]:
        activation_densities = {}
        for module_name in self.model.components:
            counts = getattr(self, f"component_activation_counts_{module_name}")
            activation_densities[module_name] = counts / self.n_tokens

        fig = plot_component_activation_density(activation_densities)
        return {"figures/component_activation_density": fig}

    @override
    def reset(self) -> None:
        super().reset()


class PermutedCIPlots(Metric):
    slow = True
    is_differentiable: bool | None = False

    def __init__(
        self,
        model: ComponentModel,
        config: Config,
        identity_patterns: list[str] | None = None,
        dense_patterns: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.config = config
        self.model_device = next(iter(model.parameters())).device
        self.identity_patterns = identity_patterns
        self.dense_patterns = dense_patterns

        self.batch_shape: torch.Size | tuple[int, ...] | None = None

        assert config.task_config.task_name != "lm", (
            "PermutedCIPlots currently only works with models that take float inputs (not lms). "
        )

    @override
    def update(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
        **kwargs: Any,
    ) -> None:
        if self.batch_shape is None:
            self.batch_shape = batch.shape

    @override
    def compute(self) -> Mapping[str, Image.Image]:
        assert self.batch_shape is not None, "haven't seen any inputs yet"

        figures = plot_causal_importance_vals(
            model=self.model,
            batch_shape=self.batch_shape,
            device=self.model_device,
            input_magnitude=0.75,
            sigmoid_type=self.config.sigmoid_type,
            identity_patterns=self.identity_patterns,
            dense_patterns=self.dense_patterns,
            sampling=self.config.sampling,
        )[0]

        return {f"figures/{k}": v for k, v in figures.items()}

    @override
    def reset(self) -> None:
        super().reset()
        self.batch_shape = None


class UVPlots(Metric):
    slow = True
    is_differentiable: bool | None = False

    def __init__(
        self,
        model: ComponentModel,
        config: Config,
        identity_patterns: list[str] | None = None,
        dense_patterns: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.config = config
        self.model_device = next(iter(model.parameters())).device
        self.identity_patterns = identity_patterns
        self.dense_patterns = dense_patterns

        self.batch_shape: torch.Size | tuple[int, ...] | None = None

        assert config.task_config.task_name != "lm", (
            "UVPlots currently only works with models that take float inputs (not lms). "
        )

    @override
    def update(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
        **kwargs: Any,
    ) -> None:
        if self.batch_shape is None:
            self.batch_shape = batch.shape

    @override
    def compute(self) -> Mapping[str, Image.Image]:
        assert self.batch_shape is not None, "haven't seen any inputs yet"

        all_perm_indices = plot_causal_importance_vals(
            model=self.model,
            batch_shape=self.batch_shape,
            device=self.model_device,
            input_magnitude=0.75,
            sigmoid_type=self.config.sigmoid_type,
            identity_patterns=self.identity_patterns,
            dense_patterns=self.dense_patterns,
            sampling=self.config.sampling,
        )[1]

        uv_matrices = plot_UV_matrices(
            components=self.model.components, all_perm_indices=all_perm_indices
        )

        return {"figures/uv_matrices": uv_matrices}

    @override
    def reset(self) -> None:
        super().reset()
        self.batch_shape = None


class IdentityCIError(Metric):
    slow = True
    is_differentiable: bool | None = False

    def __init__(
        self,
        model: ComponentModel,
        config: Config,
        identity_ci: list[dict[str, str | int]] | None = None,
        dense_ci: list[dict[str, str | int]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.config = config
        self.model_device = next(iter(model.parameters())).device
        self.identity_ci = identity_ci
        self.dense_ci = dense_ci

        self.batch_shape: torch.Size | tuple[int, ...] | None = None

        assert config.task_config.task_name != "lm", (
            "IdentityCIError currently only works with models that take float inputs (not lms). "
        )

    @override
    def update(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
        **kwargs: Any,
    ) -> None:
        if self.batch_shape is None:
            self.batch_shape = batch.shape

    @override
    def compute(self) -> Mapping[str, float]:
        assert self.batch_shape is not None, "haven't seen any inputs yet"

        # Create target solution from config parameters
        target_solution = make_target_ci_solution(
            identity_ci=self.identity_ci,
            dense_ci=self.dense_ci,
        )

        if target_solution is None:
            return {}

        # Get causal importance arrays using single active features
        ci_arrays, _ = get_single_feature_causal_importances(
            model=self.model,
            batch_shape=self.batch_shape,
            device=self.model_device,
            input_magnitude=0.75,
            sampling=self.config.sampling,
            sigmoid_type=self.config.sigmoid_type,
        )

        target_metrics = compute_target_metrics(
            causal_importances=ci_arrays,
            target_solution=target_solution,
        )

        return target_metrics

    @override
    def reset(self) -> None:
        super().reset()
        self.batch_shape = None


class CIMeanPerComponent(Metric):
    slow = True
    is_differentiable: bool | None = False

    def __init__(self, model: ComponentModel, config: Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.config = config
        self.model_device = next(iter(model.parameters())).device

        for module_name in model.components:
            self.add_state(
                f"component_ci_sums_{module_name}",
                default=torch.zeros(model.C, device=self.model_device),
                dist_reduce_fx="sum",
            )
            self.add_state(
                f"samples_seen_{module_name}",
                default=torch.tensor(0),
                dist_reduce_fx="sum",
            )

    @override
    def update(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
        **kwargs: Any,
    ) -> None:
        for module_name, ci_vals in ci.items():
            n_batch_dims = ci_vals.ndim - 1
            batch_indices = tuple(range(n_batch_dims))
            batch_size = ci_vals.shape[:n_batch_dims].numel()

            samples_seen = getattr(self, f"samples_seen_{module_name}")
            samples_seen += batch_size

            ci_sums = getattr(self, f"component_ci_sums_{module_name}")
            ci_sums += ci_vals.sum(dim=batch_indices)

    @override
    def compute(self) -> Mapping[str, Image.Image]:
        """Calculate the mean CI per component across all ranks."""
        mean_component_cis = {}
        for module_name in self.model.components:
            ci_sums = getattr(self, f"component_ci_sums_{module_name}")
            samples_seen = getattr(self, f"samples_seen_{module_name}")
            mean_component_cis[module_name] = ci_sums / samples_seen

        img_linear, img_log = plot_mean_component_cis_both_scales(mean_component_cis)

        return {
            "figures/ci_mean_per_component": img_linear,
            "figures/ci_mean_per_component_log": img_log,
        }

    @override
    def reset(self) -> None:
        super().reset()


class SubsetReconstructionLoss(Metric):
    """Compute reconstruction loss for specific subsets of components."""

    slow = False
    is_differentiable: bool | None = False

    def __init__(
        self,
        model: ComponentModel,
        config: Config,
        include_patterns: dict[str, list[str]] | None = None,
        exclude_patterns: dict[str, list[str]] | None = None,
        use_all_ones_for_non_replaced: bool = False,
        n_mask_samples: int = 5,
        **kwargs: Any,
    ) -> None:
        """Initialize SubsetReconstructionLoss.

        Args:
            include_patterns: Dict mapping subset names to patterns for modules to REPLACE
                            e.g., {"layer_0_only": ["model.layers.0.*"]}
            exclude_patterns: Dict mapping subset names to patterns for modules to EXCLUDE from replacement
                            e.g., {"all_but_layer_0": ["model.layers.0.*"]}
            use_all_ones_for_non_replaced: If True, use all-ones mask for non-replaced modules
            n_mask_samples: Number of stochastic mask samples to average over
        """
        super().__init__(**kwargs)
        self.model = model
        self.config = config
        self.use_all_ones_for_non_replaced = use_all_ones_for_non_replaced
        self.n_mask_samples = n_mask_samples
        self.include_patterns = include_patterns or {}
        self.exclude_patterns = exclude_patterns or {}

        if not self.include_patterns and not self.exclude_patterns:
            raise ValueError(
                "At least one of include_patterns or exclude_patterns must be provided"
            )

        self.losses: dict[str, list[float]] = defaultdict(list)

    @override
    def update(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, " d_out d_in"]] | None = None,
        **kwargs: Any,
    ) -> None:
        if weight_deltas is None and self.config.use_delta_component:
            weight_deltas = calc_weight_deltas(self.model, device=target_out.device)
        elif not self.config.use_delta_component:
            weight_deltas = {}
        if weight_deltas is None:
            weight_deltas = {}

        losses = self._calc_subset_losses(batch, target_out, ci, weight_deltas)
        for key, value in losses.items():
            self.losses[key].append(value)

    def _get_masked_model_outputs(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        masks_list: list[StochasticMasks],
        weight_deltas: dict[str, Tensor],
        active: list[str],
        all_modules: list[str],
    ) -> list[Float[Tensor, "... vocab"]]:
        outputs: list[Float[Tensor, "... vocab"]] = []

        for masks in masks_list:
            stoch_masks = masks.component_masks
            weight_delta_masks = masks.weight_delta_masks
            masks_dict = {}
            for m in all_modules:
                if m in active:
                    masks_dict[m] = stoch_masks[m]
                elif self.use_all_ones_for_non_replaced:
                    masks_dict[m] = torch.ones_like(stoch_masks[m])

            if self.config.use_delta_component and weight_deltas:
                weight_deltas_and_masks = {}
                for m in all_modules:
                    if m in active:
                        weight_deltas_and_masks[m] = (weight_deltas[m], weight_delta_masks[m])
            else:
                weight_deltas_and_masks = None

            outputs.append(
                self.model(
                    batch,
                    mode="components",
                    mask_infos=make_mask_infos(masks_dict, weight_deltas_and_masks),
                )
            )

        return outputs

    def _calc_subset_losses(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Tensor],
    ) -> Mapping[str, float]:
        assert batch.ndim == 2, "Batch must be 2D (batch, seq_len)"

        # Setup CE calculation
        masked_batch = batch.clone()
        masked_batch[:, 0] = -100
        flat_masked_batch = masked_batch.flatten()

        def ce_vs_labels(logits: Tensor) -> float:
            flat_logits = einops.rearrange(logits, "b seq_len vocab -> (b seq_len) vocab")
            return F.cross_entropy(
                flat_logits[:-1], flat_masked_batch[1:], ignore_index=-100
            ).item()

        def kl_vs_target(logits: Tensor) -> float:
            return calc_kl_divergence_lm(pred=logits, target=target_out).item()

        # Compute baselines for CE unrecovered
        target_ce = ce_vs_labels(target_out)
        zero_masks = {k: torch.zeros_like(v) for k, v in ci.items()}
        zero_out = self.model(batch, mode="components", mask_infos=make_mask_infos(zero_masks))
        zero_ce = ce_vs_labels(zero_out)

        # Generate stochastic masks
        masks_list = calc_stochastic_masks(ci, self.n_mask_samples, self.config.sampling)

        results = {}
        all_modules = list(ci.keys())

        # Process include patterns
        for name, patterns in self.include_patterns.items():
            active = [m for m in all_modules if any(fnmatch(m, p) for p in patterns)]

            outputs = self._get_masked_model_outputs(
                batch=batch,
                masks_list=masks_list,
                weight_deltas=weight_deltas,
                active=active,
                all_modules=all_modules,
            )
            kl_losses = [kl_vs_target(out) for out in outputs]
            ce_losses = [ce_vs_labels(out) for out in outputs]

            mean_kl = sum(kl_losses) / len(kl_losses)
            mean_ce = sum(ce_losses) / len(ce_losses)
            ce_unrec = (mean_ce - target_ce) / (zero_ce - target_ce) if zero_ce != target_ce else 0

            suffix = "_all_ones" if self.use_all_ones_for_non_replaced else ""
            results[f"subset/{name}/kl{suffix}"] = mean_kl
            results[f"subset/{name}/ce{suffix}"] = mean_ce
            results[f"subset/{name}/ce_unrec{suffix}"] = ce_unrec

        # Process exclude patterns
        for name, exclude_patterns in self.exclude_patterns.items():
            active = [m for m in all_modules if not any(fnmatch(m, p) for p in exclude_patterns)]

            outputs = self._get_masked_model_outputs(
                batch=batch,
                masks_list=masks_list,
                weight_deltas=weight_deltas,
                active=active,
                all_modules=all_modules,
            )
            kl_losses = [kl_vs_target(out) for out in outputs]
            ce_losses = [ce_vs_labels(out) for out in outputs]

            mean_kl = sum(kl_losses) / len(kl_losses)
            mean_ce = sum(ce_losses) / len(ce_losses)
            ce_unrec = (mean_ce - target_ce) / (zero_ce - target_ce) if zero_ce != target_ce else 0

            suffix = "_all_ones" if self.use_all_ones_for_non_replaced else ""
            results[f"subset/{name}/kl{suffix}"] = mean_kl
            results[f"subset/{name}/ce{suffix}"] = mean_ce
            results[f"subset/{name}/ce_unrec{suffix}"] = ce_unrec

        return results

    @override
    def compute(self) -> Mapping[str, float]:
        # Compute averages for all metrics
        results = {k: sum(v) / len(v) for k, v in self.losses.items()}

        # Find worst (highest) metrics across all subsets
        metrics_by_type = {"kl": {}, "ce": {}, "ce_unrec": {}}

        for key, value in results.items():
            if not key.startswith("subset/"):
                continue

            parts = key.split("/")
            if len(parts) != 3:
                continue

            subset_name = parts[1]
            metric_type = parts[2]

            # Skip all_ones variants for worst tracking
            if metric_type.endswith("_all_ones"):
                continue

            # Group metrics by type
            if metric_type == "kl":
                metrics_by_type["kl"][subset_name] = value
            elif metric_type == "ce":
                metrics_by_type["ce"][subset_name] = value
            elif metric_type == "ce_unrec":
                metrics_by_type["ce_unrec"][subset_name] = value

        # Add worst metrics to results
        for metric_type, subset_values in metrics_by_type.items():
            if subset_values:
                worst_subset = max(subset_values, key=lambda k: subset_values[k])
                worst_value = subset_values[worst_subset]
                results[f"subset_worst/{metric_type}"] = worst_value
                results[f"subset_worst/{metric_type}_subset"] = worst_subset

        return results

    @override
    def reset(self) -> None:
        super().reset()
        self.losses = defaultdict(list)


# --- Loss metrics (per-term) ----------------------------------------------------------------- #


class FaithfulnessLoss(Metric):
    slow = False
    is_differentiable: bool | None = True

    sum_faithfulness: Float[Tensor, ""]
    total_params: int  # n_params in each batch summed over all batches

    def __init__(
        self,
        model: ComponentModel,
        _config: Config,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model

        self.add_state("sum_faithfulness", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_params", default=torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, " d_out d_in"]] | None = None,
        **kwargs: Any,
    ) -> None:
        if weight_deltas is None:
            weight_deltas = calc_weight_deltas(self.model, device=target_out.device)

        for delta in weight_deltas.values():
            self.sum_faithfulness += (delta**2).sum()
            self.total_params += delta.numel()

    @override
    def compute(self) -> Float[Tensor, ""]:
        assert self.total_params > 0, "No batches seen"
        return self.sum_faithfulness / self.total_params


class CIReconLayerwiseLoss(Metric):
    slow = False
    is_differentiable: bool | None = True

    sum_loss: Float[Tensor, ""]
    n_tokens: int

    def __init__(self, model: ComponentModel, config: Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.config = config

        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_tokens", default=torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
        **kwargs: Any,
    ) -> None:
        loss = calc_masked_recon_layerwise_loss(
            model=self.model,
            batch=batch,
            mask_infos_list=[make_mask_infos(ci)],
            target_out=target_out,
            loss_type=self.config.output_loss_type,
            device=str(target_out.device),
        )

        denom = (
            target_out.numel()
            if self.config.output_loss_type == "mse"
            else target_out.shape[:-1].numel()
        )

        self.sum_loss += loss
        self.n_tokens += denom

    @override
    def compute(self) -> Float[Tensor, ""]:
        if self.n_tokens == 0:
            return torch.tensor(0.0)
        return self.sum_loss / self.n_tokens


class StochasticReconLayerwiseLoss(Metric):
    slow = False
    is_differentiable: bool | None = True

    sum_loss: Float[Tensor, ""]
    n_tokens: int

    def __init__(
        self,
        model: ComponentModel,
        config: Config,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.config = config

        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_tokens", default=torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, " d_out d_in"]] | None = None,
        **kwargs: Any,
    ) -> None:
        if weight_deltas is None and self.config.use_delta_component:
            weight_deltas = calc_weight_deltas(self.model, device=target_out.device)

        stoch_masks_list = calc_stochastic_masks(
            causal_importances=ci,
            n_mask_samples=self.config.n_mask_samples,
            sampling=self.config.sampling,
        )

        mask_infos_list = []
        for stoch_masks in stoch_masks_list:
            deltas_and_masks = (
                {k: (weight_deltas[k], stoch_masks.weight_delta_masks[k]) for k in ci}
                if self.config.use_delta_component and weight_deltas is not None
                else None
            )
            mask_infos_list.append(
                make_mask_infos(
                    masks=stoch_masks.component_masks, weight_deltas_and_masks=deltas_and_masks
                )
            )

        loss = calc_masked_recon_layerwise_loss(
            model=self.model,
            batch=batch,
            mask_infos_list=mask_infos_list,
            target_out=target_out,
            loss_type=self.config.output_loss_type,
            device=str(target_out.device),
        )

        denom = (
            target_out.numel()
            if self.config.output_loss_type == "mse"
            else target_out.shape[:-1].numel()
        )

        self.sum_loss += loss
        self.n_tokens += denom

    @override
    def compute(self) -> Float[Tensor, ""]:
        if self.n_tokens == 0:
            return torch.tensor(0.0)
        return self.sum_loss / self.n_tokens


class CIReconLoss(Metric):
    slow = False
    is_differentiable: bool | None = True

    sum_loss: Float[Tensor, ""]
    n_tokens: int

    def __init__(self, model: ComponentModel, config: Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.config = config

        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_tokens", default=torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
        **kwargs: Any,
    ) -> None:
        loss = calc_masked_recon_loss(
            model=self.model,
            batch=batch,
            mask_infos_list=[make_mask_infos(ci)],
            target_out=target_out,
            loss_type=self.config.output_loss_type,
            device=str(target_out.device),
        )

        denom = (
            target_out.numel()
            if self.config.output_loss_type == "mse"
            else target_out.shape[:-1].numel()
        )

        self.sum_loss += loss
        self.n_tokens += denom

    @override
    def compute(self) -> Float[Tensor, ""]:
        if self.n_tokens == 0:
            return torch.tensor(0.0)
        return self.sum_loss / self.n_tokens


class ImportanceMinimalityLoss(Metric):
    slow = False
    is_differentiable: bool | None = True

    sum_imp_min: Float[Tensor, " C"]
    n_examples: int

    def __init__(
        self,
        *args: Any,
        pnorm: Any,  # Not yet cast to float
        eps: Any,  # Not yet cast to float
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pnorm = float(pnorm)
        self.eps = float(eps)

        self.add_state("sum_imp_min", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_examples", torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(self, ci_upper_leaky: dict[str, Float[Tensor, "... C"]], **kwargs: Any) -> None:
        """Calculate the summed importance minimality values on the upper leaky causal importances.

        Args:
            batch: Batch of data.
            target_out: Target model output.
            ci: Dictionary of causal importances for each layer.
            ci_upper_leaky: Dictionary of causal importances upper leaky relu for each layer.

        Returns:
            The importance minimality loss on the upper leaky relu causal importances.
        """

        for layer_ci_upper_leaky in ci_upper_leaky.values():
            # Note, the paper uses an absolute value but our layer_ci_upper_leaky is already > 0
            self.sum_imp_min += ((layer_ci_upper_leaky + self.eps) ** self.pnorm).sum()
            self.n_examples += layer_ci_upper_leaky.shape[:-1].numel()

    @override
    def compute(self) -> Float[Tensor, ""]:
        return self.sum_imp_min / self.n_examples


class StochasticReconLoss(Metric):
    slow = False
    is_differentiable: bool | None = True

    sum_stochastic_recon: Float[Tensor, ""]
    n_examples: int

    def __init__(self, model: ComponentModel, config: Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.config = config
        self.add_state("sum_stochastic_recon", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_examples", torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
        ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, " d_out d_in"]] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Calculate the stochastic recon loss before reduction."""
        if weight_deltas is None and self.config.use_delta_component:
            weight_deltas = calc_weight_deltas(self.model, device=target_out.device)
        elif not self.config.use_delta_component:
            weight_deltas = {}

        stochastic_masks_list = calc_stochastic_masks(
            causal_importances=ci,
            n_mask_samples=self.config.n_mask_samples,
            sampling=self.config.sampling,
        )
        mask_infos_list = []
        for stochastic_masks in stochastic_masks_list:
            deltas_and_masks = (
                {
                    key: (weight_deltas[key], stochastic_masks.weight_delta_masks[key])
                    for key in weight_deltas
                }
                if self.config.use_delta_component and weight_deltas
                else None
            )
            mask_infos_list.append(
                make_mask_infos(
                    masks=stochastic_masks.component_masks, weight_deltas_and_masks=deltas_and_masks
                )
            )
        for mask_infos in mask_infos_list:
            out = self.model(batch, mode="components", mask_infos=mask_infos)
            if self.config.output_loss_type == "mse":
                loss = ((out - target_out) ** 2).sum()
            else:
                loss = calc_kl_divergence_lm(pred=out, target=target_out, reduce=False).sum()
            self.n_examples += out.shape[:-1].numel()
            self.sum_stochastic_recon += loss

    @override
    def compute(self) -> Float[Tensor, ""]:
        return self.sum_stochastic_recon / self.n_examples


METRICS = {
    cls.__name__: cls
    for cls in [
        CI_L0,
        CEandKLLosses,
        CIHistograms,
        ComponentActivationDensity,
        PermutedCIPlots,
        UVPlots,
        IdentityCIError,
        CIMeanPerComponent,
        SubsetReconstructionLoss,
        FaithfulnessLoss,
        CIReconLoss,
        StochasticReconLoss,
        CIReconLayerwiseLoss,
        StochasticReconLayerwiseLoss,
        ImportanceMinimalityLoss,
    ]
}
