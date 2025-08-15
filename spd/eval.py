"""Metrics and figures for SPD experiments.

This file contains metrics and visualizations that can be logged during SPD optimization.
These can be selected and configured in the Config.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator, Mapping
from typing import Any, ClassVar, override

import einops
import torch
import torch.nn.functional as F
from einops import reduce
from jaxtyping import Float, Int
from PIL import Image
from torch import Tensor

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.plotting import (
    get_single_feature_causal_importances,
    plot_causal_importance_vals,
    plot_ci_values_histograms,
    plot_component_activation_density,
    plot_UV_matrices,
)
from spd.utils.component_utils import calc_ci_l_zero, calc_stochastic_masks
from spd.utils.general_utils import calc_kl_divergence_lm, extract_batch_data
from spd.utils.target_ci_solutions import compute_target_metrics, make_target_ci_solution


class StreamingEval(ABC):
    SLOW: ClassVar[bool]

    @abstractmethod
    def __init__(self, model: ComponentModel, config: Config, **kwargs: Any): ...

    @abstractmethod
    def watch_batch(
        self,
        batch: Tensor,
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
    ) -> None: ...

    @abstractmethod
    def compute(self) -> Mapping[str, float | Image.Image]: ...


class CI_L0(StreamingEval):
    SLOW = False

    def __init__(self, model: ComponentModel, config: Config):
        self.l0_threshold = config.ci_alive_threshold
        self.l0s = defaultdict[str, list[float]](list)

    @override
    def watch_batch(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
    ) -> None:
        for layer_name, layer_ci in ci.items():
            l0_val = calc_ci_l_zero(layer_ci, self.l0_threshold)
            self.l0s[layer_name].append(l0_val)

    @override
    def compute(self) -> Mapping[str, float]:
        out = {}
        for layer_name, l0s in self.l0s.items():
            out[f"l0_{self.l0_threshold}/{layer_name}"] = sum(l0s) / len(l0s)
        return out


class CEandKLLosses(StreamingEval):
    SLOW = False

    def __init__(self, model: ComponentModel, config: Config, rounding_threshold: float):
        self.model = model
        self.rounding_threshold = rounding_threshold
        self.ce_losses = defaultdict[str, list[float]](list)

    @override
    def watch_batch(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
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
        masked_batch[:, 0] = -100  # F.cross_entropy ignores -99
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
        ci_masked_logits = self.model(batch, mode="components", masks=ci)
        ci_masked_ce_loss = ce_vs_labels(ci_masked_logits)
        ci_masked_kl_loss = kl_vs_target(ci_masked_logits)

        # we use the regular stochastic masks
        stoch_masks = calc_stochastic_masks(ci, n_mask_samples=1)[0]
        stoch_masked_logits = self.model(batch, mode="components", masks=stoch_masks)
        stoch_masked_ce_loss = ce_vs_labels(stoch_masked_logits)
        stoch_masked_kl_loss = kl_vs_target(stoch_masked_logits)

        # we use all components
        nonmask = {k: torch.ones_like(v) for k, v in ci.items()}
        unmasked_logits = self.model(batch, mode="components", masks=nonmask)
        unmasked_ce_loss = ce_vs_labels(unmasked_logits)
        unmasked_kl_loss = kl_vs_target(unmasked_logits)

        # we use completely random masks
        rand_masks = {layer: torch.rand_like(v) for layer, v in ci.items()}
        random_masked_logits = self.model(batch, mode="components", masks=rand_masks)
        random_masked_ce_loss = ce_vs_labels(random_masked_logits)
        random_masked_kl_loss = kl_vs_target(random_masked_logits)

        # we use rounded causal importances as masks
        rounded_ci = {k: (v > self.rounding_threshold).float() for k, v in ci.items()}
        rounded_masked_logits = self.model(batch, mode="components", masks=rounded_ci)
        rounded_masked_ce_loss = ce_vs_labels(rounded_masked_logits)
        rounded_masked_kl_loss = kl_vs_target(rounded_masked_logits)

        # we zero all the components
        zero_masks = {k: torch.zeros_like(v) for k, v in ci.items()}
        zero_masked_logits = self.model(batch, mode="components", masks=zero_masks)
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
            # "ce/target": target_model_ce_loss,
            # "ce/ci_masked": ci_masked_ce_loss,
            # "ce/unmasked": unmasked_ce_loss,
            # "ce/stoch_masked": stoch_masked_ce_loss,
            # "ce/random_masked": random_masked_ce_loss,
            # "ce/rounded_masked": rounded_masked_ce_loss,
            # "ce/zero_masked": zero_masked_ce_loss,
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
            # no zero masked ce_unrecovered because it's tautologically 100%
        }

    @override
    def compute(self) -> Mapping[str, float]:
        return {k: sum(v) / len(v) for k, v in self.ce_losses.items()}


class CIHistograms(StreamingEval):
    SLOW = True

    def __init__(self, model: ComponentModel, config: Config, n_batches_accum: int | None = None):
        self.causal_importances = defaultdict[str, list[Float[Tensor, "... C"]]](list)
        self.n_batches_accum = n_batches_accum
        self.batches_seen = 0

    @override
    def watch_batch(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
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


class ComponentActivationDensity(StreamingEval):
    SLOW = True

    def __init__(self, model: ComponentModel, config: Config):
        self.model = model
        self.config = config
        self.device = next(iter(model.parameters())).device

        self.n_tokens = 0
        self.component_activation_counts: dict[str, Float[Tensor, " C"]] = {
            module_name: torch.zeros(model.C, device=self.device)
            for module_name in model.components
        }

    @override
    def watch_batch(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
    ) -> None:
        n_tokens = next(iter(ci.values())).shape[:-1].numel()
        self.n_tokens += n_tokens

        for module_name, ci_vals in ci.items():
            active_components = ci_vals > self.config.ci_alive_threshold
            n_activations_per_component = reduce(active_components, "... C -> C", "sum")
            self.component_activation_counts[module_name] += n_activations_per_component

    @override
    def compute(self) -> Mapping[str, Image.Image]:
        activation_densities = {
            module_name: self.component_activation_counts[module_name] / self.n_tokens
            for module_name in self.model.components
        }
        fig = plot_component_activation_density(activation_densities)
        return {"figures/component_activation_density": fig}


class PermutedCIPlots(StreamingEval):
    SLOW = True

    def __init__(
        self,
        model: ComponentModel,
        config: Config,
        identity_patterns: list[str] | None = None,
        dense_patterns: list[str] | None = None,
    ):
        self.model = model
        self.config = config
        self.device = next(iter(model.parameters())).device
        self.identity_patterns = identity_patterns
        self.dense_patterns = dense_patterns

        self.batch_shape = None
        assert config.task_config.task_name != "lm", (
            "PermutedCIPlots currently only works with models that take float inputs (not lms). "
        )

    @override
    def watch_batch(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
    ) -> None:
        if self.batch_shape is None:
            self.batch_shape = batch.shape

    @override
    def compute(self) -> Mapping[str, Image.Image]:
        assert self.batch_shape is not None, "haven't seen any inputs yet"

        figures = plot_causal_importance_vals(
            model=self.model,
            batch_shape=self.batch_shape,
            device=self.device,
            input_magnitude=0.75,
            sigmoid_type=self.config.sigmoid_type,
            identity_patterns=self.identity_patterns,
            dense_patterns=self.dense_patterns,
        )[0]

        return {f"figures/{k}": v for k, v in figures.items()}


class UVPlots(StreamingEval):
    SLOW = True

    def __init__(
        self,
        model: ComponentModel,
        config: Config,
        identity_patterns: list[str] | None = None,
        dense_patterns: list[str] | None = None,
    ):
        self.model = model
        self.config = config
        self.device = next(iter(model.parameters())).device
        self.identity_patterns = identity_patterns
        self.dense_patterns = dense_patterns

        self.batch_shape = None
        assert config.task_config.task_name != "lm", (
            "UVPlots currently only works with models that take float inputs (not lms). "
        )

    @override
    def watch_batch(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
    ) -> None:
        if self.batch_shape is None:
            self.batch_shape = batch.shape

    @override
    def compute(self) -> Mapping[str, Image.Image]:
        assert self.batch_shape is not None, "haven't seen any inputs yet"

        all_perm_indices = plot_causal_importance_vals(
            model=self.model,
            batch_shape=self.batch_shape,
            device=self.device,
            input_magnitude=0.75,
            sigmoid_type=self.config.sigmoid_type,
            identity_patterns=self.identity_patterns,
            dense_patterns=self.dense_patterns,
        )[1]

        uv_matrices = plot_UV_matrices(
            components=self.model.components, all_perm_indices=all_perm_indices
        )

        return {"figures/uv_matrices": uv_matrices}


class IdentityCIError(StreamingEval):
    SLOW = True

    def __init__(
        self,
        model: ComponentModel,
        config: Config,
        identity_ci: list[dict[str, str | int]] | None = None,
        dense_ci: list[dict[str, str | int]] | None = None,
    ):
        self.model = model
        self.config = config
        self.device = next(iter(model.parameters())).device
        self.identity_ci = identity_ci
        self.dense_ci = dense_ci

        self.batch_shape = None
        assert config.task_config.task_name != "lm", (
            "IdentityCIError currently only works with models that take float inputs (not lms). "
        )

    @override
    def watch_batch(
        self,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: dict[str, Float[Tensor, "... C"]],
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
            device=self.device,
            input_magnitude=0.75,
            sigmoid_type=self.config.sigmoid_type,
        )

        target_metrics = compute_target_metrics(
            causal_importances=ci_arrays,
            target_solution=target_solution,
        )

        return target_metrics


EVAL_CLASSES = {
    cls.__name__: cls
    for cls in [
        CI_L0,
        CEandKLLosses,
        CIHistograms,
        ComponentActivationDensity,
        PermutedCIPlots,
        UVPlots,
        IdentityCIError,
    ]
}


def evaluate(
    model: ComponentModel,
    eval_iterator: Iterator[Int[Tensor, "..."]]
    | Iterator[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    device: str | torch.device,
    config: Config,
    run_slow: bool,
    n_steps: int,
) -> dict[str, float | Image.Image]:
    evals: list[StreamingEval] = []
    for eval_config in config.eval_metrics:
        eval_cls = EVAL_CLASSES[eval_config.classname]
        if not run_slow and eval_cls.SLOW:
            continue
        evals.append(eval_cls(model, config, **eval_config.extra_init_kwargs))

    for _ in range(n_steps):
        # Do the common work:
        batch = extract_batch_data(next(eval_iterator))
        batch = batch.to(device)
        target_out, pre_weight_acts = model(
            batch, mode="pre_forward_cache", module_names=model.target_module_paths
        )
        ci, _ci_upper_leaky = model.calc_causal_importances(
            pre_weight_acts, sigmoid_type=config.sigmoid_type
        )

        for eval in evals:
            eval.watch_batch(batch=batch, target_out=target_out, ci=ci)

    out: dict[str, float | Image.Image] = {}
    all_dicts = [eval.compute() for eval in evals]
    for d in all_dicts:
        if set(d.keys()).intersection(out.keys()):
            raise ValueError(f"Keys {set(d.keys()).intersection(out.keys())} already in output")
        out.update(d)

    return out
