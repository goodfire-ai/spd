"""Figures for SPD experiments.

This file contains visualizations that can be logged during SPD optimization.
These can be selected and configured in the config file.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import override

import torch
from einops import reduce
from jaxtyping import Float, Int
from matplotlib import pyplot as plt
from torch import Tensor

from spd.configs import (
    CIHistogramsFigureConfig,
    Config,
    MeanComponentActivationCountsFigureConfig,
    UVandIdentityCIFigureConfig,
)
from spd.models.component_model import ComponentModel
from spd.models.sigmoids import SigmoidTypes
from spd.plotting import (
    plot_causal_importance_vals,
    plot_ci_histograms,
    plot_UV_matrices,
)
from spd.utils.general_utils import extract_batch_data


@dataclass
class FigureInput:
    ci: dict[str, Float[Tensor, "... C"]]
    batch: Int[Tensor, "..."] | Float[Tensor, "..."]


class StreamingFigureCreator(ABC):
    @abstractmethod
    def watch(self, inputs: FigureInput) -> None: ...

    @abstractmethod
    def compute(self) -> Mapping[str, plt.Figure]: ...


class CIHistograms(StreamingFigureCreator):
    def __init__(self):
        self.causal_importances = defaultdict[str, list[Float[Tensor, "... C"]]](list)

    @override
    def watch(self, inputs: FigureInput) -> None:
        for k, v in inputs.ci.items():
            self.causal_importances[k].append(v)

    @override
    def compute(self) -> Mapping[str, plt.Figure]:
        combined_causal_importances = {k: torch.cat(v) for k, v in self.causal_importances.items()}
        return plot_ci_histograms(causal_importances=combined_causal_importances)


class MeanComponentActivationCounts(StreamingFigureCreator):
    def __init__(self, model: ComponentModel, device: str | torch.device, threshold: float):
        self.model = model
        self.device = device
        self.threshold = threshold

        self.n_tokens = 0
        self.component_activation_counts: dict[str, Float[Tensor, " C"]] = {
            module_name: torch.zeros(model.C, device=device) for module_name in model.components
        }

    @override
    def watch(self, inputs: FigureInput) -> None:
        n_tokens = next(iter(inputs.ci.values())).shape[:-1].numel()
        self.n_tokens += n_tokens

        for module_name, ci_vals in inputs.ci.items():
            active_components = ci_vals > self.threshold
            n_activations_per_component = reduce(active_components, "... C -> C", "sum")
            self.component_activation_counts[module_name] += n_activations_per_component

    @staticmethod
    def _plot_mean_component_activation_counts(
        name: str,
        mean_component_activation_counts: Float[Tensor, " C"],
    ) -> plt.Figure:
        """Plots the mean activation counts for each component module, returning a dict of figures."""
        fig = plt.figure(figsize=(8, 6))
        plt.hist(mean_component_activation_counts.detach().cpu().numpy(), bins=100)
        plt.yscale("log")
        plt.title(name)
        plt.xlabel("Mean Activation Count")
        plt.ylabel("Frequency")

        plt.tight_layout()
        return fig

    @override
    def compute(self) -> Mapping[str, plt.Figure]:
        out = {}
        for module_name in self.model.components:
            mean_counts = self.component_activation_counts[module_name] / self.n_tokens
            out[f"{module_name}/mean_component_activation_counts"] = (
                self._plot_mean_component_activation_counts(
                    name=module_name,
                    mean_component_activation_counts=mean_counts,
                )
            )
        return out


class UVandIdentityCI(StreamingFigureCreator):
    def __init__(
        self,
        model: ComponentModel,
        device: str | torch.device,
        sigmoid_type: SigmoidTypes,
    ):
        self.model = model
        self.device = device
        self.sigmoid_type: SigmoidTypes = sigmoid_type
        self.batch_shape = None

    @override
    def watch(self, inputs: FigureInput) -> None:
        self.batch_shape = inputs.batch.shape

    @override
    def compute(self) -> Mapping[str, plt.Figure]:
        assert self.batch_shape is not None, "haven't seen any inputs yet"

        figures, all_perm_indices = plot_causal_importance_vals(
            model=self.model,
            batch_shape=self.batch_shape,
            device=self.device,
            input_magnitude=0.75,
            sigmoid_type=self.sigmoid_type,
        )

        uv_matrices = plot_UV_matrices(
            components=self.model.components, all_perm_indices=all_perm_indices
        )

        return {
            **figures,
            **uv_matrices,
        }


def create_figures(
    model: ComponentModel,
    eval_iterator: Iterator[Int[Tensor, "..."]]
    | Iterator[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    device: str | torch.device,
    config: Config,
    n_eval_steps: int,
) -> Mapping[str, plt.Figure]:
    figure_creators: list[StreamingFigureCreator] = []

    for figure_config in config.figures:
        match figure_config:
            case CIHistogramsFigureConfig():
                figure_creators.append(CIHistograms())
            case MeanComponentActivationCountsFigureConfig():
                figure_creators.append(
                    MeanComponentActivationCounts(model, device, config.ci_alive_threshold)
                )
            case UVandIdentityCIFigureConfig():
                figure_creators.append(UVandIdentityCI(model, device, config.sigmoid_type))

    for _ in range(n_eval_steps):
        # Do the work that's common between figures
        batch = extract_batch_data(next(eval_iterator))
        batch = batch.to(device)
        _, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
            batch, module_names=model.target_module_paths
        )
        ci, _ci_upper_leaky = model.calc_causal_importances(
            pre_weight_acts, sigmoid_type=config.sigmoid_type
        )

        inputs = FigureInput(ci=ci, batch=batch)

        for figure_creator in figure_creators:
            figure_creator.watch(inputs)

    out: dict[str, plt.Figure] = {}
    for figure_creator in figure_creators:
        out.update(figure_creator.compute())

    return out
