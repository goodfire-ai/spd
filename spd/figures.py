"""Figures for SPD experiments.

This file contains visualizations that can be logged during SPD optimization.
These can be selected and configured in the config file.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator, Mapping
from typing import Any, override

import torch
from einops import reduce
from jaxtyping import Float, Int
from matplotlib import pyplot as plt
from torch import Tensor

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.plotting import (
    plot_causal_importance_vals,
    plot_ci_histograms,
    plot_mean_component_activation_counts,
    plot_UV_matrices,
)
from spd.utils.general_utils import extract_batch_data


class StreamingFigureCreator(ABC):
    @abstractmethod
    def __init__(self, model: ComponentModel, config: Config, **kwargs: Any): ...

    @abstractmethod
    def watch_batch(
        self,
        ci: dict[str, Float[Tensor, "... C"]],
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    ) -> None: ...

    @abstractmethod
    def compute(self) -> Mapping[str, plt.Figure]: ...


class CIHistograms(StreamingFigureCreator):
    def __init__(self, model: ComponentModel, config: Config):
        self.causal_importances = defaultdict[str, list[Float[Tensor, "... C"]]](list)

    @override
    def watch_batch(
        self,
        ci: dict[str, Float[Tensor, "... C"]],
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    ) -> None:
        for k, v in ci.items():
            self.causal_importances[k].append(v)

    @override
    def compute(self) -> Mapping[str, plt.Figure]:
        combined_causal_importances = {k: torch.cat(v) for k, v in self.causal_importances.items()}
        fig = plot_ci_histograms(causal_importances=combined_causal_importances)
        return {"causal_importances_hist": fig}


class MeanComponentActivationCounts(StreamingFigureCreator):
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
        ci: dict[str, Float[Tensor, "... C"]],
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    ) -> None:
        n_tokens = next(iter(ci.values())).shape[:-1].numel()
        self.n_tokens += n_tokens

        for module_name, ci_vals in ci.items():
            active_components = ci_vals > self.config.ci_alive_threshold
            n_activations_per_component = reduce(active_components, "... C -> C", "sum")
            self.component_activation_counts[module_name] += n_activations_per_component

    @override
    def compute(self) -> Mapping[str, plt.Figure]:
        mean_counts_per_module = {
            module_name: self.component_activation_counts[module_name] / self.n_tokens
            for module_name in self.model.components
        }
        fig = plot_mean_component_activation_counts(mean_counts_per_module)
        return {"MeanComponentActivationCounts": fig}


class UVandIdentityCI(StreamingFigureCreator):
    def __init__(
        self,
        model: ComponentModel,
        config: Config,
    ):
        self.model = model
        self.config = config
        self.device = next(iter(model.parameters())).device

        self.batch_shape = None

    @override
    def watch_batch(
        self,
        ci: dict[str, Float[Tensor, "... C"]],
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    ) -> None:
        if self.batch_shape is None:
            self.batch_shape = batch.shape

    @override
    def compute(self) -> Mapping[str, plt.Figure]:
        assert self.batch_shape is not None, "haven't seen any inputs yet"

        figures, all_perm_indices = plot_causal_importance_vals(
            model=self.model,
            batch_shape=self.batch_shape,
            device=self.device,
            input_magnitude=0.75,
            sigmoid_type=self.config.sigmoid_type,
        )

        uv_matrices_fig = plot_UV_matrices(
            components=self.model.components, all_perm_indices=all_perm_indices
        )

        return {
            **figures,
            "uv_matrices": uv_matrices_fig,
        }


FIGURE_CLASSES = {cls.__name__: cls for cls in StreamingFigureCreator.__subclasses__()}


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
        figure_cls = FIGURE_CLASSES[figure_config.classname]
        figure_creators.append(figure_cls(model, config, **figure_config.extra_init_kwargs))

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

        for figure_creator in figure_creators:
            figure_creator.watch_batch(ci=ci, batch=batch)

    out: dict[str, plt.Figure] = {}
    all_dicts = [figure_creator.compute() for figure_creator in figure_creators]
    for d in all_dicts:
        if set(d.keys()).intersection(out.keys()):
            raise ValueError(f"Keys {set(d.keys()).intersection(out.keys())} already in output")
        out.update(d)

    return out
