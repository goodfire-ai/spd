"""Utilities for parameter sweeps in merge ensemble analysis."""

from __future__ import annotations

from typing import Any, Literal

import matplotlib.pyplot as plt
from jaxtyping import Float
from torch import Tensor

from spd.clustering.merge import (
    MergeConfig,
    MergeHistoryEnsemble,
    MergePlotConfig,
    merge_iteration_ensemble,
)
from spd.clustering.plotting.merge import plot_dists_distribution


def sweep_merge_parameter(
    activations: Float[Tensor, "samples c_components"],
    parameter_name: Literal["alpha", "check_threshold", "pop_component_prob"],
    parameter_values: list[float],
    base_config: dict[str, Any] | None = None,
    ensemble_size: int = 16,
    component_labels: list[str] | None = None,
    plot_config: MergePlotConfig | None = None,
    figsize: tuple[int, int] = (16, 10),
    plot_mode: Literal["points", "dist"] = "dist",
) -> tuple[dict[float, MergeHistoryEnsemble], plt.Figure, plt.Axes]:
    """Run ensemble merge iterations for different values of a single parameter.

    Args:
        activations: Component activations tensor
        parameter_name: Name of the parameter to sweep over
        parameter_values: List of values to test for the parameter
        base_config: Base configuration for MergeConfig (parameter_name will be overridden)
        ensemble_size: Number of ensemble members to generate for each parameter value
        component_labels: Optional labels for components
        plot_config: Optional plot configuration for merge iterations
        figsize: Figure size for the comparison plot
        plot_mode: Plot mode for distance distribution

    Returns:
        Tuple of:
        - Dictionary mapping parameter values to MergeEnsemble objects
        - Figure object
        - Axes object
    """
    # Default base configuration
    default_base = {
        "activation_threshold": None,
        "alpha": 1.0,
        "iters": 140,
        "check_threshold": 0.1,
        "pop_component_prob": 0.1,
        "rank_cost_fn": lambda x: 1.0,
        "stopping_condition": None,
    }

    # Merge with user-provided base config
    config_dict = {**default_base, **(base_config or {})}

    # Default plot config that skips intermediate plots
    if plot_config is None:
        plot_config = MergePlotConfig(
            plot_every=999,
            plot_every_min=999,
            save_pdf=False,
            plot_final=False,
        )

    # Create figure for comparison
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Store results
    ensembles: dict[float, MergeHistoryEnsemble] = {}

    # Run sweep
    for value in parameter_values:
        print(f"{parameter_name}: {value}")

        # Update the swept parameter
        config_dict[parameter_name] = value
        merge_config = MergeConfig(**config_dict)

        # Run ensemble
        ensemble = merge_iteration_ensemble(
            activations=activations,
            component_labels=component_labels,
            merge_config=merge_config,
            ensemble_size=ensemble_size,
        )

        ensembles[value] = ensemble

        print(f"  Got ensemble with {ensemble.n_iters} iterations, {ensemble.n_ensemble} members")

        # Get distances and plot
        distances = ensemble.get_distances()
        print(f"  Distances shape: {distances.shape}")

        # Format label based on parameter name
        if parameter_name == "alpha":
            label = f"$\\alpha={value:.4f}$"
        elif parameter_name == "check_threshold":
            label = f"$c={value:.4f}$"
        elif parameter_name == "pop_component_prob":
            label = f"$p={value:.4f}$"
        else:
            label = f"{parameter_name}={value}"

        plot_dists_distribution(
            distances=distances,
            mode=plot_mode,
            label=label,
            ax=ax,
        )

    # Finalize plot
    ax.legend()
    ax.set_title(f"Distance distribution vs {parameter_name}")
    plt.tight_layout()

    return ensembles, fig, ax


def sweep_multiple_parameters(
    activations: Float[Tensor, "samples c_components"],
    parameter_sweeps: dict[str, list[float]],
    base_config: dict[str, Any] | None = None,
    ensemble_size: int = 16,
    component_labels: list[str] | None = None,
    plot_config: MergePlotConfig | None = None,
    figsize: tuple[int, int] = (16, 10),
    plot_mode: Literal["points", "dist"] = "dist",
) -> dict[str, tuple[dict[float, MergeHistoryEnsemble], plt.Figure, plt.Axes]]:
    """Run multiple parameter sweeps and create comparison plots.

    Args:
        activations: Component activations tensor
        parameter_sweeps: Dictionary mapping parameter names to lists of values
        base_config: Base configuration for MergeConfig
        ensemble_size: Number of ensemble members to generate
        component_labels: Optional labels for components
        plot_config: Optional plot configuration for merge iterations
        figsize: Figure size for each comparison plot
        plot_mode: Plot mode for distance distribution

    Returns:
        Dictionary mapping parameter names to (ensembles, figure, axes) tuples
    """
    results = {}

    for param_name, param_values in parameter_sweeps.items():
        ensembles, fig, ax = sweep_merge_parameter(
            activations=activations,
            parameter_name=param_name,
            parameter_values=param_values,
            base_config=base_config,
            ensemble_size=ensemble_size,
            component_labels=component_labels,
            plot_config=plot_config,
            figsize=figsize,
            plot_mode=plot_mode,
        )
        results[param_name] = (ensembles, fig, ax)

    return results
