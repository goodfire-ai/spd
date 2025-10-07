#!/usr/bin/env python3
"""
Visualize component matrices from a trained SPD model.

This script loads a trained SPD model and creates heatmaps of the rank-one matrices
(V @ U) for each component, using red for positive values, blue for negative values,
and white for zero.

Usage:
    python spd/scripts/component_visualization/component_visualization.py spd/scripts/component_visualization/component_visualization_config.yaml
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from PIL import Image
from pydantic import Field
from torch import Tensor

from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import BaseModel, load_config


class ComponentVisualizationConfig(BaseModel):
    """Configuration for component visualization script."""

    model_path: str = Field(
        ..., description="Path to the trained SPD model (wandb:project/run_id or local path)"
    )
    threshold: float = Field(
        default=0.01, description="Threshold for considering a value as 'active' (default: 0.01)"
    )
    figsize: tuple[float, float] = Field(
        default=(8, 6), description="Figure size per component (width height) (default: 8 6)"
    )
    dpi: int = Field(default=150, description="DPI for the figures (default: 150)")
    device: str = Field(default="auto", description="Device to use (default: auto)")

    output_dir: str | None = Field(
        default=None,
        description="Directory to save results (defaults to 'out' directory relative to script location)",
    )


def create_rank_one_matrices(components: dict[str, Any]) -> dict[str, Float[Tensor, "d_out d_in"]]:
    """Create rank-one matrices by computing V @ U for each component.

    Args:
        components: Dictionary mapping module names to Components objects

    Returns:
        Dictionary mapping module names to rank-one matrices
    """
    rank_one_matrices = {}

    for module_name, component in components.items():
        # Compute V @ U to get the rank-one matrix
        # V has shape (d_in, C), U has shape (C, d_out)
        # So V @ U has shape (d_in, d_out)
        print(f"Debug: {module_name} - V shape: {component.V.shape}, U shape: {component.U.shape}")
        rank_one_matrix = torch.matmul(component.V, component.U)
        rank_one_matrices[module_name] = rank_one_matrix

    return rank_one_matrices


def plot_component_matrices(
    rank_one_matrices: dict[str, Float[Tensor, "d_out d_in"]],
    output_path: str | Path | None = None,
    figsize_per_component: tuple[float, float] = (8, 6),
    dpi: int = 150,
) -> Image.Image:
    """Plot component matrices as heatmaps.

    Args:
        rank_one_matrices: Dictionary mapping module names to rank-one matrices
        output_path: Optional path to save the figure
        figsize_per_component: Figure size per component (width, height)
        dpi: DPI for the figure

    Returns:
        PIL Image of the plot
    """
    n_components = len(rank_one_matrices)

    # Calculate grid layout
    n_cols = min(3, n_components)  # Max 3 columns
    n_rows = (n_components + n_cols - 1) // n_cols  # Ceiling division

    # Create figure
    fig_width = figsize_per_component[0] * n_cols
    fig_height = figsize_per_component[1] * n_rows
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_width, fig_height), dpi=dpi, squeeze=False
    )

    # Flatten axes for easier indexing
    axes_flat = axes.flatten()

    # Find global min/max for consistent colorbar
    all_values = torch.cat([matrix.flatten() for matrix in rank_one_matrices.values()])
    vmin = all_values.min().item()
    vmax = all_values.max().item()

    # Ensure symmetric colorbar around zero
    abs_max = max(abs(vmin), abs(vmax))
    vmin, vmax = -abs_max, abs_max

    # Plot each component matrix
    for i, (module_name, matrix) in enumerate(rank_one_matrices.items()):
        ax = axes_flat[i]

        # Convert to numpy and transpose to match expected orientation
        matrix_np = matrix.detach().cpu().numpy()

        # Create heatmap with custom colormap
        im = ax.imshow(
            matrix_np,
            cmap="RdBu_r",  # Red-Blue reversed (red=positive, blue=negative)
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )

        # Set labels and title
        ax.set_title(f"{module_name}\nShape: {matrix.shape}", fontsize=10)
        ax.set_xlabel("Output dimension")
        ax.set_ylabel("Input dimension")

        # Add colorbar for each subplot
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_components, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {output_path}")

    # Convert to PIL Image
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = Image.fromarray(img_array)

    plt.close(fig)
    return img


def plot_individual_components(
    rank_one_matrices: dict[str, Float[Tensor, "d_out d_in"]],
    output_dir: str | Path,
    figsize: tuple[float, float] = (10, 8),
    dpi: int = 150,
) -> None:
    """Plot each component matrix individually.

    Args:
        rank_one_matrices: Dictionary mapping module names to rank-one matrices
        output_dir: Directory to save individual plots
        figsize: Figure size for individual plots
        dpi: DPI for the figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for module_name, matrix in rank_one_matrices.items():
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Convert to numpy
        matrix_np = matrix.detach().cpu().numpy()

        # Find symmetric colorbar range
        vmin, vmax = matrix_np.min(), matrix_np.max()
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max

        # Create heatmap
        im = ax.imshow(
            matrix_np, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest"
        )

        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Set labels and title
        ax.set_title(f"{module_name}\nShape: {matrix.shape}", fontsize=12)
        ax.set_xlabel("Output dimension")
        ax.set_ylabel("Input dimension")
        ax.grid(True, alpha=0.3)

        # Save individual plot
        safe_name = module_name.replace(".", "_").replace("/", "_")
        output_path = output_dir / f"{safe_name}_component_matrix.png"
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Individual plot saved to: {output_path}")

        plt.close(fig)


def create_activation_heatmap(
    activation_matrix: Float[Tensor, "n_features n_components"],
    module_name: str,
    output_dir: str = "activation_plots",
    figsize: tuple[float, float] = (12, 8),
    dpi: int = 150,
) -> None:
    """Create a heatmap showing which components activate for which inputs.

    Args:
        activation_matrix: Binary matrix (n_features, n_components) where 1 = active, 0 = inactive
        module_name: Name of the module for the plot title
        output_dir: Directory to save the plot
        figsize: Figure size
        dpi: DPI for the figure
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Convert to numpy
    matrix_np = activation_matrix.detach().cpu().numpy()

    # Create heatmap using the same colormap as upper leaky plots
    im = ax.imshow(
        matrix_np,
        cmap="Reds",  # Same colormap as causal_importances_upper_leaky plots
        aspect="auto",
        interpolation="nearest",
    )

    # Set labels
    ax.set_xlabel("Component Index")
    ax.set_ylabel("Input Feature Index")
    ax.set_title(f"{module_name} - Component Activation Pattern\n(Red=Active, White=Inactive)")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Activation (0=Inactive, 1=Active)")

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Save plot
    safe_name = module_name.replace(".", "_").replace("/", "_")
    output_path = output_dir / f"{safe_name}_activation_pattern.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Activation pattern plot saved to: {output_path}")

    plt.close(fig)


def analyze_component_behavior(
    model: ComponentModel,
    device: str,
    output_dir: Path,
    n_features: int | None = None,
    input_magnitude: float = 0.75,
    threshold: float = 0.1,
) -> None:
    """Analyze component behavior by computing gate activations for different inputs.

    This is similar to how causal importances are computed in the evaluation.

    Args:
        model: The trained ComponentModel
        device: Device to run on
        n_features: Number of input features to test (if None, will get from model)
        input_magnitude: Magnitude of input features
        threshold: Threshold for considering a gate as "active"
    """
    print("\n" + "=" * 60)
    print("COMPONENT GATE ACTIVATION ANALYSIS")
    print("=" * 60)

    model.eval()

    # Get n_features from the model if not provided
    if n_features is None:
        if hasattr(model.target_model, "config"):
            n_features = model.target_model.config.n_features
        else:
            # Fallback: get from W_E shape
            n_features = model.target_model.W_E.shape[0]

    print(f"Using n_features: {n_features}")

    # Create test inputs - one-hot vectors for each feature
    test_inputs = torch.eye(n_features, device=device) * input_magnitude

    # Get pre-weight activations by running through target model with input caching
    with torch.no_grad():
        _, pre_weight_acts = model(test_inputs, cache_type="input")

        # Compute causal importances
        causal_importances, _ = model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            sigmoid_type=model.config.sigmoid_type if hasattr(model, "config") else "leaky_hard",
            sampling="continuous",
            detach_inputs=True,
        )

    # Analyze each module
    for module_name, ci in causal_importances.items():
        print(f"\n{module_name}:")
        print(f"  Causal importance shape: {ci.shape}")

        # ci has shape (n_features, C) where C is number of components
        n_components = ci.shape[1]
        n_features = ci.shape[0]

        # For each component, count how many inputs activate it above threshold
        component_stats = []
        for c in range(n_components):
            component_ci = ci[:, c]  # Shape: (n_features,)
            active_inputs = torch.sum(component_ci > threshold).item()
            total_inputs = component_ci.shape[0]
            activation_ratio = active_inputs / total_inputs

            max_ci = torch.max(component_ci).item()
            mean_ci = torch.mean(component_ci).item()

            component_stats.append(
                {
                    "component": c,
                    "active_inputs": active_inputs,
                    "activation_ratio": activation_ratio,
                    "max_ci": max_ci,
                    "mean_ci": mean_ci,
                }
            )

        # Sort components by activation ratio (most active first)
        component_stats.sort(key=lambda x: x["activation_ratio"], reverse=True)

        # Print top components
        print("  Top 10 most active components:")
        for stats in component_stats[:10]:
            c = stats["component"]
            active_inputs = stats["active_inputs"]
            activation_ratio = stats["activation_ratio"]
            max_ci = stats["max_ci"]
            mean_ci = stats["mean_ci"]

            print(
                f"    Component {c}: {active_inputs}/{n_features} inputs activate gate ({activation_ratio:.3f})"
            )
            print(f"      Max CI: {max_ci:.4f}, Mean CI: {mean_ci:.4f}")

            if activation_ratio > 0.8:  # Component gate activates for most inputs
                print(
                    f"      🚨 BROAD ACTIVATION: Component {c} gate activates for {activation_ratio:.1%} of inputs!"
                )
            elif activation_ratio > 0.5:  # Component gate activates for more than half the inputs
                print(
                    f"      ⚠️  WARNING: Component {c} gate activates for {activation_ratio:.1%} of inputs!"
                )

        # Analyze input-specific patterns
        print("\n  Input-specific analysis:")
        for input_idx in range(min(10, n_features)):  # Check first 10 inputs
            input_ci = ci[input_idx, :]  # Shape: (n_components,)
            active_components = torch.sum(input_ci > threshold).item()
            max_ci_val = torch.max(input_ci).item()

            print(
                f"    Input {input_idx}: {active_components} components active (max CI: {max_ci_val:.4f})"
            )

            # Show which components are active for this input
            active_component_indices = torch.where(input_ci > threshold)[0].tolist()
            if len(active_component_indices) > 0:
                print(f"      Active components: {active_component_indices}")

        # Find the "universal" component (activates for most inputs)
        universal_components = [s for s in component_stats if s["activation_ratio"] > 0.8]
        if universal_components:
            print("\n  🚨 UNIVERSAL COMPONENTS (activate for >80% of inputs):")
            for stats in universal_components:
                c = stats["component"]
                activation_ratio = stats["activation_ratio"]
                print(f"    Component {c}: {activation_ratio:.1%} of inputs")

        # Count how many inputs have exactly 1, 2, 3+ components active
        input_component_counts = torch.sum(ci > threshold, dim=1)  # Shape: (n_features,)
        unique_counts, counts = torch.unique(input_component_counts, return_counts=True)
        print("\n  Input activation patterns:")
        for count, freq in zip(unique_counts.tolist(), counts.tolist(), strict=False):
            print(f"    {freq} inputs have exactly {count} components active")

        # Create a binary activation matrix for visualization
        activation_matrix = (ci > threshold).float()  # Shape: (n_features, n_components)

        # Save activation matrix for further analysis
        torch.save(
            activation_matrix, output_dir / f"{module_name.replace('.', '_')}_activation_matrix.pt"
        )
        print(
            f"  Saved activation matrix to: {output_dir / f'{module_name.replace(".", "_")}_activation_matrix.pt'}"
        )

        # Create activation pattern heatmap
        create_activation_heatmap(
            activation_matrix, module_name, output_dir=output_dir / "activation_plots"
        )


def main(config_path_or_obj: str | ComponentVisualizationConfig = None) -> None:
    """Main function for component visualization."""
    config = load_config(config_path_or_obj, config_model=ComponentVisualizationConfig)

    # Set device
    device = get_device() if config.device == "auto" else config.device
    logger.info(f"Using device: {device}")
    logger.info(f"Loading model from: {config.model_path}")

    # Extract model ID from the model path for directory organization
    if config.model_path.startswith("wandb:"):
        # Extract run ID from wandb path (e.g., "wandb:goodfire/spd/fi1phj8l" -> "fi1phj8l")
        model_id = config.model_path.split("/")[-1]
    else:
        # For local paths, use the directory name
        model_id = Path(config.model_path).name

    # Set up output directory with model ID subdirectory
    if config.output_dir is None:
        base_output_dir = Path(__file__).parent / "out"
    else:
        base_output_dir = Path(config.output_dir)

    output_dir_path = base_output_dir / model_id
    output_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir_path}")

    # Load the model
    try:
        run_info = SPDRunInfo.from_path(config.model_path)
        model = ComponentModel.from_run_info(run_info)
        model.to(device)
        model.eval()
        logger.info(f"Successfully loaded model with {len(model.components)} component modules")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    # Create rank-one matrices
    logger.info("Creating rank-one matrices...")
    rank_one_matrices = create_rank_one_matrices(model.components)

    # Analyze component behavior using gate activations
    analyze_component_behavior(model, device, output_dir_path, threshold=config.threshold)

    # Create combined plot
    output_path = output_dir_path / "component_matrices.png"
    logger.info(f"Creating combined plot: {output_path}")
    plot_component_matrices(
        rank_one_matrices,
        output_path=output_path,
        figsize_per_component=config.figsize,
        dpi=config.dpi,
    )

    # Create individual plots
    individual_plots_dir = output_dir_path / "component_plots"
    logger.info(f"Creating individual plots in: {individual_plots_dir}")
    plot_individual_components(
        rank_one_matrices, individual_plots_dir, figsize=config.figsize, dpi=config.dpi
    )

    logger.info("Visualization complete!")


if __name__ == "__main__":
    # Use the config file directly
    config_path = Path(__file__).parent / "component_visualization_config.yaml"
    main(config_path)
