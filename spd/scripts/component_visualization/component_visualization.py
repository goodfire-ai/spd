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
from spd.utils.target_ci_solutions import permute_to_identity


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
    generate_both_permutations: bool = Field(
        default=True,
        description="Generate both permuted and non-permuted activation plots (default: True)",
    )

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


def analyze_component_behavior(
    model: ComponentModel,
    device: str,
    output_dir: Path,
    n_features: int | None = None,
    input_magnitudes: list[float] | None = None,
    threshold: float = 0.1,
    generate_both_permutations: bool = True,
) -> None:
    """Analyze component behavior by computing gate activations for different inputs.

    This is similar to how causal importances are computed in the evaluation.

    Args:
        model: The trained ComponentModel
        device: Device to run on
        n_features: Number of input features to test (if None, will get from model)
        input_magnitudes: List of input magnitudes to test (default: [0.0, 0.25, 0.75, 1.0])
        threshold: Threshold for considering a gate as "active"
    """
    print("\n" + "=" * 60)
    print("COMPONENT GATE ACTIVATION ANALYSIS")
    print("=" * 60)

    model.eval()

    # Set default input magnitudes if not provided (include negative values)
    if input_magnitudes is None:
        input_magnitudes = [-1.0, -0.75, -0.25, 0.0, 0.25, 0.75, 1.0]

    # Get n_features from the model if not provided
    if n_features is None:
        if hasattr(model.target_model, "config"):
            n_features = model.target_model.config.n_features
        else:
            # Fallback: get from W_E shape
            n_features = model.target_model.W_E.shape[0]

    print(f"Using n_features: {n_features}")
    print(f"Testing input magnitudes: {input_magnitudes}")

    # Store results for all magnitudes
    all_causal_importances = {}
    all_perm_indices = {}

    # Process each input magnitude
    for magnitude in input_magnitudes:
        print(f"\nProcessing magnitude: {magnitude}")

        # Create test inputs - one-hot vectors for each feature (match evals exactly)
        test_inputs = torch.eye(n_features, device=device) * magnitude

        # Get pre-weight activations by running through target model with input caching
        with torch.no_grad():
            pre_weight_acts = model(test_inputs, cache_type="input").cache

            # Compute causal importances (use upper_leaky like evals)
            _, causal_importances = model.calc_causal_importances(
                pre_weight_acts=pre_weight_acts,
                sigmoid_type=model.config.sigmoid_type
                if hasattr(model, "config")
                else "leaky_hard",
                sampling="continuous",
                detach_inputs=False,  # Match evals behavior
            )

        # Store original causal importances
        all_causal_importances[magnitude] = causal_importances

        if generate_both_permutations:
            # Apply sorting using the same logic as evals
            sorted_causal_importances = {}
            perm_indices = {}

            for layer_name, ci_vals in causal_importances.items():
                # Use permute_to_identity for sorting (same as evals)
                sorted_ci, perm_idx = permute_to_identity(ci_vals)
                sorted_causal_importances[layer_name] = sorted_ci
                perm_indices[layer_name] = perm_idx

            # Store permuted data separately
            all_causal_importances[f"{magnitude}_permuted"] = sorted_causal_importances
            all_perm_indices[magnitude] = perm_indices

    # Create side-by-side comparison plots for each module
    for module_name in all_causal_importances[input_magnitudes[0]]:
        print(f"\n{module_name}:")

        # Collect CI values for all magnitudes
        ci_data = {}
        for magnitude in input_magnitudes:
            ci_data[magnitude] = all_causal_importances[magnitude][module_name]

        # Print analysis for the first magnitude (0.75 by default)
        primary_magnitude = 0.75 if 0.75 in input_magnitudes else input_magnitudes[0]
        ci = ci_data[primary_magnitude]
        print(f"  Causal importance shape: {ci.shape}")

        # Find active components (above threshold)
        active_mask = ci > threshold
        active_counts = active_mask.sum(dim=0)  # Count per component

        # Sort components by activity
        sorted_indices = torch.argsort(active_counts, descending=True)
        top_components = sorted_indices[:10]  # Top 10 most active

        print("  Top 10 most active components:")
        for comp_idx in top_components:
            count = active_counts[comp_idx].item()
            percentage = count / ci.shape[0] * 100
            max_ci = ci[:, comp_idx].max().item()
            mean_ci = ci[:, comp_idx].mean().item()

            print(
                f"    Component {comp_idx.item()}: {count}/{ci.shape[0]} inputs activate gate ({percentage:.3f})"
            )
            print(f"      Max CI: {max_ci:.4f}, Mean CI: {mean_ci:.4f}")

            # Warning for highly active components
            if percentage > 50:
                print(
                    f"      ⚠️  WARNING: Component {comp_idx.item()} gate activates for {percentage:.1f}% of inputs!"
                )

        # Input-specific analysis
        print("\n  Input-specific analysis:")
        for input_idx in range(min(10, ci.shape[0])):  # Show first 10 inputs
            input_ci = ci[input_idx]
            active_components = torch.where(input_ci > threshold)[0]
            max_ci = input_ci.max().item()

            print(
                f"    Input {input_idx}: {len(active_components)} components active (max CI: {max_ci:.4f})"
            )
            if len(active_components) > 0:
                active_list = active_components.cpu().numpy().tolist()
                print(f"      Active components: {active_list}")

        # Activation pattern analysis
        n_active_per_input = active_mask.sum(dim=1)
        unique_counts, counts = torch.unique(n_active_per_input, return_counts=True)

        print("\n  Input activation patterns:")
        for n_active, count in zip(unique_counts, counts, strict=False):
            print(f"    {count} inputs have exactly {n_active.item()} components active")

        # Save activation matrix for primary magnitude
        activation_matrix_path = (
            output_dir / f"{module_name.replace('.', '_')}_activation_matrix.pt"
        )
        torch.save(ci.cpu(), activation_matrix_path)
        print(f"  Saved activation matrix to: {activation_matrix_path}")

        # Create side-by-side activation pattern plots
        if generate_both_permutations:
            # Create non-permuted plots
            non_permuted_data = {
                mag: all_causal_importances[mag][module_name] for mag in input_magnitudes
            }
            plot_multi_magnitude_activation_patterns(
                non_permuted_data, module_name, output_dir, input_magnitudes, is_permuted=False
            )

            # Create permuted plots
            permuted_data = {
                mag: all_causal_importances[f"{mag}_permuted"][module_name]
                for mag in input_magnitudes
            }
            plot_multi_magnitude_activation_patterns(
                permuted_data, module_name, output_dir, input_magnitudes, is_permuted=True
            )
        else:
            # Create only non-permuted plots (original behavior)
            plot_multi_magnitude_activation_patterns(
                ci_data, module_name, output_dir, input_magnitudes, is_permuted=False
            )


def plot_multi_magnitude_activation_patterns(
    ci_data: dict[float, Float[Tensor, "n_features n_components"]],
    module_name: str,
    output_dir: Path,
    input_magnitudes: list[float],
    is_permuted: bool = False,
) -> None:
    """Create side-by-side activation pattern plots for multiple input magnitudes.

    Args:
        ci_data: Dictionary mapping magnitude to causal importance tensor
        module_name: Name of the module being analyzed
        output_dir: Directory to save plots
        input_magnitudes: List of input magnitudes used
        is_permuted: Whether the data has been permuted to identity (affects file naming)
    """
    n_magnitudes = len(input_magnitudes)

    # Create figure with subplots for each magnitude
    fig, axes = plt.subplots(1, n_magnitudes, figsize=(4 * n_magnitudes, 8))
    if n_magnitudes == 1:
        axes = [axes]

    # Create activation matrices for each magnitude (plot raw values like evals)
    for i, magnitude in enumerate(input_magnitudes):
        ci = ci_data[magnitude]
        # Plot raw causal importance values (no threshold) like evals do
        ax = axes[i]
        im = ax.imshow(ci.cpu().numpy(), cmap="Reds", aspect="auto")
        ax.set_title(f"Input Magnitude: {magnitude}")
        ax.set_xlabel("Component Index")
        ax.set_ylabel("Input Feature Index")

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Set overall title
    permutation_suffix = "_permuted" if is_permuted else "_non_permuted"
    fig.suptitle(
        f"Activation Patterns - {module_name}{permutation_suffix.replace('_', ' ').title()}",
        fontsize=14,
    )
    plt.tight_layout()

    # Save plot
    output_path = (
        output_dir
        / "activation_plots"
        / f"{module_name.replace('.', '_')}_multi_magnitude_activation_pattern{permutation_suffix}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Multi-magnitude activation pattern plot saved to: {output_path}")


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
    analyze_component_behavior(
        model,
        device,
        output_dir_path,
        threshold=config.threshold,
        generate_both_permutations=config.generate_both_permutations,
    )

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
