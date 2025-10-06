#!/usr/bin/env python3
"""
Plot ResidMLP neuron activations and causal importance as input magnitude increases.

This script creates plots showing how individual neuron activations and causal importance
values respond as we gradually increase the magnitude of a one-hot input vector from 0 to max_magnitude.

The x-axis represents the input magnitude, and the y-axes show:
1. Individual neuron activations in the ResidMLP layers
2. Causal importance function values for gates that actually activate

Usage:
    python spd/scripts/magnitude_sweep/magnitude_sweep.py spd/scripts/magnitude_sweep/magnitude_sweep_config.yaml
    python spd/scripts/magnitude_sweep/magnitude_sweep.py --model_path="wandb:..." --feature_idx=0
"""

from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from pydantic import Field
from torch import Tensor

from spd.experiments.resid_mlp.models import ResidMLP
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import LinearCiFn, MLPCiFn, VectorMLPCiFn, VectorSharedMLPCiFn
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import BaseModel, load_config


class MagnitudeSweepConfig(BaseModel):
    """Configuration for magnitude sweep plotting script."""

    model_path: str = Field(
        ..., description="Path to the trained SPD model (wandb:project/run_id or local path)"
    )
    feature_idx: int = Field(default=0, description="Which feature to activate (default: 0)")
    n_steps: int = Field(
        default=100, description="Number of steps from 0 to max_magnitude (default: 100)"
    )
    max_magnitude: float = Field(default=2.0, description="Maximum input magnitude (default: 2.0)")
    figsize_per_subplot: tuple[float, float] = Field(
        default=(2, 1.5), description="Figure size per subplot (width height)"
    )
    dpi: int = Field(default=150, description="DPI for figures (default: 150)")
    ci_threshold: float = Field(
        default=0.1, description="CI threshold for active gates (default: 0.1)"
    )
    pre_activation: bool = Field(
        default=False,
        description="Show pre-activation values (before ReLU) instead of post-activation",
    )
    device: str = Field(default="auto", description="Device to use (default: auto)")

    output_dir: str | None = Field(
        default=None,
        description="Directory to save results (defaults to 'out' directory relative to script location)",
    )


def get_residmlp_activations(
    model: ResidMLP,
    input_tensor: Float[Tensor, "batch n_features"],
    return_intermediate: bool = True,
    return_pre_activation: bool = False,
) -> dict[str, Float[Tensor, "batch d_mlp"]]:
    """Get intermediate activations from ResidMLP model.

    Args:
        model: The ResidMLP model
        input_tensor: Input tensor of shape (batch, n_features)
        return_intermediate: Whether to return intermediate activations
        return_pre_activation: Whether to return pre-activation values (before ReLU)

    Returns:
        Dictionary mapping layer names to their activations (after or before activation function)
    """
    activations = {}

    # Embed the input
    residual = torch.matmul(input_tensor, model.W_E)

    # Forward through each layer
    for i, layer in enumerate(model.layers):
        # Get pre-activation values
        mid_pre_act = layer.mlp_in(residual)  # pyright: ignore[reportCallIssue]

        # Apply activation function
        mid_act = model.act_fn(mid_pre_act)

        # Store activations for this layer
        if return_intermediate:
            if return_pre_activation:
                activations[f"layers.{i}.mlp_in"] = mid_pre_act  # Before ReLU
            else:
                activations[f"layers.{i}.mlp_in"] = mid_act  # After ReLU

        # Get output and add to residual
        out = layer.mlp_out(mid_act)  # pyright: ignore[reportCallIssue]
        residual = residual + out

    return activations


def compute_magnitude_sweep_data(
    model: ComponentModel,
    device: str,
    n_features: int,
    feature_idx: int = 0,
    n_steps: int = 100,
    max_magnitude: float = 2.0,
    pre_activation: bool = False,
) -> tuple[
    dict[str, Float[Tensor, "n_steps d_mlp"]],
    dict[str, Float[Tensor, "n_steps n_components"]],
    dict[str, Float[Tensor, "n_steps n_features"]],
    dict[str, Float[Tensor, "n_steps n_components"]],
    dict[str, Float[Tensor, "n_steps d_in"]],
]:
    """Compute neuron activations and causal importance as input magnitude increases.

    Args:
        model: The trained ComponentModel containing ResidMLP
        device: Device to run on
        n_features: Number of input features
        feature_idx: Which feature to activate (default: 0)
        n_steps: Number of steps from 0 to max_magnitude
        max_magnitude: Maximum input magnitude

    Returns:
        Tuple of (activations_dict, causal_importance_dict, output_responses_dict, gate_outputs_dict, gate_inputs_dict) where:
        - activations_dict maps layer names to activation tensors of shape (n_steps, d_mlp)
        - causal_importance_dict maps layer names to CI tensors of shape (n_steps, n_components)
        - output_responses_dict maps layer names to output tensors of shape (n_steps, n_features)
        - gate_outputs_dict maps layer names to pre-sigmoid gate outputs of shape (n_steps, n_components)
        - gate_inputs_dict maps layer names to gate inputs (inner acts) of shape (n_steps, d_in)
    """
    model.eval()

    # Create magnitude steps
    magnitudes = torch.linspace(-max_magnitude, max_magnitude, n_steps, device=device)

    # Get the target ResidMLP model
    target_model = model.target_model
    assert isinstance(target_model, ResidMLP), "Model must be a ResidMLP"

    # Initialize storage
    activations = {}
    causal_importances = {}
    output_responses = {}
    gate_outputs = {}
    gate_inputs = {}

    # Get layer names and dimensions
    layer_names = []
    for i in range(target_model.config.n_layers):
        layer_names.append(f"layers.{i}.mlp_in")

    # Initialize tensors
    for layer_name in layer_names:
        d_mlp = target_model.config.d_mlp
        n_components = model.components[layer_name].U.shape[0]
        # For gate inputs, we need to determine the actual input dimension based on CI function type
        # We'll initialize with a placeholder and resize later
        activations[layer_name] = torch.zeros(n_steps, d_mlp, device=device)
        causal_importances[layer_name] = torch.zeros(n_steps, n_components, device=device)
        output_responses[layer_name] = torch.zeros(n_steps, n_features, device=device)
        gate_outputs[layer_name] = torch.zeros(n_steps, n_components, device=device)
        # Initialize gate_inputs with a placeholder - will be resized when we know the actual dimension
        gate_inputs[layer_name] = None

    print(f"Computing magnitude sweep for feature {feature_idx}...")
    print(f"Magnitude range: 0 to {max_magnitude} in {n_steps} steps")

    # For each magnitude step
    for step_idx, magnitude in enumerate(magnitudes):
        if step_idx % 20 == 0:
            print(f"  Step {step_idx}/{n_steps} (magnitude={magnitude:.3f})")

        # Create one-hot input with specified magnitude
        input_tensor = torch.zeros(1, n_features, device=device)
        input_tensor[0, feature_idx] = magnitude

        with torch.no_grad():
            # Get ResidMLP activations
            residmlp_acts = get_residmlp_activations(
                target_model, input_tensor, return_pre_activation=pre_activation
            )

            # Get the full model output
            model_output = target_model(input_tensor)  # Shape: (1, n_features)

            # Get pre-weight activations for ComponentModel
            _, pre_weight_acts = model(
                input_tensor, mode="input_cache", module_names=list(model.components.keys())
            )

            # Calculate causal importances
            ci_dict, _ = model.calc_causal_importances(
                pre_weight_acts=pre_weight_acts,
                sigmoid_type="leaky_hard",
                sampling="continuous",
                detach_inputs=True,
            )

            # Calculate pre-sigmoid gate outputs (ci_fn outputs before sigmoid)
            gate_outputs_dict = {}
            for param_name in pre_weight_acts:
                acts = pre_weight_acts[param_name]
                ci_fns = model.ci_fns[param_name]

                match ci_fns:
                    case MLPCiFn() | LinearCiFn():
                        ci_fn_input = model.components[param_name].get_inner_acts(acts)
                    case VectorMLPCiFn() | VectorSharedMLPCiFn():
                        ci_fn_input = acts
                    case _:
                        raise ValueError(f"Unknown ci_fn type: {type(ci_fns)}")

                if True:  # detach_inputs
                    ci_fn_input = ci_fn_input.detach()

                gate_outputs_dict[param_name] = ci_fns(ci_fn_input)

            # Store results
            for layer_name in layer_names:
                if layer_name in residmlp_acts:
                    activations[layer_name][step_idx] = residmlp_acts[layer_name][
                        0
                    ]  # [0] for batch dimension
                if layer_name in ci_dict:
                    causal_importances[layer_name][step_idx] = ci_dict[layer_name][
                        0
                    ]  # [0] for batch dimension
                if layer_name in gate_outputs_dict:
                    gate_outputs[layer_name][step_idx] = gate_outputs_dict[layer_name][
                        0
                    ]  # [0] for batch dimension
                # Store output response for this layer (same for all layers since it's the final output)
                output_responses[layer_name][step_idx] = model_output[0]  # [0] for batch dimension

                # Store gate inputs (inner acts) - need to compute these
                if layer_name in pre_weight_acts:
                    acts = pre_weight_acts[layer_name]
                    ci_fns = model.ci_fns[layer_name]

                    # Get gate input based on ci_fn type
                    match ci_fns:
                        case MLPCiFn() | LinearCiFn():
                            # For MLP/Linear CI functions, get_inner_acts returns (..., C) - one per component
                            gate_input = model.components[layer_name].get_inner_acts(acts)
                        case VectorMLPCiFn() | VectorSharedMLPCiFn():
                            # For Vector CI functions, all components use the same input
                            gate_input = acts
                        case _:
                            gate_input = acts  # Default fallback

                    # Initialize gate_inputs tensor if not already done
                    if gate_inputs[layer_name] is None:
                        gate_input_dim = gate_input.shape[-1]  # Get the last dimension
                        gate_inputs[layer_name] = torch.zeros(
                            n_steps, gate_input_dim, device=device
                        )

                    gate_inputs[layer_name][step_idx] = gate_input[0]  # [0] for batch dimension

    return activations, causal_importances, output_responses, gate_outputs, gate_inputs


def plot_unified_grid(
    activations: dict[str, Float[Tensor, "n_steps d_mlp"]],
    output_responses: dict[str, Float[Tensor, "n_steps n_features"]],
    causal_importances: dict[str, Float[Tensor, "n_steps n_components"]],
    gate_outputs: dict[str, Float[Tensor, "n_steps n_components"]],
    gate_inputs: dict[str, Float[Tensor, "n_steps d_in"]],
    magnitudes: Float[Tensor, "..."],
    feature_idx: int,
    output_dir: str = "magnitude_sweep_plots",
    figsize_per_subplot: tuple[float, float] = (2, 1.5),
    dpi: int = 150,
    ci_threshold: float = 0.1,
) -> None:
    """Create unified grid with neurons, output, causal importance functions, gate outputs, and gate inputs.

    Args:
        activations: Dictionary of activation data for each layer
        output_responses: Dictionary of output responses for each layer
        causal_importances: Dictionary of causal importance data for each layer
        gate_outputs: Dictionary of pre-sigmoid gate outputs for each layer
        gate_inputs: Dictionary of gate inputs (inner acts) for each layer
        magnitudes: Magnitude values for x-axis
        feature_idx: Which feature was activated
        output_dir: Directory to save plots
        figsize_per_subplot: Figure size per subplot
        dpi: DPI for figures
        ci_threshold: Threshold for considering a gate as "active"
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    magnitudes_np = magnitudes.detach().cpu().numpy()

    for layer_name in activations:
        print(f"Creating unified grid for {layer_name}...")

        # Get data
        acts = activations[layer_name].detach().cpu().numpy()  # Shape: (n_steps, d_mlp)
        outputs = (
            output_responses[layer_name].detach().cpu().numpy()
        )  # Shape: (n_steps, n_features)
        ci = causal_importances[layer_name].detach().cpu().numpy()  # Shape: (n_steps, n_components)
        gate_outs = (
            gate_outputs[layer_name].detach().cpu().numpy()
        )  # Shape: (n_steps, n_components)
        gate_ins = gate_inputs[layer_name].detach().cpu().numpy()  # Shape: (n_steps, d_in)

        _, d_mlp = acts.shape
        n_components = ci.shape[1]

        # Find active components
        max_ci_per_component = np.max(ci, axis=0)
        active_components = np.where(max_ci_per_component > ci_threshold)[0]

        print(f"  Found {len(active_components)} active components (out of {n_components})")

        # Calculate total number of subplots needed
        # Limit to reasonable number: 50 neurons + 1 output + max 10 active CI + max 10 gate outputs + gate inputs for active components only
        max_ci_plot = min(10, len(active_components))
        max_gate_inputs_plot = min(
            10, len(active_components)
        )  # Only plot gate inputs for active components
        total_subplots = d_mlp + 1 + max_ci_plot + max_ci_plot + max_gate_inputs_plot

        print(
            f"  Plotting: {d_mlp} neurons + 1 output + {max_ci_plot} CI + {max_ci_plot} gate outputs + {max_gate_inputs_plot} gate inputs (for active components) = {total_subplots} total subplots"
        )

        # Create grid layout (aim for roughly square grid)
        n_cols = int(np.ceil(np.sqrt(total_subplots)))
        n_rows = (total_subplots + n_cols - 1) // n_cols

        # Calculate figure size
        fig_width = n_cols * figsize_per_subplot[0]
        fig_height = n_rows * figsize_per_subplot[1]

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(fig_width, fig_height), dpi=dpi, sharex=True
        )

        # Ensure axes is 2D
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        subplot_idx = 0

        # Plot each neuron in its own subplot
        for neuron_idx in range(d_mlp):
            row = subplot_idx // n_cols
            col = subplot_idx % n_cols
            ax = axes[row, col]

            # Plot neuron activation
            ax.plot(magnitudes_np, acts[:, neuron_idx], "b-", linewidth=1.5, alpha=0.8)

            # Customize subplot
            ax.set_title(f"Neuron {neuron_idx}", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)

            # Set y-axis limits based on data range
            y_min, y_max = np.min(acts[:, neuron_idx]), np.max(acts[:, neuron_idx])
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

            subplot_idx += 1

        # Plot output dimension in its own subplot
        if subplot_idx < n_rows * n_cols:
            row = subplot_idx // n_cols
            col = subplot_idx % n_cols
            ax = axes[row, col]

            # Plot output response
            ax.plot(magnitudes_np, outputs[:, feature_idx], "r-", linewidth=1.5, alpha=0.8)

            # Customize subplot
            ax.set_title(f"Output Feature {feature_idx}", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)

            # Set y-axis limits
            y_min, y_max = np.min(outputs[:, feature_idx]), np.max(outputs[:, feature_idx])
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

            subplot_idx += 1

        # Plot each active causal importance component in its own subplot (limit to 10)
        for _, comp_idx in enumerate(active_components[:max_ci_plot]):
            if subplot_idx >= n_rows * n_cols:
                break

            row = subplot_idx // n_cols
            col = subplot_idx % n_cols
            ax = axes[row, col]

            # Plot causal importance
            ax.plot(magnitudes_np, ci[:, comp_idx], "g-", linewidth=1.5, alpha=0.8)

            # Customize subplot
            ax.set_title(f"CI Component {comp_idx}", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)

            # Set y-axis limits
            y_min, y_max = np.min(ci[:, comp_idx]), np.max(ci[:, comp_idx])
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

            subplot_idx += 1

        # Plot each active gate output (pre-sigmoid) in its own subplot (limit to 10)
        for _, comp_idx in enumerate(active_components[:max_ci_plot]):
            if subplot_idx >= n_rows * n_cols:
                break

            row = subplot_idx // n_cols
            col = subplot_idx % n_cols
            ax = axes[row, col]

            # Plot gate output
            ax.plot(magnitudes_np, gate_outs[:, comp_idx], "orange", linewidth=1.5, alpha=0.8)

            # Customize subplot
            ax.set_title(f"Gate Output {comp_idx}", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)

            # Set y-axis limits
            y_min, y_max = np.min(gate_outs[:, comp_idx]), np.max(gate_outs[:, comp_idx])
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

            subplot_idx += 1

        # Plot gate inputs (inner acts) for active components only (limit to 10)
        for _, comp_idx in enumerate(active_components[:max_gate_inputs_plot]):
            if subplot_idx >= n_rows * n_cols:
                break

            row = subplot_idx // n_cols
            col = subplot_idx % n_cols
            ax = axes[row, col]

            # Plot gate input for this active component
            if gate_ins.shape[1] == n_components:
                # MLPGates case: gate_ins has shape (n_steps, n_components) - one per component
                input_idx = comp_idx
                ax.plot(magnitudes_np, gate_ins[:, input_idx], "purple", linewidth=1.5, alpha=0.8)
                ax.set_title(f"Gate Input {input_idx} (CI {comp_idx})", fontsize=8)
            else:
                # VectorMLPGates case: gate_ins has shape (n_steps, d_in) - shared across components
                # Plot the first few dimensions of the shared input
                input_idx = comp_idx % gate_ins.shape[1]
                ax.plot(magnitudes_np, gate_ins[:, input_idx], "purple", linewidth=1.5, alpha=0.8)
                ax.set_title(f"Gate Input {input_idx} (Shared, CI {comp_idx})", fontsize=8)

            # Customize subplot
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)

            # Set y-axis limits
            y_min, y_max = np.min(gate_ins[:, input_idx]), np.max(gate_ins[:, input_idx])
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

            subplot_idx += 1

        # Hide unused subplots
        for i in range(subplot_idx, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)

        # Add labels to the entire figure
        fig.suptitle(
            f"{layer_name} - Unified Grid: Neurons, Output, and Active CI Components\n(Feature {feature_idx} active)",
            fontsize=12,
        )

        # Add x and y labels to the entire figure
        fig.text(0.5, 0.02, "Input Magnitude", ha="center", fontsize=10)
        fig.text(0.02, 0.5, "Activation Value", va="center", rotation="vertical", fontsize=10)

        # Add legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="blue", linewidth=1.5, label="Neuron Activations"),
            Line2D([0], [0], color="red", linewidth=1.5, label=f"Output Feature {feature_idx}"),
            Line2D([0], [0], color="green", linewidth=1.5, label="Active CI Components"),
            Line2D([0], [0], color="orange", linewidth=1.5, label="Gate Outputs (Pre-sigmoid)"),
            Line2D([0], [0], color="purple", linewidth=1.5, label="Gate Inputs (Inner Acts)"),
        ]
        fig.legend(handles=legend_elements, loc="upper right", fontsize=10)

        # Save plot
        safe_layer_name = layer_name.replace(".", "_")
        plot_filename = f"unified_grid_feature_{feature_idx}_{safe_layer_name}.png"
        plot_path = output_dir_path / plot_filename
        plt.savefig(plot_path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        print(f"  Saved unified grid: {plot_path}")


def main(config_path_or_obj: str | MagnitudeSweepConfig = None) -> None:
    """Main function for magnitude sweep plotting."""
    config = load_config(config_path_or_obj, config_model=MagnitudeSweepConfig)

    # Set up output directory
    if config.output_dir is None:
        base_output_dir = Path(__file__).parent / "out"
    else:
        base_output_dir = Path(config.output_dir)

    # Create model-specific subdirectory
    if "wandb:" in config.model_path:
        # Extract run ID from wandb URL (e.g., "wandb://entity/project/run_id" -> "run_id")
        model_id = config.model_path.split("/")[-1]
    elif "/wandb/" in config.model_path:
        # Extract run ID from local wandb path (e.g., "./wandb/6hk3uciu/files/model.pth" -> "6hk3uciu")
        path_parts = config.model_path.split("/")
        wandb_idx = path_parts.index("wandb")
        if wandb_idx + 1 < len(path_parts):
            model_id = path_parts[wandb_idx + 1]
        else:
            model_id = Path(config.model_path).stem
    else:
        # For other local paths, use filename without extension
        model_id = Path(config.model_path).stem

    output_dir = base_output_dir / model_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get device
    device = get_device() if config.device == "auto" else config.device
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from: {config.model_path}")
    try:
        run_info = SPDRunInfo.from_path(config.model_path)
        model = ComponentModel.from_run_info(run_info)
        model.to(device)
        model.eval()
        logger.info(f"Successfully loaded model with {len(model.components)} component modules")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    # Get n_features from model config
    target_model = model.target_model
    assert isinstance(target_model, ResidMLP), "Model must be a ResidMLP"
    n_features = target_model.config.n_features
    logger.info(f"Using n_features: {n_features}")
    logger.info(f"Model config: {target_model.config}")

    # Validate feature index
    if config.feature_idx >= n_features:
        raise ValueError(f"feature_idx {config.feature_idx} >= n_features {n_features}")

    # Compute magnitude sweep data
    logger.info(f"Computing magnitude sweep for feature {config.feature_idx}...")
    activations, causal_importances, output_responses, gate_outputs, gate_inputs = (
        compute_magnitude_sweep_data(
            model=model,
            device=device,
            n_features=n_features,
            feature_idx=config.feature_idx,
            n_steps=config.n_steps,
            max_magnitude=config.max_magnitude,
            pre_activation=config.pre_activation,
        )
    )

    # Create magnitude array for plotting (symmetric range)
    magnitudes = torch.linspace(
        -config.max_magnitude, config.max_magnitude, config.n_steps, device=device
    )

    # Create plots
    logger.info("Creating unified grid plots...")
    plot_unified_grid(
        activations=activations,
        output_responses=output_responses,
        causal_importances=causal_importances,
        gate_outputs=gate_outputs,
        gate_inputs=gate_inputs,
        magnitudes=magnitudes,
        feature_idx=config.feature_idx,
        output_dir=str(output_dir),
        figsize_per_subplot=config.figsize_per_subplot,
        dpi=config.dpi,
        ci_threshold=config.ci_threshold,
    )

    logger.info(f"All plots saved to: {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
