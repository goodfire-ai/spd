#!/usr/bin/env python3
"""
Plot ResidMLP neuron activations and causal importance as input magnitude increases.

This script creates plots showing how individual neuron activations and causal importance
values respond as we gradually increase the magnitude of a one-hot input vector from 0 to max_magnitude.

The x-axis represents the input magnitude, and the y-axes show:
1. Individual neuron activations in the ResidMLP layers
2. Causal importance function values for gates that actually activate
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from spd.experiments.resid_mlp.models import ResidMLP
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device


def get_residmlp_activations(
    model: ResidMLP, 
    input_tensor: Float[Tensor, "batch n_features"],
    return_intermediate: bool = True,
    return_pre_activation: bool = False
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
        mid_pre_act = layer.mlp_in(residual)
        
        # Apply activation function
        mid_act = model.act_fn(mid_pre_act)
        
        # Store activations for this layer
        if return_intermediate:
            if return_pre_activation:
                activations[f"layers.{i}.mlp_in"] = mid_pre_act  # Before ReLU
            else:
                activations[f"layers.{i}.mlp_in"] = mid_act  # After ReLU
        
        # Get output and add to residual
        out = layer.mlp_out(mid_act)
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
) -> tuple[dict[str, Float[Tensor, "n_steps d_mlp"]], dict[str, Float[Tensor, "n_steps n_components"]], dict[str, Float[Tensor, "n_steps n_features"]], dict[str, Float[Tensor, "n_steps n_components"]], dict[str, Float[Tensor, "n_steps d_in"]]]:
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
        d_in = model.components[layer_name].V.shape[0]  # Input dimension for gates
        activations[layer_name] = torch.zeros(n_steps, d_mlp, device=device)
        causal_importances[layer_name] = torch.zeros(n_steps, n_components, device=device)
        output_responses[layer_name] = torch.zeros(n_steps, n_features, device=device)
        gate_outputs[layer_name] = torch.zeros(n_steps, n_components, device=device)
        gate_inputs[layer_name] = torch.zeros(n_steps, d_in, device=device)
    
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
            residmlp_acts = get_residmlp_activations(target_model, input_tensor, return_pre_activation=pre_activation)
            
            # Get the full model output
            model_output = target_model(input_tensor)  # Shape: (1, n_features)
            
            # Get pre-weight activations for ComponentModel
            _, pre_weight_acts = model(
                input_tensor,
                mode="input_cache",
                module_names=list(model.components.keys())
            )
            
            # Calculate causal importances
            ci_dict, _ = model.calc_causal_importances(
                pre_weight_acts=pre_weight_acts,
                sigmoid_type='leaky_hard',
                sampling='continuous',
                detach_inputs=True
            )
            
            # Calculate pre-sigmoid gate outputs
            gate_outputs_dict = model.calc_pre_sigmoid_gate_outputs(
                pre_weight_acts=pre_weight_acts,
                detach_inputs=True
            )
            
            # Store results
            for layer_name in layer_names:
                if layer_name in residmlp_acts:
                    activations[layer_name][step_idx] = residmlp_acts[layer_name][0]  # [0] for batch dimension
                if layer_name in ci_dict:
                    causal_importances[layer_name][step_idx] = ci_dict[layer_name][0]  # [0] for batch dimension
                if layer_name in gate_outputs_dict:
                    gate_outputs[layer_name][step_idx] = gate_outputs_dict[layer_name][0]  # [0] for batch dimension
                # Store output response for this layer (same for all layers since it's the final output)
                output_responses[layer_name][step_idx] = model_output[0]  # [0] for batch dimension
                
                # Store gate inputs (inner acts) - need to compute these
                if layer_name in pre_weight_acts:
                    acts = pre_weight_acts[layer_name]
                    gates = model.gates[layer_name]
                    
                    # Get gate input based on gate type
                    try:
                        from spd.models.gates import MLPGates, VectorMLPGates, VectorSharedMLPGate
                        match gates:
                            case MLPGates():
                                # For MLPGates, get_inner_acts returns (..., C) - one per component
                                gate_input = model.components[layer_name].get_inner_acts(acts)
                            case VectorMLPGates() | VectorSharedMLPGate():
                                # For Vector gates, all components use the same input
                                gate_input = acts
                            case _:
                                gate_input = acts  # Default fallback
                    except ImportError:
                        # Fallback if gates module not available
                        gate_input = acts
                    
                    gate_inputs[layer_name][step_idx] = gate_input[0]  # [0] for batch dimension
    
    return activations, causal_importances, output_responses, gate_outputs, gate_inputs


def plot_unified_grid(
    activations: dict[str, Float[Tensor, "n_steps d_mlp"]],
    output_responses: dict[str, Float[Tensor, "n_steps n_features"]],
    causal_importances: dict[str, Float[Tensor, "n_steps n_components"]],
    gate_outputs: dict[str, Float[Tensor, "n_steps n_components"]],
    gate_inputs: dict[str, Float[Tensor, "n_steps d_in"]],
    magnitudes: Float[Tensor, "n_steps"],
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    magnitudes_np = magnitudes.detach().cpu().numpy()
    
    for layer_name in activations:
        print(f"Creating unified grid for {layer_name}...")
        
        # Get data
        acts = activations[layer_name].detach().cpu().numpy()  # Shape: (n_steps, d_mlp)
        outputs = output_responses[layer_name].detach().cpu().numpy()  # Shape: (n_steps, n_features)
        ci = causal_importances[layer_name].detach().cpu().numpy()  # Shape: (n_steps, n_components)
        gate_outs = gate_outputs[layer_name].detach().cpu().numpy()  # Shape: (n_steps, n_components)
        gate_ins = gate_inputs[layer_name].detach().cpu().numpy()  # Shape: (n_steps, d_in)
        
        n_steps, d_mlp = acts.shape
        n_features = outputs.shape[1]
        n_components = ci.shape[1]
        d_in = gate_ins.shape[1]
        
        # Find active components
        max_ci_per_component = np.max(ci, axis=0)
        active_components = np.where(max_ci_per_component > ci_threshold)[0]
        
        print(f"  Found {len(active_components)} active components (out of {n_components})")
        
        # Calculate total number of subplots needed
        # Limit to reasonable number: 50 neurons + 1 output + max 10 active CI + max 10 gate outputs + gate inputs for active components only
        max_ci_plot = min(10, len(active_components))
        max_gate_inputs_plot = min(10, len(active_components))  # Only plot gate inputs for active components
        total_subplots = d_mlp + 1 + max_ci_plot + max_ci_plot + max_gate_inputs_plot
        
        print(f"  Plotting: {d_mlp} neurons + 1 output + {max_ci_plot} CI + {max_ci_plot} gate outputs + {max_gate_inputs_plot} gate inputs (for active components) = {total_subplots} total subplots")
        
        # Create grid layout (aim for roughly square grid)
        n_cols = int(np.ceil(np.sqrt(total_subplots)))
        n_rows = (total_subplots + n_cols - 1) // n_cols
        
        # Calculate figure size
        fig_width = n_cols * figsize_per_subplot[0]
        fig_height = n_rows * figsize_per_subplot[1]
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), dpi=dpi, sharex=True)
        
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
            ax.plot(magnitudes_np, acts[:, neuron_idx], 'b-', linewidth=1.5, alpha=0.8)
            
            # Customize subplot
            ax.set_title(f'Neuron {neuron_idx}', fontsize=8)
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
            ax.plot(magnitudes_np, outputs[:, feature_idx], 'r-', linewidth=1.5, alpha=0.8)
            
            # Customize subplot
            ax.set_title(f'Output Feature {feature_idx}', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)
            
            # Set y-axis limits
            y_min, y_max = np.min(outputs[:, feature_idx]), np.max(outputs[:, feature_idx])
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            
            subplot_idx += 1
        
        # Plot each active causal importance component in its own subplot (limit to 10)
        for i, comp_idx in enumerate(active_components[:max_ci_plot]):
            if subplot_idx >= n_rows * n_cols:
                break
                
            row = subplot_idx // n_cols
            col = subplot_idx % n_cols
            ax = axes[row, col]
            
            # Plot causal importance
            ax.plot(magnitudes_np, ci[:, comp_idx], 'g-', linewidth=1.5, alpha=0.8)
            
            # Customize subplot
            ax.set_title(f'CI Component {comp_idx}', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)
            
            # Set y-axis limits
            y_min, y_max = np.min(ci[:, comp_idx]), np.max(ci[:, comp_idx])
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            
            subplot_idx += 1
        
        # Plot each active gate output (pre-sigmoid) in its own subplot (limit to 10)
        for i, comp_idx in enumerate(active_components[:max_ci_plot]):
            if subplot_idx >= n_rows * n_cols:
                break
                
            row = subplot_idx // n_cols
            col = subplot_idx % n_cols
            ax = axes[row, col]
            
            # Plot gate output
            ax.plot(magnitudes_np, gate_outs[:, comp_idx], 'orange', linewidth=1.5, alpha=0.8)
            
            # Customize subplot
            ax.set_title(f'Gate Output {comp_idx}', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=6)
            
            # Set y-axis limits
            y_min, y_max = np.min(gate_outs[:, comp_idx]), np.max(gate_outs[:, comp_idx])
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            
            subplot_idx += 1
        
        # Plot gate inputs (inner acts) for active components only (limit to 10)
        for i, comp_idx in enumerate(active_components[:max_gate_inputs_plot]):
            if subplot_idx >= n_rows * n_cols:
                break
                
            row = subplot_idx // n_cols
            col = subplot_idx % n_cols
            ax = axes[row, col]
            
            # Plot gate input for this active component
            if gate_ins.shape[1] == n_components:
                # MLPGates case: gate_ins has shape (n_steps, n_components) - one per component
                input_idx = comp_idx
                ax.plot(magnitudes_np, gate_ins[:, input_idx], 'purple', linewidth=1.5, alpha=0.8)
                ax.set_title(f'Gate Input {input_idx} (CI {comp_idx})', fontsize=8)
            else:
                # VectorMLPGates case: gate_ins has shape (n_steps, d_in) - shared across components
                # Plot the first few dimensions of the shared input
                input_idx = comp_idx % gate_ins.shape[1]
                ax.plot(magnitudes_np, gate_ins[:, input_idx], 'purple', linewidth=1.5, alpha=0.8)
                ax.set_title(f'Gate Input {input_idx} (Shared, CI {comp_idx})', fontsize=8)
            
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
        fig.suptitle(f'{layer_name} - Unified Grid: Neurons, Output, and Active CI Components\n(Feature {feature_idx} active)', fontsize=12)
        
        # Add x and y labels to the entire figure
        fig.text(0.5, 0.02, 'Input Magnitude', ha='center', fontsize=10)
        fig.text(0.02, 0.5, 'Activation Value', va='center', rotation='vertical', fontsize=10)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=1.5, label='Neuron Activations'),
            Line2D([0], [0], color='red', linewidth=1.5, label=f'Output Feature {feature_idx}'),
            Line2D([0], [0], color='green', linewidth=1.5, label='Active CI Components'),
            Line2D([0], [0], color='orange', linewidth=1.5, label='Gate Outputs (Pre-sigmoid)'),
            Line2D([0], [0], color='purple', linewidth=1.5, label='Gate Inputs (Inner Acts)')
        ]
        fig.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Save plot
        safe_layer_name = layer_name.replace('.', '_')
        plot_filename = f"unified_grid_feature_{feature_idx}_{safe_layer_name}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        print(f"  Saved unified grid: {plot_path}")


def plot_magnitude_sweep(
    activations: dict[str, Float[Tensor, "n_steps d_mlp"]],
    causal_importances: dict[str, Float[Tensor, "n_steps n_components"]],
    magnitudes: Float[Tensor, "n_steps"],
    feature_idx: int,
    output_dir: str = "magnitude_sweep_plots",
    figsize: tuple[float, float] = (15, 10),
    dpi: int = 150,
    max_neurons_per_plot: int = 50,
    ci_threshold: float = 0.1,
) -> None:
    """Create plots showing activations and causal importance vs input magnitude.
    
    Args:
        activations: Dictionary of activation data
        causal_importances: Dictionary of causal importance data
        magnitudes: Magnitude values for x-axis
        feature_idx: Which feature was activated
        output_dir: Directory to save plots
        figsize: Figure size
        dpi: DPI for figures
        max_neurons_per_plot: Maximum number of neurons to show per plot
        ci_threshold: Threshold for considering a gate as "active"
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    magnitudes_np = magnitudes.detach().cpu().numpy()
    
    for layer_name in activations:
        print(f"Creating plots for {layer_name}...")
        
        # Get data
        acts = activations[layer_name].detach().cpu().numpy()  # Shape: (n_steps, d_mlp)
        ci = causal_importances[layer_name].detach().cpu().numpy()  # Shape: (n_steps, n_components)
        
        n_steps, d_mlp = acts.shape
        n_components = ci.shape[1]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)
        
        # Plot 1: Neuron Activations
        ax1.set_title(f"{layer_name} - Neuron Activations vs Input Magnitude\n(Feature {feature_idx} active)")
        
        # Plot a subset of neurons to avoid overcrowding
        neuron_indices = np.linspace(0, d_mlp-1, min(max_neurons_per_plot, d_mlp), dtype=int)
        
        for i, neuron_idx in enumerate(neuron_indices):
            color = plt.cm.viridis(i / len(neuron_indices))
            ax1.plot(
                magnitudes_np, 
                acts[:, neuron_idx], 
                color=color, 
                alpha=0.7, 
                linewidth=1,
                label=f"Neuron {neuron_idx}" if i < 10 else None  # Only label first 10
            )
        
        ax1.set_ylabel("Neuron Activation")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, loc='upper left')
        
        # Plot 2: Causal Importance (only for active gates)
        ax2.set_title(f"{layer_name} - Causal Importance vs Input Magnitude\n(Only gates with CI > {ci_threshold} shown)")
        
        # Find components that are active at any point
        max_ci_per_component = np.max(ci, axis=0)
        active_components = np.where(max_ci_per_component > ci_threshold)[0]
        
        print(f"  Found {len(active_components)} active components (out of {n_components})")
        
        if len(active_components) > 0:
            for i, comp_idx in enumerate(active_components):
                color = plt.cm.plasma(i / len(active_components))
                ax2.plot(
                    magnitudes_np, 
                    ci[:, comp_idx], 
                    color=color, 
                    alpha=0.8, 
                    linewidth=2,
                    label=f"Component {comp_idx}" if i < 15 else None  # Only label first 15
                )
            
            ax2.legend(fontsize=8, loc='upper left')
        else:
            ax2.text(0.5, 0.5, f"No components with CI > {ci_threshold}", 
                    transform=ax2.transAxes, ha='center', va='center')
        
        ax2.set_xlabel("Input Magnitude")
        ax2.set_ylabel("Causal Importance")
        ax2.grid(True, alpha=0.3)
        
        # Set consistent x-axis limits
        ax1.set_xlim(0, magnitudes_np.max())
        ax2.set_xlim(0, magnitudes_np.max())
        
        # Add some statistics
        max_act = np.max(acts)
        max_ci = np.max(ci)
        fig.suptitle(f"Max Activation: {max_act:.3f}, Max CI: {max_ci:.3f}", fontsize=10)
        
        # Save plot
        safe_layer_name = layer_name.replace('.', '_')
        plot_filename = f"magnitude_sweep_feature_{feature_idx}_{safe_layer_name}.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        print(f"  Saved plot: {plot_path}")


def create_summary_plot(
    activations: dict[str, Float[Tensor, "n_steps d_mlp"]],
    causal_importances: dict[str, Float[Tensor, "n_steps n_components"]],
    magnitudes: Float[Tensor, "n_steps"],
    feature_idx: int,
    output_dir: str = "magnitude_sweep_plots",
    figsize: tuple[float, float] = (20, 12),
    dpi: int = 150,
    ci_threshold: float = 0.1,
) -> None:
    """Create a summary plot showing all layers in a grid layout."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    magnitudes_np = magnitudes.detach().cpu().numpy()
    layer_names = list(activations.keys())
    n_layers = len(layer_names)
    
    # Create subplot grid
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi, sharex=True)
    
    # Ensure axes is always a 2D array for consistent indexing
    if n_layers == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 and n_cols > 1:
        axes = axes.reshape(1, -1)
    elif n_rows > 1 and n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, layer_name in enumerate(layer_names):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Get data
        acts = activations[layer_name].detach().cpu().numpy()
        ci = causal_importances[layer_name].detach().cpu().numpy()
        
        # Plot mean activation and mean causal importance (only for active gates)
        mean_acts = np.mean(acts, axis=1)
        
        # Find active components (those with CI > threshold at any point)
        max_ci_per_component = np.max(ci, axis=0)
        active_components = np.where(max_ci_per_component > ci_threshold)[0]
        
        if len(active_components) > 0:
            mean_ci = np.mean(ci[:, active_components], axis=1)
        else:
            mean_ci = np.zeros_like(mean_acts)
        
        ax2 = ax.twinx()
        
        # Plot activations
        line1 = ax.plot(magnitudes_np, mean_acts, 'b-', alpha=0.7, linewidth=2, label='Mean Activation')
        ax.set_ylabel('Mean Neuron Activation', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        
        # Plot causal importance
        if len(active_components) > 0:
            line2 = ax2.plot(magnitudes_np, mean_ci, 'r-', alpha=0.7, linewidth=2, label=f'Mean CI (Active Gates, n={len(active_components)})')
        else:
            line2 = ax2.plot(magnitudes_np, mean_ci, 'r-', alpha=0.7, linewidth=2, label='Mean CI (No Active Gates)')
        ax2.set_ylabel('Mean Causal Importance', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        ax.set_title(f"{layer_name}")
        ax.set_xlabel("Input Magnitude")
        ax.grid(True, alpha=0.3)
        
        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=8)
    
    # Hide unused subplots
    for i in range(n_layers, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)
    
    fig.suptitle(f"Summary: Mean Activations and Causal Importance vs Input Magnitude\n(Feature {feature_idx} active)", fontsize=14)
    
    # Save summary plot
    plot_filename = f"magnitude_sweep_summary_feature_{feature_idx}.png"
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print(f"Saved summary plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot ResidMLP magnitude sweep")
    parser.add_argument("model_path", help="Path to the trained SPD model (wandb:project/run_id or local path)")
    parser.add_argument("--output-dir", default="spd/scripts/results/magnitude_sweep_plots", help="Output directory for plots")
    parser.add_argument("--device", default="auto", help="Device to use (default: auto)")
    parser.add_argument("--feature-idx", type=int, default=0, help="Which feature to activate (default: 0)")
    parser.add_argument("--n-steps", type=int, default=100, help="Number of steps from 0 to max_magnitude (default: 100)")
    parser.add_argument("--max-magnitude", type=float, default=2.0, help="Maximum input magnitude (default: 2.0)")
    parser.add_argument("--figsize", nargs=2, type=float, default=[15, 10], help="Figure size (width height)")
    parser.add_argument("--figsize-per-subplot", nargs=2, type=float, default=[2, 1.5], help="Figure size per subplot (width height)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for figures (default: 150)")
    parser.add_argument("--max-neurons", type=int, default=50, help="Max neurons per plot (default: 50)")
    parser.add_argument("--ci-threshold", type=float, default=0.1, help="CI threshold for active gates (default: 0.1)")
    parser.add_argument("--summary-only", action="store_true", help="Only create summary plot")
    parser.add_argument("--pre-activation", action="store_true", help="Show pre-activation values (before ReLU) instead of post-activation")
    
    args = parser.parse_args()
    
    # Get device
    if args.device == "auto":
        device = get_device()
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    try:
        run_info = SPDRunInfo.from_path(args.model_path)
        model = ComponentModel.from_run_info(run_info)
        model.to(device)
        model.eval()
        print(f"Successfully loaded model with {len(model.components)} component modules")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Get n_features from model config
    target_model = model.target_model
    assert isinstance(target_model, ResidMLP), "Model must be a ResidMLP"
    n_features = target_model.config.n_features
    print(f"Using n_features: {n_features}")
    print(f"Model config: {target_model.config}")
    
    # Validate feature index
    if args.feature_idx >= n_features:
        print(f"Error: feature_idx {args.feature_idx} >= n_features {n_features}")
        return 1
    
    # Compute magnitude sweep data
    print(f"\nComputing magnitude sweep for feature {args.feature_idx}...")
    activations, causal_importances, output_responses, gate_outputs, gate_inputs = compute_magnitude_sweep_data(
        model=model,
        device=device,
        n_features=n_features,
        feature_idx=args.feature_idx,
        n_steps=args.n_steps,
        max_magnitude=args.max_magnitude,
        pre_activation=args.pre_activation,
    )
    
    # Create magnitude array for plotting (symmetric range)
    magnitudes = torch.linspace(-args.max_magnitude, args.max_magnitude, args.n_steps, device=device)
    
    # Create plots
    if not args.summary_only:
        print("\nCreating detailed plots...")
        plot_unified_grid(
            activations=activations,
            output_responses=output_responses,
            causal_importances=causal_importances,
            gate_outputs=gate_outputs,
            gate_inputs=gate_inputs,
            magnitudes=magnitudes,
            feature_idx=args.feature_idx,
            output_dir=args.output_dir,
            figsize_per_subplot=args.figsize_per_subplot,
            dpi=args.dpi,
            ci_threshold=args.ci_threshold,
        )
    
    print("\nCreating summary plot...")
    create_summary_plot(
        activations=activations,
        causal_importances=causal_importances,
        magnitudes=magnitudes,
        feature_idx=args.feature_idx,
        output_dir=args.output_dir,
        figsize=(20, 12),
        dpi=args.dpi,
        ci_threshold=args.ci_threshold,
    )
    
    print(f"\nAll plots saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
