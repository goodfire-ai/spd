#!/usr/bin/env python3
"""
Plot causal importance functions as input magnitude increases.

This script creates plots showing how each component's causal importance
responds as we gradually increase one input dimension from 0 to 1.
"""

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device


def compute_causal_importance_sweep(
    model: ComponentModel,
    device: str,
    n_features: int,
    n_steps: int = 50,
    max_magnitude: float = 1.0,
) -> dict[str, Float[Tensor, "n_steps n_features n_components"]]:
    """Compute causal importance for each input dimension as magnitude increases.
    
    Args:
        model: The trained ComponentModel
        device: Device to run on
        n_features: Number of input features
        n_steps: Number of steps from 0 to max_magnitude
        max_magnitude: Maximum input magnitude
        
    Returns:
        Dictionary mapping module names to causal importance tensors
        Shape: (n_steps, n_features, n_components)
    """
    model.eval()
    
    # Create magnitude steps
    magnitudes = torch.linspace(0, max_magnitude, n_steps, device=device)
    
    causal_importances = {}
    
    for module_name, component in model.components.items():
        print(f"Computing causal importance sweep for {module_name}...")
        
        # Initialize tensor to store results
        n_components = component.U.shape[0]
        ci_sweep = torch.zeros(n_steps, n_features, n_components, device=device)
        
        # For each input dimension
        for feature_idx in range(n_features):
            # Create input with zeros except for current feature
            input_tensor = torch.zeros(n_features, device=device)
            
            # For each magnitude step
            for step_idx, magnitude in enumerate(magnitudes):
                input_tensor[feature_idx] = magnitude
                
                # Reshape for model input (add batch dimension)
                input_batch = input_tensor.unsqueeze(0)  # Shape: (1, n_features)
                
                # Compute pre-weight activations and causal importances
                with torch.no_grad():
                    _, pre_weight_acts = model(
                        input_batch,
                        mode="input_cache", 
                        module_names=[module_name]
                    )
                    
                    # Compute causal importances for this module
                    ci_dict, _ = model.calc_causal_importances(
                        pre_weight_acts=pre_weight_acts,
                        sigmoid_type=model.config.sigmoid_type if hasattr(model, 'config') else 'leaky_hard',
                        sampling='continuous',
                        detach_inputs=True
                    )
                    
                    # Store causal importance (gate activation values)
                    ci_sweep[step_idx, feature_idx, :] = ci_dict[module_name][0]  # [0] for batch dimension
        
        causal_importances[module_name] = ci_sweep
        print(f"  Shape: {ci_sweep.shape}")
    
    return causal_importances


def plot_causal_importance_sweep(
    ci_sweep: dict[str, Float[Tensor, "n_steps n_features n_components"]],
    output_dir: str = "causal_importance_sweep_plots",
    figsize: tuple[float, float] = (16, 12),
    dpi: int = 150,
    max_components_per_subplot: int = 20,
) -> None:
    """Create one plot per layer showing causal importance sweep with subplots.
    
    Args:
        ci_sweep: Dictionary of causal importance sweep data
        output_dir: Directory to save plots
        figsize: Figure size
        dpi: DPI for figures
        max_components_per_subplot: Maximum number of components to show per subplot
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group modules by layer
    layer_modules = {}
    for module_name, ci_data in ci_sweep.items():
        # Extract layer number from module name (e.g., "layers.0.mlp_in" -> 0)
        layer_num = int(module_name.split('.')[1])
        if layer_num not in layer_modules:
            layer_modules[layer_num] = {}
        layer_modules[layer_num][module_name] = ci_data
    
    for layer_num, modules in layer_modules.items():
        print(f"Creating plot for layer {layer_num}...")
        
        # Create subplots for this layer
        n_modules = len(modules)
        fig, axes = plt.subplots(
            n_modules, 1,
            figsize=(figsize[0], figsize[1] * n_modules),
            dpi=dpi,
            sharex=True
        )
        
        if n_modules == 1:
            axes = [axes]
        
        for module_idx, (module_name, ci_data) in enumerate(modules.items()):
            ax = axes[module_idx]
            
            # ci_data has shape (n_steps, n_features, n_components)
            n_steps, n_features, n_components = ci_data.shape
            
            # Convert to numpy for plotting
            ci_np = ci_data.detach().cpu().numpy()
            
            # Create magnitude steps for x-axis
            magnitudes = np.linspace(0, 1.0, n_steps)
            
            # Plot a subset of components to avoid overcrowding
            n_components_to_plot = min(max_components_per_subplot, n_components)
            component_indices = np.linspace(0, n_components-1, n_components_to_plot, dtype=int)
            
            # Plot each selected component
            for comp_idx in component_indices:
                # Get data for this component across all features and steps
                comp_data = ci_np[:, :, comp_idx]  # Shape: (n_steps, n_features)
                
                # Plot each input feature as a separate line
                for feature_idx in range(n_features):
                    ax.plot(
                        magnitudes,
                        comp_data[:, feature_idx],
                        alpha=0.6,
                        linewidth=0.8,
                        label=f"Feature {feature_idx}" if comp_idx == component_indices[0] else None
                    )
            
            # Customize subplot
            ax.set_ylabel(f"{module_name}\nCausal Importance")
            ax.grid(True, alpha=0.3)
            
            # Add legend only for first subplot
            if module_idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                if len(handles) > 10:  # Too many features, show subset
                    step = max(1, len(handles) // 10)
                    ax.legend(handles[::step], labels[::step], loc='upper right', ncol=2, fontsize=8)
                else:
                    ax.legend(loc='upper right', fontsize=8)
        
        # Set common x-label
        axes[-1].set_xlabel("Input Magnitude")
        
        # Set main title
        fig.suptitle(f"Layer {layer_num} - Causal Importance vs Input Magnitude\n(Showing {n_components_to_plot} components per module)", fontsize=14)
        
        # Save plot
        plot_filename = f"layer_{layer_num}_causal_importance_sweep.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        print(f"  Saved plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot causal importance sweep")
    parser.add_argument("model_path", help="Path to the trained SPD model (wandb:project/run_id or local path)")
    parser.add_argument("--output-dir", default="spd/scripts/results/causal_importance_sweep_plots", help="Output directory for plots")
    parser.add_argument("--device", default="auto", help="Device to use (default: auto)")
    parser.add_argument("--n-steps", type=int, default=50, help="Number of steps from 0 to 1 (default: 50)")
    parser.add_argument("--max-magnitude", type=float, default=1.0, help="Maximum input magnitude (default: 1.0)")
    parser.add_argument("--figsize", nargs=2, type=float, default=[16, 12], help="Figure size (width height)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for figures (default: 150)")
    parser.add_argument("--max-components", type=int, default=20, help="Max components per subplot (default: 20)")
    
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
    n_features = model.target_model.config.n_features
    print(f"Using n_features: {n_features}")
    
    # Compute causal importance sweep
    print("\nComputing causal importance sweep...")
    ci_sweep = compute_causal_importance_sweep(
        model=model,
        device=device,
        n_features=n_features,
        n_steps=args.n_steps,
        max_magnitude=args.max_magnitude,
    )
    
    # Create plots
    print("\nCreating layer-based plots...")
    plot_causal_importance_sweep(
        ci_sweep=ci_sweep,
        output_dir=args.output_dir,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        max_components_per_subplot=args.max_components,
    )
    
    print(f"\nAll plots saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
