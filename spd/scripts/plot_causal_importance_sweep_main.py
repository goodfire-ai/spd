#!/usr/bin/env python3
"""
Plot causal importance functions as input magnitude increases.

This script creates plots showing how each component's causal importance
responds as we gradually increase one input dimension from 0 to 1.
"""

import argparse
from pathlib import Path

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
    figsize: tuple[float, float] = (20, 20),
    dpi: int = 150,
    max_components_per_plot: int = 20,
) -> None:
    """Create 10x10 grid plots for each module showing causal importance sweep.
    
    Each cell in the grid represents one input dimension, and contains
    multiple lines (one per component) showing how causal importance changes
    as input magnitude increases.
    
    Args:
        ci_sweep: Dictionary of causal importance sweep data
        output_dir: Directory to save plots
        figsize: Figure size for each grid
        dpi: DPI for figures
        max_components_per_plot: Maximum number of components to show per plot
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for module_name, ci_data in ci_sweep.items():
        print(f"Creating 10x10 grid plot for {module_name}...")
        
        # ci_data has shape (n_steps, n_features, n_components)
        n_steps, n_features, n_components = ci_data.shape
        
        # Convert to numpy for plotting
        ci_np = ci_data.detach().cpu().numpy()
        
        # Create magnitude steps for x-axis
        magnitudes = np.linspace(0, 1.0, n_steps)
        
        # Create 10x10 grid of subplots
        fig, axes = plt.subplots(10, 10, figsize=figsize, dpi=dpi)
        axes = axes.flatten()  # Flatten to 1D for easier indexing
        
        # Plot ALL components to see the full picture
        component_indices = np.arange(n_components)
        
        # Create colormap for components
        cmap = plt.cm.cool
        colors = [cmap(i / (n_components - 1)) for i in range(n_components)]
        
        # For each input dimension (feature)
        for feature_idx in range(min(100, n_features)):  # Limit to 100 features for 10x10 grid
            ax = axes[feature_idx]
            
            # Plot each component for this input dimension
            for comp_idx in component_indices:
                # Get data for this component and this feature across all steps
                comp_data = ci_np[:, feature_idx, comp_idx]  # Shape: (n_steps,)
                
                ax.plot(
                    magnitudes,
                    comp_data,
                    color=colors[comp_idx],
                    alpha=0.6,
                    linewidth=0.5,
                    label=f"Comp {comp_idx}" if feature_idx == 0 else None
                )
            
            # Customize subplot
            ax.set_title(f"Input {feature_idx}", fontsize=8)
            ax.tick_params(labelsize=6)
            # Remove gridlines and axis labels
            
            # Set y-axis limits to be consistent
            ax.set_ylim(0, 1)
        
        # Hide unused subplots
        for i in range(min(100, n_features), 100):
            axes[i].set_visible(False)
        
        # Add legend only once - show subset of components to avoid overcrowding
        handles, labels = axes[0].get_legend_handles_labels()
        if len(handles) > 20:  # Too many components, show subset
            step = max(1, len(handles) // 20)
            fig.legend(handles[::step], labels[::step], loc='upper right', fontsize=6, ncol=2)
        else:
            fig.legend(handles, labels, loc='upper right', fontsize=6, ncol=2)
        
        # Set main title
        fig.suptitle(f"{module_name} - Causal Importance vs Input Magnitude\n(10x10 grid: each cell = one input dimension, lines = components)", fontsize=14)
        
        # Save plot
        plot_filename = f"{module_name.replace('.', '_')}_causal_importance_sweep.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        print(f"  Saved plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot causal importance sweep")
    parser.add_argument("model_path", help="Path to the trained SPD model (wandb:project/run_id or local path)")
    parser.add_argument("--output-dir", default="spd/scripts/results/causal_importance_sweep_plots", help="Output directory for plots")
    parser.add_argument("--device", default="auto", help="Device to use (default: auto)")
    parser.add_argument("--n-steps", type=int, default=200, help="Number of steps from 0 to 1 (default: 200)")
    parser.add_argument("--max-magnitude", type=float, default=1.0, help="Maximum input magnitude (default: 1.0)")
    parser.add_argument("--figsize", nargs=2, type=float, default=[25, 25], help="Figure size (width height)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for figures (default: 150)")
    parser.add_argument("--max-components", type=int, default=20, help="Max components per plot (default: 20)")
    
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
        max_components_per_plot=args.max_components,
    )
    
    print(f"\nAll plots saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())