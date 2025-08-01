#!/usr/bin/env python3
"""Test GPU memory leak specifically with the plotting functions from spd.plotting."""

import gc
import io
import time
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Use non-interactive backend
matplotlib.use('Agg')


def get_detailed_gpu_memory():
    """Get detailed GPU memory statistics."""
    if not torch.cuda.is_available():
        return {}
    
    return {
        'allocated': torch.cuda.memory_allocated() / 1024 / 1024,
        'reserved': torch.cuda.memory_reserved() / 1024 / 1024,
        'max_allocated': torch.cuda.max_memory_allocated() / 1024 / 1024,
        'max_reserved': torch.cuda.max_memory_reserved() / 1024 / 1024,
    }


def _render_figure(fig: plt.Figure) -> Image.Image:
    """Render figure to PIL Image (same as in spd.plotting)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    buf.close()
    return img


def test_plot_with_tensors(use_log_scale: bool, n_iterations: int = 50):
    """Test plotting with PyTorch tensors, mimicking the actual plotting functions."""
    print(f"\n{'='*60}")
    print(f"Testing tensor plotting with log_scale={use_log_scale}")
    print(f"{'='*60}")
    
    memory_history = []
    
    # Initial state
    gc.collect()
    torch.cuda.empty_cache()
    initial_mem = get_detailed_gpu_memory()
    memory_history.append(initial_mem)
    print(f"Initial: allocated={initial_mem.get('allocated', 0):.2f} MB, "
          f"reserved={initial_mem.get('reserved', 0):.2f} MB")
    
    for i in range(n_iterations):
        # Create GPU tensors similar to what's used in plotting.py
        n_components = 100
        n_modules = 3
        
        # Simulate component activation density data
        activation_densities = {}
        for j in range(n_modules):
            # Create tensor on GPU
            density = torch.randn(n_components, device='cuda')
            activation_densities[f'module_{j}'] = density
        
        # Create figure similar to plot_component_activation_density
        fig, axs = plt.subplots(
            n_modules,
            1,
            figsize=(5, 5 * n_modules),
            squeeze=False,
        )
        axs = axs.flatten()
        
        for j, (module_name, density) in enumerate(activation_densities.items()):
            ax = axs[j]
            # Move to CPU for plotting
            ax.hist(density.detach().cpu().numpy(), bins=100)
            
            if use_log_scale:
                ax.set_yscale("log")  # The suspected problem
                
            ax.set_title(module_name)
            ax.set_xlabel("Activation density")
            ax.set_ylabel("Frequency")
        
        fig.tight_layout()
        
        # Render to image (same as in actual code)
        img = _render_figure(fig)
        
        # Cleanup
        plt.close(fig)
        plt.close('all')
        del img
        del fig
        del axs
        
        # Delete tensors
        for tensor in activation_densities.values():
            del tensor
        del activation_densities
        
        # Memory check every 5 iterations
        if (i + 1) % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            current_mem = get_detailed_gpu_memory()
            memory_history.append(current_mem)
            
            allocated_diff = current_mem.get('allocated', 0) - initial_mem.get('allocated', 0)
            reserved_diff = current_mem.get('reserved', 0) - initial_mem.get('reserved', 0)
            
            print(f"Iter {i+1}: allocated={current_mem.get('allocated', 0):.2f} MB (+{allocated_diff:.2f}), "
                  f"reserved={current_mem.get('reserved', 0):.2f} MB (+{reserved_diff:.2f})")
    
    # Final cleanup
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(0.5)
    
    final_mem = get_detailed_gpu_memory()
    memory_history.append(final_mem)
    
    print(f"\nFinal: allocated={final_mem.get('allocated', 0):.2f} MB, "
          f"reserved={final_mem.get('reserved', 0):.2f} MB")
    print(f"Total allocated increase: {final_mem.get('allocated', 0) - initial_mem.get('allocated', 0):.2f} MB")
    print(f"Total reserved increase: {final_mem.get('reserved', 0) - initial_mem.get('reserved', 0):.2f} MB")
    
    return memory_history


def test_histogram_function(use_log_scale: bool, n_iterations: int = 50):
    """Test the histogram plotting function specifically."""
    print(f"\n{'='*60}")
    print(f"Testing histogram function with log_scale={use_log_scale}")
    print(f"{'='*60}")
    
    memory_history = []
    
    # Initial state
    gc.collect()
    torch.cuda.empty_cache()
    initial_mem = get_detailed_gpu_memory()
    memory_history.append(initial_mem)
    
    for i in range(n_iterations):
        # Create causal importances similar to plot_ci_values_histograms
        causal_importances = {}
        for j in range(3):
            # Shape: (batch, components)
            ci = torch.randn(64, 128, device='cuda')
            causal_importances[f'layer.{j}'] = ci
        
        # Plot histograms
        n_layers = len(causal_importances)
        fig, axs = plt.subplots(
            n_layers,
            1,
            figsize=(6, 5 * n_layers),
            squeeze=False,
        )
        axs = axs.flatten()
        
        for j, (layer_name_raw, layer_ci) in enumerate(causal_importances.items()):
            layer_name = layer_name_raw.replace(".", "_")
            ax = axs[j]
            ax.hist(layer_ci.flatten().cpu().numpy(), bins=100)
            ax.set_title(f"Causal importances for {layer_name}")
            ax.set_xlabel("Causal importance value")
            
            if use_log_scale:
                ax.set_yscale("log")
                
            ax.set_ylabel("Frequency")
        
        fig.tight_layout()
        
        # Render
        img = _render_figure(fig)
        
        # Cleanup
        plt.close(fig)
        plt.close('all')
        del img
        del fig
        del axs
        
        # Delete tensors
        for tensor in causal_importances.values():
            del tensor
        del causal_importances
        
        # Check memory
        if (i + 1) % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            current_mem = get_detailed_gpu_memory()
            memory_history.append(current_mem)
            
            if i == 4:  # First check
                print(f"After {i+1} plots:")
                print(f"  Allocated: {current_mem.get('allocated', 0):.2f} MB")
                print(f"  Reserved: {current_mem.get('reserved', 0):.2f} MB")
    
    # Final check
    gc.collect()
    torch.cuda.empty_cache()
    final_mem = get_detailed_gpu_memory()
    
    print(f"\nFinal results:")
    print(f"  Allocated: {initial_mem.get('allocated', 0):.2f} -> {final_mem.get('allocated', 0):.2f} MB "
          f"(+{final_mem.get('allocated', 0) - initial_mem.get('allocated', 0):.2f})")
    print(f"  Reserved: {initial_mem.get('reserved', 0):.2f} -> {final_mem.get('reserved', 0):.2f} MB "
          f"(+{final_mem.get('reserved', 0) - initial_mem.get('reserved', 0):.2f})")
    
    return memory_history


def diagnose_matplotlib_internals():
    """Check matplotlib's internal state for potential issues."""
    print(f"\n{'='*60}")
    print("Matplotlib diagnostics")
    print(f"{'='*60}")
    
    print(f"Backend: {matplotlib.get_backend()}")
    print(f"Interactive: {matplotlib.is_interactive()}")
    print(f"Number of figures before: {len(plt.get_fignums())}")
    
    # Create and close a figure with log scale
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [10, 100, 1000])
    ax.set_yscale('log')
    plt.close(fig)
    
    print(f"Number of figures after close: {len(plt.get_fignums())}")
    
    # Check for any remaining axes
    import matplotlib._pylab_helpers as pylab_helpers
    print(f"Active figures: {pylab_helpers.Gcf.get_all_fig_managers()}")
    
    # Force cleanup
    plt.close('all')
    gc.collect()


def main():
    """Run comprehensive memory leak tests."""
    print("GPU Memory Leak Analysis for matplotlib log scale in SPD")
    print(f"PyTorch: {torch.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")
    print(f"NumPy: {np.__version__}")
    
    if not torch.cuda.is_available():
        print("No GPU available!")
        return
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    torch.cuda.reset_peak_memory_stats()
    
    # Run diagnostics first
    diagnose_matplotlib_internals()
    
    # Test scenarios
    print("\n" + "="*80)
    print("RUNNING TESTS")
    print("="*80)
    
    # Test 1: Tensor plotting with log scale
    test_plot_with_tensors(use_log_scale=True, n_iterations=30)
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)
    
    # Test 2: Tensor plotting without log scale
    test_plot_with_tensors(use_log_scale=False, n_iterations=30)
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)
    
    # Test 3: Histogram function with log scale
    test_histogram_function(use_log_scale=True, n_iterations=30)
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)
    
    # Test 4: Histogram function without log scale
    test_histogram_function(use_log_scale=False, n_iterations=30)
    
    print("\n" + "="*80)
    print("TESTS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()