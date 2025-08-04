#!/usr/bin/env python3
"""Test that exactly mimics the SPD plotting pattern to reproduce the memory leak."""

import gc
import io
import time
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.use('Agg')


def get_memory_info():
    """Get detailed memory information."""
    if not torch.cuda.is_available():
        return {}
    
    return {
        'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
        'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
        'allocated_gb': torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
        'reserved_gb': torch.cuda.memory_reserved() / 1024 / 1024 / 1024,
    }


def print_memory_diff(label, start_mem, current_mem):
    """Print memory difference."""
    alloc_diff = current_mem['allocated_mb'] - start_mem['allocated_mb']
    reserved_diff = current_mem['reserved_mb'] - start_mem['reserved_mb']
    
    print(f"{label}:")
    print(f"  Allocated: {current_mem['allocated_mb']:.2f} MB ({alloc_diff:+.2f} MB)")
    print(f"  Reserved: {current_mem['reserved_mb']:.2f} MB ({reserved_diff:+.2f} MB)")
    
    if abs(alloc_diff) > 100:  # More than 100MB
        print(f"  ⚠️  SIGNIFICANT LEAK: {alloc_diff:.2f} MB")


def _render_figure(fig):
    """Exact copy of the render function from spd.plotting."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    buf.close()
    return img


def simulate_plot_component_activation_density(use_log_scale=True, n_modules=3):
    """Simulate the exact pattern from plot_component_activation_density."""
    # Create activation density data on GPU
    component_activation_density = {}
    for i in range(n_modules):
        # Tensor on GPU
        density = torch.randn(100, device='cuda')
        component_activation_density[f'module_{i}'] = density
    
    # Create figure exactly as in the original
    fig, axs = plt.subplots(
        n_modules,
        1,
        figsize=(5, 5 * n_modules),
        squeeze=False,
    )
    
    axs = axs.flatten()
    
    # Plot histograms
    for i, (module_name, density) in enumerate(component_activation_density.items()):
        ax = axs[i]
        ax.hist(density.detach().cpu().numpy(), bins=100)
        
        if use_log_scale:
            ax.set_yscale("log")  # The suspected issue
            
        ax.set_title(module_name)
        ax.set_xlabel("Activation density")
        ax.set_ylabel("Frequency")
    
    fig.tight_layout()
    
    # Render exactly as in original
    fig_img = _render_figure(fig)
    plt.close(fig)
    
    # Clean up tensors
    for tensor in component_activation_density.values():
        del tensor
    del component_activation_density
    del fig_img
    
    return True


def simulate_plot_ci_values_histograms(use_log_scale=True, n_layers=3):
    """Simulate the exact pattern from plot_ci_values_histograms."""
    # Create causal importance data on GPU
    causal_importances = {}
    for i in range(n_layers):
        # Shape: (batch, components)
        ci = torch.randn(64, 128, device='cuda')
        causal_importances[f'layer.{i}'] = ci
    
    # Create figure exactly as in original
    fig, axs = plt.subplots(
        n_layers,
        1,
        figsize=(6, 5 * n_layers),
        squeeze=False,
    )
    
    axs = axs.flatten()
    
    for i, (layer_name_raw, layer_ci) in enumerate(causal_importances.items()):
        layer_name = layer_name_raw.replace(".", "_")
        ax = axs[i]
        ax.hist(layer_ci.flatten().cpu().numpy(), bins=100)
        ax.set_title(f"Causal importances for {layer_name}")
        ax.set_xlabel("Causal importance value")
        
        if use_log_scale:
            ax.set_yscale("log")  # The suspected issue
            
        ax.set_ylabel("Frequency")
    
    fig.tight_layout()
    
    # Render exactly as in original
    fig_img = _render_figure(fig)
    plt.close(fig)
    
    # Clean up
    for tensor in causal_importances.values():
        del tensor
    del causal_importances
    del fig_img
    
    return True


def run_extended_test(plot_function, function_name, use_log_scale=True, iterations=500):
    """Run an extended test with many iterations."""
    print(f"\n{'='*80}")
    print(f"EXTENDED TEST: {function_name} with log_scale={use_log_scale}")
    print(f"Iterations: {iterations}")
    print(f"{'='*80}")
    
    # Clear everything first
    gc.collect()
    torch.cuda.empty_cache()
    plt.close('all')
    
    start_mem = get_memory_info()
    print_memory_diff("Initial", start_mem, start_mem)
    
    memory_history = [start_mem['allocated_mb']]
    
    for i in range(iterations):
        # Run the plotting function
        plot_function(use_log_scale=use_log_scale)
        
        # Periodic checks
        if (i + 1) % 50 == 0:
            # Force cleanup
            gc.collect()
            torch.cuda.empty_cache()
            
            current_mem = get_memory_info()
            memory_history.append(current_mem['allocated_mb'])
            print_memory_diff(f"After {i+1} iterations", start_mem, current_mem)
            
            # Check if memory is growing
            if len(memory_history) > 2:
                recent_growth = memory_history[-1] - memory_history[-2]
                if recent_growth > 10:
                    print(f"  ⚠️  Memory grew by {recent_growth:.2f} MB in last 50 iterations")
    
    # Final cleanup
    plt.close('all')
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)
    
    final_mem = get_memory_info()
    print_memory_diff("Final", start_mem, final_mem)
    
    total_leak = final_mem['allocated_mb'] - start_mem['allocated_mb']
    print(f"\nTotal memory leak: {total_leak:.2f} MB")
    print(f"Average per iteration: {total_leak / iterations:.4f} MB")
    
    return total_leak, memory_history


def test_with_actual_spd_context():
    """Test with conditions more similar to actual SPD runs."""
    print(f"\n{'='*80}")
    print("TESTING WITH SPD-LIKE CONTEXT")
    print(f"{'='*80}")
    
    # Create some persistent tensors like SPD would have
    print("Creating persistent model-like tensors...")
    persistent_tensors = []
    
    # Simulate model weights
    for i in range(10):
        weight = torch.randn(512, 512, device='cuda')
        persistent_tensors.append(weight)
    
    # Simulate activations cache
    activations = []
    for i in range(5):
        act = torch.randn(64, 512, device='cuda')
        activations.append(act)
    
    initial_mem = get_memory_info()
    print(f"Memory after loading 'model': {initial_mem['allocated_mb']:.2f} MB")
    
    # Now run plotting operations
    print("\nRunning plotting operations with model in memory...")
    
    for i in range(100):
        # Simulate SPD workflow
        
        # 1. Generate some new tensors (like causal importances)
        new_tensors = []
        for j in range(3):
            ci = torch.randn(64, 128, device='cuda')
            new_tensors.append(ci)
        
        # 2. Plot with log scale
        simulate_plot_ci_values_histograms(use_log_scale=True)
        
        # 3. Plot activation densities
        simulate_plot_component_activation_density(use_log_scale=True)
        
        # 4. Clean up iteration tensors (but keep model)
        for t in new_tensors:
            del t
        
        if (i + 1) % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            current_mem = get_memory_info()
            leak = current_mem['allocated_mb'] - initial_mem['allocated_mb']
            print(f"  Iter {i+1}: {current_mem['allocated_mb']:.2f} MB (leak: {leak:+.2f} MB)")
    
    # Final cleanup
    print("\nCleaning up everything...")
    for t in persistent_tensors + activations:
        del t
    
    gc.collect()
    torch.cuda.empty_cache()
    final_mem = get_memory_info()
    
    print(f"Final memory: {final_mem['allocated_mb']:.2f} MB")


def main():
    """Run comprehensive SPD-pattern memory leak tests."""
    print("SPD EXACT PATTERN MEMORY LEAK TEST")
    print(f"PyTorch: {torch.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")
    print(f"NumPy: {np.__version__}")
    
    if not torch.cuda.is_available():
        print("No GPU available!")
        return
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    torch.cuda.reset_peak_memory_stats()
    
    # Test 1: Component activation density plots
    leak1_log, _ = run_extended_test(
        simulate_plot_component_activation_density,
        "plot_component_activation_density",
        use_log_scale=True,
        iterations=200
    )
    
    time.sleep(2)
    
    leak1_linear, _ = run_extended_test(
        simulate_plot_component_activation_density,
        "plot_component_activation_density",
        use_log_scale=False,
        iterations=200
    )
    
    time.sleep(2)
    
    # Test 2: CI histograms
    leak2_log, _ = run_extended_test(
        simulate_plot_ci_values_histograms,
        "plot_ci_values_histograms",
        use_log_scale=True,
        iterations=200
    )
    
    time.sleep(2)
    
    leak2_linear, _ = run_extended_test(
        simulate_plot_ci_values_histograms,
        "plot_ci_values_histograms",
        use_log_scale=False,
        iterations=200
    )
    
    # Test 3: With SPD-like context
    test_with_actual_spd_context()
    
    # Summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"plot_component_activation_density:")
    print(f"  With log scale: {leak1_log:.2f} MB leaked")
    print(f"  Without log scale: {leak1_linear:.2f} MB leaked")
    print(f"  Difference: {abs(leak1_log - leak1_linear):.2f} MB")
    
    print(f"\nplot_ci_values_histograms:")
    print(f"  With log scale: {leak2_log:.2f} MB leaked")
    print(f"  Without log scale: {leak2_linear:.2f} MB leaked")
    print(f"  Difference: {abs(leak2_log - leak2_linear):.2f} MB")
    
    # Peak memory stats
    print(f"\nPeak memory stats:")
    print(f"  Peak allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
    print(f"  Peak reserved: {torch.cuda.max_memory_reserved() / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()