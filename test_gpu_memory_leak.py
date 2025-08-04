#!/usr/bin/env python3
"""Test script to reproduce GPU memory leak with matplotlib log scale."""

import gc
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

# Use non-interactive backend to avoid display issues
matplotlib.use('Agg')


def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def create_plot_with_log_scale(data, plot_idx):
    """Create a plot with log scale on y-axis."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.hist(data, bins=50)
    ax.set_yscale("log")  # The suspected culprit
    ax.set_title(f"Plot {plot_idx} - Log Scale")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency (log)")
    
    # Close everything
    plt.close(fig)
    plt.close('all')
    
    # Explicitly delete references
    del fig
    del ax


def create_plot_without_log_scale(data, plot_idx):
    """Create a plot without log scale on y-axis."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.hist(data, bins=50)
    # No log scale
    ax.set_title(f"Plot {plot_idx} - Linear Scale")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    
    # Close everything
    plt.close(fig)
    plt.close('all')
    
    # Explicitly delete references
    del fig
    del ax


def test_memory_leak(n_iterations=100, use_log_scale=True, use_gpu_data=True):
    """Test for memory leaks when creating many plots."""
    print(f"\n{'='*60}")
    print(f"Testing with log_scale={use_log_scale}, use_gpu_data={use_gpu_data}")
    print(f"{'='*60}")
    
    memory_usage = []
    
    # Initial memory measurement
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    initial_memory = get_gpu_memory_usage()
    memory_usage.append(initial_memory)
    print(f"Initial GPU memory: {initial_memory:.2f} MB")
    
    for i in range(n_iterations):
        # Generate data
        if use_gpu_data:
            # Create data on GPU then move to CPU for plotting
            gpu_data = torch.randn(10000, device='cuda' if torch.cuda.is_available() else 'cpu')
            data = gpu_data.cpu().numpy()
            del gpu_data
        else:
            # Create data directly on CPU
            data = np.random.randn(10000)
        
        # Create plot
        if use_log_scale:
            create_plot_with_log_scale(data, i)
        else:
            create_plot_without_log_scale(data, i)
        
        # Clean up
        del data
        
        # Measure memory every 10 iterations
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            current_memory = get_gpu_memory_usage()
            memory_usage.append(current_memory)
            print(f"After {i+1} plots: GPU memory = {current_memory:.2f} MB "
                  f"(+{current_memory - initial_memory:.2f} MB)")
    
    # Final cleanup and measurement
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    time.sleep(0.5)  # Give time for cleanup
    final_memory = get_gpu_memory_usage()
    memory_usage.append(final_memory)
    
    print(f"\nFinal GPU memory: {final_memory:.2f} MB")
    print(f"Total memory increase: {final_memory - initial_memory:.2f} MB")
    print(f"Average increase per plot: {(final_memory - initial_memory) / n_iterations:.4f} MB")
    
    return memory_usage


def main():
    """Run the memory leak tests."""
    print("GPU Memory Leak Test for matplotlib log scale")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Matplotlib version: {matplotlib.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        
        # Test different scenarios
        scenarios = [
            (True, True),   # log scale with GPU data
            (False, True),  # no log scale with GPU data
            (True, False),  # log scale with CPU data
            (False, False), # no log scale with CPU data
        ]
        
        results = {}
        for use_log, use_gpu in scenarios:
            key = f"log={use_log}, gpu={use_gpu}"
            results[key] = test_memory_leak(
                n_iterations=100,
                use_log_scale=use_log,
                use_gpu_data=use_gpu
            )
            
            # Extra cleanup between tests
            gc.collect()
            torch.cuda.empty_cache()
            plt.close('all')
            time.sleep(1)
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for key, memory_usage in results.items():
            initial = memory_usage[0]
            final = memory_usage[-1]
            leak = final - initial
            print(f"{key}: {leak:.2f} MB leaked ({initial:.2f} -> {final:.2f} MB)")
            
    else:
        print("No GPU available. Cannot test GPU memory leaks.")


if __name__ == "__main__":
    main()