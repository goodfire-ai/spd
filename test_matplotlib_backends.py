#!/usr/bin/env python3
"""Test if the memory leak is related to matplotlib backend or configuration."""

import gc
import os
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np

# Try different backends
BACKENDS = ['Agg', 'svg', 'pdf']


def test_backend(backend, use_log=True, iterations=50):
    """Test a specific backend."""
    matplotlib.use(backend, force=True)
    print(f"\nTesting backend: {backend}, log_scale={use_log}")
    
    if not torch.cuda.is_available():
        print("No GPU!")
        return
    
    torch.cuda.empty_cache()
    gc.collect()
    
    start_mem = torch.cuda.memory_allocated() / 1024 / 1024
    
    for i in range(iterations):
        # Create GPU data
        data = torch.randn(10000, device='cuda')
        
        # Plot
        fig, ax = plt.subplots()
        ax.hist(data.cpu().numpy(), bins=50)
        if use_log:
            ax.set_yscale('log')
        
        # Save to memory
        fig.canvas.draw()
        
        # Cleanup
        plt.close(fig)
        del data
        
        if (i + 1) % 10 == 0:
            gc.collect()
            curr_mem = torch.cuda.memory_allocated() / 1024 / 1024
            print(f"  Iter {i+1}: {curr_mem:.2f} MB (+{curr_mem - start_mem:.2f})")
    
    gc.collect()
    torch.cuda.empty_cache()
    final_mem = torch.cuda.memory_allocated() / 1024 / 1024
    leak = final_mem - start_mem
    print(f"  Total leak: {leak:.2f} MB")
    return leak


def test_matplotlib_cache():
    """Test if matplotlib's internal caching causes issues."""
    print("\nTesting matplotlib cache settings...")
    
    # Check current cache settings
    print(f"Current rcParams['path.simplify']: {plt.rcParams['path.simplify']}")
    print(f"Current rcParams['path.simplify_threshold']: {plt.rcParams['path.simplify_threshold']}")
    print(f"Current rcParams['agg.path.chunksize']: {plt.rcParams['agg.path.chunksize']}")
    
    # Try with disabled caching
    plt.rcParams['path.simplify'] = False
    plt.rcParams['agg.path.chunksize'] = 0
    
    matplotlib.use('Agg')
    
    if torch.cuda.is_available():
        print("\nTesting with cache disabled...")
        test_backend('Agg', use_log=True, iterations=30)


def check_versions_and_env():
    """Check versions and environment variables."""
    print("Environment and Version Check:")
    print(f"Python: {os.sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")
    print(f"NumPy: {np.__version__}")
    
    # Check relevant env vars
    env_vars = ['MPLBACKEND', 'MPLCONFIGDIR', 'CUDA_VISIBLE_DEVICES']
    for var in env_vars:
        print(f"{var}: {os.environ.get(var, 'Not set')}")


def test_different_plot_types():
    """Test if the leak is specific to histograms."""
    print("\nTesting different plot types with log scale...")
    
    if not torch.cuda.is_available():
        return
    
    matplotlib.use('Agg')
    
    plot_types = [
        ('histogram', lambda ax, data: ax.hist(data, bins=50)),
        ('line', lambda ax, data: ax.plot(data)),
        ('scatter', lambda ax, data: ax.scatter(range(len(data)), data)),
        ('bar', lambda ax, data: ax.bar(range(min(100, len(data))), data[:100])),
    ]
    
    for plot_name, plot_func in plot_types:
        torch.cuda.empty_cache()
        gc.collect()
        
        start_mem = torch.cuda.memory_allocated() / 1024 / 1024
        
        for i in range(30):
            data = torch.randn(1000, device='cuda').cpu().numpy()
            
            fig, ax = plt.subplots()
            plot_func(ax, data)
            ax.set_yscale('log')
            
            plt.close(fig)
            del data
        
        gc.collect()
        torch.cuda.empty_cache()
        
        final_mem = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"  {plot_name}: {final_mem - start_mem:.2f} MB leaked")


def main():
    """Run all tests."""
    check_versions_and_env()
    
    # Test different backends
    print("\n" + "="*60)
    print("Testing different backends")
    print("="*60)
    
    for backend in BACKENDS:
        try:
            test_backend(backend, use_log=True, iterations=30)
            test_backend(backend, use_log=False, iterations=30)
        except Exception as e:
            print(f"  Error with {backend}: {e}")
    
    # Test matplotlib cache
    test_matplotlib_cache()
    
    # Test different plot types
    test_different_plot_types()


if __name__ == "__main__":
    main()