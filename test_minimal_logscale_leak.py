#!/usr/bin/env python3
"""Minimal test to isolate the log scale memory leak."""

import gc
import matplotlib
import matplotlib.pyplot as plt
import torch

matplotlib.use('Agg')


def test_minimal(use_log: bool, iterations: int = 100):
    """Minimal test case."""
    if not torch.cuda.is_available():
        print("No GPU!")
        return
    
    print(f"\nTesting with log_scale={use_log}")
    
    # Create a GPU tensor once
    data = torch.randn(10000, device='cuda')
    
    # Initial memory
    torch.cuda.empty_cache()
    gc.collect()
    mem_start = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"Start: {mem_start:.2f} MB")
    
    for i in range(iterations):
        # Create figure
        fig, ax = plt.subplots()
        
        # Plot histogram of GPU data
        ax.hist(data.cpu().numpy(), bins=50)
        
        if use_log:
            ax.set_yscale('log')
        
        # Close immediately
        plt.close(fig)
        
        # Check memory every 20 iterations
        if (i + 1) % 20 == 0:
            gc.collect()
            mem_now = torch.cuda.memory_allocated() / 1024 / 1024
            print(f"  After {i+1}: {mem_now:.2f} MB (+{mem_now - mem_start:.2f})")
    
    # Final
    plt.close('all')
    gc.collect()
    torch.cuda.empty_cache()
    mem_end = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"End: {mem_end:.2f} MB (leaked: {mem_end - mem_start:.2f} MB)")
    
    # Clean up data tensor
    del data


def main():
    print("Minimal GPU Memory Leak Test")
    print(f"PyTorch: {torch.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")
    
    # Test both cases
    test_minimal(use_log=True, iterations=100)
    test_minimal(use_log=False, iterations=100)


if __name__ == "__main__":
    main()