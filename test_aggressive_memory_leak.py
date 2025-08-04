#!/usr/bin/env python3
"""Aggressive test to find conditions that trigger the memory leak."""

import gc
import io
import time
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

matplotlib.use('Agg')


def monitor_memory(label: str):
    """Print current memory usage with a label."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        print(f"{label}: allocated={allocated:.2f} MB, reserved={reserved:.2f} MB")
        return allocated, reserved
    return 0, 0


def aggressive_test(use_log_scale: bool, iterations: int = 200):
    """Aggressive test with various data patterns and sizes."""
    print(f"\n{'='*80}")
    print(f"AGGRESSIVE TEST: log_scale={use_log_scale}, iterations={iterations}")
    print(f"{'='*80}")
    
    torch.cuda.empty_cache()
    gc.collect()
    start_alloc, start_reserved = monitor_memory("Start")
    
    leaks_detected = []
    
    for i in range(iterations):
        # Vary data characteristics to trigger edge cases
        n_samples = random.choice([1000, 10000, 50000])
        n_components = random.choice([50, 100, 200])
        n_modules = random.choice([1, 3, 5])
        
        # Create different data patterns
        data_pattern = random.choice(['normal', 'sparse', 'extreme', 'zeros'])
        
        tensors_created = []
        
        # Simulate the plotting workflow
        fig, axs = plt.subplots(n_modules, 1, figsize=(5, 5 * n_modules), squeeze=False)
        axs = axs.flatten()
        
        for j in range(n_modules):
            # Create data on GPU
            if data_pattern == 'normal':
                data = torch.randn(n_samples, device='cuda')
            elif data_pattern == 'sparse':
                data = torch.zeros(n_samples, device='cuda')
                sparse_idx = torch.randint(0, n_samples, (n_samples // 10,))
                data[sparse_idx] = torch.randn(n_samples // 10, device='cuda') * 10
            elif data_pattern == 'extreme':
                data = torch.randn(n_samples, device='cuda') * 1000
                data[0] = 1e-10  # Add tiny value for log scale
            else:  # zeros
                data = torch.zeros(n_samples, device='cuda')
                data[0] = 1  # Avoid log(0)
            
            tensors_created.append(data)
            
            # Plot histogram
            ax = axs[j]
            cpu_data = data.detach().cpu().numpy()
            
            # Handle edge cases for log scale
            if use_log_scale and data_pattern in ['zeros', 'extreme']:
                cpu_data = np.maximum(cpu_data, 1e-10)
            
            try:
                ax.hist(cpu_data, bins=random.choice([50, 100, 200]))
                if use_log_scale:
                    ax.set_yscale('log')
                ax.set_title(f"Module {j} - {data_pattern}")
            except Exception as e:
                print(f"  Plot error at iter {i}: {e}")
        
        # Render to image
        buf = io.BytesIO()
        try:
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=random.choice([100, 150, 300]))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            buf.close()
            del img
        except Exception as e:
            print(f"  Render error at iter {i}: {e}")
        
        # Cleanup
        plt.close(fig)
        plt.close('all')
        del fig, axs
        
        # Delete tensors
        for tensor in tensors_created:
            del tensor
        tensors_created.clear()
        
        # Periodic memory check
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            
            curr_alloc, curr_reserved = monitor_memory(f"  Iter {i+1}")
            
            # Check for significant leaks
            alloc_increase = curr_alloc - start_alloc
            if alloc_increase > 10:  # More than 10MB increase
                leaks_detected.append((i+1, alloc_increase))
                print(f"    ⚠️  LEAK DETECTED: +{alloc_increase:.2f} MB")
    
    # Final cleanup and check
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(0.5)
    
    final_alloc, final_reserved = monitor_memory("Final")
    total_leak = final_alloc - start_alloc
    
    print(f"\nSUMMARY:")
    print(f"  Total allocated memory leak: {total_leak:.2f} MB")
    print(f"  Reserved memory increase: {final_reserved - start_reserved:.2f} MB")
    print(f"  Significant leaks detected: {len(leaks_detected)}")
    if leaks_detected:
        print(f"  Leak iterations: {[x[0] for x in leaks_detected[:5]]}...")
    
    return total_leak, leaks_detected


def test_specific_edge_cases():
    """Test specific edge cases that might trigger leaks."""
    print(f"\n{'='*80}")
    print("TESTING SPECIFIC EDGE CASES")
    print(f"{'='*80}")
    
    test_cases = [
        ("Empty histogram with log scale", lambda: torch.zeros(1000, device='cuda')),
        ("Single value histogram with log scale", lambda: torch.ones(1000, device='cuda') * 0.1),
        ("Negative values with log scale", lambda: torch.randn(1000, device='cuda') - 2),
        ("Very small positive values", lambda: torch.rand(1000, device='cuda') * 1e-10),
        ("Mixed tiny and large values", lambda: torch.cat([
            torch.ones(500, device='cuda') * 1e-10,
            torch.ones(500, device='cuda') * 1e10
        ])),
    ]
    
    for test_name, data_gen in test_cases:
        print(f"\nTesting: {test_name}")
        torch.cuda.empty_cache()
        gc.collect()
        
        start_mem = torch.cuda.memory_allocated() / 1024 / 1024
        
        for i in range(20):
            data = data_gen()
            
            fig, ax = plt.subplots()
            try:
                # Ensure no negative values for log scale
                cpu_data = data.cpu().numpy()
                cpu_data = np.maximum(cpu_data, 1e-10)
                
                ax.hist(cpu_data, bins=50)
                ax.set_yscale('log')
            except Exception as e:
                print(f"  Error: {e}")
            
            plt.close(fig)
            del data
        
        gc.collect()
        torch.cuda.empty_cache()
        end_mem = torch.cuda.memory_allocated() / 1024 / 1024
        
        print(f"  Memory change: {end_mem - start_mem:.2f} MB")


def test_concurrent_plotting():
    """Test if leak occurs with concurrent tensor operations."""
    print(f"\n{'='*80}")
    print("TESTING CONCURRENT OPERATIONS")
    print(f"{'='*80}")
    
    torch.cuda.empty_cache()
    gc.collect()
    start_mem = torch.cuda.memory_allocated() / 1024 / 1024
    
    # Keep some tensors alive while plotting
    persistent_tensors = [torch.randn(10000, device='cuda') for _ in range(10)]
    
    for i in range(50):
        # Create new tensors while old ones exist
        temp_tensors = [torch.randn(5000, device='cuda') for _ in range(5)]
        
        # Plot with log scale
        fig, ax = plt.subplots()
        data = torch.cat(temp_tensors).cpu().numpy()
        ax.hist(data, bins=100)
        ax.set_yscale('log')
        
        plt.close(fig)
        
        # Delete only temp tensors
        for t in temp_tensors:
            del t
        
        if (i + 1) % 10 == 0:
            gc.collect()
            curr_mem = torch.cuda.memory_allocated() / 1024 / 1024
            print(f"  Iter {i+1}: {curr_mem:.2f} MB (+{curr_mem - start_mem:.2f})")
    
    # Clean up persistent tensors
    for t in persistent_tensors:
        del t
    
    gc.collect()
    torch.cuda.empty_cache()
    final_mem = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"  Final leak: {final_mem - start_mem:.2f} MB")


def main():
    """Run all aggressive tests."""
    print("AGGRESSIVE GPU MEMORY LEAK DETECTION")
    print(f"PyTorch: {torch.__version__}")
    print(f"Matplotlib: {matplotlib.__version__}")
    
    if not torch.cuda.is_available():
        print("No GPU available!")
        return
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    torch.cuda.reset_peak_memory_stats()
    
    # Run aggressive tests
    print("\n" + "="*80)
    print("Running aggressive tests...")
    
    # Test with log scale
    log_leak, log_leaks = aggressive_test(use_log_scale=True, iterations=100)
    
    # Clean between tests
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)
    
    # Test without log scale
    linear_leak, linear_leaks = aggressive_test(use_log_scale=False, iterations=100)
    
    # Test edge cases
    test_specific_edge_cases()
    
    # Test concurrent operations
    test_concurrent_plotting()
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Log scale total leak: {log_leak:.2f} MB ({len(log_leaks)} spikes)")
    print(f"Linear scale total leak: {linear_leak:.2f} MB ({len(linear_leaks)} spikes)")
    print(f"Difference: {abs(log_leak - linear_leak):.2f} MB")
    
    if abs(log_leak - linear_leak) > 5:
        print("\n⚠️  SIGNIFICANT DIFFERENCE DETECTED BETWEEN LOG AND LINEAR SCALE!")


if __name__ == "__main__":
    main()