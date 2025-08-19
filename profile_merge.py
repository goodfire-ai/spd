#!/usr/bin/env python3
"""Profile the merge iteration loop to identify performance bottlenecks."""

import cProfile
import io
import pstats

# Add the project to path
import time
from contextlib import contextmanager

import torch
from line_profiler import LineProfiler

from spd.clustering.compute_costs import (
    compute_merge_costs,
    recompute_coacts_merge_pair,
)
from spd.clustering.merge import merge_iteration
from spd.clustering.merge_config import MergeConfig


@contextmanager
def profile_context(sort_by="cumulative", top_n=30):
    """Context manager for profiling code blocks."""
    pr = cProfile.Profile()
    pr.enable()
    start_time = time.perf_counter()

    try:
        yield pr
    finally:
        pr.disable()
        end_time = time.perf_counter()

        print(f"\n{'=' * 60}")
        print(f"Profiling Results (Total time: {end_time - start_time:.2f}s)")
        print(f"{'=' * 60}\n")

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(sort_by)
        ps.print_stats(top_n)
        print(s.getvalue())


def create_test_data(n_samples=1000, n_components=100):
    """Create synthetic test data for profiling."""
    print(f"Creating test data: {n_samples} samples, {n_components} components")

    # Create random activations (sparse, as in real use case)
    sparsity = 0.1  # 10% of activations are non-zero
    activations = torch.rand(n_samples, n_components)
    mask = torch.rand(n_samples, n_components) < sparsity
    activations = activations * mask.float()

    # Create component labels
    component_labels = [f"comp_{i}" for i in range(n_components)]

    return activations, component_labels


def profile_merge_with_line_profiler():
    """Profile specific functions line-by-line."""
    print("\n" + "=" * 60)
    print("Line-by-line profiling of key functions")
    print("=" * 60 + "\n")

    # Create test data
    n_samples = 500
    n_components = 50
    activations, component_labels = create_test_data(n_samples, n_components)

    # Create config
    config = MergeConfig(
        iters=10,  # Fewer iterations for line profiling
        alpha=1.0,
        pop_chance=0.0,  # Disable popping for simpler profiling
    )

    # Setup line profiler
    lp = LineProfiler()

    # Add functions to profile
    lp.add_function(compute_merge_costs)
    lp.add_function(recompute_coacts_merge_pair)

    # Wrap the main function
    merge_iteration_wrapped = lp(merge_iteration)

    # Run the profiled function
    print("Running line profiler...")
    merge_iteration_wrapped(
        activations=activations,
        merge_config=config,
        component_labels=component_labels,
    )

    # Print results
    lp.print_stats()


def profile_with_different_sizes():
    """Profile with different data sizes to understand scaling."""
    print("\n" + "=" * 60)
    print("Profiling with different data sizes")
    print("=" * 60 + "\n")

    sizes = [
        (500, 50, 10),  # samples, components, iterations
        (1000, 100, 20),
        (2000, 200, 30),
    ]

    for n_samples, n_components, n_iters in sizes:
        print(
            f"\n--- Size: {n_samples} samples, {n_components} components, {n_iters} iterations ---"
        )

        activations, component_labels = create_test_data(n_samples, n_components)
        config = MergeConfig(iters=n_iters, alpha=1.0, pop_chance=0.0)

        start = time.perf_counter()
        merge_iteration(
            activations=activations,
            merge_config=config,
            component_labels=component_labels,
        )
        end = time.perf_counter()

        total_time = end - start
        time_per_iter = total_time / n_iters
        print(f"Total time: {total_time:.2f}s")
        print(f"Time per iteration: {time_per_iter:.3f}s")
        print(f"Iterations per second: {1 / time_per_iter:.2f}")


def analyze_gpu_usage():
    """Check if operations are using GPU efficiently."""
    print("\n" + "=" * 60)
    print("GPU Usage Analysis")
    print("=" * 60 + "\n")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")

        # Create test data on GPU
        n_samples, n_components = 1000, 100
        activations = torch.rand(n_samples, n_components, device=device) * 0.1
        component_labels = [f"comp_{i}" for i in range(n_components)]

        config = MergeConfig(iters=20, alpha=1.0, pop_chance=0.0)

        # Profile GPU execution
        torch.cuda.synchronize()
        start = time.perf_counter()

        merge_iteration(
            activations=activations,
            merge_config=config,
            component_labels=component_labels,
        )

        torch.cuda.synchronize()
        end = time.perf_counter()

        print(f"GPU execution time: {end - start:.2f}s")
        print(f"Memory allocated after: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    else:
        print("No GPU available - running CPU analysis")

        # Check if operations are vectorized
        n_samples, n_components = 1000, 100
        activations = torch.rand(n_samples, n_components) * 0.1

        # Check if using efficient operations
        print("\nChecking tensor operation efficiency:")

        # Test coactivation computation
        start = time.perf_counter()
        _coact = activations.T @ activations
        end = time.perf_counter()
        print(f"Coactivation matrix computation: {end - start:.4f}s")

        # Test boolean masking
        mask = activations > 0
        start = time.perf_counter()
        _masked = mask.float() @ mask.float().T
        end = time.perf_counter()
        print(f"Boolean mask multiplication: {end - start:.4f}s")


def main():
    """Main profiling function."""
    print("Starting merge iteration profiling\n")
    from spd.clustering.scripts.s2_run_clustering import cli

    with profile_context(sort_by="cumulative", top_n=20):
        # uv run python /home/miv/projects/MATS/spd/spd/clustering/scripts/s2_run_clustering.py --config spd/clustering/configs/test-simplestories.json --dataset-path /home/miv/projects/MATS/spd/data/clustering/task_lm-w_ioprgffh-a1-i2-b1-n1-h_f0f9d4/batches/batch_00.npz --save-dir /home/miv/projects/MATS/spd/data/clustering/task_lm-w_ioprgffh-a1-i2-b1-n1-h_f0f9d4/merge_history --device cuda
        cli(
            [
                "--config",
                "spd/clustering/configs/test-simplestories.json",
                "--dataset-path",
                "/home/miv/projects/MATS/spd/data/clustering/task_lm-w_ioprgffh-a1-i2-b1-n1-h_f0f9d4/batches/batch_00.npz",
                "--save-dir",
                "/home/miv/projects/MATS/spd/data/clustering/task_lm-w_ioprgffh-a1-i2-b1-n1-h_f0f9d4/merge_history",
                "--device",
                "cuda",
                "--override-json-fd",
                "1",
            ]
        )

    # 2. Line profiling of key functions
    # profile_merge_with_line_profiler()

    # 3. Scaling analysis
    # profile_with_different_sizes()

    # # 4. GPU usage analysis
    # analyze_gpu_usage()


if __name__ == "__main__":
    main()
