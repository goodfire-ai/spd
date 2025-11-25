"""Benchmark script for activation context collection.

Usage:
    source .venv/bin/activate
    python spd/app/backend/lib/benchmark_activation_contexts.py [--original] [--profile] [--flamegraph]

Uses a real W&B run for accurate profiling. Pass --original to benchmark the
original implementation (rename files to swap).

Flame graphs are saved to spd/app/backend/lib/flamegraphs/
"""

import argparse
import cProfile
import pstats
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import torch

from spd.app.backend.services.run_context_service import RunContextService, TrainRunContext
from spd.log import logger

# Default W&B run for benchmarking
DEFAULT_WANDB_PATH = "goodfire/spd/runs/jyo9duz5"

# Paths for swapping implementations
LIB_DIR = Path(__file__).parent
MAIN_FILE = LIB_DIR / "activation_contexts.py"
ORIGINAL_FILE = LIB_DIR / "activation_contexts_original.py"
OPTIMIZED_FILE = LIB_DIR / "activation_contexts_optimized.py"
FLAMEGRAPH_DIR = LIB_DIR / "flamegraphs"


def swap_implementation(use_original: bool) -> None:
    """Swap the activation_contexts.py file to use original or optimized version."""
    source = ORIGINAL_FILE if use_original else OPTIMIZED_FILE
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    shutil.copy(source, MAIN_FILE)
    logger.info(f"Swapped to {'original' if use_original else 'optimized'} implementation")


@dataclass
class BenchmarkParams:
    n_batches: int = 20
    batch_size: int = 32
    importance_threshold: float = 0.1
    n_tokens_either_side: int = 10
    topk_examples: int = 50


def run_benchmark(
    run_context: TrainRunContext,
    params: BenchmarkParams,
    get_activations_fn: Any,
) -> dict[str, Any]:
    """Run the activation collection and return timing info."""
    start = time.perf_counter()

    results = list(
        get_activations_fn(
            run_context=run_context,
            importance_threshold=params.importance_threshold,
            n_batches=params.n_batches,
            n_tokens_either_side=params.n_tokens_either_side,
            batch_size=params.batch_size,
            topk_examples=params.topk_examples,
        )
    )

    elapsed = time.perf_counter() - start

    # Extract the final result
    complete_result = next((r for t, r in results if t == "complete"), None)

    return {
        "elapsed_seconds": elapsed,
        "n_batches": params.n_batches,
        "batch_size": params.batch_size,
        "result": complete_result,
    }


def profile_with_cprofile(
    run_context: TrainRunContext,
    params: BenchmarkParams,
    get_activations_fn: Any,
) -> str:
    """Profile with cProfile and return formatted stats."""
    profiler = cProfile.Profile()
    profiler.enable()

    list(
        get_activations_fn(
            run_context=run_context,
            importance_threshold=params.importance_threshold,
            n_batches=params.n_batches,
            n_tokens_either_side=params.n_tokens_either_side,
            batch_size=params.batch_size,
            topk_examples=params.topk_examples,
        )
    )

    profiler.disable()

    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(50)

    return stream.getvalue()


def main():
    parser = argparse.ArgumentParser(description="Benchmark activation context collection")
    parser.add_argument(
        "--original",
        action="store_true",
        help="Use the original (unoptimized) implementation",
    )
    parser.add_argument(
        "--wandb-path",
        default=DEFAULT_WANDB_PATH,
        help=f"W&B run path (default: {DEFAULT_WANDB_PATH})",
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=5,
        help="Number of batches to process (default: 5)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run cProfile profiling",
    )
    parser.add_argument(
        "--flamegraph",
        action="store_true",
        help="Generate flame graph using py-spy (requires sudo or ptrace permissions)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Activation Context Collection Benchmark")
    print("=" * 60)

    # Swap implementation if needed
    impl_name = "ORIGINAL" if args.original else "OPTIMIZED"
    swap_implementation(args.original)

    # Import after swapping
    from spd.app.backend.lib.activation_contexts import get_activations_data_streaming

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nDevice: {device}")
    print(f"Implementation: {impl_name}")

    # Load real run context
    print(f"\nLoading W&B run: {args.wandb_path}")
    service = RunContextService()
    service.load_run(args.wandb_path)
    run_context = service.train_run_context
    assert run_context is not None

    # Get seq_len from config
    seq_len: int = getattr(run_context.config.task_config, "max_seq_len", 128)

    print(f"  Model loaded on: {device}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Components (C): {run_context.config.C}")

    params = BenchmarkParams(
        n_batches=args.n_batches,
        batch_size=32,
        importance_threshold=0.1,
        n_tokens_either_side=10,
        topk_examples=50,
    )

    print("\nBenchmark params:")
    print(f"  n_batches: {params.n_batches}")
    print(f"  batch_size: {params.batch_size}")
    print(f"  importance_threshold: {params.importance_threshold}")
    print(f"  n_tokens_either_side: {params.n_tokens_either_side}")
    print(f"  topk_examples: {params.topk_examples}")
    print(f"  Total tokens: {params.n_batches * params.batch_size * seq_len:,}")

    # Basic timing
    print("\n" + "-" * 40)
    print(f"Basic Timing ({impl_name})")
    print("-" * 40)
    result = run_benchmark(run_context, params, get_activations_data_streaming)
    print(f"Total time: {result['elapsed_seconds']:.3f}s")
    print(f"Time per batch: {result['elapsed_seconds'] / params.n_batches * 1000:.1f}ms")
    tokens_per_sec = params.n_batches * params.batch_size * seq_len / result["elapsed_seconds"]
    print(f"Throughput: {tokens_per_sec:,.0f} tokens/sec")

    # cProfile output
    if args.profile:
        print("\n" + "-" * 40)
        print("cProfile Output (top 50 functions)")
        print("-" * 40)

        profile_output = profile_with_cprofile(run_context, params, get_activations_data_streaming)
        print(profile_output)

    # Flame graph generation
    if args.flamegraph:
        print("\n" + "-" * 40)
        print("Generating Flame Graph")
        print("-" * 40)

        FLAMEGRAPH_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        impl_suffix = "original" if args.original else "optimized"
        svg_path = FLAMEGRAPH_DIR / f"flamegraph_{impl_suffix}_{timestamp}.svg"

        # Create a small script to run the benchmark function
        runner_script = LIB_DIR / "_flamegraph_runner.py"
        runner_script.write_text(f'''
import sys
sys.path.insert(0, "{Path.cwd()}")

from spd.app.backend.services.run_context_service import RunContextService
from spd.app.backend.lib.activation_contexts import get_activations_data_streaming

service = RunContextService()
service.load_run("{args.wandb_path}")
run_context = service.train_run_context

list(get_activations_data_streaming(
    run_context=run_context,
    importance_threshold={params.importance_threshold},
    n_batches={params.n_batches},
    n_tokens_either_side={params.n_tokens_either_side},
    batch_size={params.batch_size},
    topk_examples={params.topk_examples},
))
''')

        print("Running py-spy... (this may require sudo)")
        try:
            result = subprocess.run(
                [
                    "py-spy",
                    "record",
                    "-o",
                    str(svg_path),
                    "--native",
                    "--",
                    "python",
                    str(runner_script),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"Flame graph saved to: {svg_path}")
            else:
                print(f"py-spy failed: {result.stderr}")
                if "permissions" in result.stderr.lower() or "ptrace" in result.stderr.lower():
                    print("Try running with: sudo -E env PATH=$PATH python ...")
        finally:
            runner_script.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
