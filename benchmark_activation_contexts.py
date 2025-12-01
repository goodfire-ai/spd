"""Benchmark script for activation_contexts.py optimizations."""

import cProfile
import pstats
import time
from io import StringIO

from spd.app.backend.lib import activation_contexts
from spd.app.backend.lib import activation_contexts_v2
from spd.app.backend.services.run_context_service import RunContextService

# Parameters
RUN_ID = "goodfire/spd/c0k3z78g"
N_BATCHES = 5
BATCH_SIZE = 2
CI_THRESHOLD = 0.0
N_TOKENS_EITHER_SIDE = 10
TOPK_EXAMPLES = 1000


def run_benchmark(module, run_context, name: str):
    """Run benchmark for a specific implementation."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")

    start_time = time.perf_counter()

    generator = module.get_activations_data_streaming(
        run_context=run_context,
        importance_threshold=CI_THRESHOLD,
        n_batches=N_BATCHES,
        n_tokens_either_side=N_TOKENS_EITHER_SIDE,
        batch_size=BATCH_SIZE,
        topk_examples=TOPK_EXAMPLES,
    )

    progress_count = 0
    result = None
    for msg_type, data in generator:
        if msg_type == "progress":
            progress_count += 1
        elif msg_type == "complete":
            result = data

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    print(f"✓ Completed in {elapsed:.3f}s")
    print(f"  Progress updates: {progress_count}")

    if result:
        total_subcomponents = sum(len(layer) for layer in result.layers.values())
        print(f"  Total subcomponents: {total_subcomponents}")

    return elapsed, result


def profile_v2(run_context):
    """Profile V2 to find remaining bottlenecks."""
    print("\n" + "=" * 60)
    print("Profiling V2 (optimized)")
    print("=" * 60)

    profiler = cProfile.Profile()
    profiler.enable()

    generator = activation_contexts_v2.get_activations_data_streaming(
        run_context=run_context,
        importance_threshold=CI_THRESHOLD,
        n_batches=N_BATCHES,
        n_tokens_either_side=N_TOKENS_EITHER_SIDE,
        batch_size=BATCH_SIZE,
        topk_examples=TOPK_EXAMPLES,
    )

    for msg_type, data in generator:
        pass

    profiler.disable()

    # Print top 30 cumulative time functions
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(30)
    print(stream.getvalue())


def main():
    print("Loading run context...")
    service = RunContextService()
    service.load_run(RUN_ID)
    run_context = service.train_run_context
    assert run_context is not None

    print(f"\nConfig:")
    print(f"  Run: {RUN_ID}")
    print(f"  Batches: {N_BATCHES}, Batch size: {BATCH_SIZE}")
    print(f"  CI threshold: {CI_THRESHOLD}, Tokens either side: {N_TOKENS_EITHER_SIDE}")
    print(f"  Top-k examples: {TOPK_EXAMPLES}")

    # Run both implementations
    elapsed_v1, result_v1 = run_benchmark(activation_contexts, run_context, "V1 (current)")
    elapsed_v2, result_v2 = run_benchmark(activation_contexts_v2, run_context, "V2 (optimized)")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"V1 (current):   {elapsed_v1:.3f}s")
    print(f"V2 (optimized): {elapsed_v2:.3f}s")
    speedup = elapsed_v1 / elapsed_v2 if elapsed_v2 > 0 else float("inf")
    print(f"Speedup: {speedup:.2f}x")

    # Quick sanity check - same number of layers and subcomponents
    if result_v1 and result_v2:
        v1_subcomps = sum(len(layer) for layer in result_v1.layers.values())
        v2_subcomps = sum(len(layer) for layer in result_v2.layers.values())
        print(f"\nSanity check:")
        print(f"  V1 subcomponents: {v1_subcomps}")
        print(f"  V2 subcomponents: {v2_subcomps}")
        print(f"  Match: {v1_subcomps == v2_subcomps}")

    # Profile V2
    profile_v2(run_context)


if __name__ == "__main__":
    main()
