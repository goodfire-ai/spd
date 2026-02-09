"""Benchmark streaming vs non-streaming data loading for Pile training.

Compares:
1. Pre-cached: Pre-fetch N batches from streaming, then train from memory (simulates non-streaming)
2. Streaming: Train directly from the streaming data loader

This avoids downloading the entire Pile dataset while giving a fair throughput comparison.
"""

import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn

from spd.data import create_data_loader
from spd.pretrain.models import MODEL_CLASSES
from spd.pretrain.train import Config, load_config

CONFIG_PATH = Path("spd/pretrain/configs/pile_llama_simple_mlp-4L-768.yaml")

WARMUP_STEPS = 10
BENCHMARK_STEPS = 50
PREFETCH_BATCHES = WARMUP_STEPS + BENCHMARK_STEPS + 10  # Extra margin


def setup_model(config: Config, device: str) -> nn.Module:
    model_cls = MODEL_CLASSES[config.model.model_type]
    model: nn.Module = model_cls(config.model)
    model.to(device)
    model.train()
    if config.compile:
        model = torch.compile(model)  # type: ignore
    return model


def benchmark_precached(
    model: nn.Module,
    cached_batches: list[torch.Tensor],
    B: int,
    T: int,
    device: str,
    ctx: torch.amp.autocast | nullcontext,
    config: Config,
) -> dict[str, float]:
    """Benchmark training with pre-cached data (simulates non-streaming)."""
    optimizer = model.configure_optimizers(  # type: ignore
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(0.9, 0.95),
        device_type=device,
        zero_stage=0,
    )

    # Warmup
    for i in range(WARMUP_STEPS):
        optimizer.zero_grad(set_to_none=True)
        bat = cached_batches[i].to(device)
        x = bat[:, :-1].contiguous()
        y = bat[:, 1:].contiguous()
        with ctx:
            _, loss = model(x, y, return_logits=False)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

    # Benchmark
    step_times = []
    data_times = []
    compute_times = []
    for i in range(WARMUP_STEPS, WARMUP_STEPS + BENCHMARK_STEPS):
        t0 = time.perf_counter()
        bat = cached_batches[i].to(device)
        t_data = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        x = bat[:, :-1].contiguous()
        y = bat[:, 1:].contiguous()
        with ctx:
            _, loss = model(x, y, return_logits=False)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        data_times.append(t_data - t0)
        compute_times.append(t1 - t_data)
        step_times.append(t1 - t0)

    return {
        "avg_step_ms": sum(step_times) / len(step_times) * 1000,
        "avg_data_ms": sum(data_times) / len(data_times) * 1000,
        "avg_compute_ms": sum(compute_times) / len(compute_times) * 1000,
        "median_step_ms": sorted(step_times)[len(step_times) // 2] * 1000,
        "min_step_ms": min(step_times) * 1000,
        "max_step_ms": max(step_times) * 1000,
        "tokens_per_sec": B * T / (sum(step_times) / len(step_times)),
    }


def benchmark_streaming(
    model: nn.Module,
    train_iter,
    B: int,
    T: int,
    device: str,
    ctx: torch.amp.autocast | nullcontext,
    config: Config,
) -> dict[str, float]:
    """Benchmark training with streaming data loader."""
    optimizer = model.configure_optimizers(  # type: ignore
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(0.9, 0.95),
        device_type=device,
        zero_stage=0,
    )

    # Warmup
    for _ in range(WARMUP_STEPS):
        optimizer.zero_grad(set_to_none=True)
        bat = next(train_iter)["input_ids"].to(torch.long)
        bat = bat.view(B, T + 1).to(device)
        x = bat[:, :-1].contiguous()
        y = bat[:, 1:].contiguous()
        with ctx:
            _, loss = model(x, y, return_logits=False)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

    # Benchmark
    step_times = []
    data_times = []
    compute_times = []
    for _ in range(BENCHMARK_STEPS):
        t0 = time.perf_counter()
        bat = next(train_iter)["input_ids"].to(torch.long)
        bat = bat.view(B, T + 1).to(device)
        t_data = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        x = bat[:, :-1].contiguous()
        y = bat[:, 1:].contiguous()
        with ctx:
            _, loss = model(x, y, return_logits=False)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        data_times.append(t_data - t0)
        compute_times.append(t1 - t_data)
        step_times.append(t1 - t0)

    return {
        "avg_step_ms": sum(step_times) / len(step_times) * 1000,
        "avg_data_ms": sum(data_times) / len(data_times) * 1000,
        "avg_compute_ms": sum(compute_times) / len(compute_times) * 1000,
        "median_step_ms": sorted(step_times)[len(step_times) // 2] * 1000,
        "min_step_ms": min(step_times) * 1000,
        "max_step_ms": max(step_times) * 1000,
        "tokens_per_sec": B * T / (sum(step_times) / len(step_times)),
    }


def main():
    config = load_config(CONFIG_PATH, config_model=Config)

    T = config.train_dataset_config.n_ctx
    train_dataset_config = config.train_dataset_config.model_copy(
        update={"n_ctx": config.train_dataset_config.n_ctx + 1}
    )

    device = "cuda"
    B = 64  # Reduced from config.batch_size (1024) to fit on single GPU

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
        config.dtype
    ]
    ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    torch.manual_seed(45)
    torch.cuda.manual_seed(45)
    if config.tensorcores:
        torch.set_float32_matmul_precision("high")

    print(f"Config: batch_size={B}, n_ctx={T}, dtype={config.dtype}")
    print(f"Warmup steps: {WARMUP_STEPS}, Benchmark steps: {BENCHMARK_STEPS}")
    print()

    # --- Phase 1: Create streaming data loader and pre-fetch batches ---
    print("=" * 60)
    print("Setting up streaming data loader and pre-fetching batches...")
    print("=" * 60)

    train_loader, _ = create_data_loader(
        dataset_config=train_dataset_config,
        batch_size=B,
        buffer_size=1000,
        global_seed=0,
    )
    train_iter = iter(train_loader)

    print(f"Pre-fetching {PREFETCH_BATCHES} batches from streaming...")
    t_prefetch_start = time.perf_counter()
    cached_batches = []
    for i in range(PREFETCH_BATCHES):
        bat = next(train_iter)["input_ids"].to(torch.long)
        cached_batches.append(bat.view(B, T + 1))
        if (i + 1) % 10 == 0:
            print(f"  Pre-fetched {i + 1}/{PREFETCH_BATCHES} batches")
    t_prefetch_end = time.perf_counter()
    print(f"Pre-fetching took {t_prefetch_end - t_prefetch_start:.1f}s")
    print()

    # --- Phase 2: Benchmark with pre-cached data ---
    print("=" * 60)
    print("Benchmarking PRE-CACHED (simulated non-streaming)...")
    print("=" * 60)

    model = setup_model(config, device)
    precached_results = benchmark_precached(model, cached_batches, B, T, device, ctx, config)
    del model
    torch.cuda.empty_cache()

    print(f"  Avg step:    {precached_results['avg_step_ms']:.2f} ms")
    print(f"  Avg data:    {precached_results['avg_data_ms']:.2f} ms")
    print(f"  Avg compute: {precached_results['avg_compute_ms']:.2f} ms")
    print(f"  Median step: {precached_results['median_step_ms']:.2f} ms")
    print(
        f"  Min/Max:     {precached_results['min_step_ms']:.2f} / {precached_results['max_step_ms']:.2f} ms"
    )
    print(f"  Tokens/sec:  {precached_results['tokens_per_sec']:.0f}")
    print()

    # --- Phase 3: Benchmark with streaming ---
    print("=" * 60)
    print("Benchmarking STREAMING...")
    print("=" * 60)

    # Create a fresh streaming data loader
    train_loader2, _ = create_data_loader(
        dataset_config=train_dataset_config,
        batch_size=B,
        buffer_size=1000,
        global_seed=0,
    )
    train_iter2 = iter(train_loader2)

    model = setup_model(config, device)
    streaming_results = benchmark_streaming(model, train_iter2, B, T, device, ctx, config)
    del model
    torch.cuda.empty_cache()

    print(f"  Avg step:    {streaming_results['avg_step_ms']:.2f} ms")
    print(f"  Avg data:    {streaming_results['avg_data_ms']:.2f} ms")
    print(f"  Avg compute: {streaming_results['avg_compute_ms']:.2f} ms")
    print(f"  Median step: {streaming_results['median_step_ms']:.2f} ms")
    print(
        f"  Min/Max:     {streaming_results['min_step_ms']:.2f} / {streaming_results['max_step_ms']:.2f} ms"
    )
    print(f"  Tokens/sec:  {streaming_results['tokens_per_sec']:.0f}")
    print()

    # --- Summary ---
    print("=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    overhead_ms = streaming_results["avg_step_ms"] - precached_results["avg_step_ms"]
    overhead_pct = overhead_ms / precached_results["avg_step_ms"] * 100
    data_overhead_ms = streaming_results["avg_data_ms"] - precached_results["avg_data_ms"]

    print(f"{'Metric':<25} {'Pre-cached':>12} {'Streaming':>12} {'Overhead':>12}")
    print("-" * 61)
    print(
        f"{'Avg step (ms)':<25} {precached_results['avg_step_ms']:>12.2f} "
        f"{streaming_results['avg_step_ms']:>12.2f} {overhead_ms:>+12.2f}"
    )
    print(
        f"{'Avg data load (ms)':<25} {precached_results['avg_data_ms']:>12.2f} "
        f"{streaming_results['avg_data_ms']:>12.2f} {data_overhead_ms:>+12.2f}"
    )
    print(
        f"{'Avg compute (ms)':<25} {precached_results['avg_compute_ms']:>12.2f} "
        f"{streaming_results['avg_compute_ms']:>12.2f} "
        f"{streaming_results['avg_compute_ms'] - precached_results['avg_compute_ms']:>+12.2f}"
    )
    print(
        f"{'Tokens/sec':<25} {precached_results['tokens_per_sec']:>12.0f} "
        f"{streaming_results['tokens_per_sec']:>12.0f} "
        f"{streaming_results['tokens_per_sec'] - precached_results['tokens_per_sec']:>+12.0f}"
    )
    print()
    print(f"Streaming overhead: {overhead_ms:+.2f} ms/step ({overhead_pct:+.1f}%)")
    if abs(overhead_pct) < 5:
        print("=> Streaming overhead is NEGLIGIBLE (<5%)")
    elif overhead_pct < 20:
        print("=> Streaming overhead is MODERATE (5-20%)")
    else:
        print("=> Streaming overhead is SIGNIFICANT (>20%)")


if __name__ == "__main__":
    main()
