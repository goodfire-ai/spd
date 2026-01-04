"""Minimal benchmark for LinearComponents masked forward/backward pass.

Tests torch.compile() efficiency on the core LinearComponents operation:
  out = (x @ V * mask) @ U
"""

import time

import fire
import torch
import torch.nn as nn
from torch import Tensor

from spd.models.components import LinearComponents

torch.set_float32_matmul_precision("high")


def benchmark_linear_components(
    d_in: int = 512,
    d_out: int = 512,
    C: int = 100,
    batch_size: int = 64,
    seq_len: int = 256,
    steps: int = 200,
    warmup: int = 50,
    compile_mode: str = "default",
) -> None:
    """Benchmark LinearComponents masked forward and backward pass.

    Args:
        d_in: Input dimension
        d_out: Output dimension
        C: Number of components
        batch_size: Batch size
        seq_len: Sequence length
        steps: Number of benchmark steps
        warmup: Number of warmup steps
        compile_mode: torch.compile mode (default, reduce-overhead, max-autotune)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"d_in={d_in}, d_out={d_out}, C={C}, batch_size={batch_size}, seq_len={seq_len}")
    print(f"Warmup: {warmup}, Steps: {steps}")
    print()

    results = {}

    for use_compile in [False, True]:
        mode_name = "compiled" if use_compile else "eager"
        print(f"{'=' * 60}")
        print(f"Running: {mode_name}")
        print(f"{'=' * 60}")

        # Create LinearComponents
        components = LinearComponents(C=C, d_in=d_in, d_out=d_out, bias=None)
        components.to(device)

        # Define the forward function we want to benchmark
        def forward_backward(
            x: Tensor, mask: Tensor, components: nn.Module, target: Tensor
        ) -> Tensor:
            out = components(x, mask=mask)
            loss = (out - target).pow(2).mean()
            return loss

        if use_compile:
            mode = None if compile_mode == "default" else compile_mode
            print(f"Compiling (mode={mode})...")
            compile_start = time.perf_counter()
            forward_backward = torch.compile(forward_backward, fullgraph=True, mode=mode)
            print(f"torch.compile() call took {time.perf_counter() - compile_start:.2f}s")

        # Setup optimizer
        optimizer = torch.optim.AdamW(components.parameters(), lr=1e-4)

        # Warmup
        print(f"Warming up ({warmup} steps)...")
        for _ in range(warmup):
            optimizer.zero_grad()
            x = torch.randn(batch_size, seq_len, d_in, device=device)
            mask = torch.rand(batch_size, seq_len, C, device=device)
            target = torch.randn(batch_size, seq_len, d_out, device=device)

            loss = forward_backward(x, mask, components, target)
            loss.backward()
            optimizer.step()

        # Benchmark
        print(f"Benchmarking ({steps} steps)...")
        if device == "cuda":
            torch.cuda.synchronize()

        step_times = []
        for _ in range(steps):
            x = torch.randn(batch_size, seq_len, d_in, device=device)
            mask = torch.rand(batch_size, seq_len, C, device=device)
            target = torch.randn(batch_size, seq_len, d_out, device=device)

            if device == "cuda":
                torch.cuda.synchronize()
            step_start = time.perf_counter()

            optimizer.zero_grad()
            loss = forward_backward(x, mask, components, target)
            loss.backward()
            optimizer.step()

            if device == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - step_start)

        avg_time = sum(step_times) / len(step_times)
        min_time = min(step_times)
        max_time = max(step_times)

        results[mode_name] = {"avg": avg_time, "min": min_time, "max": max_time}

        print(f"\n{mode_name.upper()} Results:")
        print(f"  Avg: {avg_time * 1000:.3f}ms")
        print(f"  Min: {min_time * 1000:.3f}ms")
        print(f"  Max: {max_time * 1000:.3f}ms")
        print()

        del components, optimizer
        if device == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print(f"{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    eager_time = results["eager"]["avg"]
    compiled_time = results["compiled"]["avg"]
    speedup = eager_time / compiled_time
    print(f"Eager:    {eager_time * 1000:.3f}ms")
    print(f"Compiled: {compiled_time * 1000:.3f}ms")
    print(f"Speedup:  {speedup:.2f}x ({(speedup - 1) * 100:.1f}%)")


def benchmark_raw_einsum(
    d_in: int = 512,
    d_out: int = 512,
    C: int = 100,
    batch_size: int = 64,
    seq_len: int = 256,
    steps: int = 200,
    warmup: int = 50,
) -> None:
    """Benchmark raw einsum operations (equivalent to LinearComponents) to isolate overhead.

    The core operation is: out = (x @ V * mask) @ U
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"d_in={d_in}, d_out={d_out}, C={C}, batch_size={batch_size}, seq_len={seq_len}")
    print()
    print("Testing RAW einsum: out = (x @ V * mask) @ U")
    print()

    results = {}

    for use_compile in [False, True]:
        mode_name = "compiled" if use_compile else "eager"
        print(f"{'=' * 60}")
        print(f"Running: {mode_name}")
        print(f"{'=' * 60}")

        # Create parameters directly
        V = nn.Parameter(torch.randn(d_in, C, device=device) * 0.02)
        U = nn.Parameter(torch.randn(C, d_out, device=device) * 0.02)

        def forward_backward(x: Tensor, mask: Tensor, V: Tensor, U: Tensor, target: Tensor) -> Tensor:
            inner = x @ V  # (batch, seq, C)
            masked = inner * mask
            out = masked @ U  # (batch, seq, d_out)
            loss = (out - target).pow(2).mean()
            return loss

        if use_compile:
            print("Compiling...")
            compile_start = time.perf_counter()
            forward_backward = torch.compile(forward_backward, fullgraph=True)
            print(f"torch.compile() call took {time.perf_counter() - compile_start:.2f}s")

        optimizer = torch.optim.AdamW([V, U], lr=1e-4)

        # Warmup
        print(f"Warming up ({warmup} steps)...")
        for _ in range(warmup):
            optimizer.zero_grad()
            x = torch.randn(batch_size, seq_len, d_in, device=device)
            mask = torch.rand(batch_size, seq_len, C, device=device)
            target = torch.randn(batch_size, seq_len, d_out, device=device)

            loss = forward_backward(x, mask, V, U, target)
            loss.backward()
            optimizer.step()

        # Benchmark
        print(f"Benchmarking ({steps} steps)...")
        if device == "cuda":
            torch.cuda.synchronize()

        step_times = []
        for _ in range(steps):
            x = torch.randn(batch_size, seq_len, d_in, device=device)
            mask = torch.rand(batch_size, seq_len, C, device=device)
            target = torch.randn(batch_size, seq_len, d_out, device=device)

            if device == "cuda":
                torch.cuda.synchronize()
            step_start = time.perf_counter()

            optimizer.zero_grad()
            loss = forward_backward(x, mask, V, U, target)
            loss.backward()
            optimizer.step()

            if device == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - step_start)

        avg_time = sum(step_times) / len(step_times)
        results[mode_name] = {"avg": avg_time}

        print(f"{mode_name.upper()}: {avg_time * 1000:.3f}ms")
        print()

        del V, U, optimizer
        if device == "cuda":
            torch.cuda.empty_cache()

    speedup = results["eager"]["avg"] / results["compiled"]["avg"]
    print(f"Speedup: {speedup:.2f}x ({(speedup - 1) * 100:.1f}%)")


def benchmark_no_mask(
    d_in: int = 512,
    d_out: int = 512,
    C: int = 100,
    batch_size: int = 64,
    seq_len: int = 256,
    steps: int = 200,
    warmup: int = 50,
) -> None:
    """Benchmark LinearComponents WITHOUT mask to see if masking is the bottleneck."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"d_in={d_in}, d_out={d_out}, C={C}, batch_size={batch_size}, seq_len={seq_len}")
    print()
    print("Testing LinearComponents WITHOUT mask")
    print()

    results = {}

    for use_compile in [False, True]:
        mode_name = "compiled" if use_compile else "eager"
        print(f"{'=' * 60}")
        print(f"Running: {mode_name}")
        print(f"{'=' * 60}")

        components = LinearComponents(C=C, d_in=d_in, d_out=d_out, bias=None)
        components.to(device)

        def forward_backward(x: Tensor, components: nn.Module, target: Tensor) -> Tensor:
            out = components(x, mask=None)
            loss = (out - target).pow(2).mean()
            return loss

        if use_compile:
            print("Compiling...")
            compile_start = time.perf_counter()
            forward_backward = torch.compile(forward_backward, fullgraph=True)
            print(f"torch.compile() call took {time.perf_counter() - compile_start:.2f}s")

        optimizer = torch.optim.AdamW(components.parameters(), lr=1e-4)

        # Warmup
        print(f"Warming up ({warmup} steps)...")
        for _ in range(warmup):
            optimizer.zero_grad()
            x = torch.randn(batch_size, seq_len, d_in, device=device)
            target = torch.randn(batch_size, seq_len, d_out, device=device)

            loss = forward_backward(x, components, target)
            loss.backward()
            optimizer.step()

        # Benchmark
        print(f"Benchmarking ({steps} steps)...")
        if device == "cuda":
            torch.cuda.synchronize()

        step_times = []
        for _ in range(steps):
            x = torch.randn(batch_size, seq_len, d_in, device=device)
            target = torch.randn(batch_size, seq_len, d_out, device=device)

            if device == "cuda":
                torch.cuda.synchronize()
            step_start = time.perf_counter()

            optimizer.zero_grad()
            loss = forward_backward(x, components, target)
            loss.backward()
            optimizer.step()

            if device == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - step_start)

        avg_time = sum(step_times) / len(step_times)
        results[mode_name] = {"avg": avg_time}

        print(f"{mode_name.upper()}: {avg_time * 1000:.3f}ms")
        print()

        del components, optimizer
        if device == "cuda":
            torch.cuda.empty_cache()

    speedup = results["eager"]["avg"] / results["compiled"]["avg"]
    print(f"Speedup: {speedup:.2f}x ({(speedup - 1) * 100:.1f}%)")


def benchmark_forward_only(
    d_in: int = 512,
    d_out: int = 512,
    C: int = 100,
    batch_size: int = 64,
    seq_len: int = 256,
    steps: int = 500,
    warmup: int = 100,
    compile_mode: str = "default",
) -> None:
    """Benchmark LinearComponents forward pass only (no backward)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"d_in={d_in}, d_out={d_out}, C={C}, batch_size={batch_size}, seq_len={seq_len}")
    print(f"Compile mode: {compile_mode}")
    print()
    print("Testing FORWARD ONLY (no backward)")
    print()

    results = {}

    for use_compile in [False, True]:
        mode_name = "compiled" if use_compile else "eager"
        print(f"{'=' * 60}")
        print(f"Running: {mode_name}")
        print(f"{'=' * 60}")

        components = LinearComponents(C=C, d_in=d_in, d_out=d_out, bias=None)
        components.to(device)

        def forward_fn(x: Tensor, mask: Tensor, components: nn.Module) -> Tensor:
            return components(x, mask=mask)

        if use_compile:
            mode = None if compile_mode == "default" else compile_mode
            print(f"Compiling (mode={mode})...")
            compile_start = time.perf_counter()
            forward_fn = torch.compile(forward_fn, fullgraph=True, mode=mode)
            print(f"torch.compile() call took {time.perf_counter() - compile_start:.2f}s")

        # Fixed mask for fair comparison
        mask = torch.rand(batch_size, seq_len, C, device=device)

        # Warmup
        print(f"Warming up ({warmup} steps)...")
        with torch.no_grad():
            for _ in range(warmup):
                x = torch.randn(batch_size, seq_len, d_in, device=device)
                _ = forward_fn(x, mask, components)

        # Benchmark
        print(f"Benchmarking ({steps} steps)...")
        if device == "cuda":
            torch.cuda.synchronize()

        step_times = []
        with torch.no_grad():
            for _ in range(steps):
                x = torch.randn(batch_size, seq_len, d_in, device=device)

                if device == "cuda":
                    torch.cuda.synchronize()
                step_start = time.perf_counter()

                _ = forward_fn(x, mask, components)

                if device == "cuda":
                    torch.cuda.synchronize()
                step_times.append(time.perf_counter() - step_start)

        avg_time = sum(step_times) / len(step_times)
        results[mode_name] = {"avg": avg_time}

        print(f"{mode_name.upper()}: {avg_time * 1000:.3f}ms")
        print()

        del components
        if device == "cuda":
            torch.cuda.empty_cache()

    speedup = results["eager"]["avg"] / results["compiled"]["avg"]
    print(f"Speedup: {speedup:.2f}x ({(speedup - 1) * 100:.1f}%)")


class MaskedLinearComponents(nn.Module):
    """Wrapper that stores mask as a module property for simpler forward signature."""

    def __init__(self, C: int, d_in: int, d_out: int):
        super().__init__()
        self.components = LinearComponents(C=C, d_in=d_in, d_out=d_out, bias=None)
        self.mask: Tensor | None = None

    def set_mask(self, mask: Tensor) -> None:
        self.mask = mask

    def forward(self, x: Tensor) -> Tensor:
        return self.components(x, mask=self.mask)


def benchmark_module_compile(
    d_in: int = 512,
    d_out: int = 512,
    C: int = 512,
    batch_size: int = 64,
    seq_len: int = 256,
    steps: int = 200,
    warmup: int = 50,
    compile_mode: str = "reduce-overhead",
) -> None:
    """Benchmark compiling the module directly instead of a function wrapper."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"d_in={d_in}, d_out={d_out}, C={C}, batch_size={batch_size}, seq_len={seq_len}")
    print(f"Compile mode: {compile_mode}")
    print()
    print("Testing MODULE compile (mask as property)")
    print()

    results = {}

    for use_compile in [False, True]:
        mode_name = "compiled" if use_compile else "eager"
        print(f"{'=' * 60}")
        print(f"Running: {mode_name}")
        print(f"{'=' * 60}")

        model = MaskedLinearComponents(C=C, d_in=d_in, d_out=d_out)
        model.to(device)

        if use_compile:
            mode = None if compile_mode == "default" else compile_mode
            print(f"Compiling module (mode={mode})...")
            compile_start = time.perf_counter()
            model = torch.compile(model, fullgraph=True, mode=mode)  # type: ignore
            print(f"torch.compile() call took {time.perf_counter() - compile_start:.2f}s")

        base_model = model._orig_mod if use_compile else model  # type: ignore
        optimizer = torch.optim.AdamW(base_model.components.parameters(), lr=1e-4)

        # Warmup with fixed mask
        print(f"Warming up ({warmup} steps)...")
        warmup_mask = torch.rand(batch_size, seq_len, C, device=device)
        base_model.set_mask(warmup_mask)
        for _ in range(warmup):
            optimizer.zero_grad()
            x = torch.randn(batch_size, seq_len, d_in, device=device)
            target = torch.randn(batch_size, seq_len, d_out, device=device)

            out = model(x)
            loss = (out - target).pow(2).mean()
            loss.backward()
            optimizer.step()

        # Benchmark with FIXED mask (no recompilation)
        print(f"Benchmarking ({steps} steps) with FIXED mask...")
        if device == "cuda":
            torch.cuda.synchronize()

        fixed_mask = torch.rand(batch_size, seq_len, C, device=device)
        base_model.set_mask(fixed_mask)

        step_times = []
        for _ in range(steps):
            x = torch.randn(batch_size, seq_len, d_in, device=device)
            target = torch.randn(batch_size, seq_len, d_out, device=device)

            if device == "cuda":
                torch.cuda.synchronize()
            step_start = time.perf_counter()

            optimizer.zero_grad()
            out = model(x)
            loss = (out - target).pow(2).mean()
            loss.backward()
            optimizer.step()

            if device == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - step_start)

        avg_time = sum(step_times) / len(step_times)
        results[mode_name] = {"avg": avg_time}

        print(f"{mode_name.upper()}: {avg_time * 1000:.3f}ms")
        print()

        del model, base_model, optimizer
        if device == "cuda":
            torch.cuda.empty_cache()

    speedup = results["eager"]["avg"] / results["compiled"]["avg"]
    print(f"Speedup: {speedup:.2f}x ({(speedup - 1) * 100:.1f}%)")


def benchmark_pure_function(
    d_in: int = 512,
    d_out: int = 512,
    C: int = 512,
    batch_size: int = 64,
    seq_len: int = 256,
    steps: int = 200,
    warmup: int = 50,
    compile_mode: str = "reduce-overhead",
) -> None:
    """Benchmark a pure function (no module state) with explicit parameters."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"d_in={d_in}, d_out={d_out}, C={C}, batch_size={batch_size}, seq_len={seq_len}")
    print(f"Compile mode: {compile_mode}")
    print()
    print("Testing PURE FUNCTION: out = (x @ V * mask) @ U")
    print()

    def masked_linear(x: Tensor, V: Tensor, U: Tensor, mask: Tensor) -> Tensor:
        """Pure function: (x @ V * mask) @ U"""
        inner = x @ V
        masked = inner * mask
        return masked @ U

    results = {}

    for use_compile in [False, True]:
        mode_name = "compiled" if use_compile else "eager"
        print(f"{'=' * 60}")
        print(f"Running: {mode_name}")
        print(f"{'=' * 60}")

        V = nn.Parameter(torch.randn(d_in, C, device=device) * 0.02)
        U = nn.Parameter(torch.randn(C, d_out, device=device) * 0.02)

        fn = masked_linear
        if use_compile:
            mode = None if compile_mode == "default" else compile_mode
            print(f"Compiling (mode={mode})...")
            compile_start = time.perf_counter()
            fn = torch.compile(masked_linear, fullgraph=True, mode=mode)
            print(f"torch.compile() call took {time.perf_counter() - compile_start:.2f}s")

        optimizer = torch.optim.AdamW([V, U], lr=1e-4)

        # Warmup
        print(f"Warming up ({warmup} steps)...")
        for _ in range(warmup):
            optimizer.zero_grad()
            x = torch.randn(batch_size, seq_len, d_in, device=device)
            mask = torch.rand(batch_size, seq_len, C, device=device)
            target = torch.randn(batch_size, seq_len, d_out, device=device)

            out = fn(x, V, U, mask)
            loss = (out - target).pow(2).mean()
            loss.backward()
            optimizer.step()

        # Benchmark
        print(f"Benchmarking ({steps} steps)...")
        if device == "cuda":
            torch.cuda.synchronize()

        step_times = []
        for _ in range(steps):
            x = torch.randn(batch_size, seq_len, d_in, device=device)
            mask = torch.rand(batch_size, seq_len, C, device=device)
            target = torch.randn(batch_size, seq_len, d_out, device=device)

            if device == "cuda":
                torch.cuda.synchronize()
            step_start = time.perf_counter()

            optimizer.zero_grad()
            out = fn(x, V, U, mask)
            loss = (out - target).pow(2).mean()
            loss.backward()
            optimizer.step()

            if device == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - step_start)

        avg_time = sum(step_times) / len(step_times)
        results[mode_name] = {"avg": avg_time}

        print(f"{mode_name.upper()}: {avg_time * 1000:.3f}ms")
        print()

        del V, U, optimizer
        if device == "cuda":
            torch.cuda.empty_cache()

    speedup = results["eager"]["avg"] / results["compiled"]["avg"]
    print(f"Speedup: {speedup:.2f}x ({(speedup - 1) * 100:.1f}%)")


def benchmark_fp16(
    d_in: int = 512,
    d_out: int = 512,
    C: int = 512,
    batch_size: int = 64,
    seq_len: int = 256,
    steps: int = 200,
    warmup: int = 50,
    compile_mode: str = "reduce-overhead",
) -> None:
    """Benchmark with fp16 to see if tensor cores + compile helps."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"d_in={d_in}, d_out={d_out}, C={C}, batch_size={batch_size}, seq_len={seq_len}")
    print(f"Compile mode: {compile_mode}")
    print()
    print("Testing FP16: out = (x @ V * mask) @ U")
    print()

    def masked_linear(x: Tensor, V: Tensor, U: Tensor, mask: Tensor) -> Tensor:
        inner = x @ V
        masked = inner * mask
        return masked @ U

    results = {}

    for use_compile in [False, True]:
        mode_name = "compiled" if use_compile else "eager"
        print(f"{'=' * 60}")
        print(f"Running: {mode_name}")
        print(f"{'=' * 60}")

        V = nn.Parameter(torch.randn(d_in, C, device=device, dtype=torch.float16) * 0.02)
        U = nn.Parameter(torch.randn(C, d_out, device=device, dtype=torch.float16) * 0.02)

        fn = masked_linear
        if use_compile:
            mode = None if compile_mode == "default" else compile_mode
            print(f"Compiling (mode={mode})...")
            compile_start = time.perf_counter()
            fn = torch.compile(masked_linear, fullgraph=True, mode=mode)
            print(f"torch.compile() call took {time.perf_counter() - compile_start:.2f}s")

        optimizer = torch.optim.AdamW([V, U], lr=1e-4)

        # Warmup
        print(f"Warming up ({warmup} steps)...")
        for _ in range(warmup):
            optimizer.zero_grad()
            x = torch.randn(batch_size, seq_len, d_in, device=device, dtype=torch.float16)
            mask = torch.rand(batch_size, seq_len, C, device=device, dtype=torch.float16)
            target = torch.randn(batch_size, seq_len, d_out, device=device, dtype=torch.float16)

            out = fn(x, V, U, mask)
            loss = (out - target).pow(2).mean()
            loss.backward()
            optimizer.step()

        # Benchmark
        print(f"Benchmarking ({steps} steps)...")
        if device == "cuda":
            torch.cuda.synchronize()

        step_times = []
        for _ in range(steps):
            x = torch.randn(batch_size, seq_len, d_in, device=device, dtype=torch.float16)
            mask = torch.rand(batch_size, seq_len, C, device=device, dtype=torch.float16)
            target = torch.randn(batch_size, seq_len, d_out, device=device, dtype=torch.float16)

            if device == "cuda":
                torch.cuda.synchronize()
            step_start = time.perf_counter()

            optimizer.zero_grad()
            out = fn(x, V, U, mask)
            loss = (out - target).pow(2).mean()
            loss.backward()
            optimizer.step()

            if device == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - step_start)

        avg_time = sum(step_times) / len(step_times)
        results[mode_name] = {"avg": avg_time}

        print(f"{mode_name.upper()}: {avg_time * 1000:.3f}ms")
        print()

        del V, U, optimizer
        if device == "cuda":
            torch.cuda.empty_cache()

    speedup = results["eager"]["avg"] / results["compiled"]["avg"]
    print(f"Speedup: {speedup:.2f}x ({(speedup - 1) * 100:.1f}%)")


if __name__ == "__main__":
    fire.Fire(
        {
            "components": benchmark_linear_components,
            "einsum": benchmark_raw_einsum,
            "no_mask": benchmark_no_mask,
            "forward": benchmark_forward_only,
            "module": benchmark_module_compile,
            "pure": benchmark_pure_function,
            "fp16": benchmark_fp16,
        }
    )
