"""Minimal toy benchmark to isolate torch.compile() performance.

This tests a single linear layer with component decomposition to understand
why torch.compile() isn't providing more speedup.
"""

import time
from typing import override

import fire
import torch
import torch.nn as nn
from torch import Tensor

from spd.models.component_model import ComponentModel
from spd.models.components import ComponentsMaskInfo, make_mask_infos
from spd.utils.module_utils import ModulePathInfo

torch.set_float32_matmul_precision("high")


class ToyModel(nn.Module):
    """Simple model with a single linear layer."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out, bias=False)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


def benchmark_toy(
    d_in: int = 512,
    d_out: int = 512,
    C: int = 100,
    batch_size: int = 64,
    seq_len: int = 256,
    steps: int = 100,
    warmup: int = 20,
) -> None:
    """Benchmark a toy model with component decomposition.

    Args:
        d_in: Input dimension
        d_out: Output dimension
        C: Number of components
        batch_size: Batch size
        seq_len: Sequence length
        steps: Number of benchmark steps
        warmup: Number of warmup steps
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

        # Create fresh model
        target_model = ToyModel(d_in, d_out)
        target_model.eval()
        target_model.requires_grad_(False)

        model = ComponentModel(
            target_model=target_model,
            module_path_info=[ModulePathInfo(module_path="linear", C=C)],
            ci_fn_type="mlp",
            ci_fn_hidden_dims=[32],
            pretrained_model_output_attr=None,
            sigmoid_type="leaky_hard",
        )
        model.to(device)

        if use_compile:
            print("Compiling model...")
            compile_start = time.perf_counter()
            model = torch.compile(model, fullgraph=False)  # type: ignore
            print(f"torch.compile() call took {time.perf_counter() - compile_start:.2f}s")

        component_model = model._orig_mod if use_compile else model  # type: ignore

        # Setup optimizer
        params = list(component_model.components["linear"].parameters())
        params += list(component_model.ci_fns["linear"].parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-4)

        # Warmup
        print(f"Warming up ({warmup} steps)...")
        warmup_start = time.perf_counter()
        for _ in range(warmup):
            optimizer.zero_grad()

            # Random input
            x = torch.randn(batch_size, seq_len, d_in, device=device)

            # Forward with caching
            out, cache = model(x, cache_type="input")  # type: ignore

            # Calculate CI
            ci = component_model.calc_causal_importances(
                pre_weight_acts=cache,
                detach_inputs=False,
                sampling="continuous",
            )

            # Create binary mask from CI
            mask = (ci.lower_leaky["linear"] > 0.5).float()
            mask_infos = make_mask_infos({"linear": mask})

            # Forward with mask
            masked_out = model(x, mask_infos=mask_infos)  # type: ignore

            # Simple MSE loss
            loss = (masked_out - out.detach()).pow(2).mean()
            loss.backward()
            optimizer.step()

        warmup_time = time.perf_counter() - warmup_start
        print(f"Warmup took {warmup_time:.2f}s ({warmup_time / warmup * 1000:.1f}ms/step)")

        # Benchmark
        print(f"Benchmarking ({steps} steps)...")
        torch.cuda.synchronize() if device == "cuda" else None

        step_times = []
        for _ in range(steps):
            step_start = time.perf_counter()

            optimizer.zero_grad()
            x = torch.randn(batch_size, seq_len, d_in, device=device)

            out, cache = model(x, cache_type="input")  # type: ignore

            ci = component_model.calc_causal_importances(
                pre_weight_acts=cache,
                detach_inputs=False,
                sampling="continuous",
            )

            mask = (ci.lower_leaky["linear"] > 0.5).float()
            mask_infos = make_mask_infos({"linear": mask})

            masked_out = model(x, mask_infos=mask_infos)  # type: ignore

            loss = (masked_out - out.detach()).pow(2).mean()
            loss.backward()
            optimizer.step()

            if device == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - step_start)

        avg_time = sum(step_times) / len(step_times)
        min_time = min(step_times)
        max_time = max(step_times)

        results[mode_name] = {
            "avg": avg_time,
            "min": min_time,
            "max": max_time,
        }

        print(f"\n{mode_name.upper()} Results:")
        print(f"  Avg: {avg_time * 1000:.2f}ms")
        print(f"  Min: {min_time * 1000:.2f}ms")
        print(f"  Max: {max_time * 1000:.2f}ms")
        print()

        del model, component_model, optimizer
        torch.cuda.empty_cache() if device == "cuda" else None

    # Summary
    print(f"{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    eager_time = results["eager"]["avg"]
    compiled_time = results["compiled"]["avg"]
    speedup = eager_time / compiled_time
    print(f"Eager:    {eager_time * 1000:.2f}ms")
    print(f"Compiled: {compiled_time * 1000:.2f}ms")
    print(f"Speedup:  {speedup:.2f}x ({(speedup - 1) * 100:.1f}%)")


def benchmark_raw_linear(
    d_in: int = 512,
    d_out: int = 512,
    batch_size: int = 64,
    seq_len: int = 256,
    steps: int = 100,
    warmup: int = 20,
) -> None:
    """Benchmark a raw linear layer (no SPD) to establish baseline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"d_in={d_in}, d_out={d_out}, batch_size={batch_size}, seq_len={seq_len}")
    print(f"Warmup: {warmup}, Steps: {steps}")
    print()
    print("This tests a RAW linear layer without SPD to establish baseline.")
    print()

    results = {}

    for use_compile in [False, True]:
        mode_name = "compiled" if use_compile else "eager"
        print(f"{'=' * 60}")
        print(f"Running: {mode_name}")
        print(f"{'=' * 60}")

        model = nn.Linear(d_in, d_out, bias=False).to(device)

        if use_compile:
            print("Compiling model...")
            model = torch.compile(model, fullgraph=False)  # type: ignore

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Warmup
        print(f"Warming up ({warmup} steps)...")
        for _ in range(warmup):
            optimizer.zero_grad()
            x = torch.randn(batch_size, seq_len, d_in, device=device)
            out = model(x)
            loss = out.pow(2).mean()
            loss.backward()
            optimizer.step()

        # Benchmark
        print(f"Benchmarking ({steps} steps)...")
        torch.cuda.synchronize() if device == "cuda" else None

        step_times = []
        for _ in range(steps):
            step_start = time.perf_counter()

            optimizer.zero_grad()
            x = torch.randn(batch_size, seq_len, d_in, device=device)
            out = model(x)
            loss = out.pow(2).mean()
            loss.backward()
            optimizer.step()

            if device == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - step_start)

        avg_time = sum(step_times) / len(step_times)
        results[mode_name] = {"avg": avg_time}

        print(f"{mode_name.upper()}: {avg_time * 1000:.2f}ms")
        print()

        del model, optimizer
        torch.cuda.empty_cache() if device == "cuda" else None

    speedup = results["eager"]["avg"] / results["compiled"]["avg"]
    print(f"Speedup: {speedup:.2f}x ({(speedup - 1) * 100:.1f}%)")


def benchmark_forward_only(
    d_in: int = 512,
    d_out: int = 512,
    C: int = 100,
    batch_size: int = 64,
    seq_len: int = 256,
    steps: int = 200,
    warmup: int = 50,
) -> None:
    """Benchmark just the forward pass (no backward) to isolate compilation benefit."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"d_in={d_in}, d_out={d_out}, C={C}, batch_size={batch_size}, seq_len={seq_len}")
    print(f"Warmup: {warmup}, Steps: {steps}")
    print()
    print("Testing FORWARD ONLY (no backward)")
    print()

    results = {}

    for use_compile in [False, True]:
        mode_name = "compiled" if use_compile else "eager"
        print(f"{'=' * 60}")
        print(f"Running: {mode_name}")
        print(f"{'=' * 60}")

        # Create fresh model
        target_model = ToyModel(d_in, d_out)
        target_model.eval()
        target_model.requires_grad_(False)

        model = ComponentModel(
            target_model=target_model,
            module_path_info=[ModulePathInfo(module_path="linear", C=C)],
            ci_fn_type="mlp",
            ci_fn_hidden_dims=[32],
            pretrained_model_output_attr=None,
            sigmoid_type="leaky_hard",
        )
        model.to(device)

        if use_compile:
            print("Compiling model...")
            compile_start = time.perf_counter()
            model = torch.compile(model, fullgraph=False)  # type: ignore
            print(f"torch.compile() call took {time.perf_counter() - compile_start:.2f}s")

        # Prepare fixed mask
        mask = torch.ones(batch_size, seq_len, C, device=device)
        mask_infos = make_mask_infos({"linear": mask})

        # Warmup
        print(f"Warming up ({warmup} steps)...")
        with torch.no_grad():
            for _ in range(warmup):
                x = torch.randn(batch_size, seq_len, d_in, device=device)
                _ = model(x, mask_infos=mask_infos)  # type: ignore

        # Benchmark
        print(f"Benchmarking ({steps} steps)...")
        torch.cuda.synchronize() if device == "cuda" else None

        step_times = []
        with torch.no_grad():
            for _ in range(steps):
                x = torch.randn(batch_size, seq_len, d_in, device=device)

                torch.cuda.synchronize() if device == "cuda" else None
                step_start = time.perf_counter()

                _ = model(x, mask_infos=mask_infos)  # type: ignore

                torch.cuda.synchronize() if device == "cuda" else None
                step_times.append(time.perf_counter() - step_start)

        avg_time = sum(step_times) / len(step_times)
        results[mode_name] = {"avg": avg_time}

        print(f"{mode_name.upper()}: {avg_time * 1000:.3f}ms")
        print()

        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    speedup = results["eager"]["avg"] / results["compiled"]["avg"]
    print(f"Speedup: {speedup:.2f}x ({(speedup - 1) * 100:.1f}%)")


def benchmark_masked_module_direct(
    d_in: int = 512,
    d_out: int = 512,
    C: int = 100,
    batch_size: int = 64,
    seq_len: int = 256,
    steps: int = 200,
    warmup: int = 50,
) -> None:
    """Benchmark MaskedModule directly to isolate its performance."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"d_in={d_in}, d_out={d_out}, C={C}, batch_size={batch_size}, seq_len={seq_len}")
    print()
    print("Testing MaskedModule.forward() DIRECTLY")
    print()

    from spd.models.components import LinearComponents
    from spd.models.masked_module import MaskedModule

    results = {}

    for use_compile in [False, True]:
        mode_name = "compiled" if use_compile else "eager"
        print(f"{'=' * 60}")
        print(f"Running: {mode_name}")
        print(f"{'=' * 60}")

        # Create MaskedModule directly
        base = nn.Linear(d_in, d_out, bias=False)
        base.requires_grad_(False)
        components = LinearComponents(C=C, d_in=d_in, d_out=d_out, bias=None)

        masked_module = MaskedModule(
            module_name="test",
            base=base,
            components=components,
        )
        masked_module.to(device)

        # Set up state for active forward
        mask = torch.ones(batch_size, seq_len, C, device=device)
        mask_info = ComponentsMaskInfo(component_mask=mask)
        masked_module.set_runtime_state(
            active=True,
            mask_info=mask_info,
            cache_type="none",
            cache=None,
        )

        if use_compile:
            print("Compiling MaskedModule...")
            compile_start = time.perf_counter()
            masked_module = torch.compile(masked_module, fullgraph=False)  # type: ignore
            print(f"torch.compile() call took {time.perf_counter() - compile_start:.2f}s")

        # Warmup
        print(f"Warming up ({warmup} steps)...")
        with torch.no_grad():
            for _ in range(warmup):
                x = torch.randn(batch_size, seq_len, d_in, device=device)
                _ = masked_module(x)

        # Benchmark
        print(f"Benchmarking ({steps} steps)...")
        torch.cuda.synchronize() if device == "cuda" else None

        step_times = []
        with torch.no_grad():
            for _ in range(steps):
                x = torch.randn(batch_size, seq_len, d_in, device=device)

                torch.cuda.synchronize() if device == "cuda" else None
                step_start = time.perf_counter()

                _ = masked_module(x)

                torch.cuda.synchronize() if device == "cuda" else None
                step_times.append(time.perf_counter() - step_start)

        avg_time = sum(step_times) / len(step_times)
        results[mode_name] = {"avg": avg_time}

        print(f"{mode_name.upper()}: {avg_time * 1000:.3f}ms")
        print()

        del masked_module
        torch.cuda.empty_cache() if device == "cuda" else None

    speedup = results["eager"]["avg"] / results["compiled"]["avg"]
    print(f"Speedup: {speedup:.2f}x ({(speedup - 1) * 100:.1f}%)")


if __name__ == "__main__":
    fire.Fire(
        {
            "toy": benchmark_toy,
            "raw": benchmark_raw_linear,
            "forward": benchmark_forward_only,
            "masked": benchmark_masked_module_direct,
        }
    )
