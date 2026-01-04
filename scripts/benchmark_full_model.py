"""Benchmark torch.compile on a full ComponentModel (e.g., SS Llama).

Tests the masked forward/backward pass on a real model rather than isolated components.
"""

import time
from pathlib import Path

import fire
import torch
import torch.nn as nn
from simple_stories_train.run_info import RunInfo as SSRunInfo
from torch import Tensor

from spd.configs import Config
from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos
from spd.utils.general_utils import resolve_class, set_seed
from spd.utils.module_utils import expand_module_patterns

torch.set_float32_matmul_precision("high")


def load_model_and_config(
    config_path: str = "spd/experiments/lm/ss_llama_simple_config.yaml",
) -> tuple[nn.Module, Config]:
    """Load the target model and config."""
    config = Config.from_file(Path(config_path))

    pretrained_model_class = resolve_class(config.pretrained_model_class)
    assert config.pretrained_model_name is not None

    if config.pretrained_model_class.startswith("simple_stories_train"):
        run_info = SSRunInfo.from_path(config.pretrained_model_name)
        target_model = pretrained_model_class.from_run_info(run_info)
    else:
        target_model = pretrained_model_class.from_pretrained(config.pretrained_model_name)

    target_model.eval()
    target_model.requires_grad_(False)

    return target_model, config


def benchmark_full_model(
    config_path: str = "spd/experiments/lm/ss_llama_simple_config.yaml",
    batch_size: int = 32,
    seq_len: int = 256,
    steps: int = 50,
    warmup: int = 10,
    compile_mode: str = "reduce-overhead",
) -> None:
    """Benchmark ComponentModel forward/backward with full model.

    Args:
        config_path: Path to experiment config
        batch_size: Batch size
        seq_len: Sequence length
        steps: Number of benchmark steps
        warmup: Number of warmup steps
        compile_mode: torch.compile mode
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: {config_path}")
    print(f"batch_size={batch_size}, seq_len={seq_len}")
    print(f"Warmup: {warmup}, Steps: {steps}")
    print(f"Compile mode: {compile_mode}")
    print()

    set_seed(42)

    print("Loading model...")
    target_model, config = load_model_and_config(config_path)
    print(f"Model loaded: {type(target_model).__name__}")

    results = {}

    for use_compile in [False, True]:
        mode_name = "compiled" if use_compile else "eager"
        print(f"\n{'=' * 60}")
        print(f"Running: {mode_name}")
        print(f"{'=' * 60}")

        # Reload target model fresh each time (ComponentModel modifies it)
        target_model, config = load_model_and_config(config_path)

        # Create fresh ComponentModel
        module_path_info = expand_module_patterns(target_model, config.all_module_info)

        model = ComponentModel(
            target_model=target_model,
            module_path_info=module_path_info,
            ci_fn_type=config.ci_fn_type,
            ci_fn_hidden_dims=config.ci_fn_hidden_dims,
            pretrained_model_output_attr=config.pretrained_model_output_attr,
            sigmoid_type=config.sigmoid_type,
        )
        model.to(device)

        print(f"ComponentModel created with {len(model.target_module_paths)} modules:")
        for path in model.target_module_paths[:5]:
            print(f"  - {path} (C={model.module_to_c[path]})")
        if len(model.target_module_paths) > 5:
            print(f"  ... and {len(model.target_module_paths) - 5} more")

        if use_compile:
            mode = None if compile_mode == "default" else compile_mode
            print(f"\nCompiling model (mode={mode})...")
            compile_start = time.perf_counter()
            model = torch.compile(model, fullgraph=False, mode=mode)  # type: ignore
            print(f"torch.compile() call took {time.perf_counter() - compile_start:.2f}s")

        component_model = model._orig_mod if use_compile else model  # type: ignore

        # Setup optimizer
        params = []
        for name in component_model.target_module_paths:
            params.extend(component_model.components[name].parameters())
            params.extend(component_model.ci_fns[name].parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-4)

        # Create sample mask
        def create_masks() -> dict[str, Tensor]:
            masks = {}
            for name in component_model.target_module_paths:
                C = component_model.module_to_c[name]
                masks[name] = torch.rand(batch_size, seq_len, C, device=device)
            return masks

        # Warmup
        print(f"\nWarming up ({warmup} steps)...")
        warmup_start = time.perf_counter()
        for i in range(warmup):
            optimizer.zero_grad()

            # Random token input
            x = torch.randint(0, 1000, (batch_size, seq_len), device=device)
            masks = create_masks()
            mask_infos = make_mask_infos(masks)

            # Forward pass with mask
            out = model(x, mask_infos=mask_infos)  # type: ignore

            # Simple loss
            loss = out.mean()
            loss.backward()
            optimizer.step()

            if (i + 1) % 5 == 0:
                print(f"  Warmup step {i + 1}/{warmup}")

        warmup_time = time.perf_counter() - warmup_start
        print(f"Warmup took {warmup_time:.2f}s ({warmup_time / warmup * 1000:.1f}ms/step)")

        # Benchmark
        print(f"\nBenchmarking ({steps} steps)...")
        if device == "cuda":
            torch.cuda.synchronize()

        step_times = []
        for i in range(steps):
            x = torch.randint(0, 1000, (batch_size, seq_len), device=device)
            masks = create_masks()
            mask_infos = make_mask_infos(masks)

            if device == "cuda":
                torch.cuda.synchronize()
            step_start = time.perf_counter()

            optimizer.zero_grad()
            out = model(x, mask_infos=mask_infos)  # type: ignore
            loss = out.mean()
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
        print(f"  Avg: {avg_time * 1000:.2f}ms")
        print(f"  Min: {min_time * 1000:.2f}ms")
        print(f"  Max: {max_time * 1000:.2f}ms")

        del model, component_model, optimizer
        if device == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    eager_time = results["eager"]["avg"]
    compiled_time = results["compiled"]["avg"]
    speedup = eager_time / compiled_time
    print(f"Eager:    {eager_time * 1000:.2f}ms")
    print(f"Compiled: {compiled_time * 1000:.2f}ms")
    print(f"Speedup:  {speedup:.2f}x ({(speedup - 1) * 100:.1f}%)")


if __name__ == "__main__":
    fire.Fire(benchmark_full_model)
