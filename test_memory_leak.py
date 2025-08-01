#!/usr/bin/env python
"""Simple script to test GPU memory usage during LM decomposition."""

import gc
from pathlib import Path

import torch

from spd.configs import Config
from spd.utils.general_utils import load_config


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.4f} GB, Reserved: {reserved:.4f} GB")
    else:
        print("No GPU available")


def main():
    # Load config
    config_path = Path("/mnt/polished-lake/home/braun/spd/spd/experiments/lm/ss_mlp_config.yaml")
    config = load_config(config_path, config_model=Config)

    # Reduce steps for testing
    config.steps = 5100  # Just past one eval cycle
    config.eval_freq = 1000
    config.slow_eval_freq = 5000
    config.train_log_freq = 100

    # Disable wandb for this test
    config.wandb_project = None

    print("Initial memory state:")
    print_gpu_memory()

    # Import and run the decomposition
    from spd.experiments.lm.lm_decomposition import main as lm_main

    print("\nStarting LM decomposition...")
    print(
        f"Steps: {config.steps}, Eval freq: {config.eval_freq}, Slow eval freq: {config.slow_eval_freq}"
    )

    # Track memory at key points
    import spd.run_spd

    original_eval = spd.eval.eval

    def wrapped_eval(*args, **kwargs):
        print("\n=== Before eval ===")
        print_gpu_memory()

        result = original_eval(*args, **kwargs)

        print("=== After eval ===")
        print_gpu_memory()

        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()

        print("=== After GC ===")
        print_gpu_memory()

        return result

    # Monkey patch the eval function
    spd.eval.eval = wrapped_eval

    try:
        lm_main(config)
    finally:
        # Restore original
        spd.eval.eval = original_eval

    print("\n=== Final memory state ===")
    print_gpu_memory()


if __name__ == "__main__":
    main()
