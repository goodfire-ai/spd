#!/usr/bin/env python
"""Test script to isolate which evaluation metric causes memory leaks."""

from pathlib import Path

import torch
import yaml

from spd.configs import Config


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def test_eval_metrics():
    """Test different evaluation metric configurations to isolate memory leak."""

    config_path = Path("/mnt/polished-lake/home/braun/spd/spd/experiments/lm/ss_mlp_config.yaml")

    # Test configurations - each disables different metrics
    test_configs = [
        {"name": "No eval metrics", "metrics": []},
        {"name": "Only CI_L0", "metrics": [{"classname": "CI_L0"}]},
        {
            "name": "Only CEandKLLosses",
            "metrics": [
                {"classname": "CEandKLLosses", "extra_init_kwargs": {"rounding_threshold": 0.1}}
            ],
        },
        {"name": "Only CIHistograms", "metrics": [{"classname": "CIHistograms"}]},
        {
            "name": "Only ComponentActivationDensity",
            "metrics": [{"classname": "ComponentActivationDensity"}],
        },
        {
            "name": "All metrics",
            "metrics": [
                {"classname": "CIHistograms"},
                {"classname": "ComponentActivationDensity"},
                {"classname": "CI_L0"},
                {"classname": "CEandKLLosses", "extra_init_kwargs": {"rounding_threshold": 0.1}},
            ],
        },
    ]

    results = []

    for test_config in test_configs:
        print(f"\n{'=' * 60}")
        print(f"Testing: {test_config['name']}")
        print(f"{'=' * 60}")

        # Load base config
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        # Modify for testing
        config_dict["eval_metrics"] = test_config["metrics"]
        config_dict["steps"] = 6000  # Run through one full eval cycle
        config_dict["eval_freq"] = 1000
        config_dict["slow_eval_freq"] = 5000
        config_dict["wandb_project"] = None  # Disable wandb
        config_dict["train_log_freq"] = 1000  # Reduce logging

        config = Config(**config_dict)

        # Clear GPU memory
        torch.cuda.empty_cache()

        initial_memory = get_gpu_memory_mb()
        print(f"Initial GPU memory: {initial_memory:.2f} MB")

        try:
            # Import here to ensure fresh state
            from spd.experiments.lm.lm_decomposition import main as lm_main

            # Run the experiment
            lm_main(config)

            final_memory = get_gpu_memory_mb()
            memory_increase = final_memory - initial_memory

            print(f"Final GPU memory: {final_memory:.2f} MB")
            print(f"Memory increase: {memory_increase:.2f} MB")

            results.append(
                {
                    "config": test_config["name"],
                    "initial_memory": initial_memory,
                    "final_memory": final_memory,
                    "increase": memory_increase,
                }
            )

        except Exception as e:
            print(f"Error during test: {e}")
            results.append({"config": test_config["name"], "error": str(e)})

        # Force cleanup
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    for result in results:
        if "error" in result:
            print(f"{result['config']}: ERROR - {result['error']}")
        else:
            print(f"{result['config']}: +{result['increase']:.2f} MB")

    # Find the metric causing the most memory increase
    valid_results = [r for r in results if "increase" in r]
    if valid_results:
        worst_result = max(valid_results, key=lambda x: x["increase"])
        print(
            f"\nWorst memory increase: {worst_result['config']} with +{worst_result['increase']:.2f} MB"
        )


if __name__ == "__main__":
    test_eval_metrics()
