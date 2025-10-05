#!/usr/bin/env python3
"""
Example usage of the magnitude sweep plotting script.

This script demonstrates how to use plot_residmlp_magnitude_sweep.py
with different parameters and model configurations.
"""

import subprocess
import sys
from pathlib import Path


def run_magnitude_sweep_example():
    """Run the magnitude sweep script with example parameters."""

    # Example model path - you'll need to replace this with an actual trained model
    # This could be a wandb path like "wandb:your-project/run-id" or a local path
    model_path = "wandb:your-project/your-run-id"  # Replace with actual model path

    # Base command
    cmd = [
        sys.executable,
        "spd/scripts/magnitude_sweep/magnitude_sweep.py",
        "--model_path",
        model_path,
        "--output_dir",
        "spd/scripts/magnitude_sweep/out/example_magnitude_sweep",
        "--feature_idx",
        "0",  # Activate feature 0
        "--n_steps",
        "200",  # More steps for smoother curves
        "--max_magnitude",
        "3.0",  # Go up to magnitude 3
        "--ci_threshold",
        "0.05",  # Lower threshold to see more gates
        "--figsize_per_subplot",
        "18",
        "12",  # Larger figure size
        "--dpi",
        "200",  # Higher DPI for better quality
    ]

    print("Running magnitude sweep with example parameters...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Script completed successfully!")
        print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("❌ Script failed!")
        print("Error:", e.stderr)
        return False

    return True


def run_multiple_features_example():
    """Run the script for multiple different features."""

    model_path = "wandb:your-project/your-run-id"  # Replace with actual model path
    features_to_test = [0, 5, 10, 15, 20]  # Test different features

    for feature_idx in features_to_test:
        print(f"\n{'=' * 50}")
        print(f"Running magnitude sweep for feature {feature_idx}")
        print(f"{'=' * 50}")

        cmd = [
            sys.executable,
            "spd/scripts/magnitude_sweep/magnitude_sweep.py",
            "--model_path",
            model_path,
            "--output_dir",
            f"spd/scripts/magnitude_sweep/out/feature_{feature_idx}_sweep",
            "--feature_idx",
            str(feature_idx),
            "--n_steps",
            "100",
            "--max_magnitude",
            "2.0",
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ Feature {feature_idx} completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Feature {feature_idx} failed!")
            print("Error:", e.stderr)


def run_high_resolution_example():
    """Run with high resolution settings for publication-quality plots."""

    model_path = "wandb:your-project/your-run-id"  # Replace with actual model path

    cmd = [
        sys.executable,
        "spd/scripts/magnitude_sweep/magnitude_sweep.py",
        "--model_path",
        model_path,
        "--output_dir",
        "spd/scripts/magnitude_sweep/out/high_res_sweep",
        "--feature_idx",
        "0",
        "--n_steps",
        "500",  # Very high resolution
        "--max_magnitude",
        "2.0",
        "--ci_threshold",
        "0.01",  # Very low threshold
        "--figsize_per_subplot",
        "24",
        "16",  # Very large figures
        "--dpi",
        "300",  # Publication quality DPI
    ]

    print("Running high-resolution magnitude sweep...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ High-resolution sweep completed successfully!")
        print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("❌ High-resolution sweep failed!")
        print("Error:", e.stderr)


if __name__ == "__main__":
    print("ResidMLP Magnitude Sweep Examples")
    print("=" * 40)

    # Check if the main script exists
    script_path = Path("spd/scripts/magnitude_sweep/magnitude_sweep.py")
    if not script_path.exists():
        print("❌ Main script not found at spd/scripts/magnitude_sweep/magnitude_sweep.py")
        print("Please make sure you're running this from the project root directory.")
        sys.exit(1)

    print("Available examples:")
    print("1. Basic example with default parameters")
    print("2. Multiple features example")
    print("3. High-resolution example")
    print("4. All examples")

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == "1":
        run_magnitude_sweep_example()
    elif choice == "2":
        run_multiple_features_example()
    elif choice == "3":
        run_high_resolution_example()
    elif choice == "4":
        run_magnitude_sweep_example()
        run_multiple_features_example()
        run_high_resolution_example()
    else:
        print("Invalid choice. Please run the script again and choose 1-4.")

    print("\n" + "=" * 50)
    print("Examples completed!")
    print("Check the output directories for generated plots.")
    print("=" * 50)
