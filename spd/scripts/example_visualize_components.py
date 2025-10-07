#!/usr/bin/env python3
"""
Example script showing how to use the component visualization tools.

This demonstrates how to analyze different SPD model runs and compare their
component activation patterns.
"""

import subprocess
import sys
from pathlib import Path


def run_visualization(model_path: str, output_name: str, threshold: float = 0.1):
    """Run component visualization for a specific model."""
    print(f"\n{'=' * 60}")
    print(f"Analyzing model: {model_path}")
    print(f"Output name: {output_name}")
    print(f"Threshold: {threshold}")
    print(f"{'=' * 60}")

    # Create output directory for this specific run
    output_dir = Path("spd/scripts/component_visualization/out") / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create config file for this visualization
    config_content = f"""# Example configuration for component visualization
model_path: "{model_path}"
threshold: {threshold}
figsize: [10, 8]
output_dir: "{output_dir}"
"""

    config_path = output_dir / "example_config.yaml"
    with open(config_path, "w") as f:
        f.write(config_content)

    # Run the visualization script
    cmd = [
        sys.executable,
        "spd/scripts/component_visualization/component_visualization.py",
        str(config_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Visualization completed successfully!")

        # Results are already in the correct directory due to --output_dir parameter
        print(f"Results saved to: {output_dir}")

    except subprocess.CalledProcessError as e:
        print(f"Error running visualization: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")


def main():
    """Run visualizations for different model configurations."""

    # Example model runs to analyze
    models_to_analyze = [
        ("wandb:goodfire/spd/runs/2ki9tfsx", "resid_mlp1_analysis"),
        # Add more model runs here as needed
        # ("wandb:goodfire/spd/another_run", "another_run_analysis"),
    ]

    print("SPD Component Visualization Examples")
    print("=" * 50)

    for model_path, output_name in models_to_analyze:
        try:
            run_visualization(model_path, output_name, threshold=0.1)
        except Exception as e:
            print(f"Failed to analyze {model_path}: {e}")
            continue

    print("\n" + "=" * 60)
    print("All visualizations completed!")
    print("Check the results directories for generated plots and analysis files.")
    print("=" * 60)


if __name__ == "__main__":
    main()
