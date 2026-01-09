"""Plot component activations vs component ID for high-CI datapoints.

Creates a scatter plot where:
- X-axis: Component ID (categorical/discrete)
- Y-axis: Component activation (normalized per-component to mean=0, std=1)
- Filter: Only includes datapoints where CI > threshold (default 1)
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spd.harvest.loaders import load_activation_contexts


def extract_activations_above_threshold(
    run_id: str,
    ci_threshold: float,
) -> dict[str, list[float]]:
    """Extract component activations for positions where CI > threshold.

    Args:
        run_id: WandB run ID (e.g., "s-7884efcc")
        ci_threshold: Minimum CI value to include

    Returns:
        Dict mapping component_key to list of activation values
    """
    contexts = load_activation_contexts(run_id)
    assert contexts is not None, f"No harvest data found for run {run_id}"

    activations_by_component: dict[str, list[float]] = defaultdict(list)

    for component_key, component_data in contexts.items():
        for example in component_data.activation_examples:
            for ci_val, act_val in zip(example.ci_values, example.component_acts, strict=True):
                if ci_val > ci_threshold:
                    activations_by_component[component_key].append(act_val)

    return dict(activations_by_component)


def normalize_per_component(
    activations_by_component: dict[str, list[float]],
) -> dict[str, np.ndarray]:
    """Normalize each component's activations to mean=0, std=1.

    Args:
        activations_by_component: Dict mapping component_key to list of activations

    Returns:
        Dict mapping component_key to normalized activation array
    """
    normalized = {}
    for key, acts in activations_by_component.items():
        arr = np.array(acts)
        if len(arr) > 1:
            mean = arr.mean()
            std = arr.std()
            if std > 0:
                normalized[key] = (arr - mean) / std
            else:
                normalized[key] = arr - mean
        elif len(arr) == 1:
            normalized[key] = arr - arr.mean()
        # Skip empty arrays
    return normalized


def create_scatter_plot(
    normalized_by_component: dict[str, np.ndarray],
    output_path: Path | None = None,
) -> None:
    """Create scatter plot of component activations.

    Args:
        normalized_by_component: Dict mapping component_key to normalized activations
        output_path: Path to save plot (if None, displays interactively)
    """
    sorted_keys = sorted(normalized_by_component.keys())
    key_to_id = {key: i for i, key in enumerate(sorted_keys)}

    x_vals = []
    y_vals = []
    for key, acts in normalized_by_component.items():
        component_id = key_to_id[key]
        x_vals.extend([component_id] * len(acts))
        y_vals.extend(acts.tolist())

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.scatter(x_vals, y_vals, alpha=0.3, s=1, marker=".")
    ax.set_xlabel("Component ID")
    ax.set_ylabel("Normalized Component Activation")
    ax.set_title("Component Activations (CI > threshold, per-component normalized)")

    n_components = len(sorted_keys)
    n_points = len(x_vals)
    ax.text(
        0.02,
        0.98,
        f"Components: {n_components}\nDatapoints: {n_points}",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="WandB run ID (e.g., 's-7884efcc')")
    parser.add_argument(
        "--ci-threshold",
        type=float,
        default=1.0,
        help="Minimum CI value to include (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save plot (if not provided, displays interactively)",
    )
    args = parser.parse_args()

    print(f"Loading activation contexts for run {args.run_id}...")
    activations = extract_activations_above_threshold(args.run_id, args.ci_threshold)

    n_components_with_data = len(activations)
    n_total_points = sum(len(v) for v in activations.values())
    print(
        f"Found {n_total_points} datapoints across {n_components_with_data} components with CI > {args.ci_threshold}"
    )

    if n_total_points == 0:
        print("No datapoints found above threshold. Try lowering --ci-threshold.")
        return

    print("Normalizing per-component...")
    normalized = normalize_per_component(activations)

    print("Creating scatter plot...")
    create_scatter_plot(normalized, args.output)


if __name__ == "__main__":
    main()
