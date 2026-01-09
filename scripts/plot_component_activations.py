"""Plot component activations vs component ID for high-CI datapoints.

Creates a scatter plot where:
- X-axis: Component ID (categorical/discrete)
- Y-axis: Component activation (normalized per-component to mean=0, std=1)
- Filter: Only includes datapoints where CI > threshold (default 1)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spd.harvest.schemas import (
    ActivationExample,
    ComponentData,
    ComponentTokenPMI,
    get_activation_contexts_dir,
)


def load_activation_contexts_streaming(
    run_id: str,
    max_components: int | None = None,
) -> dict[str, ComponentData]:
    """Load activation contexts with optional early stopping."""
    ctx_dir = get_activation_contexts_dir(run_id)
    path = ctx_dir / "components.jsonl"
    assert path.exists(), f"No harvest data found for run {run_id}"

    components: dict[str, ComponentData] = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if max_components is not None and i >= max_components:
                break
            data = json.loads(line)
            data["activation_examples"] = [
                ActivationExample(
                    token_ids=ex["token_ids"],
                    ci_values=ex["ci_values"],
                    component_acts=ex.get("component_acts", [0.0] * len(ex["token_ids"])),
                )
                for ex in data["activation_examples"]
            ]
            data["input_token_pmi"] = ComponentTokenPMI(**data["input_token_pmi"])
            data["output_token_pmi"] = ComponentTokenPMI(**data["output_token_pmi"])
            comp = ComponentData(**data)
            components[comp.component_key] = comp
    return components


def extract_activations_above_threshold(
    run_id: str,
    ci_threshold: float,
    max_components: int | None = None,
) -> dict[str, list[float]]:
    """Extract component activations for positions where CI > threshold.

    Args:
        run_id: WandB run ID (e.g., "s-7884efcc")
        ci_threshold: Minimum CI value to include
        max_components: If set, only load this many components (for faster testing)

    Returns:
        Dict mapping component_key to list of activation values
    """
    contexts = load_activation_contexts_streaming(run_id, max_components)

    activations_by_component: dict[str, list[float]] = defaultdict(list)

    for component_key, component_data in contexts.items():
        for example in component_data.activation_examples:
            for ci_val, act_val in zip(example.ci_values, example.component_acts, strict=True):
                if ci_val > ci_threshold:
                    activations_by_component[component_key].append(act_val)

    return activations_by_component


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
            normalized[key] = np.zeros(1)
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

    x_vals = []
    y_vals = []
    for component_id, key in enumerate(sorted_keys):
        acts = normalized_by_component[key]
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
    parser.add_argument(
        "--max-components",
        type=int,
        help="Limit to N components (for faster testing)",
    )
    args = parser.parse_args()

    print(f"Loading activation contexts for run {args.run_id}...")
    activations = extract_activations_above_threshold(
        args.run_id, args.ci_threshold, args.max_components
    )

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
