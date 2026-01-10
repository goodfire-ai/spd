"""Plot component activations vs component ID for high-CI datapoints.

Creates scatter plots (one per layer) where:
- X-axis: Component rank (ordered by median normalized activation)
- Y-axis: Component activation (normalized per-component to [0, 1])
- Filter: Only plots datapoints where CI > threshold
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


def load_activation_contexts(run_id: str) -> dict[str, ComponentData]:
    """Load all activation contexts."""
    ctx_dir = get_activation_contexts_dir(run_id)
    path = ctx_dir / "components.jsonl"
    assert path.exists(), f"No harvest data found for run {run_id}"

    components: dict[str, ComponentData] = {}
    with open(path) as f:
        for line in f:
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


def extract_activations(
    contexts: dict[str, ComponentData],
    ci_threshold: float,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, list[float]]]]:
    """Extract component activations, separating all vs above-threshold.

    Returns:
        Tuple of:
        - all_activations: layer -> component_key -> all activation values (for normalization)
        - filtered_activations: layer -> component_key -> activations where CI > threshold
    """
    all_activations: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    filtered_activations: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for component_key, component_data in contexts.items():
        layer = component_data.layer
        for example in component_data.activation_examples:
            for ci_val, act_val in zip(example.ci_values, example.component_acts, strict=True):
                all_activations[layer][component_key].append(act_val)
                if ci_val > ci_threshold:
                    filtered_activations[layer][component_key].append(act_val)

    return dict(all_activations), dict(filtered_activations)


def normalize_per_component(
    all_activations: dict[str, list[float]],
    filtered_activations: dict[str, list[float]],
) -> dict[str, np.ndarray]:
    """Normalize filtered activations to [0, 1] using min-max from all activations."""
    normalized = {}
    for key, filtered_acts in filtered_activations.items():
        if not filtered_acts:
            continue
        all_acts = np.array(all_activations[key])
        filtered_arr = np.array(filtered_acts)
        min_val = all_acts.min()
        max_val = all_acts.max()
        if max_val > min_val:
            normalized[key] = (filtered_arr - min_val) / (max_val - min_val)
        else:
            normalized[key] = np.full_like(filtered_arr, 0.5)
    return normalized


def order_by_median(normalized: dict[str, np.ndarray]) -> list[str]:
    """Order component keys by median of their normalized activations (descending)."""
    medians = [(key, np.median(acts)) for key, acts in normalized.items()]
    medians.sort(key=lambda x: x[1], reverse=True)
    return [key for key, _ in medians]


def create_layer_scatter_plot(
    normalized_by_key: dict[str, np.ndarray],
    ordered_keys: list[str],
    layer_name: str,
    output_path: Path,
) -> None:
    """Create scatter plot for a single layer."""
    x_vals = []
    y_vals = []
    for rank, key in enumerate(ordered_keys):
        acts = normalized_by_key[key]
        x_vals.extend([rank] * len(acts))
        y_vals.extend(acts.tolist())

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.scatter(x_vals, y_vals, alpha=0.3, s=1, marker=".")
    ax.set_xlabel("Component Rank (by median activation)")
    ax.set_ylabel("Normalized Component Activation")
    ax.set_title(f"{layer_name}")

    n_components = len(ordered_keys)
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
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="WandB run ID (e.g., 's-7884efcc')")
    parser.add_argument(
        "--ci-threshold",
        type=float,
        default=0.1,
        help="Minimum CI value to include (default: 0.1)",
    )
    args = parser.parse_args()

    output_dir = Path("scripts/outputs") / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading activation contexts for run {args.run_id}...")
    contexts = load_activation_contexts(args.run_id)
    print(f"Loaded {len(contexts)} components")

    print("Extracting activations...")
    all_by_layer, filtered_by_layer = extract_activations(contexts, args.ci_threshold)

    n_layers = len(filtered_by_layer)
    n_total = sum(sum(len(v) for v in layer.values()) for layer in filtered_by_layer.values())
    print(f"Found {n_total} datapoints across {n_layers} layers with CI > {args.ci_threshold}")

    if n_total == 0:
        print("No datapoints found above threshold. Try lowering --ci-threshold.")
        return

    print(f"Creating per-layer plots in {output_dir}/...")
    for layer_name in sorted(all_by_layer.keys()):
        all_acts = all_by_layer[layer_name]
        filtered_acts = filtered_by_layer.get(layer_name, {})
        normalized = normalize_per_component(all_acts, filtered_acts)
        if not normalized:
            continue
        ordered_keys = order_by_median(normalized)
        safe_name = layer_name.replace(".", "_")
        output_path = output_dir / f"{safe_name}.png"
        create_layer_scatter_plot(normalized, ordered_keys, layer_name, output_path)
        print(f"  {output_path}")

    print("Done!")


if __name__ == "__main__":
    main()
