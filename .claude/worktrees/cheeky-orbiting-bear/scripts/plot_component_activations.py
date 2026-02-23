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
import torch

from spd.harvest.schemas import (
    ActivationExample,
    ComponentData,
    ComponentTokenPMI,
    get_activation_contexts_dir,
)
from spd.settings import SPD_OUT_DIR


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


def load_firing_counts(run_id: str) -> dict[str, int]:
    """Load pre-calculated firing counts from harvest data."""
    token_stats_path = SPD_OUT_DIR / "harvest" / run_id / "correlations" / "token_stats.pt"
    assert token_stats_path.exists(), f"No token stats found for run {run_id}"

    data = torch.load(token_stats_path)
    component_keys = data["component_keys"]
    firing_counts = data["firing_counts"]

    return {key: int(count) for key, count in zip(component_keys, firing_counts, strict=True)}


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


def order_by_frequency(
    normalized: dict[str, np.ndarray], firing_counts: dict[str, int]
) -> list[str]:
    """Order component keys by pre-calculated firing counts (descending)."""
    freqs = [(key, firing_counts.get(key, 0)) for key in normalized]
    freqs.sort(key=lambda x: x[1], reverse=True)
    return [key for key, _ in freqs]


def create_layer_scatter_plot(
    normalized_by_key: dict[str, np.ndarray],
    ordered_keys: list[str],
    layer_name: str,
    run_id: str,
    output_path: Path,
    x_label: str = "Component Rank (by median activation)",
    y_label: str = "Normalized Component Activation",
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
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"Layer: {layer_name}   |||   Run id: {run_id}")

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

    base_output_dir = Path("scripts/outputs") / args.run_id / "component-act-scatter"
    output_dir_median = base_output_dir / "order-by-median"
    output_dir_freq = base_output_dir / "order-by-freq"
    output_dir_median.mkdir(parents=True, exist_ok=True)
    output_dir_freq.mkdir(parents=True, exist_ok=True)

    print(f"Loading activation contexts for run {args.run_id}...")
    contexts = load_activation_contexts(args.run_id)
    print(f"Loaded {len(contexts)} components")

    print("Loading firing counts...")
    firing_counts = load_firing_counts(args.run_id)

    print("Extracting activations...")
    all_by_layer, filtered_by_layer = extract_activations(contexts, args.ci_threshold)

    n_layers = len(filtered_by_layer)
    n_total = sum(sum(len(v) for v in layer.values()) for layer in filtered_by_layer.values())
    print(f"Found {n_total} datapoints across {n_layers} layers with CI > {args.ci_threshold}")

    if n_total == 0:
        print("No datapoints found above threshold. Try lowering --ci-threshold.")
        return

    # Create plots ordered by median normalized activation
    print(f"Creating per-layer plots (ordered by median) in {output_dir_median}/...")
    for layer_name in sorted(all_by_layer.keys()):
        all_acts = all_by_layer[layer_name]
        filtered_acts = filtered_by_layer.get(layer_name, {})
        normalized = normalize_per_component(all_acts, filtered_acts)
        if not normalized:
            continue
        ordered_keys = order_by_median(normalized)
        safe_name = layer_name.replace(".", "_")
        output_path = output_dir_median / f"{safe_name}.png"
        create_layer_scatter_plot(normalized, ordered_keys, layer_name, args.run_id, output_path)
        print(f"  {output_path}")

    # Create plots ordered by CI activation frequency (with abs distance from midpoint)
    print(f"Creating per-layer plots (ordered by frequency) in {output_dir_freq}/...")
    for layer_name in sorted(all_by_layer.keys()):
        all_acts = all_by_layer[layer_name]
        filtered_acts = filtered_by_layer.get(layer_name, {})
        normalized = normalize_per_component(all_acts, filtered_acts)
        if not normalized:
            continue
        # Transform to absolute distance from midpoint
        abs_from_midpoint = {key: np.abs(acts - 0.5) for key, acts in normalized.items()}
        ordered_keys = order_by_frequency(abs_from_midpoint, firing_counts)
        safe_name = layer_name.replace(".", "_")
        output_path = output_dir_freq / f"{safe_name}.png"
        create_layer_scatter_plot(
            abs_from_midpoint,
            ordered_keys,
            layer_name,
            args.run_id,
            output_path,
            x_label="Component Rank (by firing frequency)",
            y_label="|Normalized Component Activation - 0.5|",
        )
        print(f"  {output_path}")

    print("Done!")


if __name__ == "__main__":
    main()
