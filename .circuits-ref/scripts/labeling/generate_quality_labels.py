#!/usr/bin/env python3
"""
Generate high-quality neuron labels from neurons_to_label_100.json.

Key improvements over previous approach:
1. OUTPUT_FUNCTION: Names SPECIFIC downstream neurons and their weights
2. INPUT_TRIGGER: Focuses on POSITIVE/ACTIVATING connections, not just inhibitors
3. MECHANISM: Includes actual weight values for key connections
4. COMPLETE_FUNCTION: Tells a story of signal flow with weights
"""

import json
from collections import defaultdict
from pathlib import Path


def parse_source_id(source: str) -> tuple:
    """Parse source ID like '15_1816_26' -> (layer=15, neuron=1816, position=26)
    Also handles 'E_128000_0' for embeddings -> ('E', embedding_id, position)
    """
    parts = source.split("_")
    if len(parts) == 3:
        # Check for embedding sources
        if parts[0] == "E":
            return "E", int(parts[1]), int(parts[2])
        try:
            return int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            return None, None, None
    return None, None, None


def parse_target_id(target: str) -> tuple:
    """Parse target ID like '29_12010_26' or 'L_7505_29' -> (layer/L, id, position)"""
    parts = target.split("_")
    if len(parts) == 3:
        # Check if it's a logit target (L_token_position)
        if parts[0] == "L":
            return "L", int(parts[1]), int(parts[2])
        else:
            return int(parts[0]), int(parts[1]), int(parts[2])
    return None, None, None


def aggregate_connections_by_neuron(connections: list, is_upstream: bool = True) -> dict:
    """
    Aggregate connections by neuron ID, summing weights across positions.
    Returns dict: {neuron_id: {"total_weight": float, "max_weight": float, "positions": list, "count": int}}
    """
    by_neuron = defaultdict(lambda: {"total_weight": 0, "weights": [], "positions": [], "count": 0})

    for conn in connections:
        if is_upstream:
            key = conn.get("source", "")
            weight = conn.get("avg_weight", 0)
        else:
            key = conn.get("target", "")
            weight = conn.get("avg_weight", 0)

        if is_upstream:
            layer, neuron, pos = parse_source_id(key)
        else:
            layer, neuron, pos = parse_target_id(key)

        if layer is None:
            continue

        if layer == "E":
            neuron_key = f"Emb/{neuron}"
        elif layer == "L":
            neuron_key = f"Logit/{neuron}"
        else:
            neuron_key = f"L{layer}/N{neuron}"

        by_neuron[neuron_key]["total_weight"] += weight
        by_neuron[neuron_key]["weights"].append(weight)
        by_neuron[neuron_key]["positions"].append(pos)
        by_neuron[neuron_key]["count"] += 1

    # Calculate max and average weights
    for key in by_neuron:
        weights = by_neuron[key]["weights"]
        by_neuron[key]["max_weight"] = max(weights, key=abs) if weights else 0
        by_neuron[key]["avg_weight"] = sum(weights) / len(weights) if weights else 0

    return dict(by_neuron)


def get_activators(upstream: dict) -> list:
    """Get neurons with positive total weight (activators)."""
    activators = []
    for neuron_id, data in upstream.items():
        if data["total_weight"] > 0.01:  # Only significant activators
            activators.append((neuron_id, data["total_weight"], data["max_weight"]))
    return sorted(activators, key=lambda x: -x[1])  # Sort by total weight


def get_inhibitors(upstream: dict) -> list:
    """Get neurons with negative total weight (inhibitors)."""
    inhibitors = []
    for neuron_id, data in upstream.items():
        if data["total_weight"] < -0.01:  # Only significant inhibitors
            inhibitors.append((neuron_id, data["total_weight"], data["max_weight"]))
    return sorted(inhibitors, key=lambda x: x[1])  # Sort by most negative


def get_downstream_neurons(downstream: dict) -> list:
    """Get downstream neuron connections (not logits)."""
    neurons = []
    for target_id, data in downstream.items():
        if not target_id.startswith("Logit/"):
            neurons.append((target_id, data["total_weight"], data["max_weight"]))
    return sorted(neurons, key=lambda x: -abs(x[1]))  # Sort by absolute weight


def get_logit_effects(downstream: dict, logit_effects: list) -> list:
    """Get logit effects with token names."""
    # Use the logit_effects directly since they have token names
    token_weights = defaultdict(list)
    for effect in logit_effects:
        token = effect.get("token", "")
        weight = effect.get("avg_weight", 0)
        token_weights[token].append(weight)

    # Aggregate by token
    result = []
    for token, weights in token_weights.items():
        total = sum(weights)
        max_w = max(weights, key=abs)
        result.append((token, total, max_w))

    return sorted(result, key=lambda x: -abs(x[1]))


def format_weight(w: float) -> str:
    """Format weight with sign."""
    return f"+{w:.2f}" if w > 0 else f"{w:.2f}"


def classify_weight(w: float, is_logit: bool = False) -> str:
    """Classify weight strength."""
    aw = abs(w)
    if is_logit:
        if aw > 3.0:
            return "VERY STRONG"
        elif aw > 1.0:
            return "STRONG"
        elif aw > 0.3:
            return "MODERATE"
        else:
            return "WEAK"
    else:
        if aw > 0.5:
            return "VERY STRONG"
        elif aw > 0.1:
            return "STRONG"
        elif aw > 0.03:
            return "MODERATE"
        else:
            return "WEAK"


def determine_functional_role(neuron_data: dict, activators: list, inhibitors: list,
                               downstream_neurons: list, logit_effects: list) -> str:
    """Determine the functional role of a neuron."""
    layer = neuron_data.get("layer", 0)

    # Check if it has significant logit effects
    has_strong_logits = any(abs(t[1]) > 1.0 for t in logit_effects)

    # Late layers with logit effects
    if layer >= 29 and has_strong_logits:
        # Check token types
        tokens = [t[0].lower() for t in logit_effects[:5]]
        if any(t in tokens for t in [" ben", " cancer", " ben", " car"]):
            return "answer_formatting"
        if any(t in tokens for t in [" syn", " hist", " micro", " pump", " test"]):
            return "domain_detection"
        return "semantic_retrieval"

    # Mid-layer neurons that are hubs
    if layer >= 12 and layer <= 28:
        # Check if it's a routing hub (many downstream connections, no logits)
        if len(downstream_neurons) > 3 and not has_strong_logits:
            return "syntactic_routing"

    # Early layer neurons
    if layer <= 3:
        return "syntactic_routing"

    # Default based on logit effects
    if has_strong_logits:
        return "semantic_retrieval"

    return "syntactic_routing"


def generate_label(neuron_data: dict) -> dict:
    """Generate a high-quality label for a single neuron."""
    neuron_id = neuron_data.get("neuron_id", "Unknown")
    layer = neuron_data.get("layer", 0)

    # Aggregate upstream connections
    upstream = aggregate_connections_by_neuron(
        neuron_data.get("top_upstream_sources", []),
        is_upstream=True
    )

    # Aggregate downstream connections
    downstream = aggregate_connections_by_neuron(
        neuron_data.get("top_downstream_targets", []),
        is_upstream=False
    )

    # Get categorized connections
    activators = get_activators(upstream)
    inhibitors = get_inhibitors(upstream)
    downstream_neurons = get_downstream_neurons(downstream)
    logit_effects = get_logit_effects(downstream, neuron_data.get("logit_effects", []))

    # === Build OUTPUT_FUNCTION ===
    output_parts = []

    # Logit effects first (for late layer neurons)
    if logit_effects:
        top_logits = logit_effects[:3]
        for token, total, max_w in top_logits:
            action = "Promotes" if total > 0 else "Suppresses"
            output_parts.append(f"{action} '{token}' ({format_weight(total)})")

    # Then downstream neurons
    if downstream_neurons:
        for target, total, max_w in downstream_neurons[:3]:
            action = "activates" if total > 0 else "inhibits"
            output_parts.append(f"{action} {target} ({format_weight(total)})")

    if not output_parts:
        output_function = "No significant downstream effects"
    else:
        output_function = "; ".join(output_parts[:3])

    # === Build INPUT_TRIGGER ===
    input_parts = []

    # Focus on ACTIVATORS, not inhibitors
    if activators:
        act_strs = []
        for src, total, max_w in activators[:3]:
            act_strs.append(f"{src} ({format_weight(total)})")
        input_parts.append("Activated by " + ", ".join(act_strs))

    # Note inhibitors separately
    if inhibitors:
        # Only mention L1/N2427 if it's actually an inhibitor
        l1_inhib = next((i for i in inhibitors if "L1/N2427" in i[0]), None)
        other_inhibs = [i for i in inhibitors[:3] if "L1/N2427" not in i[0]]

        if l1_inhib:
            input_parts.append(f"L1/N2427 inhibits ({format_weight(l1_inhib[1])})")
        if other_inhibs:
            inhib_strs = [f"{src} ({format_weight(total)})" for src, total, _ in other_inhibs[:2]]
            input_parts.append("also inhibited by " + ", ".join(inhib_strs))

    if not input_parts:
        input_trigger = "No significant upstream inputs"
    else:
        input_trigger = "; ".join(input_parts)

    # === Build MECHANISM ===
    mechanism_parts = []

    # Describe strongest downstream effects with weight classification
    if downstream_neurons:
        for target, total, max_w in downstream_neurons[:2]:
            strength = classify_weight(total)
            action = "activates" if total > 0 else "inhibits"
            mechanism_parts.append(f"{strength} {action} {target} ({format_weight(total)})")

    if logit_effects:
        for token, total, max_w in logit_effects[:2]:
            strength = classify_weight(total, is_logit=True)
            action = "promotes" if total > 0 else "suppresses"
            mechanism_parts.append(f"{strength} {action} '{token}' ({format_weight(total)})")

    mechanism = "; ".join(mechanism_parts) if mechanism_parts else "Weak distributed effects"

    # === Build COMPLETE_FUNCTION ===
    complete_parts = []

    # Start with activation condition
    if activators:
        top_act = activators[0]
        complete_parts.append(f"When {top_act[0]} ({format_weight(top_act[1])}) provides activation")

        if inhibitors and "L1/N2427" in inhibitors[0][0]:
            complete_parts.append(f"that overcomes L1/N2427 inhibition ({format_weight(inhibitors[0][1])})")
    elif inhibitors:
        top_inhib = inhibitors[0]
        if top_inhib[1] < -0.1:
            complete_parts.append(f"When not inhibited by {top_inhib[0]} ({format_weight(top_inhib[1])})")

    # Describe output
    if downstream_neurons or logit_effects:
        complete_parts.append("this neuron")

        if downstream_neurons:
            top_ds = downstream_neurons[0]
            action = "routes to" if top_ds[1] > 0 else "blocks"
            complete_parts.append(f"{action} {top_ds[0]} ({format_weight(top_ds[1])})")

        if logit_effects:
            top_logit = logit_effects[0]
            action = "promoting" if top_logit[1] > 0 else "suppressing"
            complete_parts.append(f"{action} '{top_logit[0]}' token ({format_weight(top_logit[1])})")

    complete_function = " ".join(complete_parts) if complete_parts else "Minimal signal transformation"

    # === Determine functional role ===
    functional_role = determine_functional_role(
        neuron_data, activators, inhibitors, downstream_neurons, logit_effects
    )

    # === Determine confidence ===
    has_clear_signal = (
        (len(activators) > 0 or len(inhibitors) > 0) and
        (len(downstream_neurons) > 0 or len(logit_effects) > 0)
    )
    has_strong_effects = (
        any(abs(t[1]) > 0.3 for t in downstream_neurons[:3]) if downstream_neurons else False
    ) or (
        any(abs(t[1]) > 2.0 for t in logit_effects[:3]) if logit_effects else False
    )

    if has_strong_effects:
        confidence = "high"
    elif has_clear_signal:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "output_function": output_function,
        "mechanism": mechanism,
        "input_trigger": input_trigger,
        "complete_function": complete_function,
        "functional_role": functional_role,
        "confidence": confidence
    }


def main():
    # Load the neuron data
    data_path = Path("data/neurons_to_label_100.json")
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        return

    with open(data_path) as f:
        data = json.load(f)

    neurons = data.get("neurons", [])
    print(f"Processing {len(neurons)} neurons...")

    # Generate labels for each neuron
    labels = {}
    layers_present = set()

    for neuron_data in neurons:
        neuron_id = neuron_data.get("neuron_id", "")
        if not neuron_id:
            continue

        label = generate_label(neuron_data)
        labels[neuron_id] = label
        layers_present.add(neuron_data.get("layer", 0))

        print(f"  {neuron_id}: {label['output_function'][:60]}...")

    # Build output structure
    output = {
        "metadata": {
            "pass1_order": "L31->L0",
            "pass2_order": "L0->L31",
            "weight_interpretation": {
                "logit": "|w|>0.3 VERY STRONG, |w|>0.1 STRONG, |w|>0.03 MODERATE",
                "neuron": "|w|>0.5 VERY STRONG, |w|>0.1 STRONG, |w|>0.03 MODERATE"
            },
            "total_neurons": len(labels),
            "layers_present": sorted(layers_present, reverse=True)
        },
        "labels": labels
    }

    # Save
    output_path = Path("data/neuron_labels_2pass.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved {len(labels)} labels to {output_path}")

    # Print some examples
    print("\n" + "="*60)
    print("SAMPLE LABELS")
    print("="*60)

    sample_ids = ["L27/N8140", "L31/N9886", "L1/N2427", "L12/N13860"]
    for sid in sample_ids:
        if sid in labels:
            print(f"\n{sid}:")
            for key, val in labels[sid].items():
                print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
