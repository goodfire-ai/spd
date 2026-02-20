#!/usr/bin/env python3
"""
Compositional Neuron Labeling

Labels neurons in topological order so each neuron's label is built from
already-labeled neurons:

Pass 1 (Late→Early): Output functions
  - L31: "promotes 'dopamine' token"
  - L30: "activates dopamine-promoter neurons"
  - L29: "amplifies neurotransmitter pathway"

Pass 2 (Early→Late): Input triggers
  - L0: "fires on 'Parkinson' token"
  - L1: "fires on disease name tokens"
  - L12: "fires in dopamine-related disease context"

Complete Function: Combines input trigger + output function
"""

import argparse
import json
from collections import defaultdict

from transformers import AutoTokenizer


def load_edge_stats(path: str) -> dict:
    """Load edge statistics."""
    with open(path) as f:
        return json.load(f)


def build_neuron_index(profiles: list) -> dict:
    """Build index from neuron_id -> profile."""
    index = {}
    for p in profiles:
        nid = f"L{p['layer']}/N{p['neuron']}"
        index[nid] = p
    return index


def decode_token(token_id: int, tokenizer) -> str:
    """Decode a token ID to string."""
    try:
        return tokenizer.decode([token_id])
    except:
        return f"<token_{token_id}>"


def parse_source_target(s: str) -> tuple:
    """Parse source/target format like '1_2427_0' or 'E_128000_0' or 'L_128000_0'."""
    parts = s.split('_')
    if parts[0] == 'E':
        return ('emb', int(parts[1]), int(parts[2]))
    elif parts[0] == 'L':
        return ('logit', int(parts[1]), int(parts[2]))
    else:
        return ('neuron', int(parts[0]), int(parts[1]), int(parts[2]))


def get_l0_token_label(profile: dict, tokenizer) -> str:
    """Get semantic label for L0 neuron based on embedding sources."""
    sources = profile.get('top_upstream_sources', [])

    token_weights = defaultdict(float)
    for src in sources:
        parsed = parse_source_target(src['source'])
        if parsed[0] == 'emb':
            token_id = parsed[1]
            token = decode_token(token_id, tokenizer)
            token_weights[token] += src['avg_weight']

    if not token_weights:
        return "unknown input pattern"

    # Get top tokens
    sorted_tokens = sorted(token_weights.items(), key=lambda x: -x[1])[:3]

    # Format label
    if len(sorted_tokens) == 1 or sorted_tokens[0][1] > 2 * sorted_tokens[1][1]:
        # Single dominant token
        return f"fires on {repr(sorted_tokens[0][0])}"
    else:
        # Multiple tokens
        tokens = [repr(t[0]) for t in sorted_tokens[:2]]
        return f"fires on {'/'.join(tokens)}"


def get_logit_label(profile: dict, tokenizer) -> str:
    """Get label for neuron based on logit effects."""
    logits = profile.get('logit_effects', [])
    if not logits:
        return None

    # Parse logit effects
    effects = []
    for entry in logits[:5]:
        token = entry.get('token', entry.get('target', ''))
        weight = entry.get('avg_weight', entry.get('weight', 0))
        if abs(weight) > 0.1:  # Only strong effects
            direction = "promotes" if weight > 0 else "suppresses"
            effects.append((token, weight, direction))

    if not effects:
        return None

    # Group by direction
    promotes = [(t, w) for t, w, d in effects if d == "promotes"]
    suppresses = [(t, w) for t, w, d in effects if d == "suppresses"]

    if promotes and (not suppresses or promotes[0][1] > abs(suppresses[0][1])):
        tokens = [repr(t) for t, w in promotes[:2]]
        return f"promotes {'/'.join(tokens)} tokens"
    elif suppresses:
        tokens = [repr(t) for t, w in suppresses[:2]]
        return f"suppresses {'/'.join(tokens)} tokens"

    return None


def get_downstream_label(profile: dict, labels: dict, threshold: float = 0.1) -> str:
    """Get label based on effects on already-labeled downstream neurons."""
    targets = profile.get('top_downstream_targets', [])

    effects = []
    for tgt in targets:
        parsed = parse_source_target(tgt['target'])
        if parsed[0] == 'neuron':
            layer, neuron = parsed[1], parsed[2]
            nid = f"L{layer}/N{neuron}"
            weight = tgt['avg_weight']

            if nid in labels and abs(weight) > threshold:
                downstream_label = labels[nid].get('output_function', '')
                if downstream_label:
                    direction = "activates" if weight > 0 else "inhibits"
                    effects.append((nid, weight, direction, downstream_label))

    if not effects:
        return None

    # Sort by absolute weight
    effects.sort(key=lambda x: -abs(x[1]))

    # Build label from top effects
    parts = []
    for nid, weight, direction, downstream_label in effects[:3]:
        # Simplify the downstream label for composition
        simplified = simplify_label(downstream_label)
        parts.append(f"{direction} {simplified} ({nid}, w={weight:.2f})")

    return "; ".join(parts)


def get_upstream_label(profile: dict, labels: dict, threshold: float = 0.05) -> str:
    """Get label based on inputs from already-labeled upstream neurons."""
    sources = profile.get('top_upstream_sources', [])

    effects = []
    for src in sources:
        parsed = parse_source_target(src['source'])
        if parsed[0] == 'neuron':
            layer, neuron = parsed[1], parsed[2]
            nid = f"L{layer}/N{neuron}"
            weight = src['avg_weight']

            if nid in labels and abs(weight) > threshold:
                upstream_label = labels[nid].get('input_trigger', labels[nid].get('output_function', ''))
                if upstream_label:
                    direction = "activated by" if weight > 0 else "inhibited by"
                    effects.append((nid, weight, direction, upstream_label))

    if not effects:
        return None

    # Separate activators and inhibitors
    activators = [(n, w, d, l) for n, w, d, l in effects if w > 0]
    inhibitors = [(n, w, d, l) for n, w, d, l in effects if w < 0]

    # Sort by weight
    activators.sort(key=lambda x: -x[1])
    inhibitors.sort(key=lambda x: x[1])

    parts = []

    # Focus on activators for input trigger
    if activators:
        for nid, weight, _, upstream_label in activators[:2]:
            simplified = simplify_label(upstream_label)
            parts.append(f"when {simplified} ({nid}, w=+{weight:.2f})")

    # Mention strong inhibition
    if inhibitors and abs(inhibitors[0][1]) > 0.3:
        nid, weight, _, upstream_label = inhibitors[0]
        simplified = simplify_label(upstream_label)
        parts.append(f"gated by {simplified} ({nid}, w={weight:.2f})")

    return "; ".join(parts) if parts else None


def simplify_label(label: str) -> str:
    """Simplify a label for composition."""
    # Remove verbose parts
    label = label.replace("fires on ", "")
    label = label.replace("promotes ", "")
    label = label.replace("suppresses ", "")
    label = label.replace("activates ", "")
    label = label.replace("inhibits ", "")

    # Truncate if too long
    if len(label) > 50:
        label = label[:47] + "..."

    return label


def synthesize_semantic_label(output_function: str, input_trigger: str, logit_effect: str) -> str:
    """Create a semantic complete_function from components."""
    parts = []

    if input_trigger:
        parts.append(f"Input: {input_trigger}")

    if output_function:
        parts.append(f"Output: {output_function}")

    if logit_effect:
        parts.append(f"Logit effect: {logit_effect}")

    return " | ".join(parts) if parts else "Unknown function"


def determine_functional_role(profile: dict, output_function: str, input_trigger: str) -> str:
    """Determine functional role based on patterns."""
    layer = profile['layer']

    # Check for logit effects
    has_logit = bool(profile.get('logit_effects'))

    # Keywords for detection
    semantic_keywords = ['cancer', 'brain', 'heart', 'immune', 'disease', 'blood', 'hormone']
    format_keywords = ['Answer', 'begin_of_text', 'user', 'assistant']

    combined = (output_function or '') + ' ' + (input_trigger or '')
    combined_lower = combined.lower()

    has_semantic = any(kw in combined_lower for kw in semantic_keywords)
    has_format = any(kw in combined for kw in format_keywords)

    # Determine role
    if has_logit and layer >= 28:
        if has_semantic:
            return "semantic_retrieval"
        else:
            return "answer_formatting"
    elif has_semantic:
        return "domain_detection"
    elif has_format:
        return "syntactic_routing"
    elif layer < 5:
        return "input_encoding"
    elif layer > 25:
        return "output_preparation"
    else:
        return "intermediate_routing"


def run_compositional_labeling(edge_stats_path: str, output_path: str):
    """Run the full compositional labeling pipeline."""
    print("Loading data...")
    data = load_edge_stats(edge_stats_path)
    profiles = data['profiles']

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')

    # Build index
    neuron_index = build_neuron_index(profiles)
    print(f"Indexed {len(neuron_index)} neurons")

    # Group by layer
    by_layer = defaultdict(list)
    for p in profiles:
        by_layer[p['layer']].append(p)

    layers = sorted(by_layer.keys())
    print(f"Layers: {min(layers)} to {max(layers)}")

    # Initialize labels
    labels = {}

    # ========================================
    # PASS 1: Late → Early (Output Functions)
    # ========================================
    print("\n=== PASS 1: Output Functions (Late → Early) ===")

    for layer in reversed(layers):
        neurons = by_layer[layer]
        print(f"\nProcessing L{layer} ({len(neurons)} neurons)...")

        for p in neurons:
            nid = f"L{p['layer']}/N{p['neuron']}"

            # Initialize label
            if nid not in labels:
                labels[nid] = {}

            # Strategy depends on layer
            if layer == 0:
                # L0: Use token triggers
                labels[nid]['output_function'] = get_l0_token_label(p, tokenizer)
            else:
                # Check for direct logit effects first
                logit_label = get_logit_label(p, tokenizer)

                # Then check downstream effects
                downstream_label = get_downstream_label(p, labels)

                # Combine
                if logit_label and downstream_label:
                    labels[nid]['output_function'] = f"{logit_label}; via {downstream_label}"
                elif logit_label:
                    labels[nid]['output_function'] = logit_label
                elif downstream_label:
                    labels[nid]['output_function'] = downstream_label
                else:
                    labels[nid]['output_function'] = "weak/unclear output effects"

    # ========================================
    # PASS 2: Early → Late (Input Triggers)
    # ========================================
    print("\n=== PASS 2: Input Triggers (Early → Late) ===")

    for layer in layers:
        neurons = by_layer[layer]
        print(f"\nProcessing L{layer} ({len(neurons)} neurons)...")

        for p in neurons:
            nid = f"L{p['layer']}/N{p['neuron']}"

            if layer == 0:
                # L0: Same as output (token detectors)
                labels[nid]['input_trigger'] = labels[nid].get('output_function', 'unknown')
            else:
                # Use upstream labels
                upstream_label = get_upstream_label(p, labels)
                if upstream_label:
                    labels[nid]['input_trigger'] = upstream_label
                else:
                    labels[nid]['input_trigger'] = "unclear upstream pattern"

    # ========================================
    # PASS 3: Synthesize Complete Functions
    # ========================================
    print("\n=== PASS 3: Synthesizing Complete Functions ===")

    for nid, label in labels.items():
        p = neuron_index[nid]

        output_fn = label.get('output_function', '')
        input_tr = label.get('input_trigger', '')
        logit_effect = get_logit_label(p, tokenizer)

        label['complete_function'] = synthesize_semantic_label(output_fn, input_tr, logit_effect)
        label['functional_role'] = determine_functional_role(p, output_fn, input_tr)
        label['layer'] = p['layer']
        label['neuron'] = p['neuron']
        label['appearance_count'] = p.get('appearance_count', 0)

    # ========================================
    # Save Results
    # ========================================
    print(f"\n=== Saving to {output_path} ===")

    result = {
        "metadata": {
            "pass1_order": "L31->L0 (output functions)",
            "pass2_order": "L0->L31 (input triggers)",
            "labeling_method": "compositional",
            "total_neurons": len(labels),
            "layers": sorted(set(l['layer'] for l in labels.values()))
        },
        "labels": labels
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Saved {len(labels)} labels")

    # Print sample labels
    print("\n=== Sample Labels ===")
    samples = [
        "L0/N13305",  # cancer
        "L0/N14326",  # brain
        "L1/N2427",   # hub
        "L12/N13860", # mid-layer
        "L27/N8140",  # late mid
        "L31/N5493",  # late layer
    ]

    for nid in samples:
        if nid in labels:
            l = labels[nid]
            print(f"\n{nid}:")
            print(f"  Output: {l.get('output_function', 'N/A')}")
            print(f"  Input: {l.get('input_trigger', 'N/A')}")
            print(f"  Role: {l.get('functional_role', 'N/A')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compositional neuron labeling")
    parser.add_argument("--edge-stats", default="data/medical_edge_stats_v2_enriched.json",
                        help="Path to edge statistics JSON")
    parser.add_argument("-o", "--output", default="data/neuron_labels_compositional.json",
                        help="Output path for labels")

    args = parser.parse_args()
    run_compositional_labeling(args.edge_stats, args.output)
