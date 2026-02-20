#!/usr/bin/env python3
"""
Compositional labeling for MVP subgraph.

Pass 1 (L31→L0): Label output functions
Pass 2 (L0→L31): Label input triggers
Pass 3: Synthesize complete labels
"""

import json
from collections import defaultdict


def get_layer(nid):
    return int(nid.split('/')[0][1:])

def load_subgraph(path='data/mvp_subgraph.json'):
    with open(path) as f:
        return json.load(f)

def build_adjacency(subgraph):
    """Build upstream and downstream adjacency lists."""
    upstream = defaultdict(list)   # target -> [(source, weight)]
    downstream = defaultdict(list) # source -> [(target, weight)]

    for e in subgraph['edges']:
        src, tgt, w = e['source'], e['target'], e['weight']
        upstream[tgt].append((src, w))
        downstream[src].append((tgt, w))

    # Sort by absolute weight
    for nid in upstream:
        upstream[nid].sort(key=lambda x: -abs(x[1]))
    for nid in downstream:
        downstream[nid].sort(key=lambda x: -abs(x[1]))

    return upstream, downstream

def format_weight(w):
    """Format weight for display."""
    return f"{w:+.2f}" if w >= 0 else f"{w:.2f}"

def format_neuron_ref(nid, labels):
    """Format neuron reference with its label."""
    if nid in labels:
        # Get a short description
        label = labels[nid]
        if 'short_label' in label:
            return f"{nid} ({label['short_label']})"
        elif 'output_function' in label:
            short = label['output_function'][:30]
            return f"{nid} ({short})"
    return nid

# =============================================================================
# PASS 1: Output Functions (L31 → L0)
# =============================================================================

def pass1_label_l31(nid, neuron_data, labels):
    """Label L31 neuron by logit effects."""
    logits = neuron_data.get('logit_effects', [])
    if not logits:
        return "weak logit effects"

    effects = []
    for l in logits[:2]:
        token = l.get('token', '?')
        weight = l.get('weight', 0)
        if abs(weight) > 0.1:
            direction = "promotes" if weight > 0 else "suppresses"
            effects.append(f"{direction} '{token}' (w={format_weight(weight)})")

    if effects:
        label = "; ".join(effects)
        # Create short label
        first_effect = logits[0]
        direction = "+" if first_effect['weight'] > 0 else "-"
        short = f"{direction}'{first_effect['token']}'"
        return label, short

    return "weak logit effects", "weak"

def pass1_label_late(nid, neuron_data, downstream, labels):
    """Label L28-L30 neuron by logit effects + downstream to labeled neurons."""
    parts = []

    # Check logit effects (if any)
    logits = neuron_data.get('logit_effects', [])
    logit_str = None
    if logits:
        for l in logits[:1]:
            token = l.get('token', '?')
            weight = l.get('weight', 0)
            if abs(weight) > 0.05:
                direction = "promotes" if weight > 0 else "suppresses"
                logit_str = f"{direction} '{token}'"
                parts.append(logit_str)

    # Check downstream to labeled neurons
    down = downstream.get(nid, [])
    for tgt, w in down[:3]:
        if tgt in labels and abs(w) > 0.01:
            ref = format_neuron_ref(tgt, labels)
            direction = "activates" if w > 0 else "inhibits"
            parts.append(f"{direction} {ref}")

    if parts:
        short = logit_str if logit_str else parts[0][:20]
        return "; ".join(parts), short

    return "routes to output", "router"

def pass1_label_mid(nid, neuron_data, downstream, labels):
    """Label mid-layer neuron by effect on labeled downstream neurons."""
    down = downstream.get(nid, [])

    effects = []
    for tgt, w in down[:4]:
        if tgt in labels and abs(w) > 0.01:
            ref = format_neuron_ref(tgt, labels)
            direction = "activates" if w > 0 else "inhibits"
            effects.append(f"{direction} {ref} (w={format_weight(w)})")

    if effects:
        # Short label from first strong effect
        first_tgt = down[0][0] if down else None
        if first_tgt and first_tgt in labels:
            short = f"→{labels[first_tgt].get('short_label', first_tgt)}"
        else:
            short = "router"
        return "; ".join(effects[:3]), short

    return "weak downstream effects", "weak"

def pass1_label_l0(nid, neuron_data, downstream, labels):
    """Label L0 neuron by token + downstream effects."""
    tokens = neuron_data.get('input_tokens', [])

    # Token label
    if tokens:
        token = tokens[0].get('token', '?')
        token_label = f"fires on '{token}'"
        short = f"'{token}'"
    else:
        token_label = "unknown token"
        short = "?"

    # Downstream effects
    down = downstream.get(nid, [])
    down_effects = []
    for tgt, w in down[:3]:
        if tgt in labels and abs(w) > 0.02:
            ref = format_neuron_ref(tgt, labels)
            down_effects.append(f"activates {ref}")

    if down_effects:
        full_label = f"{token_label}; {'; '.join(down_effects[:2])}"
    else:
        full_label = token_label

    return full_label, short

def run_pass1(subgraph, upstream, downstream):
    """Run Pass 1: Label output functions from L31 to L0."""
    print("\n" + "="*60)
    print("PASS 1: Output Functions (L31 → L0)")
    print("="*60)

    labels = {}
    neurons = subgraph['neurons']

    # Get layers in reverse order
    layers = sorted(set(get_layer(nid) for nid in neurons), reverse=True)

    for layer in layers:
        layer_neurons = [nid for nid in neurons if get_layer(nid) == layer]
        print(f"\n--- Layer {layer} ({len(layer_neurons)} neurons) ---")

        for nid in layer_neurons:
            nd = neurons[nid]

            if layer == 31:
                output_fn, short = pass1_label_l31(nid, nd, labels)
            elif layer >= 28:
                output_fn, short = pass1_label_late(nid, nd, downstream, labels)
            elif layer == 0:
                output_fn, short = pass1_label_l0(nid, nd, downstream, labels)
            else:
                output_fn, short = pass1_label_mid(nid, nd, downstream, labels)

            labels[nid] = {
                'layer': layer,
                'output_function': output_fn,
                'short_label': short,
            }

            print(f"  {nid}: {output_fn[:70]}{'...' if len(output_fn) > 70 else ''}")

    return labels

# =============================================================================
# PASS 2: Input Triggers (L0 → L31)
# =============================================================================

def pass2_label_l0(nid, neuron_data, labels):
    """Label L0 neuron input trigger (same as output - token detector)."""
    tokens = neuron_data.get('input_tokens', [])
    if tokens:
        token = tokens[0].get('token', '?')
        return f"fires on '{token}' token"
    return "unknown token pattern"

def pass2_label_early(nid, neuron_data, upstream, labels):
    """Label L1-L5 neuron by which labeled upstream neurons activate it."""
    up = upstream.get(nid, [])

    activators = []
    inhibitors = []

    for src, w in up[:5]:
        if src in labels and abs(w) > 0.01:
            ref = format_neuron_ref(src, labels)
            if w > 0:
                activators.append(f"{ref} (w={format_weight(w)})")
            else:
                inhibitors.append(f"{ref} (w={format_weight(w)})")

    parts = []
    if activators:
        parts.append(f"activated by {', '.join(activators[:2])}")
    if inhibitors:
        parts.append(f"inhibited by {', '.join(inhibitors[:1])}")

    if parts:
        return "; ".join(parts)
    return "unclear upstream pattern"

def pass2_label_mid(nid, neuron_data, upstream, labels):
    """Label mid-layer neuron by labeled upstream sources."""
    up = upstream.get(nid, [])

    sources = []
    for src, w in up[:4]:
        if src in labels and abs(w) > 0.01:
            ref = format_neuron_ref(src, labels)
            sources.append(f"{ref}")

    if sources:
        return f"receives from {', '.join(sources[:3])}"
    return "unclear upstream"

def pass2_label_late(nid, neuron_data, upstream, labels):
    """Label L28-L31 neuron by semantic upstream path."""
    up = upstream.get(nid, [])

    # Find semantic sources (trace back to find L0 concepts)
    activators = []
    inhibitors = []

    for src, w in up[:5]:
        if src in labels and abs(w) > 0.01:
            ref = format_neuron_ref(src, labels)
            if w > 0:
                activators.append(ref)
            else:
                inhibitors.append(ref)

    parts = []
    if activators:
        parts.append(f"activated by {', '.join(activators[:2])}")
    if inhibitors:
        parts.append(f"inhibited by {', '.join(inhibitors[:1])}")

    if parts:
        return "; ".join(parts)
    return "receives from semantic pathways"

def run_pass2(subgraph, upstream, downstream, labels):
    """Run Pass 2: Label input triggers from L0 to L31."""
    print("\n" + "="*60)
    print("PASS 2: Input Triggers (L0 → L31)")
    print("="*60)

    neurons = subgraph['neurons']

    # Get layers in forward order
    layers = sorted(set(get_layer(nid) for nid in neurons))

    for layer in layers:
        layer_neurons = [nid for nid in neurons if get_layer(nid) == layer]
        print(f"\n--- Layer {layer} ({len(layer_neurons)} neurons) ---")

        for nid in layer_neurons:
            nd = neurons[nid]

            if layer == 0:
                input_tr = pass2_label_l0(nid, nd, labels)
            elif layer <= 5:
                input_tr = pass2_label_early(nid, nd, upstream, labels)
            elif layer >= 28:
                input_tr = pass2_label_late(nid, nd, upstream, labels)
            else:
                input_tr = pass2_label_mid(nid, nd, upstream, labels)

            labels[nid]['input_trigger'] = input_tr

            print(f"  {nid}: {input_tr[:70]}{'...' if len(input_tr) > 70 else ''}")

    return labels

# =============================================================================
# PASS 3: Synthesize Complete Labels
# =============================================================================

def determine_functional_role(layer, output_fn, input_tr):
    """Determine functional role category."""
    if layer == 0:
        return "input_encoding"
    elif layer <= 2:
        return "domain_detection"
    elif layer >= 28:
        # Check if has logit effects
        if "promotes" in output_fn or "suppresses" in output_fn:
            return "semantic_retrieval"
        return "output_preparation"
    else:
        return "intermediate_routing"

def synthesize_complete_function(nid, label):
    """Create complete function description."""
    input_tr = label.get('input_trigger', '')
    output_fn = label.get('output_function', '')

    # Create a narrative
    if label['layer'] == 0:
        return f"Detects '{label.get('short_label', '?')}' token in input and signals to downstream aggregators"
    elif label['layer'] >= 28:
        # Output neuron
        if "promotes" in output_fn:
            token = output_fn.split("'")[1] if "'" in output_fn else "?"
            return f"When {input_tr.replace('activated by ', '').split(',')[0]} active, promotes '{token}' in output"
        elif "suppresses" in output_fn:
            token = output_fn.split("'")[1] if "'" in output_fn else "?"
            return f"When {input_tr.replace('activated by ', '').split(',')[0]} active, suppresses '{token}' in output"

    # Mid-layer
    return f"Input: {input_tr} | Output: {output_fn}"

def run_pass3(subgraph, labels):
    """Run Pass 3: Synthesize complete labels."""
    print("\n" + "="*60)
    print("PASS 3: Synthesizing Complete Labels")
    print("="*60)

    neurons = subgraph['neurons']

    for nid in labels:
        label = labels[nid]
        nd = neurons.get(nid, {})

        # Add metadata
        label['neuron'] = nd.get('neuron', 0)
        label['appearances'] = nd.get('appearances', 0)

        # Determine role
        label['functional_role'] = determine_functional_role(
            label['layer'],
            label.get('output_function', ''),
            label.get('input_trigger', '')
        )

        # Synthesize complete function
        label['complete_function'] = synthesize_complete_function(nid, label)

    # Print summary
    by_role = defaultdict(list)
    for nid, label in labels.items():
        by_role[label['functional_role']].append(nid)

    print("\nFunctional Role Summary:")
    for role in ['input_encoding', 'domain_detection', 'intermediate_routing', 'output_preparation', 'semantic_retrieval']:
        if role in by_role:
            print(f"  {role}: {len(by_role[role])} neurons")

    return labels

# =============================================================================
# Main
# =============================================================================

def main():
    print("Loading MVP subgraph...")
    subgraph = load_subgraph()
    print(f"Loaded {len(subgraph['neurons'])} neurons, {len(subgraph['edges'])} edges")

    # Build adjacency
    upstream, downstream = build_adjacency(subgraph)

    # Pass 1: Output functions
    labels = run_pass1(subgraph, upstream, downstream)

    # Pass 2: Input triggers
    labels = run_pass2(subgraph, upstream, downstream, labels)

    # Pass 3: Synthesize
    labels = run_pass3(subgraph, labels)

    # Save results
    output = {
        'metadata': {
            'method': 'compositional_labeling',
            'passes': ['output_functions (L31→L0)', 'input_triggers (L0→L31)', 'synthesis'],
            'total_neurons': len(labels),
        },
        'labels': labels
    }

    output_path = 'data/mvp_labels_compositional.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved {len(labels)} labels to {output_path}")
    print("="*60)

    # Print sample labels
    print("\n=== Sample Labels ===\n")
    samples = ['L0/N857', 'L2/N12082', 'L12/N8459', 'L24/N5326', 'L31/N9707', 'L31/N311']
    for nid in samples:
        if nid in labels:
            l = labels[nid]
            print(f"{nid} [{l.get('short_label', '?')}]:")
            print(f"  Input:  {l.get('input_trigger', 'N/A')}")
            print(f"  Output: {l.get('output_function', 'N/A')}")
            print(f"  Role:   {l.get('functional_role', 'N/A')}")
            print(f"  Complete: {l.get('complete_function', 'N/A')}")
            print()

if __name__ == "__main__":
    main()
