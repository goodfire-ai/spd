#!/usr/bin/env python3
"""
Compositional labeling for MVP subgraph - V2 with semantic propagation.

Key insight: Mid-layer neurons should inherit semantic concepts from their
upstream sources, not just be labeled "router". Trace back to L0/L2 to find
the actual semantic meaning.
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
    upstream = defaultdict(list)
    downstream = defaultdict(list)

    for e in subgraph['edges']:
        src, tgt, w = e['source'], e['target'], e['weight']
        upstream[tgt].append((src, w))
        downstream[src].append((tgt, w))

    for nid in upstream:
        upstream[nid].sort(key=lambda x: -abs(x[1]))
    for nid in downstream:
        downstream[nid].sort(key=lambda x: -abs(x[1]))

    return upstream, downstream

def format_weight(w):
    return f"{w:+.2f}"

# =============================================================================
# SEMANTIC TRACING: Find the actual semantic source for any neuron
# =============================================================================

def trace_semantic_sources(nid, upstream, root_sources, visited=None, depth=0):
    """
    Recursively trace back to find ROOT semantic sources (L0 tokens, L2 aggregators).
    Only returns actual semantic concepts, not intermediate "routes-X" labels.
    Returns list of (concept, weight_product) tuples.
    """
    if visited is None:
        visited = set()

    if nid in visited or depth > 15:
        return []
    visited.add(nid)

    # If this neuron is a root semantic source (L0 or L2), return it
    if nid in root_sources:
        return [(root_sources[nid], 1.0)]

    # Otherwise, trace upstream
    up = upstream.get(nid, [])
    sources = []

    for src, w in up[:5]:  # Top 5 upstream
        if abs(w) < 0.005:
            continue
        src_sources = trace_semantic_sources(src, upstream, root_sources, visited.copy(), depth + 1)
        for concept, upstream_w in src_sources:
            # Weight decay as we trace back
            sources.append((concept, w * upstream_w * 0.8))

    return sources

def get_top_semantic_sources(nid, upstream, semantic_sources, n=3):
    """Get top N semantic sources for a neuron."""
    sources = trace_semantic_sources(nid, upstream, semantic_sources)

    # Aggregate by concept
    concept_weights = defaultdict(float)
    for concept, w in sources:
        concept_weights[concept] += abs(w)

    # Sort by weight
    sorted_concepts = sorted(concept_weights.items(), key=lambda x: -x[1])
    return [c for c, w in sorted_concepts[:n]]

# =============================================================================
# PASS 1: Label L0 and L31 first (ground truth semantic anchors)
# =============================================================================

def pass1_semantic_anchors(subgraph, upstream, downstream):
    """Label the semantic anchors: L0 (input tokens) and L31 (output tokens)."""
    print("\n" + "="*60)
    print("PASS 1: Semantic Anchors (L0 tokens, L31 logits)")
    print("="*60)

    labels = {}
    # root_sources: ONLY L0 and L2 - these are the input semantic anchors
    # used for tracing what signals flow through the network
    root_sources = {}
    neurons = subgraph['neurons']

    # L0: Token detectors
    print("\n--- L0: Input Token Detectors ---")
    for nid in neurons:
        if get_layer(nid) != 0:
            continue
        nd = neurons[nid]
        tokens = nd.get('input_tokens', [])

        if tokens:
            token = tokens[0].get('token', '?').strip()
        else:
            token = "unknown"

        root_sources[nid] = token  # L0 IS a root source
        labels[nid] = {
            'layer': 0,
            'concept': token,
            'short_label': token,
            'semantic_source': token,
            'input_trigger': f"fires on '{token}' token",
            'output_function': f"detects '{token}', signals downstream",
            'functional_role': 'input_encoding',
        }
        print(f"  {nid}: '{token}'")

    # L31: Output logit effects (NOT added to root_sources - they are outputs, not inputs)
    print("\n--- L31: Output Token Effects ---")
    for nid in neurons:
        if get_layer(nid) != 31:
            continue
        nd = neurons[nid]
        logits = nd.get('logit_effects', [])

        if logits and abs(logits[0].get('weight', 0)) > 0.1:
            token = logits[0].get('token', '?').strip()
            weight = logits[0].get('weight', 0)
            direction = "promotes" if weight > 0 else "suppresses"
            concept = f"{direction}-{token}"
            short = f"{'+' if weight > 0 else '-'}{token}"
            output_fn = f"{direction} '{token}' (w={format_weight(weight)})"

            # Add secondary effects
            for l in logits[1:2]:
                if abs(l.get('weight', 0)) > 0.1:
                    d2 = "promotes" if l['weight'] > 0 else "suppresses"
                    output_fn += f"; {d2} '{l['token']}' (w={format_weight(l['weight'])})"
        else:
            concept = "weak-output"
            short = "weak"
            output_fn = "weak logit effects"

        # NOTE: L31 is NOT added to root_sources - it's an output, not a semantic input
        labels[nid] = {
            'layer': 31,
            'concept': concept,
            'short_label': short,
            'semantic_source': concept,
            'output_function': output_fn,
            'functional_role': 'semantic_retrieval',
            'input_trigger': None,  # Will be filled in pass 4
        }
        print(f"  {nid}: {concept}")

    return labels, root_sources

# =============================================================================
# PASS 2: Label L2 aggregators (aggregate L0 concepts)
# =============================================================================

def pass2_aggregators(subgraph, upstream, downstream, labels, semantic_sources):
    """Label L2 aggregators based on which L0 neurons feed them."""
    print("\n" + "="*60)
    print("PASS 2: Domain Aggregators (L1-L2)")
    print("="*60)

    neurons = subgraph['neurons']

    for nid in neurons:
        layer = get_layer(nid)
        if layer not in [1, 2]:
            continue

        nd = neurons[nid]
        up = upstream.get(nid, [])

        # Find L0 sources
        l0_concepts = []
        for src, w in up:
            if src in semantic_sources and get_layer(src) == 0 and w > 0.01:
                l0_concepts.append(semantic_sources[src])

        if l0_concepts:
            if len(l0_concepts) == 1:
                concept = f"{l0_concepts[0]}-signal"
            else:
                concept = f"{'/'.join(l0_concepts[:2])}-aggregator"

            input_tr = f"aggregates: {', '.join(l0_concepts[:3])}"
        else:
            concept = "domain-aggregator"
            input_tr = "aggregates domain signals"

        semantic_sources[nid] = concept
        labels[nid] = {
            'layer': layer,
            'concept': concept,
            'short_label': concept[:20],
            'semantic_source': concept,
            'input_trigger': input_tr,
            'output_function': f"routes {concept} signal downstream",
            'functional_role': 'domain_detection',
        }
        print(f"  {nid}: {concept} <- {l0_concepts[:3]}")

    return labels, semantic_sources

# =============================================================================
# PASS 3: Label all mid-layer neurons by tracing to semantic sources
# =============================================================================

def pass3_mid_layers(subgraph, upstream, downstream, labels, root_sources):
    """Label mid-layer neurons by tracing back to ROOT semantic sources (L0/L2)."""
    print("\n" + "="*60)
    print("PASS 3: Mid-Layer Routing (trace to ROOT semantic sources)")
    print("="*60)

    neurons = subgraph['neurons']

    # Process layers 30 down to 3 (reverse order so downstream is labeled first)
    for layer in range(30, 2, -1):
        layer_neurons = [nid for nid in neurons if get_layer(nid) == layer]
        if not layer_neurons:
            continue

        print(f"\n--- Layer {layer} ---")

        for nid in layer_neurons:
            nd = neurons[nid]

            # Trace back to find ROOT semantic sources (L0/L2 only)
            sources = get_top_semantic_sources(nid, upstream, root_sources, n=3)

            # Check downstream targets and their semantic meaning
            down = downstream.get(nid, [])
            down_effects = []
            down_semantic = []
            for tgt, w in down[:5]:
                tgt_label = labels.get(tgt, {})
                tgt_sem = tgt_label.get('semantic_source', None)
                if tgt_sem and tgt_sem != 'unknown':
                    direction = "activates" if w > 0 else "inhibits"
                    down_effects.append(f"{direction} {tgt} ({tgt_sem}, w={w:.2f})")
                    down_semantic.append(tgt_sem)

            if sources:
                # Use the root semantic source directly - no "routes-" prefix chains
                primary = sources[0]
                concept = f"{primary}-pathway"
                input_tr = f"carries: {', '.join(sources[:3])}"
            elif down_semantic:
                # No upstream sources, but we can infer from downstream effects
                primary_down = down_semantic[0]
                concept = f"modulates-{primary_down}"
                input_tr = f"upstream unclear; affects: {', '.join(down_semantic[:2])}"
            else:
                # No semantic sources found - check if this neuron has ANY upstream edges
                up = upstream.get(nid, [])
                if not up:
                    concept = "orphan-no-upstream"
                    input_tr = "no upstream edges in subgraph"
                else:
                    up_list = [f"{src}" for src, w in up[:3]]
                    concept = "untraced-routing"
                    input_tr = f"upstream doesn't trace to L0/L2: {up_list}"

            # Build output function from downstream effects
            if down_effects:
                output_fn = "; ".join(down_effects[:3])
            else:
                output_fn = "routes downstream"

            # Determine functional role based on layer
            if layer >= 28:
                role = 'output_preparation'
            else:
                role = 'intermediate_routing'

            # DON'T add to root_sources - mid-layer neurons are not semantic anchors
            labels[nid] = {
                'layer': layer,
                'concept': concept,
                'short_label': primary[:15] if sources else "routing",
                'semantic_source': sources[0] if sources else "unknown",
                'input_trigger': input_tr,
                'output_function': output_fn,
                'functional_role': role,
            }

            print(f"  {nid}: [{concept}] {input_tr[:50]}")

    return labels, root_sources

# =============================================================================
# PASS 4: Complete labels with full narratives
# =============================================================================

def pass4_synthesize(subgraph, upstream, downstream, labels, root_sources):
    """Create complete function narratives."""
    print("\n" + "="*60)
    print("PASS 4: Synthesizing Complete Functions")
    print("="*60)

    neurons = subgraph['neurons']

    for nid in labels:
        label = labels[nid]
        nd = neurons.get(nid, {})
        layer = label['layer']

        # Add metadata
        label['neuron'] = nd.get('neuron', 0)
        label['appearances'] = nd.get('appearances', 0)

        # Create narrative based on layer
        if layer == 0:
            token = label['concept']
            label['complete_function'] = f"Detects '{token}' token in input and activates downstream aggregators"

        elif layer <= 2:
            sources = label.get('input_trigger', '').replace('aggregates: ', '')
            label['complete_function'] = f"When [{sources}] detected, aggregates and routes signal toward output"

        elif layer == 31:
            # Output layer - get semantic input path
            semantic_src = label.get('semantic_source', 'unknown')
            # Trace back to find what activates this
            up_sources = get_top_semantic_sources(nid, upstream, root_sources, n=3)
            if up_sources:
                input_path = ', '.join(up_sources[:3])
                label['input_trigger'] = f"activated when [{input_path}] present"
            else:
                input_path = "upstream pathway"
                label['input_trigger'] = "activated by upstream pathway"

            output_fn = label.get('output_function', '')
            if "promotes" in output_fn:
                token = output_fn.split("'")[1] if "'" in output_fn else "?"
                label['complete_function'] = f"When [{input_path}] pathway active, promotes '{token}' in output"
            elif "suppresses" in output_fn:
                token = output_fn.split("'")[1] if "'" in output_fn else "?"
                label['complete_function'] = f"When [{input_path}] pathway active, suppresses '{token}' in output"
            else:
                label['complete_function'] = f"Routes [{input_path}] signals to output"

        elif layer >= 28:
            # Late layer - connect input semantic to output
            semantic_src = label.get('semantic_source', 'unknown')
            label['complete_function'] = f"Routes [{semantic_src}] signal to output layer"

        else:
            # Mid layer
            semantic_src = label.get('semantic_source', 'unknown')
            label['complete_function'] = f"Routes [{semantic_src}] signal through network"

    # Summary
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

    upstream, downstream = build_adjacency(subgraph)

    # Pass 1: Semantic anchors (L0 tokens, L31 logits)
    labels, semantic_sources = pass1_semantic_anchors(subgraph, upstream, downstream)

    # Pass 2: Aggregators (L1-L2)
    labels, semantic_sources = pass2_aggregators(subgraph, upstream, downstream, labels, semantic_sources)

    # Pass 3: Mid-layers (trace to semantic sources)
    labels, semantic_sources = pass3_mid_layers(subgraph, upstream, downstream, labels, semantic_sources)

    # Pass 4: Synthesize complete functions
    labels = pass4_synthesize(subgraph, upstream, downstream, labels, semantic_sources)

    # Save
    output = {
        'metadata': {
            'method': 'compositional_labeling_v2_semantic_propagation',
            'passes': ['semantic_anchors', 'aggregators', 'mid_layers', 'synthesis'],
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
    samples = ['L0/N857', 'L2/N12082', 'L7/N1177', 'L16/N12982', 'L22/N4489', 'L28/N447', 'L31/N9707', 'L31/N3869']
    for nid in samples:
        if nid in labels:
            l = labels[nid]
            print(f"{nid} [{l.get('concept', '?')}]:")
            print(f"  Semantic source: {l.get('semantic_source', 'N/A')}")
            print(f"  Input:    {l.get('input_trigger', 'N/A')}")
            print(f"  Output:   {l.get('output_function', 'N/A')}")
            print(f"  Complete: {l.get('complete_function', 'N/A')}")
            print()

if __name__ == "__main__":
    main()
