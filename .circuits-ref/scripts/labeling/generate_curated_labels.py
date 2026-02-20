#!/usr/bin/env python3
"""
Generate compositional labels for curated neuron set.
Labels reference other labeled neurons to tell a coherent semantic story.
"""

import json
from pathlib import Path

from transformers import AutoTokenizer


def load_data():
    with open('data/medical_edge_stats_v2_enriched.json') as f:
        return json.load(f)


def find_profile(data, layer, neuron):
    return next((x for x in data['profiles'] if x['layer'] == layer and x['neuron'] == neuron), None)


def format_neuron_ref(nid, curated, weight=None):
    """Format a neuron reference with concept name."""
    concept = curated.get(nid, {}).get('concept', 'unknown')
    if weight is not None:
        return f"{nid} ({concept}, w={weight:+.3f})"
    return f"{nid} ({concept})"


def get_upstream_connections(profile, curated_neurons):
    """Get upstream connections, highlighting curated neurons."""
    sources = profile.get('top_upstream_sources', [])
    connections = []

    for s in sources[:20]:
        parts = s['source'].split('_')
        weight = s['avg_weight']

        if parts[0] == 'E':
            connections.append(('emb', int(parts[1]), weight))
        elif parts[0].isdigit():
            src_nid = f"L{parts[0]}/N{parts[1]}"
            is_curated = src_nid in curated_neurons
            connections.append(('neuron', src_nid, weight, is_curated))

    return connections


def get_downstream_connections(profile, curated_neurons):
    """Get downstream connections, highlighting curated neurons."""
    targets = profile.get('top_downstream_targets', [])
    connections = []

    for t in targets[:20]:
        parts = t['target'].split('_')
        weight = t['avg_weight']

        if parts[0] == 'L':
            connections.append(('logit', int(parts[1]), weight))
        elif parts[0].isdigit():
            tgt_nid = f"L{parts[0]}/N{parts[1]}"
            is_curated = tgt_nid in curated_neurons
            connections.append(('neuron', tgt_nid, weight, is_curated))

    return connections


def generate_labels(data, tokenizer):
    """Generate compositional labels for curated set."""

    # Neurons to exclude from upstream/downstream references (always-present hubs)
    EXCLUDE_FROM_REFS = {'L1/N2427', 'L0/N491', 'L0/N8268', 'L0/N10585'}

    curated = {
        # L0 Token Detectors
        'L0/N491': {'type': 'token_detector', 'concept': 'BOS/start token'},
        'L0/N2765': {'type': 'token_detector', 'concept': 'Answer marker'},
        'L0/N13305': {'type': 'token_detector', 'concept': 'cancer'},
        'L0/N14326': {'type': 'token_detector', 'concept': 'brain'},
        'L0/N8694': {'type': 'token_detector', 'concept': 'hormone'},
        'L0/N1918': {'type': 'token_detector', 'concept': 'immune'},
        'L0/N14181': {'type': 'token_detector', 'concept': 'blood'},
        'L0/N8745': {'type': 'token_detector', 'concept': 'heart'},
        # L1-L2 Hubs
        'L1/N2427': {'type': 'hub', 'concept': 'global baseline inhibitor'},
        'L2/N4324': {'type': 'aggregator', 'concept': 'cancer-context'},
        'L2/N10095': {'type': 'aggregator', 'concept': 'brain-context'},
        'L2/N9521': {'type': 'aggregator', 'concept': 'hormone-context'},
        'L2/N5897': {'type': 'aggregator', 'concept': 'immune-context'},
        'L2/N13194': {'type': 'aggregator', 'concept': 'oncology-context'},
        # Mid-layer routing
        'L12/N13860': {'type': 'router', 'concept': 'semantic signal router'},
        'L15/N1816': {'type': 'router', 'concept': 'domain pathway hub'},
        'L24/N5326': {'type': 'router', 'concept': 'answer preparation hub'},
        'L27/N8140': {'type': 'router', 'concept': 'output routing hub'},
        # L31 Outputs
        'L31/N311': {'type': 'output', 'concept': 'Cancer answer'},
        'L31/N317': {'type': 'output', 'concept': 'Pituitary suppression'},
        'L31/N899': {'type': 'output', 'concept': 'Hippocampus suppression'},
        'L31/N5493': {'type': 'output', 'concept': 'brain answer'},
        'L31/N9886': {'type': 'output', 'concept': 'Benign suppression'},
        'L31/N12916': {'type': 'output', 'concept': 'antigen answer'},
        'L31/N14016': {'type': 'output', 'concept': 'Liver answer'},
    }

    labels = {}

    for nid, info in curated.items():
        parts = nid.split('/')
        layer = int(parts[0][1:])
        neuron = int(parts[1][1:])

        profile = find_profile(data, layer, neuron)
        if not profile:
            continue

        upstream = get_upstream_connections(profile, curated)
        downstream = get_downstream_connections(profile, curated)

        label = {
            'layer': layer,
            'neuron': neuron,
            'appearances': profile.get('appearance_count', 0),
            'concept': info['concept'],
            'type': info['type'],
        }

        # === INPUT TRIGGER ===
        if info['type'] == 'token_detector':
            emb_sources = [(tid, w) for typ, tid, w in upstream if typ == 'emb' and w > 0.5]
            if emb_sources:
                tokens = [tokenizer.decode([tid]) for tid, _ in emb_sources[:2]]
                label['input_trigger'] = f"fires on {'/'.join([repr(t) for t in tokens])} token(s)"
            else:
                label['input_trigger'] = f"fires on {info['concept']}-related tokens"

        elif info['type'] == 'hub':
            curated_up = [(n, w) for typ, n, w, is_cur in upstream if typ == 'neuron' and is_cur and w > 0.1 and n not in EXCLUDE_FROM_REFS]
            if curated_up:
                refs = [format_neuron_ref(n, curated, w) for n, w in curated_up[:3]]
                label['input_trigger'] = f"strongly activated by {'; '.join(refs)}"
            else:
                label['input_trigger'] = "activated by BOS/format tokens (baseline)"

        elif info['type'] == 'aggregator':
            curated_up = [(n, w) for typ, n, w, is_cur in upstream if typ == 'neuron' and is_cur and abs(w) > 0.01 and n not in EXCLUDE_FROM_REFS]
            pos = [(n, w) for n, w in curated_up if w > 0]
            neg = [(n, w) for n, w in curated_up if w < 0]
            parts_list = []
            if pos:
                refs = [format_neuron_ref(n, curated) for n, w in pos[:2]]
                parts_list.append(f"activated by {', '.join(refs)}")
            if neg:
                refs = [format_neuron_ref(n, curated) for n, w in neg[:2]]
                parts_list.append(f"inhibited by {', '.join(refs)}")
            label['input_trigger'] = '; '.join(parts_list) if parts_list else f"aggregates {info['concept']} signals from L0 detectors"

        elif info['type'] == 'router':
            curated_up = [(n, w) for typ, n, w, is_cur in upstream if typ == 'neuron' and is_cur and abs(w) > 0.01 and n not in EXCLUDE_FROM_REFS]
            activators = [(n, w) for n, w in curated_up if w > 0][:2]
            inhibitors = [(n, w) for n, w in curated_up if w < 0][:1]
            parts_list = []
            if activators:
                refs = [format_neuron_ref(n, curated, w) for n, w in activators]
                parts_list.append(f"activated by {', '.join(refs)}")
            if inhibitors:
                ref = format_neuron_ref(inhibitors[0][0], curated, inhibitors[0][1])
                parts_list.append(f"gated by {ref}")
            label['input_trigger'] = '; '.join(parts_list) if parts_list else "routes semantic signals through network"

        elif info['type'] == 'output':
            curated_up = [(n, w) for typ, n, w, is_cur in upstream if typ == 'neuron' and is_cur and abs(w) > 0.005 and n not in EXCLUDE_FROM_REFS]
            activators = [(n, w) for n, w in curated_up if w > 0][:2]
            inhibitors = [(n, w) for n, w in curated_up if w < 0][:2]
            parts_list = []
            if activators:
                refs = [format_neuron_ref(n, curated, w) for n, w in activators]
                parts_list.append(f"activated by {', '.join(refs)}")
            if inhibitors:
                refs = [format_neuron_ref(n, curated, w) for n, w in inhibitors]
                parts_list.append(f"inhibited by {', '.join(refs)}")
            label['input_trigger'] = '; '.join(parts_list) if parts_list else "receives signals from semantic pathways"

        # === OUTPUT FUNCTION ===
        if info['type'] == 'token_detector':
            curated_down = [(n, w) for typ, n, w, is_cur in downstream if typ == 'neuron' and is_cur and w > 0.01 and n not in EXCLUDE_FROM_REFS]
            if curated_down:
                refs = [format_neuron_ref(n, curated, w) for n, w in curated_down[:3]]
                label['output_function'] = f"activates {'; '.join(refs)}"
            else:
                label['output_function'] = f"signals {info['concept']} context to downstream aggregators"

        elif info['type'] == 'hub':
            curated_down = [(n, w) for typ, n, w, is_cur in downstream if typ == 'neuron' and is_cur and abs(w) > 0.1 and n not in EXCLUDE_FROM_REFS]
            inhibits = [(n, w) for n, w in curated_down if w < 0][:3]
            if inhibits:
                refs = [format_neuron_ref(n, curated, w) for n, w in inhibits]
                label['output_function'] = f"INHIBITS {', '.join(refs)}"
            else:
                label['output_function'] = "provides baseline inhibition to semantic pathways"

        elif info['type'] == 'aggregator':
            curated_down = [(n, w) for typ, n, w, is_cur in downstream if typ == 'neuron' and is_cur and abs(w) > 0.005 and n not in EXCLUDE_FROM_REFS]
            if curated_down:
                refs = [format_neuron_ref(n, curated, w) for n, w in curated_down[:3]]
                label['output_function'] = f"routes to {'; '.join(refs)}"
            else:
                label['output_function'] = f"routes {info['concept']} signal to mid-layer hubs"

        elif info['type'] == 'router':
            curated_down = [(n, w) for typ, n, w, is_cur in downstream if typ == 'neuron' and is_cur and abs(w) > 0.01 and n not in EXCLUDE_FROM_REFS]
            logit_effects = [(item[1], item[2]) for item in downstream if item[0] == 'logit' and abs(item[2]) > 0.05]
            parts_list = []
            if curated_down:
                refs = [format_neuron_ref(n, curated, w) for n, w in curated_down[:2]]
                parts_list.append(f"routes to {', '.join(refs)}")
            if logit_effects:
                tokens = [f"{repr(tokenizer.decode([tid]))} (w={w:+.2f})" for tid, w in logit_effects[:2]]
                parts_list.append(f"logit effects: {', '.join(tokens)}")
            label['output_function'] = '; '.join(parts_list) if parts_list else "routes signals toward output"

        elif info['type'] == 'output':
            logits = profile.get('logit_effects', [])
            effects = []
            for l in logits[:3]:
                token = l.get('token', '')
                weight = l.get('avg_weight', 0)
                if abs(weight) > 0.1:
                    direction = "PROMOTES" if weight > 0 else "SUPPRESSES"
                    effects.append(f"{direction} {repr(token)} (w={weight:+.2f})")
            label['output_function'] = '; '.join(effects) if effects else "weak logit effects"

        # === COMPLETE FUNCTION ===
        label['complete_function'] = f"Input: {label['input_trigger']} | Output: {label['output_function']}"

        # === FUNCTIONAL ROLE ===
        role_map = {
            'token_detector': 'input_encoding',
            'hub': 'syntactic_routing',
            'aggregator': 'domain_detection',
            'router': 'intermediate_routing',
            'output': 'semantic_retrieval',
        }
        label['functional_role'] = role_map.get(info['type'], 'other')

        labels[nid] = label

    return labels


def main():
    print("Loading data...")
    data = load_data()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')

    print("Generating labels...")
    labels = generate_labels(data, tokenizer)

    output = {
        'metadata': {
            'method': 'compositional_curated',
            'total_neurons': len(labels),
            'description': 'Hand-curated semantic pathway neurons with compositional labels',
        },
        'labels': labels
    }

    output_path = Path('data/neuron_labels_curated.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(labels)} labels to {output_path}")

    print("\n=== Sample Labels ===\n")
    for nid in ['L0/N13305', 'L2/N4324', 'L27/N8140', 'L31/N9886', 'L31/N311']:
        if nid in labels:
            l = labels[nid]
            print(f"{nid} ({l['concept']}):")
            print(f"  Input: {l['input_trigger']}")
            print(f"  Output: {l['output_function']}")
            print(f"  Role: {l['functional_role']}")
            print()


if __name__ == "__main__":
    main()
