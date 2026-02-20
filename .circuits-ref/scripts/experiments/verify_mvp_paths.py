#!/usr/bin/env python3
"""Verify path connectivity in MVP subgraph."""

import json
from collections import defaultdict, deque

with open('data/mvp_subgraph.json') as f:
    subgraph = json.load(f)

# Build adjacency
adj = defaultdict(list)
for e in subgraph['edges']:
    adj[e['source']].append((e['target'], e['weight']))

def get_layer(nid):
    return int(nid.split('/')[0][1:])

def find_paths(start, end_layer=31, max_depth=15):
    paths = []
    queue = deque([(start, [start])])

    while queue:
        node, path = queue.popleft()

        if get_layer(node) == end_layer:
            paths.append(path)
            continue

        if len(path) >= max_depth:
            continue

        for next_node, _ in adj[node]:
            if next_node not in path:
                queue.append((next_node, path + [next_node]))

    return paths

l0_neurons = [nid for nid in subgraph['neurons'] if get_layer(nid) == 0]
l31_neurons = [nid for nid in subgraph['neurons'] if get_layer(nid) == 31]

print('=== Path Analysis ===\n')
print(f'L0 neurons: {len(l0_neurons)}')
print(f'L31 neurons: {len(l31_neurons)}')

for l0 in l0_neurons:
    nd = subgraph['neurons'][l0]
    tokens = nd.get('input_tokens', [])
    token = tokens[0].get('token', '?') if tokens else '?'

    paths = find_paths(l0)
    if paths:
        paths.sort(key=len)
        shortest = paths[0]

        endpoints = set(p[-1] for p in paths)
        endpoint_info = []
        for ep in endpoints:
            ep_nd = subgraph['neurons'].get(ep, {})
            logits = ep_nd.get('logit_effects', [])
            if logits:
                endpoint_info.append(f"{ep}({logits[0]['token']})")

        path_str = " -> ".join(shortest[:8])
        if len(shortest) > 8:
            path_str += "..."

        print(f'\n{l0} [{token}]:')
        print(f'  Paths found: {len(paths)}, shortest length: {len(shortest)}')
        print(f'  Reaches: {endpoint_info}')
        print(f'  Example path: {path_str}')
    else:
        print(f'\n{l0} [{token}]: No paths to L31')

# Layer continuity
layers_present = sorted(set(get_layer(n) for n in subgraph['neurons']))
gaps = []
for i in range(len(layers_present)-1):
    gap = layers_present[i+1] - layers_present[i] - 1
    if gap > 0:
        gaps.append((layers_present[i], layers_present[i+1], gap))

print('\n\n=== Subgraph Summary ===')
print(f'Layers present: {layers_present}')
print(f'Total layers: {len(layers_present)}/32')
print(f'Gaps: {gaps if gaps else "None!"}')
