# %%
"""Plot attribution graph from saved global attributions."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import torch

# Configuration
# wandb_id = "c0k3z78g" # ss_gpt2_simple-2L
# n_blocks = 2
wandb_id = "8ynfbr38"  # ss_gpt2_simple-1L
n_blocks = 1
edge_threshold = 1e-2

# Load saved data
out_dir = Path(__file__).parent / "out"
global_attributions = torch.load(out_dir / f"global_attributions_{wandb_id}.pt")

# Reconstruct alive_indices from attribution tensor shapes
alive_indices: dict[str, list[int]] = {}
for (in_layer, out_layer), attr in global_attributions.items():
    n_alive_in, n_alive_out = attr.shape
    if in_layer not in alive_indices:
        alive_indices[in_layer] = list(range(n_alive_in))
    if out_layer not in alive_indices:
        alive_indices[out_layer] = list(range(n_alive_out))

print(f"Loaded attributions for {len(global_attributions)} layer pairs")
print(f"Total alive components: {sum(len(v) for v in alive_indices.values())}")

# Count edges before and after thresholding
total_edges = sum(attr.numel() for attr in global_attributions.values())
print(f"Total edges: {total_edges:,}")
thresholds = [1, 0.6, 0.2, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-12, 1e-15]
for threshold in thresholds:
    total_edges_threshold = sum(
        (attr > threshold).sum().item() for attr in global_attributions.values()
    )
    print(f"Edges > {threshold}: {total_edges_threshold:,}")

# %%
# Plot the attribution graph
print("\nPlotting attribution graph...")

# Define layer order within a block (network order)
layer_names_in_block = [
    "attn.q_proj",
    "attn.k_proj",
    "attn.v_proj",
    "attn.o_proj",
    "mlp.c_fc",
    "mlp.down_proj",
]

# Build full layer list in network order
all_layers = []
for block_idx in range(n_blocks):
    for layer_name in layer_names_in_block:
        all_layers.append(f"h.{block_idx}.{layer_name}")

# Create graph
G = nx.DiGraph()

# Add nodes for each (layer, component) pair
node_positions = {}
block_spacing = 6.0  # Vertical spacing between blocks

# Layer y-offsets within a block: down_proj at top, q/k/v at same level at bottom
# q_proj, k_proj, v_proj are placed side by side since they never connect to each other
layer_y_offsets = {
    "mlp.down_proj": 2.0,
    "mlp.c_fc": 1.0,
    "attn.o_proj": 0.0,
    "attn.v_proj": -1.0,  # Same y-level for q/k/v
    "attn.k_proj": -1.0,
    "attn.q_proj": -1.0,
}

# X-offsets for q/k/v to place them side by side with much more spacing
layer_x_offsets = {
    "mlp.down_proj": 0.0,
    "mlp.c_fc": 0.0,
    "attn.o_proj": 0.0,
    "attn.q_proj": -20.0,  # Left (much more spacing)
    "attn.k_proj": 0.0,  # Center
    "attn.v_proj": 20.0,  # Right (much more spacing)
}

for layer in all_layers:
    parts = layer.split(".")
    block_idx = int(parts[1])
    layer_name = ".".join(parts[2:])

    n_alive = len(alive_indices.get(layer, []))
    if n_alive == 0:
        continue

    # Block 1 on top (higher y), Block 0 on bottom (lower y)
    y_base = block_idx * block_spacing + layer_y_offsets[layer_name]
    # X-axis base depends on layer type (q/k/v are offset)
    x_base = layer_x_offsets[layer_name]

    for comp_idx, local_idx in enumerate(alive_indices.get(layer, [])):
        node_id = f"{layer}:{local_idx}"
        G.add_node(node_id, layer=layer, component=local_idx)
        # Increase spacing between nodes from 0.15 to 0.25 for less overlap
        x = x_base + (comp_idx - n_alive / 2) * 0.25
        y = y_base
        node_positions[node_id] = (x, y)

# Add edges based on attributions
edge_weights = []
for (in_layer, out_layer), attr_tensor in global_attributions.items():
    in_alive = alive_indices.get(in_layer, [])
    out_alive = alive_indices.get(out_layer, [])

    for i, in_comp in enumerate(in_alive):
        for j, out_comp in enumerate(out_alive):
            weight = attr_tensor[i, j].item()
            if weight > edge_threshold:
                in_node = f"{in_layer}:{in_comp}"
                out_node = f"{out_layer}:{out_comp}"
                if in_node in G.nodes and out_node in G.nodes:
                    G.add_edge(in_node, out_node, weight=weight)
                    edge_weights.append(weight)

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Create figure (extra wide to accommodate q/k/v side by side with large spacing)
fig, ax = plt.subplots(1, 1, figsize=(32, 12))

# Draw nodes grouped by layer
layer_colors = {
    "attn.q_proj": "#1f77b4",
    "attn.k_proj": "#2ca02c",
    "attn.v_proj": "#9467bd",
    "attn.o_proj": "#d62728",
    "mlp.c_fc": "#ff7f0e",
    "mlp.down_proj": "#8c564b",
}

for layer in all_layers:
    parts = layer.split(".")
    layer_name = ".".join(parts[2:])
    color = layer_colors.get(layer_name, "#333333")

    layer_nodes = [n for n in G.nodes if G.nodes[n].get("layer") == layer]
    if layer_nodes:
        pos_subset = {n: node_positions[n] for n in layer_nodes}
        nx.draw_networkx_nodes(
            G,
            pos_subset,
            nodelist=layer_nodes,
            node_color=color,
            node_size=100,
            alpha=0.8,
            ax=ax,
        )

# Draw edges batched by weight bucket for performance
if edge_weights:
    max_weight = max(edge_weights)
    min_weight = min(edge_weights)

    n_buckets = 10
    edge_buckets: list[list[tuple[str, str]]] = [[] for _ in range(n_buckets)]

    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 0)
        if max_weight > min_weight:
            normalized = (weight - min_weight) / (max_weight - min_weight)
        else:
            normalized = 0.5
        bucket_idx = min(int(normalized * n_buckets), n_buckets - 1)
        edge_buckets[bucket_idx].append((u, v))

    for bucket_idx, bucket_edges in enumerate(edge_buckets):
        if not bucket_edges:
            continue
        normalized = (bucket_idx + 0.5) / n_buckets
        width = 0.2 + normalized * 2.0
        alpha = 0.3 + normalized * 0.5

        nx.draw_networkx_edges(
            G,
            node_positions,
            edgelist=bucket_edges,
            width=width,
            alpha=alpha,
            edge_color="#666666",
            arrows=True,
            arrowsize=8,
            connectionstyle="arc3,rad=0.1",
            ax=ax,
        )

# Add legend
legend_elements = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10, label=name)
    for name, color in [
        ("q_proj", "#1f77b4"),
        ("k_proj", "#2ca02c"),
        ("v_proj", "#9467bd"),
        ("o_proj", "#d62728"),
        ("c_fc", "#ff7f0e"),
        ("down_proj", "#8c564b"),
    ]
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

ax.set_title("Global Attribution Graph", fontsize=14, fontweight="bold")
ax.axis("off")
plt.tight_layout()

# Save
# Make an edge threshold string in scientific notation which doesn't include decimal places
edge_threshold_str = f"{edge_threshold:.1e}".replace(".0", "")
output_path = out_dir / f"attribution_graph_{wandb_id}_edge_threshold_{edge_threshold_str}.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved to {output_path}")

plt.close(fig)

# %%
