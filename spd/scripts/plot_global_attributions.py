"""Plot attribution graph from saved global attributions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch import Tensor

from spd.scripts.model_loading import get_out_dir

if TYPE_CHECKING:
    Graph = nx.DiGraph[Any]
else:
    Graph = nx.DiGraph


@dataclass
class LayerInfo:
    """All display information for a layer type."""

    name: str
    color: str
    y_offset: float
    x_offset: float = 0.0
    legend_name: str | None = None  # If different from name

    @property
    def display_name(self) -> str:
        return self.legend_name if self.legend_name is not None else self.name


# fmt: off
LAYER_INFOS: dict[str, LayerInfo] = {
    "wte":            LayerInfo("wte",            color="#34495E", y_offset=-3.0),
    "attn.q_proj":    LayerInfo("attn.q_proj",    color="#1f77b4", y_offset=-1.0, x_offset=-20.0, legend_name="q_proj"),
    "attn.k_proj":    LayerInfo("attn.k_proj",    color="#2ca02c", y_offset=-1.0, x_offset=0.0,   legend_name="k_proj"),
    "attn.v_proj":    LayerInfo("attn.v_proj",    color="#9467bd", y_offset=-1.0, x_offset=20.0,  legend_name="v_proj"),
    "attn.o_proj":    LayerInfo("attn.o_proj",    color="#d62728", y_offset=0.0,  legend_name="o_proj"),
    "mlp.c_fc":       LayerInfo("mlp.c_fc",       color="#ff7f0e", y_offset=1.0,  legend_name="c_fc"),
    "mlp.down_proj":  LayerInfo("mlp.down_proj",  color="#8c564b", y_offset=2.0,  legend_name="down_proj"),
    "output":         LayerInfo("output",         color="#17A589", y_offset=4.0),
}
# fmt: on


def load_attributions(
    out_dir: Path, wandb_id: str
) -> tuple[dict[tuple[str, str], Tensor], dict[str, list[int]]]:
    """Load global attributions and reconstruct alive_indices from tensor shapes."""
    global_attributions: dict[tuple[str, str], Tensor] = torch.load(
        out_dir / f"global_attributions_{wandb_id}.pt"
    )

    alive_indices: dict[str, list[int]] = {}
    for (in_layer, out_layer), attr in global_attributions.items():
        n_alive_in, n_alive_out = attr.shape
        if in_layer not in alive_indices:
            alive_indices[in_layer] = list(range(n_alive_in))
        if out_layer not in alive_indices:
            alive_indices[out_layer] = list(range(n_alive_out))

    return global_attributions, alive_indices


def print_edge_statistics(global_attributions: dict[tuple[str, str], Tensor]) -> None:
    """Print statistics about edges at various thresholds."""
    total_edges = sum(attr.numel() for attr in global_attributions.values())
    print(f"Total edges: {total_edges:,}")

    thresholds = [1, 0.6, 0.2, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-12, 1e-15]
    for threshold in thresholds:
        edges_above = sum((attr > threshold).sum().item() for attr in global_attributions.values())
        print(f"Edges > {threshold}: {edges_above:,}")


def build_layer_list(n_blocks: int) -> list[str]:
    """Build full layer list in network order (wte -> blocks -> output)."""
    all_layers = ["wte"]
    for block_idx in range(n_blocks):
        for layer_name in [k for k in LAYER_INFOS if k not in ["wte", "output"]]:
            all_layers.append(f"h.{block_idx}.{layer_name}")
    all_layers.append("output")
    return all_layers


def find_nodes_with_edges(
    global_attributions: dict[tuple[str, str], Tensor],
    alive_indices: dict[str, list[int]],
    edge_threshold: float,
) -> set[str]:
    """Find all nodes that participate in edges above threshold."""
    nodes_with_edges: set[str] = set()
    for (in_layer, out_layer), attr_tensor in global_attributions.items():
        in_alive = alive_indices.get(in_layer, [])
        out_alive = alive_indices.get(out_layer, [])
        for i, in_comp in enumerate(in_alive):
            for j, out_comp in enumerate(out_alive):
                if attr_tensor[i, j].item() > edge_threshold:
                    nodes_with_edges.add(f"{in_layer}:{in_comp}")
                    nodes_with_edges.add(f"{out_layer}:{out_comp}")
    return nodes_with_edges


def parse_layer_name(layer: str, n_blocks: int) -> tuple[int, str]:
    """Parse layer name to get block_idx and base layer_name."""
    if layer == "wte":
        return 0, "wte"
    elif layer == "output":
        return n_blocks - 1, "output"
    else:
        parts = layer.split(".")
        return int(parts[1]), ".".join(parts[2:])


def build_graph(
    all_layers: list[str],
    global_attributions: dict[tuple[str, str], Tensor],
    alive_indices: dict[str, list[int]],
    nodes_with_edges: set[str],
    n_blocks: int,
    edge_threshold: float,
    block_spacing: float = 6.0,
) -> tuple[Graph, dict[str, tuple[float, float]], list[float]]:
    """Build the attribution graph with node positions and edge weights."""
    G: Graph = nx.DiGraph()
    node_positions: dict[str, tuple[float, float]] = {}

    # Add nodes
    for layer in all_layers:
        block_idx, layer_name = parse_layer_name(layer, n_blocks)
        info = LAYER_INFOS[layer_name]

        y_base = block_idx * block_spacing + info.y_offset
        x_base = info.x_offset

        layer_alive = alive_indices.get(layer, [])
        layer_nodes_with_edges = [
            (idx, comp)
            for idx, comp in enumerate(layer_alive)
            if f"{layer}:{comp}" in nodes_with_edges
        ]
        n_layer_nodes = len(layer_nodes_with_edges)

        for pos_idx, (_, local_idx) in enumerate(layer_nodes_with_edges):
            node_id = f"{layer}:{local_idx}"
            G.add_node(node_id, layer=layer, component=local_idx)
            x = x_base + (pos_idx - n_layer_nodes / 2) * 0.25
            node_positions[node_id] = (x, y_base)

    # Add edges
    edge_weights: list[float] = []
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

    return G, node_positions, edge_weights


def draw_nodes(
    ax: plt.Axes,
    G: Graph,
    node_positions: dict[str, tuple[float, float]],
    all_layers: list[str],
) -> None:
    """Draw nodes grouped by layer."""
    for layer in all_layers:
        if layer in ("wte", "output"):
            layer_name = layer
        else:
            parts = layer.split(".")
            layer_name = ".".join(parts[2:])
        color = LAYER_INFOS[layer_name].color

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


def draw_edges(
    ax: plt.Axes,
    G: Graph,
    node_positions: dict[str, tuple[float, float]],
    edge_weights: list[float],
    n_buckets: int = 10,
) -> None:
    """Draw edges batched by weight bucket for performance."""
    if not edge_weights:
        return

    max_weight = max(edge_weights)
    min_weight = min(edge_weights)

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


def add_legend(ax: plt.Axes) -> None:
    """Add legend for layer types."""
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=info.color,
            markersize=10,
            label=info.display_name,
        )
        for info in LAYER_INFOS.values()
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)


def plot_attribution_graph(
    global_attributions: dict[tuple[str, str], Tensor],
    alive_indices: dict[str, list[int]],
    n_blocks: int,
    edge_threshold: float,
    output_path: Path,
) -> None:
    """Create and save the attribution graph visualization."""
    print("\nPlotting attribution graph...")

    all_layers = build_layer_list(n_blocks)
    nodes_with_edges = find_nodes_with_edges(global_attributions, alive_indices, edge_threshold)
    G, node_positions, edge_weights = build_graph(
        all_layers, global_attributions, alive_indices, nodes_with_edges, n_blocks, edge_threshold
    )

    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    fig, ax = plt.subplots(1, 1, figsize=(32, 12))

    draw_nodes(ax, G, node_positions, all_layers)
    draw_edges(ax, G, node_positions, edge_weights)
    add_legend(ax)

    ax.set_title("Global Attribution Graph", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    # Configuration
    # wandb_id = "jyo9duz5"  # ss_gpt2_simple-1.25M (4L)
    # wandb_id = "c0k3z78g"  # ss_gpt2_simple-2L
    wandb_id = "33n6xjjt"  # ss_gpt2_simple-1L (New)
    n_blocks = 1
    edge_threshold = 1e-1

    out_dir = get_out_dir()

    global_attributions, alive_indices = load_attributions(out_dir, wandb_id)
    print(f"Loaded attributions for {len(global_attributions)} layer pairs")
    print(f"Total alive components: {sum(len(v) for v in alive_indices.values())}")

    print_edge_statistics(global_attributions)

    edge_threshold_str = f"{edge_threshold:.1e}".replace(".0", "")
    output_path = out_dir / f"attribution_graph_{wandb_id}_edge_threshold_{edge_threshold_str}.png"

    plot_attribution_graph(
        global_attributions,
        alive_indices,
        n_blocks,
        edge_threshold,
        output_path,
    )
