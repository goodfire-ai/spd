# %%
"""Plot local attribution graph from saved .pt file."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float
from matplotlib.collections import LineCollection
from torch import Tensor

from spd.scripts.model_loading import get_out_dir


@dataclass
class PairAttribution:
    source: str
    target: str
    attribution: Float[Tensor, "s_in trimmed_c_in s_out trimmed_c_out"]
    trimmed_c_in_idxs: list[int]
    trimmed_c_out_idxs: list[int]
    is_kv_to_o_pair: bool
    # Original alive masks (from model CI, before any optimization)
    # Used to show "ghost" nodes that would have been active without CI optimization
    # Shape: [n_seq, n_components] where True means alive at that (seq, component)
    original_alive_mask_in: Float[Tensor, "seq C"] | None = None
    original_alive_mask_out: Float[Tensor, "seq C"] | None = None


@dataclass
class NodeInfo:
    """Information about a node in the attribution graph."""

    layer: str
    seq_pos: int
    component_idx: int
    x: float
    y: float
    importance: float


def get_layer_order() -> list[str]:
    """Get the canonical ordering of sublayers within a block."""
    return [
        "wte",  # Word token embeddings (first layer)
        "attn.q_proj",
        "attn.v_proj",
        "attn.k_proj",
        "attn.o_proj",
        "mlp.c_fc",
        "mlp.down_proj",
    ]


# Sublayers that share a row (q, v, k displayed side by side)
QVK_SUBLAYERS = {"attn.q_proj", "attn.v_proj", "attn.k_proj"}

# Column allocation for q, v, k: (n_cols, start_col) out of 12 total columns
# Layout: Q(2) | gap(1) | V(4) | gap(1) | K(4) = 12 total
QVK_LAYOUT: dict[str, tuple[int, int]] = {
    "attn.q_proj": (2, 0),  # 2 columns, starts at 0
    "attn.v_proj": (4, 3),  # 4 columns, starts at 3 (1 col gap after q)
    "attn.k_proj": (4, 8),  # 4 columns, starts at 8 (1 col gap after v)
}
QVK_TOTAL_COLS = 12


def get_layer_color(layer: str) -> str:
    """Get color for a layer based on its type."""
    if layer == "wte":
        return "#34495E"  # Dark blue-gray for embeddings

    colors = {
        "attn.q_proj": "#E67E22",  # Orange
        "attn.k_proj": "#27AE60",  # Green
        "attn.v_proj": "#F1C40F",  # Yellow
        "attn.o_proj": "#E74C3C",  # Red
        "mlp.c_fc": "#3498DB",  # Blue
        "mlp.down_proj": "#9B59B6",  # Purple
    }
    for sublayer, color in colors.items():
        if sublayer in layer:
            return color
    return "#95A5A6"  # Gray fallback


def parse_layer_name(layer: str) -> tuple[int, str]:
    """Parse layer name into block index and sublayer type.

    E.g., "h.0.attn.q_proj" -> (0, "attn.q_proj")
         "wte" -> (-1, "wte")
         "output" -> (999, "output")
    """
    if layer == "wte":
        return -1, "wte"
    if layer == "output":
        return 999, "output"

    parts = layer.split(".")
    block_idx = int(parts[1])
    sublayer = ".".join(parts[2:])
    return block_idx, sublayer


def compute_layer_y_positions(
    attr_pairs: list[PairAttribution],
) -> dict[str, float]:
    """Compute Y position for each layer.

    Layers are ordered by block, then by sublayer type within block.
    q, v, k layers share the same row (same y position).

    Returns:
        Dict mapping layer name to y position.
    """
    # Collect all unique layers
    layers = set()
    for pair in attr_pairs:
        layers.add(pair.source)
        layers.add(pair.target)

    # Parse and sort
    layer_order = get_layer_order()
    parsed = [(layer, *parse_layer_name(layer)) for layer in layers]

    def sort_key(item: tuple[str, int, str]) -> tuple[int, int]:
        _, block_idx, sublayer = item
        sublayer_idx = layer_order.index(sublayer) if sublayer in layer_order else 999
        return (block_idx, sublayer_idx)

    sorted_layers = sorted(parsed, key=sort_key)

    # Assign Y positions, grouping q, v, k on the same row
    y_positions = {}
    current_y = 0.0
    prev_block_idx = None
    prev_was_qvk = False

    for layer, block_idx, sublayer in sorted_layers:
        is_qvk = sublayer in QVK_SUBLAYERS

        # Check if we should share y with previous layer
        if is_qvk and prev_was_qvk and block_idx == prev_block_idx:
            # Same row as previous q/v/k layer
            y_positions[layer] = current_y
        else:
            # New row
            if y_positions:  # Not the first layer
                current_y += 1.0
            y_positions[layer] = current_y

        prev_block_idx = block_idx
        prev_was_qvk = is_qvk

    return y_positions


def compute_node_importances(
    attr_pairs: list[PairAttribution],
    n_seq: int,
) -> tuple[dict[str, Float[Tensor, "seq C"]], dict[str, Float[Tensor, "seq C"]]]:
    """Compute importance values for nodes based on total attribution flow.

    Returns:
        importances: Dict mapping layer -> tensor of shape [n_seq, max_component_idx+1].
            Importance is the sum of incoming and outgoing attributions.
        original_alive_masks: Dict mapping layer -> bool tensor of shape [n_seq, max_component_idx+1].
            True means the component was alive at that (seq_pos, component) in original model CI.
    """
    # First pass: determine max component index per layer (including original alive)
    layer_max_c: dict[str, int] = {}
    for pair in attr_pairs:
        src_max = max(pair.trimmed_c_in_idxs) if pair.trimmed_c_in_idxs else 0
        tgt_max = max(pair.trimmed_c_out_idxs) if pair.trimmed_c_out_idxs else 0
        # Also consider original alive mask dimensions
        if pair.original_alive_mask_in is not None:
            src_max = max(src_max, pair.original_alive_mask_in.shape[1] - 1)
        if pair.original_alive_mask_out is not None:
            tgt_max = max(tgt_max, pair.original_alive_mask_out.shape[1] - 1)
        layer_max_c[pair.source] = max(layer_max_c.get(pair.source, 0), src_max)
        layer_max_c[pair.target] = max(layer_max_c.get(pair.target, 0), tgt_max)

    # Initialize importance tensors and original alive masks (on same device as attributions)
    device = attr_pairs[0].attribution.device if attr_pairs else "cpu"
    importances: dict[str, Float[Tensor, "seq C"]] = {}
    original_alive_masks: dict[str, Float[Tensor, "seq C"]] = {}
    for layer, max_c in layer_max_c.items():
        importances[layer] = torch.zeros(n_seq, max_c + 1, device=device)
        original_alive_masks[layer] = torch.zeros(n_seq, max_c + 1, device=device, dtype=torch.bool)

    # Accumulate attribution magnitudes and original alive masks
    for pair in attr_pairs:
        attr = pair.attribution.abs()  # [s_in, trimmed_c_in, s_out, trimmed_c_out]

        # Sum over output dimensions -> importance for source nodes
        src_importance = attr.sum(dim=(2, 3))  # [s_in, trimmed_c_in]
        for i, c_in in enumerate(pair.trimmed_c_in_idxs):
            importances[pair.source][:, c_in] += src_importance[:, i]

        # Sum over input dimensions -> importance for target nodes
        tgt_importance = attr.sum(dim=(0, 1))  # [s_out, trimmed_c_out]
        for j, c_out in enumerate(pair.trimmed_c_out_idxs):
            importances[pair.target][:, c_out] += tgt_importance[:, j]

        # Accumulate original alive masks (OR them together since multiple pairs may have same layer)
        if pair.original_alive_mask_in is not None:
            n_c = pair.original_alive_mask_in.shape[1]
            original_alive_masks[pair.source][:, :n_c] |= pair.original_alive_mask_in
        if pair.original_alive_mask_out is not None:
            n_c = pair.original_alive_mask_out.shape[1]
            original_alive_masks[pair.target][:, :n_c] |= pair.original_alive_mask_out

    return importances, original_alive_masks


def plot_local_graph(
    attr_pairs: list[PairAttribution],
    token_strings: list[str],
    min_edge_weight: float = 0.001,
    node_scale: float = 30.0,
    edge_alpha_scale: float = 0.5,
    figsize: tuple[float, float] | None = None,
    max_grid_cols: int = 12,
    output_token_labels: dict[int, str] | None = None,
    output_prob_threshold: float | None = None,
    output_probs_by_pos: dict[tuple[int, int], float] | None = None,
) -> plt.Figure:
    """Plot the local attribution graph.

    Args:
        attr_pairs: List of PairAttribution objects from compute_local_attributions.
        token_strings: List of token strings for x-axis labels.
        min_edge_weight: Minimum edge weight to display.
        node_scale: Fixed size for all nodes.
        edge_alpha_scale: Scale factor for edge transparency.
        figsize: Figure size (width, height). Auto-computed if None.
        max_grid_cols: Maximum number of columns in the grid per layer.
        output_token_labels: Dict mapping output component indices to token strings.
        output_prob_threshold: Threshold used for filtering output probabilities.
        output_probs_by_pos: Dict mapping (seq_pos, component_idx) -> probability for output layer.

    Returns:
        Matplotlib figure.
    """
    n_seq = len(token_strings)

    # Compute node importances and original alive masks (per-position)
    importances, original_alive_masks = compute_node_importances(attr_pairs, n_seq)

    # Compute layout
    layer_y = compute_layer_y_positions(attr_pairs)

    # Auto-compute figure size
    if figsize is None:
        total_height = max(layer_y.values())
        figsize = (max(16, n_seq * 2), max(8, total_height * 1.2))

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("#FAFAFA")

    # Collect all nodes and their positions
    nodes: list[NodeInfo] = []
    ghost_nodes: list[NodeInfo] = []  # Nodes originally alive at this position but now dead
    node_lookup: dict[tuple[str, int, int], NodeInfo] = {}  # (layer, seq, comp) -> NodeInfo

    # X spacing: spread tokens across the plot
    x_positions = np.linspace(0.1, 0.9, n_seq)

    # Grid layout parameters
    col_spacing = 0.012  # Horizontal spacing between columns in grid
    row_spacing = 0.08  # Vertical spacing between rows in grid

    for layer, y_center in layer_y.items():
        if layer not in importances:
            continue

        layer_imp = importances[layer]  # [n_seq, max_c+1]
        alive_mask = layer_imp > 0

        # Get original alive mask for this layer (per-position)
        layer_original_mask = original_alive_masks.get(layer)

        # Find all components that are alive at ANY sequence position (current)
        all_alive_components = torch.where(alive_mask.any(dim=0))[0].tolist()

        # Find all components that were originally alive at ANY sequence position
        originally_alive_any_pos: list[int] = []
        if layer_original_mask is not None:
            originally_alive_any_pos = torch.where(layer_original_mask.any(dim=0))[0].tolist()

        # Combine for layout purposes
        all_components_for_layout = sorted(set(all_alive_components) | set(originally_alive_any_pos))
        n_components = len(all_components_for_layout)

        if n_components == 0:
            continue

        # Check if this is a q/v/k layer that shares a row
        _, sublayer = parse_layer_name(layer)
        is_qvk = sublayer in QVK_SUBLAYERS
        is_output_layer = layer == "output"

        if is_qvk:
            # Use the allocated columns for this sublayer
            layer_max_cols, start_col = QVK_LAYOUT[sublayer]
        else:
            layer_max_cols = max_grid_cols
            start_col = 0

        # Calculate grid dimensions for this layer
        n_rows = (n_components + layer_max_cols - 1) // layer_max_cols
        n_cols = min(n_components, layer_max_cols)

        # Process each sequence position separately
        for s in range(n_seq):
            # Center the grid at this sequence position
            x_base = x_positions[s]

            if is_output_layer:
                # For output layer: only show components active at THIS position
                active_at_pos = [c for c in all_alive_components if alive_mask[s, c]]
                n_active = len(active_at_pos)
                if n_active == 0:
                    continue

                n_rows_pos = (n_active + max_grid_cols - 1) // max_grid_cols
                n_cols_pos = min(n_active, max_grid_cols)

                # Use full width of max_grid_cols for output layer to spread out labels
                max_width = (max_grid_cols - 1) * col_spacing
                output_col_spacing = max_width / max(n_cols_pos - 1, 1) if n_cols_pos > 1 else 0

                for local_idx, c in enumerate(active_at_pos):
                    col = local_idx % max_grid_cols
                    row = local_idx // max_grid_cols

                    x_offset = (col - (n_cols_pos - 1) / 2) * output_col_spacing
                    y_offset = (row - (n_rows_pos - 1) / 2) * row_spacing

                    x = x_base + x_offset
                    y = y_center + y_offset

                    imp = layer_imp[s, c].item()
                    node = NodeInfo(
                        layer=layer,
                        seq_pos=s,
                        component_idx=c,
                        x=x,
                        y=y,
                        importance=imp,
                    )
                    nodes.append(node)
                    node_lookup[(layer, s, c)] = node
            else:
                # For other layers: arrange all components in grid (including ghost nodes)
                for local_idx, c in enumerate(all_components_for_layout):
                    col = local_idx % layer_max_cols
                    row = local_idx // layer_max_cols

                    if is_qvk:
                        # Position within the allocated horizontal segment
                        # Center of this sublayer's segment within the total QVK row
                        segment_center = start_col + layer_max_cols / 2
                        total_center = QVK_TOTAL_COLS / 2
                        # Offset from center of entire row
                        segment_offset = (segment_center - total_center) * col_spacing
                        # Position within segment, centered
                        x_offset = segment_offset + (col - (n_cols - 1) / 2) * col_spacing
                    else:
                        # Position within grid, centered on sequence position
                        x_offset = (col - (n_cols - 1) / 2) * col_spacing

                    y_offset = (row - (n_rows - 1) / 2) * row_spacing

                    x = x_base + x_offset
                    y = y_center + y_offset

                    imp = layer_imp[s, c].item() if c < layer_imp.shape[1] else 0.0
                    node = NodeInfo(
                        layer=layer,
                        seq_pos=s,
                        component_idx=c,
                        x=x,
                        y=y,
                        importance=imp,
                    )

                    # Check if this is a ghost node at THIS position
                    # Ghost = originally alive at this (s, c) but not currently alive at this (s, c)
                    is_currently_alive_here = alive_mask[s, c].item() if c < alive_mask.shape[1] else False
                    is_originally_alive_here = (
                        layer_original_mask is not None
                        and c < layer_original_mask.shape[1]
                        and layer_original_mask[s, c].item()
                    )
                    is_ghost = is_originally_alive_here and not is_currently_alive_here

                    if is_ghost:
                        ghost_nodes.append(node)
                    else:
                        nodes.append(node)
                    node_lookup[(layer, s, c)] = node

    # Collect edges
    edges: list[tuple[NodeInfo, NodeInfo, float]] = []

    for pair in attr_pairs:
        attr = pair.attribution  # [s_in, trimmed_c_in, s_out, trimmed_c_out]

        for i, c_in in enumerate(pair.trimmed_c_in_idxs):
            for j, c_out in enumerate(pair.trimmed_c_out_idxs):
                for s_in in range(attr.shape[0]):
                    for s_out in range(attr.shape[2]):
                        weight = attr[s_in, i, s_out, j].abs().item()
                        if weight < min_edge_weight:
                            continue

                        src_key = (pair.source, s_in, c_in)
                        tgt_key = (pair.target, s_out, c_out)

                        if src_key in node_lookup and tgt_key in node_lookup:
                            src_node = node_lookup[src_key]
                            tgt_node = node_lookup[tgt_key]
                            edges.append((src_node, tgt_node, weight))

    edges_by_target: dict[tuple[str, int, int], list[tuple[NodeInfo, NodeInfo, float]]] = {}
    for src, tgt, w in edges:
        key = (tgt.layer, tgt.seq_pos, tgt.component_idx)
        if key not in edges_by_target:
            edges_by_target[key] = []
        edges_by_target[key].append((src, tgt, w))

    sorted_edges = []
    for target_edges in edges_by_target.values():
        sorted_edges.extend(sorted(target_edges, key=lambda e: e[2], reverse=True))
    edges = sorted_edges

    # Normalize edge weights for alpha
    if edges:
        edge_weights = [e[2] for e in edges]
        max_edge = max(edge_weights)
        if max_edge > 0:
            edges = [(s, t, w / max_edge) for s, t, w in edges]

    # Track which nodes have edges
    nodes_with_edges = set()
    for src, tgt, _ in edges:
        nodes_with_edges.add((src.layer, src.seq_pos, src.component_idx))
        nodes_with_edges.add((tgt.layer, tgt.seq_pos, tgt.component_idx))

    # Draw edges
    if edges:
        lines = []
        alphas = []
        for src, tgt, w in edges:
            lines.append([(src.x, src.y), (tgt.x, tgt.y)])
            alphas.append(w * edge_alpha_scale)

        lc = LineCollection(
            lines,
            colors=[(0.5, 0.5, 0.5, a) for a in alphas],
            linewidths=0.5,
            zorder=1,
        )
        ax.add_collection(lc)

    # Draw ghost nodes first (so they appear behind regular nodes)
    # Ghost nodes = originally alive (in model CI) but not currently alive (in optimized CI)
    for node in ghost_nodes:
        ax.scatter(
            node.x,
            node.y,
            s=node_scale,
            c="#909090",  # Darker gray than nodes-without-edges
            edgecolors="white",
            linewidths=0.5,
            zorder=1.5,  # Behind regular nodes
            alpha=0.5,
        )

    # Draw regular nodes
    for node in nodes:
        node_key = (node.layer, node.seq_pos, node.component_idx)
        has_edges = node_key in nodes_with_edges

        if has_edges:
            color = get_layer_color(node.layer)
            alpha = 0.9
        else:
            color = "#D3D3D3"
            alpha = 0.3

        ax.scatter(
            node.x,
            node.y,
            s=node_scale,
            c=color,
            edgecolors="white",
            linewidths=0.5,
            zorder=2,
            alpha=alpha,
        )

        # Add token label and probability for output layer nodes
        if node.layer == "output" and output_token_labels is not None and has_edges:
            token_label = output_token_labels.get(node.component_idx, "")
            if token_label:
                # Build label with probability if available
                label_text = repr(token_label)[1:-1]  # Strip quotes but show escape chars
                if output_probs_by_pos is not None:
                    prob = output_probs_by_pos.get((node.seq_pos, node.component_idx))
                    if prob is not None:
                        label_text = f"({prob:.2f})\n{label_text}"
                ax.annotate(
                    label_text,
                    (node.x, node.y),
                    xytext=(0, 6),
                    textcoords="offset points",
                    fontsize=6,
                    ha="center",
                    va="bottom",
                    alpha=0.8,
                )

    # Configure axes
    total_height = max(layer_y.values())
    ax.set_xlim(0, 1)
    # Extra top margin to fit output token labels with probabilities
    ax.set_ylim(-0.5, total_height + 1.0)

    # X-axis: token labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(token_strings, rotation=45, ha="right", fontsize=9)
    ax.xaxis.set_ticks_position("bottom")

    # Y-axis: layer labels (group q/v/k into single label)
    layer_names_sorted = sorted(layer_y.keys(), key=lambda x: layer_y[x])
    # Deduplicate y positions and create combined labels for q/v/k rows
    y_to_layers: dict[float, list[str]] = {}
    for layer in layer_names_sorted:
        y = layer_y[layer]
        if y not in y_to_layers:
            y_to_layers[y] = []
        y_to_layers[y].append(layer)

    layer_centers = sorted(y_to_layers.keys())
    layer_labels = []
    for y in layer_centers:
        layers_at_y = y_to_layers[y]
        if len(layers_at_y) > 1:
            # Multiple layers at same y (q/v/k row)
            # Extract block prefix and combine sublayer names
            block_idx, _ = parse_layer_name(layers_at_y[0])
            sublayers = [parse_layer_name(layer)[1] for layer in layers_at_y]
            # Order: q, v, k
            ordered = []
            for sub in ["attn.q_proj", "attn.v_proj", "attn.k_proj"]:
                if sub in sublayers:
                    ordered.append(sub.split(".")[-1])
            label = f"h.{block_idx}\n" + "/".join(ordered)
        elif layers_at_y[0] == "output" and output_prob_threshold is not None:
            label = f"output\n(prob>{output_prob_threshold})"
        else:
            label = layers_at_y[0].replace(".", "\n", 1)
        layer_labels.append(label)

    ax.set_yticks(layer_centers)
    ax.set_yticklabels(layer_labels, fontsize=9)

    # Add horizontal lines to separate layers
    for y in layer_y.values():
        ax.axhline(y=y - 0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)

    # Grid
    ax.grid(False)

    # Title
    ax.set_title("Local Attribution Graph", fontsize=14, fontweight="bold", pad=10)

    # Legend for layer colors
    layer_order = get_layer_order()
    legend_elements = []
    for sublayer in reversed(layer_order):
        color = get_layer_color(sublayer)
        legend_elements.append(
            plt.scatter([], [], c=color, s=50, label=sublayer, edgecolors="white")
        )
    # Add ghost node to legend if there are any
    if ghost_nodes:
        legend_elements.append(
            plt.scatter(
                [], [], c="#909090", s=50, label="ghost (orig. alive)", edgecolors="white"
            )
        )
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=8,
        framealpha=0.9,
    )

    plt.tight_layout()
    return fig


def load_and_plot(
    pt_path: Path,
    output_path: Path | None = None,
    **plot_kwargs: Any,
) -> plt.Figure:
    """Load attributions from .pt file and create plot.

    Args:
        pt_path: Path to the saved .pt file.
        output_path: Optional path to save the figure.
        **plot_kwargs: Additional kwargs passed to plot_local_attribution_graph.

    Returns:
        Matplotlib figure.
    """
    data = torch.load(pt_path, weights_only=False)
    attr_pairs: list[PairAttribution] = data["attr_pairs"]
    token_strings: list[str] = data["token_strings"]
    output_token_labels: dict[int, str] | None = data.get("output_token_labels")
    output_prob_threshold: float | None = data.get("output_prob_threshold")
    output_probs_by_pos: dict[tuple[int, int], float] | None = data.get("output_probs_by_pos")

    print(f"Loaded attributions from {pt_path}")
    print(f"  Prompt: {data.get('prompt', 'N/A')!r}")
    print(f"  Tokens: {token_strings}")
    print(f"  Number of layer pairs: {len(attr_pairs)}")

    fig = plot_local_graph(
        attr_pairs=attr_pairs,
        token_strings=token_strings,
        output_token_labels=output_token_labels,
        output_prob_threshold=output_prob_threshold,
        output_probs_by_pos=output_probs_by_pos,
        **plot_kwargs,
    )

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved figure to {output_path}")

    return fig


# %%
if __name__ == "__main__":
    # Configuration
    # wandb_id = "33n6xjjt"  # ss_gpt2_simple-1L (new)
    # wandb_id = "c0k3z78g"  # ss_gpt2_simple-2L
    wandb_id = "jyo9duz5"  # ss_gpt2_simple-1.25M (4L)

    out_dir = get_out_dir()
    pt_path = out_dir / f"local_attributions_{wandb_id}.pt"

    if not pt_path.exists():
        raise FileNotFoundError(
            f"Local attributions file not found: {pt_path}\n"
            "Run calc_local_attributions.py first to generate the data."
        )

    output_path = out_dir / f"local_attribution_graph_{wandb_id}.png"

    fig = load_and_plot(
        pt_path=pt_path,
        output_path=output_path,
        min_edge_weight=0.0001,
        node_scale=30.0,
        edge_alpha_scale=0.7,
    )
