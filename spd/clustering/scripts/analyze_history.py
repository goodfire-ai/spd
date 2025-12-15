"""Analyze a clustering run from a history.zip file.

Usage:
    python -m spd.clustering.scripts.analyze_history /path/to/history.zip
    python -m spd.clustering.scripts.analyze_history /path/to/history.zip --top-clusters 5
    python -m spd.clustering.scripts.analyze_history /path/to/history.zip --plot
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from spd.clustering.merge_history import MergeHistory


def parse_label(label: str) -> tuple[str, int]:
    """Parse a component label into module and index."""
    module, idx_str = label.rsplit(":", 1)
    return module, int(idx_str)


def analyze_history(
    history_path: Path,
    top_clusters: int = 5,
    show_plots: bool = False,
    max_components_to_show: int = 20,
) -> None:
    """Analyze a clustering run from a history.zip file.

    Args:
        history_path: Path to history.zip file
        top_clusters: Number of largest clusters to analyze
        show_plots: Whether to display matplotlib plots
        max_components_to_show: Max components to print per cluster
    """
    history = MergeHistory.read(history_path)

    # Basic summary
    print("=" * 60)
    print(f"CLUSTERING ANALYSIS: {history_path.name}")
    print("=" * 60)
    print(f"Source: {history_path}")
    print()

    print("=== Basic Statistics ===")
    print(f"Initial components: {history.c_components}")
    print(f"Iterations performed: {history.n_iters_current}")
    print(f"Initial clusters (iter 0): {history.initial_k_groups}")
    print(f"Final clusters (iter {history.n_iters_current - 1}): {history.final_k_groups}")
    print(f"Clusters merged: {history.initial_k_groups - history.final_k_groups}")
    print()

    # Module breakdown
    module_components: dict[str, list[int]] = defaultdict(list)
    for label in history.labels:
        module, idx = parse_label(label)
        module_components[module].append(idx)

    print("=== Module Breakdown ===")
    print(f"Total modules: {len(module_components)}")
    for module in sorted(module_components.keys()):
        components = module_components[module]
        print(f"  {module}: {len(components)} components (indices {min(components)}-{max(components)})")
    print()

    # Final iteration cluster analysis
    final_iter = history.n_iters_current - 1
    final_merge = history.merges[final_iter]
    cluster_sizes = final_merge.components_per_group

    print("=== Cluster Size Distribution (Final Iteration) ===")
    print(f"Number of clusters: {len(cluster_sizes)}")
    print(f"Total components assigned: {cluster_sizes.sum().item()}")
    print(f"Min cluster size: {cluster_sizes.min().item()}")
    print(f"Max cluster size: {cluster_sizes.max().item()}")
    print(f"Mean cluster size: {cluster_sizes.float().mean().item():.2f}")
    print(f"Median cluster size: {cluster_sizes.float().median().item():.1f}")
    print()

    # Size distribution
    size_counts: dict[int, int] = {}
    for size in cluster_sizes.tolist():
        size_counts[size] = size_counts.get(size, 0) + 1

    print("=== Size Counts ===")
    for size in sorted(size_counts.keys()):
        count = size_counts[size]
        pct = 100 * count / len(cluster_sizes)
        print(f"  Size {size}: {count} clusters ({pct:.1f}%)")
    print()

    # Analyze top N largest clusters
    unique_cluster_ids = final_merge.group_idxs.unique()

    # Get actual cluster IDs sorted by size
    cluster_id_to_size = {}
    for cid in unique_cluster_ids.tolist():
        size = (final_merge.group_idxs == cid).sum().item()
        cluster_id_to_size[cid] = size

    sorted_cluster_ids = sorted(cluster_id_to_size.keys(), key=lambda x: cluster_id_to_size[x], reverse=True)

    n_to_show = min(top_clusters, len(sorted_cluster_ids))
    print(f"=== Top {n_to_show} Largest Clusters ===")

    for rank, cluster_id in enumerate(sorted_cluster_ids[:n_to_show], 1):
        cluster_size = cluster_id_to_size[cluster_id]

        # Skip singleton clusters in detailed analysis
        if cluster_size == 1:
            print(f"\n--- Rank {rank}: Cluster ID {cluster_id} (size: {cluster_size}) ---")
            print("  [Singleton cluster]")
            continue

        print(f"\n--- Rank {rank}: Cluster ID {cluster_id} (size: {cluster_size}) ---")

        # Get component labels
        cluster_labels = history.get_cluster_component_labels(iteration=-1, cluster_id=cluster_id)

        # Module composition
        cluster_module_counts: dict[str, int] = defaultdict(int)
        for label in cluster_labels:
            module, _ = parse_label(label)
            cluster_module_counts[module] += 1

        print("  Module composition:")
        for module in sorted(cluster_module_counts.keys()):
            count = cluster_module_counts[module]
            total_in_module = len(module_components[module])
            pct = 100 * count / total_in_module
            print(f"    {module}: {count}/{total_in_module} ({pct:.1f}%)")

        # Show component labels
        if len(cluster_labels) <= max_components_to_show:
            print(f"  Components ({len(cluster_labels)} total):")
            for label in cluster_labels:
                print(f"    {label}")
        else:
            print(f"  Components (showing {max_components_to_show}/{len(cluster_labels)}):")
            for label in cluster_labels[:max_components_to_show]:
                print(f"    {label}")
            print(f"    ... and {len(cluster_labels) - max_components_to_show} more")

    print()

    # Layer-by-layer cluster membership summary
    print("=== Layer-by-Layer Cluster Membership ===")

    # Build label -> cluster mapping
    label_to_cluster = {}
    group_idxs = final_merge.group_idxs.numpy()
    for comp_idx, label in enumerate(history.labels):
        label_to_cluster[label] = group_idxs[comp_idx]

    for module in sorted(module_components.keys()):
        component_indices = module_components[module]

        # Get cluster assignments for this module's components
        cluster_ids = []
        for idx in component_indices:
            label = f"{module}:{idx}"
            if label in label_to_cluster:
                cluster_ids.append(label_to_cluster[label])

        # Count per cluster
        cluster_counts: dict[int, int] = defaultdict(int)
        for cid in cluster_ids:
            cluster_counts[cid] += 1

        singletons = sum(1 for cnt in cluster_counts.values() if cnt == 1)
        in_multi = len(cluster_ids) - singletons

        print(f"\n{module}:")
        print(f"  Components: {len(cluster_ids)} ({singletons} singletons, {in_multi} in multi-component clusters)")

        # Only show significant clusters (top N largest)
        for cid in sorted_cluster_ids[:top_clusters]:
            count_in_cluster = cluster_counts.get(cid, 0)
            if count_in_cluster > 0:
                print(f"  → In cluster {cid} (size {cluster_id_to_size[cid]}): {count_in_cluster} components")

    print()

    # First few merge pairs
    print("=== First 10 Merge Pairs ===")
    for i in range(min(10, history.n_iters_current)):
        pair = history.selected_pairs[i]
        comp_a, comp_b = pair[0], pair[1]
        label_a = history.labels[comp_a] if comp_a < len(history.labels) else f"?{comp_a}"
        label_b = history.labels[comp_b] if comp_b < len(history.labels) else f"?{comp_b}"
        print(f"  Iter {i:3d}: {label_a} + {label_b}")
    print()

    # Plots
    if show_plots:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available, skipping plots")
            return

        _, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Cluster count over iterations
        k_groups_over_time = history.merges.k_groups.numpy()[:history.n_iters_current]
        axes[0, 0].plot(range(len(k_groups_over_time)), k_groups_over_time, 'b-', linewidth=2)
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Number of Clusters")
        axes[0, 0].set_title("Cluster Count Over Merge Iterations")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Cluster size distribution (histogram)
        axes[0, 1].hist(cluster_sizes.numpy(), bins=50, color='steelblue', edgecolor='black')
        axes[0, 1].set_xlabel("Cluster Size")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title("Cluster Size Distribution")
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Module composition (all components)
        total_counts = [len(module_components[m]) for m in sorted(module_components.keys())]
        short_modules = [m.replace("h.0.", "") for m in sorted(module_components.keys())]

        axes[1, 0].barh(short_modules, total_counts, color='steelblue')
        axes[1, 0].set_xlabel("Number of Components")
        axes[1, 0].set_title("Components per Module")
        axes[1, 0].invert_yaxis()

        # Plot 4: Track largest cluster growth
        largest_cluster_size = cluster_id_to_size[sorted_cluster_ids[0]] if sorted_cluster_ids else 0
        if largest_cluster_size > 1:
            # Get the largest cluster
            largest_cluster_id = sorted_cluster_ids[0]
            largest_cluster_components = set(final_merge.components_in_group(largest_cluster_id))

            # Track how the cluster containing these components grew
            cluster_size_over_time = []
            for iter_idx in range(history.n_iters_current):
                merge = history.merges[iter_idx]
                iter_group_idxs = merge.group_idxs.numpy()

                # Find clusters containing our target components
                cluster_ids_for_target = set(iter_group_idxs[list(largest_cluster_components)])

                # Get max cluster size among those containing target components
                max_size = 0
                for cid in cluster_ids_for_target:
                    size = (iter_group_idxs == cid).sum()
                    max_size = max(max_size, size)

                cluster_size_over_time.append(max_size)

            axes[1, 1].plot(range(len(cluster_size_over_time)), cluster_size_over_time, 'r-', linewidth=2)
            axes[1, 1].set_xlabel("Iteration")
            axes[1, 1].set_ylabel("Cluster Size")
            axes[1, 1].set_title(f"Growth of Largest Cluster (ID {largest_cluster_id})")
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, "No multi-component clusters", ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("Largest Cluster Growth")

        plt.tight_layout()
        plt.suptitle(f"Clustering Analysis: {history_path.name}", y=1.02)
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a clustering run from a history.zip file"
    )
    parser.add_argument(
        "history_path",
        type=Path,
        help="Path to history.zip file",
    )
    parser.add_argument(
        "--top-clusters",
        type=int,
        default=5,
        help="Number of largest clusters to analyze (default: 5)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show matplotlib plots",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=20,
        help="Max components to show per cluster (default: 20)",
    )

    args = parser.parse_args()

    if not args.history_path.exists():
        raise FileNotFoundError(f"History file not found: {args.history_path}")

    analyze_history(
        history_path=args.history_path,
        top_clusters=args.top_clusters,
        show_plots=args.plot,
        max_components_to_show=args.max_components,
    )


if __name__ == "__main__":
    main()
