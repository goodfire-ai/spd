from collections.abc import Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from torch import Tensor


def calc_jaccard_index(
    co_occurrence_matrix: Float[Tensor, "n n"],
    marginal_counts: Float[Tensor, " n"],
) -> Float[Tensor, "n n"]:
    """Calculate the Jaccard index

     for each component based on co-occurrence matrix and marginal counts
    Jaccard index = |A ∩ B| / |A ∪ B|
    """
    union: Float[Tensor, "n n"] = (
        marginal_counts.unsqueeze(0) + marginal_counts.unsqueeze(1) - co_occurrence_matrix
    )
    jaccard_index: Float[Tensor, "n n"] = co_occurrence_matrix / union
    jaccard_index[union == 0] = 0.0  # Handle division by zero
    return jaccard_index


CoactivationResultsGroup = dict[
    Literal[
        "co_occurrence_matrix",
        "marginal_counts",
        "module_slices",
        "modules",
        "labels",
        "total_samples",
        "activation_threshold",
        "jaccard",
        "component_masks",
        "active_mask",
        "active_freq",
        "is_alive",
    ],
    Any,
]

CoactivationResults = dict[
    str,  # group key
    CoactivationResultsGroup,
]


def hierarchical_clustering(
    similarity_matrix: Float[Tensor, "n n"],
    labels: Sequence[str] | None = None,
    threshold: float = 0.5,
    criterion: Literal["distance", "maxclust"] = "distance",
    linkage_method: Literal["single", "complete", "average", "ward"] = "average",
    figsize: tuple[int, int] = (12, 6),
    cmap: str = "tab20",
    title: str | None = None,
    plot: bool = True,
) -> dict[str, Any]:
    """
    Create a hierarchical clustering dendrogram with optional ground truth labels.

    Parameters
    ----------
    similarity_matrix : array-like
        Square similarity matrix (e.g., Jaccard similarity)
    labels : array-like, optional
        Ground truth labels for each element. If provided, shows color bar
    threshold : float, default=0.5
        Distance threshold for clustering (1 - similarity)
    linkage_method : str, default='average'
        Linkage method: 'single', 'complete', 'average', 'ward'
    figsize : tuple, default=(12, 6)
        Figure size (width, height)
    cmap : str, default='tab20'
        Colormap for label categories
    title : str, optional
        Plot title

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    clusters : array
        Cluster assignments for each element
    Z : array
        The hierarchical clustering linkage matrix
    formerly:
    tuple[
        plt.Figure,
        Int[np.ndarray, " n"],  # Cluster assignments for each element
        Float[np.ndarray, "n n"],  # Hierarchical clustering linkage matrix
    ]:
    """
    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix
    condensed_dist = squareform(distance_matrix)

    # Perform hierarchical clustering
    Z_linkage = linkage(condensed_dist, method=linkage_method)

    # Get clusters
    clusters = fcluster(Z_linkage, t=threshold, criterion=criterion)

    output: dict[str, Any] = dict(
        distance_matrix=distance_matrix,
        condensed_dist=condensed_dist,
        Z_linkage=Z_linkage,
        clusters=clusters,
    )

    if plot:
        # Create figure
        if labels is not None:
            ax1: plt.Axes
            ax2: plt.Axes
            fig, (ax1, ax2) = plt.subplots(  # type: ignore
                2, 1, figsize=figsize, gridspec_kw={"height_ratios": [20, 1]}
            )
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)

        # Plot dendrogram
        dend = dendrogram(
            Z_linkage,
            ax=ax1,
            color_threshold=(threshold if criterion == "distance" else None),
            no_labels=True,
        )
        ax1.axhline(
            y=threshold if criterion == "distance" else 0,
            color="r",
            linestyle="--",
            label=f"{criterion}={threshold}",
        )
        ax1.set_ylabel("Distance (1 - Jaccard Similarity)")
        ax1.legend()

        if title:
            ax1.set_title(title)
        else:
            ax1.set_title(f"Hierarchical Clustering ({linkage_method} linkage)")

        # Add color bar if labels provided
        if labels is not None:
            # Get leaf order and create color mapping
            leaves_order: Sequence[int] = dend["leaves"]
            ordered_labels: list[str] = [labels[i] for i in leaves_order]
            unique_labels: list[str] = list(np.unique(labels))
            label_to_idx: dict[str, int] = {label: i for i, label in enumerate(unique_labels)}
            color_indices: list[int] = [label_to_idx[label] for label in ordered_labels]

            # Plot color bar
            ax2.imshow([color_indices], aspect="auto", cmap=cmap)
            ax2.set_xlabel("Subcomponent Module")
            ax2.set_xticks([])
            ax2.set_yticks([])

            # Create legend
            n_colors: int = len(unique_labels)
            if n_colors <= 20:
                colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, n_colors))
            else:
                colors = plt.cm.get_cmap("hsv")(np.linspace(0, 0.9, n_colors))

            patches: list[Patch] = [
                Patch(color=colors[i], label=str(label)) for i, label in enumerate(unique_labels)
            ]

            # Position legend
            ncol: int = min(5, n_colors)  # Limit number of columns in legend
            ax2.legend(handles=patches, loc="center", ncol=ncol, bbox_to_anchor=(0.5, -2))

        plt.tight_layout()

        output.update(
            dict(
                fig=fig,
                labels=labels,
            )
        )

    return output


def coactivation_hierarchical_clustering(
    results: CoactivationResultsGroup,
    threshold: float = 0.9,
    linkage_method: Literal["single", "complete", "average", "ward"] = "average",
    min_alive_counts: int = 0,
    plot: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """
    Convenience function to plot directly from results dictionary.

    Parameters
    ----------
    results : dict
        Results dictionary containing 'jaccard', 'marginal_counts', and 'labels'
    group_key : str, default='group_2'
        Which group to analyze
    threshold : float, default=0.9
        Distance threshold for clustering
    linkage_method : str, default='average'
        Linkage method
    **kwargs : additional arguments passed to plot_hierarchical_clustering

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    clusters : array
        Cluster assignments for alive elements
    Z : array
        The hierarchical clustering linkage matrix
    alive_mask : array
        Boolean mask of alive elements
    """
    # Extract data
    alive_mask = (results["marginal_counts"] > min_alive_counts).cpu().numpy()
    ground_truth = results["labels"][alive_mask]

    # Mask similarity matrix
    masked_similarity = results["jaccard"].cpu().numpy()[alive_mask][:, alive_mask]

    hclust_out = hierarchical_clustering(
        masked_similarity,
        labels=ground_truth,
        threshold=threshold,
        linkage_method=linkage_method,
        plot=plot,
        **kwargs,
    )
    clusters = hclust_out["clusters"]

    # type list[int] with -1 for inactive components
    clusters_nomask: list[int] = list()
    idx_mask: int = 0
    for alive in alive_mask:
        if alive:
            clusters_nomask.append(clusters[idx_mask])
            idx_mask += 1
        else:
            clusters_nomask.append(-1)

    return dict(
        **hclust_out,
        alive_mask=alive_mask,
        n_clusters=len(np.unique(clusters)),
        cluster_sizes=np.bincount(clusters)[1:],
        n_active=alive_mask.sum().item(),
        clusters_nomask=np.array(clusters_nomask),
    )
