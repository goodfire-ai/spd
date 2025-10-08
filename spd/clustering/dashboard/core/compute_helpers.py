"""Helper functions for computing cluster and component activations."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from spd.clustering.activations import ProcessedActivations

if TYPE_CHECKING:
    from spd.models.component_model import ComponentModel


@dataclass(slots=True, kw_only=True)
class ClusterActivations:
    """Vectorized cluster activations for all clusters.

    Attributes:
        activations: Tensor of shape [n_samples, n_clusters] containing cluster activations
        cluster_indices: List mapping column index to cluster index
    """

    activations: Float[Tensor, "n_samples n_clusters"]
    cluster_indices: list[int]


def compute_all_cluster_activations(
    processed: ProcessedActivations,
    cluster_components: dict[int, list[dict[str, Any]]],
    batch_size: int,
    seq_len: int,
) -> ClusterActivations:
    """Compute activations for all clusters in a vectorized manner.

    Args:
        processed: ProcessedActivations containing all component activations
        cluster_components: Dict mapping cluster_idx -> list of component info dicts
        batch_size: Batch size
        seq_len: Sequence length

    Returns:
        ClusterActivations with shape [batch_size * seq_len, n_clusters]
    """
    cluster_indices: list[int] = sorted(cluster_components.keys())
    n_clusters: int = len(cluster_indices)
    n_samples: int = batch_size * seq_len
    device: torch.device = processed.activations.device

    # Build cluster activation tensor: [n_samples, n_clusters]
    cluster_acts: Float[Tensor, "n_samples n_clusters"] = torch.zeros(
        (n_samples, n_clusters), device=device
    )

    # For each cluster, compute max activation across its components
    for cluster_col_idx, cluster_idx in enumerate(cluster_indices):
        components: list[dict[str, Any]] = cluster_components[cluster_idx]

        # Get component indices for this cluster
        comp_indices: list[int] = []
        for component_info in components:
            label: str = component_info["label"]
            comp_idx: int | None = processed.get_label_index(label)
            if comp_idx is not None:
                comp_indices.append(comp_idx)

        if not comp_indices:
            continue

        # Max activation across components for this cluster
        # processed.activations: [n_samples, n_components]
        cluster_acts[:, cluster_col_idx] = processed.activations[:, comp_indices].max(dim=1).values

    return ClusterActivations(activations=cluster_acts, cluster_indices=cluster_indices)


def compute_cluster_coactivations(
    cluster_activations_list: list[ClusterActivations],
) -> tuple[Float[np.ndarray, "n_clusters n_clusters"], list[int]]:
    """Compute coactivation matrix for clusters across all batches.

    Args:
        cluster_activations_list: List of ClusterActivations from each batch

    Returns:
        Tuple of (coactivation_matrix, cluster_indices) where coact[i,j] is the number
        of samples where both cluster i and j activate
    """
    if not cluster_activations_list:
        return np.array([[]], dtype=np.float32), []

    # All batches should have same cluster indices
    cluster_indices: list[int] = cluster_activations_list[0].cluster_indices
    _n_clusters: int = len(cluster_indices)

    # Concatenate all batch activations: [total_samples, n_clusters]
    all_acts: Float[Tensor, "total_samples n_clusters"] = torch.cat(
        [ca.activations for ca in cluster_activations_list], dim=0
    )

    # Binarize activations (1 if cluster activates, 0 otherwise)
    activation_mask: Float[Tensor, "total_samples n_clusters"] = (all_acts > 0).float()

    # Compute coactivation matrix: coact[i,j] = sum over samples of (cluster_i_active * cluster_j_active)
    # Following spd/clustering/merge.py:69
    coact: Float[Tensor, "n_clusters n_clusters"] = activation_mask.T @ activation_mask

    return coact.cpu().numpy(), cluster_indices


def compute_component_coactivations_in_cluster(
    processed: ProcessedActivations,
    component_labels: list[str],
    activation_threshold: float = 0.0,
) -> Float[np.ndarray, "n_comps n_comps"]:
    """Compute coactivation matrix for components within a cluster.

    Args:
        processed: ProcessedActivations containing all component activations
        component_labels: List of component labels in this cluster
        activation_threshold: Threshold for considering a component "active"

    Returns:
        Coactivation matrix where coact[i,j] is the number of samples where
        both component i and j activate above threshold
    """
    n_components: int = len(component_labels)

    # Get indices for these components
    comp_indices: list[int] = []
    for label in component_labels:
        comp_idx: int | None = processed.get_label_index(label)
        if comp_idx is not None:
            comp_indices.append(comp_idx)

    if not comp_indices:
        return np.zeros((n_components, n_components), dtype=np.float32)

    # Extract activations for these components: [n_samples, n_components]
    component_acts: Float[Tensor, "n_samples n_comps"] = processed.activations[:, comp_indices]

    # Binarize activations (1 if component activates above threshold, 0 otherwise)
    activation_mask: Float[Tensor, "n_samples n_comps"] = (
        component_acts > activation_threshold
    ).float()

    # Compute coactivation matrix: coact[i,j] = sum over samples of (comp_i_active * comp_j_active)
    coact: Float[Tensor, "n_comps n_comps"] = activation_mask.T @ activation_mask

    return coact.cpu().numpy()


def compute_component_cosine_similarities(
    model: "ComponentModel",
    component_labels: list[str],
) -> Float[np.ndarray, "2 n_comps n_comps"]:
    """Compute cosine similarity matrices for U and V vectors of components.

    NOTE: Not implemented - U and V vectors may have different dimensions across components.

    Args:
        model: ComponentModel containing the components with U and V matrices
        component_labels: List of component labels in format "module_name:component_idx"

    Returns:
        Array of shape (2, n_comps, n_comps) where:
        - [0, i, j] is cosine similarity between U vectors of components i and j
        - [1, i, j] is cosine similarity between V vectors of components i and j
    """
    raise NotImplementedError(
        "compute_component_cosine_similarities is not implemented - "
        "U and V vectors may have different dimensions across components"
    )

    # Code kept for reference:
    # n_components: int = len(component_labels)
    #
    # if n_components == 0:
    #     return np.zeros((2, 0, 0), dtype=np.float32)
    #
    # # Parse labels and extract U and V vectors
    # u_vectors: list[Tensor] = []
    # v_vectors: list[Tensor] = []
    #
    # for label in component_labels:
    #     # Parse label format: "module_name:component_idx"
    #     module_name, comp_idx_str = label.rsplit(":", 1)
    #     comp_idx: int = int(comp_idx_str)
    #
    #     # Get the Components object for this module
    #     if module_name not in model.components:
    #         raise ValueError(f"Module '{module_name}' not found in model.components")
    #
    #     components = model.components[module_name]
    #
    #     # Extract U vector: components.U[comp_idx, :] shape (u_dim,)
    #     u_vec: Tensor = components.U[comp_idx, :]
    #     u_vectors.append(u_vec)
    #
    #     # Extract V vector: components.V[:, comp_idx] shape (v_dim,)
    #     v_vec: Tensor = components.V[:, comp_idx]
    #     v_vectors.append(v_vec)
    #
    # # Stack vectors: [n_comps, u_dim] and [n_comps, v_dim]
    # u_matrix: Tensor = torch.stack(u_vectors, dim=0)  # [n_comps, u_dim]
    # v_matrix: Tensor = torch.stack(v_vectors, dim=0)  # [n_comps, v_dim]
    #
    # # Compute cosine similarities for U vectors
    # u_norms: Tensor = torch.norm(u_matrix, p=2, dim=1)  # [n_comps]
    # u_norms = torch.where(u_norms > 0, u_norms, torch.ones_like(u_norms))
    # u_normalized: Tensor = u_matrix / u_norms.unsqueeze(1)  # [n_comps, u_dim]
    # u_cosine_sim: Tensor = u_normalized @ u_normalized.T  # [n_comps, n_comps]
    #
    # # Compute cosine similarities for V vectors
    # v_norms: Tensor = torch.norm(v_matrix, p=2, dim=1)  # [n_comps]
    # v_norms = torch.where(v_norms > 0, v_norms, torch.ones_like(v_norms))
    # v_normalized: Tensor = v_matrix / v_norms.unsqueeze(1)  # [n_comps, v_dim]
    # v_cosine_sim: Tensor = v_normalized @ v_normalized.T  # [n_comps, n_comps]
    #
    # # Stack and return: [2, n_comps, n_comps]
    # result: np.ndarray = np.stack(
    #     [u_cosine_sim.detach().cpu().numpy(), v_cosine_sim.detach().cpu().numpy()], axis=0
    # )
    #
    # return result


@dataclass(frozen=True, slots=True, kw_only=True)
class ComponentMetrics:
    """Combined metrics for components within a cluster.

    Attributes:
        coactivations: Matrix where coact[i,j] is count of samples where both i and j activate

    TODO: Add cosine_similarities field for U/V vectors when dimension mismatch is resolved
    """

    coactivations: Float[np.ndarray, "n_comps n_comps"]


def compute_component_metrics_from_storage(
    component_labels: list[str],
    component_activations: dict[str, list[Float[np.ndarray, " n_ctx"]]],
) -> ComponentMetrics | None:
    """Compute coactivations from stored component activations.

    Args:
        component_labels: List of component labels in this cluster
        component_activations: Dict mapping component labels to their activation lists

    Returns:
        ComponentMetrics with coactivations, or None if insufficient data

    TODO: Add cosine similarities for U/V vectors when dimension mismatch is resolved
    """
    if not component_labels:
        return None

    n_comps: int = len(component_labels)

    # Build activation matrix for all components: [n_comps, n_total_samples]
    comp_act_matrix_list: list[Float[np.ndarray, "n_samples n_ctx"]] = []
    for comp_label in component_labels:
        if comp_label in component_activations:
            comp_acts_list = component_activations[comp_label]
            if comp_acts_list:
                comp_act_matrix_list.append(np.stack(comp_acts_list))

    if not comp_act_matrix_list or len(comp_act_matrix_list) != n_comps:
        # Return zero matrices if not all components have data
        return ComponentMetrics(
            coactivations=np.zeros((n_comps, n_comps), dtype=np.float32),
        )

    # Flatten to [n_total] per component, then stack to [n_comps, n_total]
    comp_act_flat: list[Float[np.ndarray, " n_total"]] = [
        arr.flatten() for arr in comp_act_matrix_list
    ]
    comp_act_matrix: Float[np.ndarray, "n_comps n_total"] = np.stack(comp_act_flat, axis=0)

    # Compute coactivations (binarized)
    comp_act_bin: Float[np.ndarray, "n_comps n_total"] = (comp_act_matrix > 0).astype(np.float32)
    coactivations: Float[np.ndarray, "n_comps n_comps"] = comp_act_bin @ comp_act_bin.T

    return ComponentMetrics(coactivations=coactivations)
