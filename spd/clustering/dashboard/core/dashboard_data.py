"""Top-level dashboard data structure."""

from typing import Any

import numpy as np
from jaxtyping import Float
from muutils.json_serialize import SerializableDataclass, serializable_dataclass, serializable_field

from spd.clustering.consts import SubComponentKey
from spd.clustering.dashboard.core.base import (
    ActivationSampleBatch,
    ClusterId,
    ClusterIdHash,
    TextSample,
    TextSampleHash,
)
from spd.clustering.dashboard.core.cluster_data import ClusterData, TrackingCriterion


@serializable_dataclass(kw_only=True)
class DashboardData(SerializableDataclass):
    """All data for the dashboard.

    Self-contained data structure that ZANJ can save/load automatically.
    Clusters contain their own activation data (no external references needed).
    """

    model_info: dict[str, Any] = serializable_field(default_factory=dict)
    clusters: dict[ClusterIdHash, ClusterData] = serializable_field(default_factory=dict)
    text_samples: dict[TextSampleHash, TextSample] = serializable_field(default_factory=dict)

    # Optional global metrics
    coactivations: Float[np.ndarray, "n_clusters n_clusters"] | None = serializable_field(
        default=None
    )
    cluster_indices: list[int] | None = serializable_field(default=None)

    @classmethod
    def create(cls, text_samples: dict[TextSampleHash, TextSample]) -> "DashboardData":
        """Initialize empty dashboard data.

        Args:
            text_samples: Text samples dict to share across clusters

        Returns:
            Empty DashboardData ready for incremental population
        """
        return cls(text_samples=text_samples)

    def add_cluster(
        self,
        cluster_id: ClusterId,
        cluster_components: list[dict[str, Any]],
        criteria: list[TrackingCriterion],
        cluster_activations: list[Float[np.ndarray, " n_ctx"]],
        cluster_text_hashes: list[TextSampleHash],
        cluster_tokens: list[list[str]],
    ) -> None:
        """Build and add a cluster to the dashboard.

        Args:
            cluster_id: ClusterId object
            cluster_components: Component info for this cluster
            criteria: Tracking criteria for top-k samples
            cluster_activations: List of activation arrays for this cluster
            cluster_text_hashes: List of text hashes for cluster activations
            cluster_tokens: List of token strings for cluster activations
        """
        cluster_hash: ClusterIdHash = cluster_id.to_string()

        # Skip if cluster has no activations
        if not cluster_activations:
            return

        # Stack cluster-level activations into batch
        acts_array: Float[np.ndarray, "batch n_ctx"] = np.stack(cluster_activations)

        activation_batch: ActivationSampleBatch = ActivationSampleBatch(
            cluster_id=cluster_id,
            text_hashes=cluster_text_hashes,
            activations=acts_array,
            tokens=cluster_tokens,
        )

        # Convert component info to ComponentInfo objects
        components_info: list[SubComponentKey] = [
            SubComponentKey(module=comp["module"], index=comp["index"])
            for comp in cluster_components
        ]

        # Generate ClusterData with stats and top-k samples (now self-contained!)
        cluster_data: ClusterData = ClusterData.generate(
            cluster_id=cluster_id,
            activation_samples=activation_batch,
            criteria=criteria,
            components=components_info,
        )

        # Store cluster (activations are now embedded in cluster_data.samples!)
        self.clusters[cluster_hash] = cluster_data

    # DEPRECATED: All methods below can be removed - ZANJ handles serialization automatically
    # The .serialize() method is provided by SerializableDataclass
