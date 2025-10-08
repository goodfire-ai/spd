"""Global activations accumulator for final DashboardData."""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

from spd.clustering.dashboard.core.base import (
    ActivationSampleBatch,
    ActivationSampleHash,
    ClusterIdHash,
    TextSampleHash,
)
from spd.clustering.dashboard.core.cluster_data import ClusterData


@dataclass(kw_only=True)
class GlobalActivationsAccumulator:
    """Accumulates activations across all clusters for final DashboardData.

    Attributes:
        activations_map: Maps activation hashes to indices in activations_list
        activations_list: List of all activation arrays
        text_hashes_list: List of text hashes corresponding to activations
        current_idx: Current index for next activation
    """

    activations_map: dict[ActivationSampleHash, int]
    activations_list: list[Float[np.ndarray, " n_ctx"]]
    text_hashes_list: list[TextSampleHash]
    current_idx: int

    @classmethod
    def create(cls) -> "GlobalActivationsAccumulator":
        """Create empty accumulator."""
        return cls(
            activations_map={},
            activations_list=[],
            text_hashes_list=[],
            current_idx=0,
        )

    def add_cluster_data(
        self,
        cluster_data: ClusterData,
        activation_batch: ActivationSampleBatch,
        component_activations_storage: dict[ClusterIdHash, dict[str, list[Float[np.ndarray, " n_ctx"]]]],
        component_text_hashes_storage: dict[ClusterIdHash, dict[str, list[TextSampleHash]]],
    ) -> None:
        """Add cluster and component activations to global storage.

        Args:
            cluster_data: ClusterData with component-level data
            activation_batch: Activation batch for cluster-level data
            component_activations_storage: Storage containing component activations
            component_text_hashes_storage: Storage containing component text hashes
        """
        cluster_hash: ClusterIdHash = cluster_data.cluster_hash

        # Add cluster-level activations
        act_hashes: list[ActivationSampleHash] = activation_batch.activation_hashes
        for i, (text_hash, acts) in enumerate(
            zip(activation_batch.text_hashes, activation_batch.activations, strict=True)
        ):
            self.activations_map[act_hashes[i]] = self.current_idx
            self.activations_list.append(acts)
            self.text_hashes_list.append(text_hash)
            self.current_idx += 1

        # Add component-level activations
        if cluster_data.component_activations:
            for comp_label in cluster_data.component_activations:
                comp_acts_list = component_activations_storage[cluster_hash][comp_label]
                comp_text_hashes = component_text_hashes_storage[cluster_hash][comp_label]

                if not comp_acts_list:
                    continue

                comp_acts_array: Float[np.ndarray, "n_samples n_ctx"] = np.stack(comp_acts_list)

                for text_hash, comp_acts in zip(comp_text_hashes, comp_acts_array, strict=True):
                    comp_act_hash = ActivationSampleHash(f"{cluster_hash}:{comp_label}:{text_hash}")
                    self.activations_map[comp_act_hash] = self.current_idx
                    self.activations_list.append(comp_acts)
                    self.text_hashes_list.append(text_hash)
                    self.current_idx += 1
