"""Top-level dashboard data structure."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from jaxtyping import Float
from muutils.spinner import SpinnerContext

from spd.clustering.dashboard.core.base import (
    ActivationSampleBatch,
    ActivationSampleHash,
    ClusterId,
    ClusterIdHash,
    ComponentActivationData,
    ComponentInfo,
    TextSample,
    TextSampleHash,
)
from spd.clustering.dashboard.core.cluster_data import ClusterData, TrackingCriterion


@dataclass(slots=True, kw_only=True)
class DashboardData:
    """All data for the dashboard.

    This class accumulates cluster data incrementally during computation,
    then finalizes the combined activations array at the end.
    """

    # Final data (populated incrementally)
    clusters: dict[ClusterIdHash, ClusterData] = field(default_factory=dict)
    text_samples: dict[TextSampleHash, TextSample] = field(default_factory=dict)

    # Accumulation state (mutable during build, finalized at end)
    _activations_list: list[Float[np.ndarray, " n_ctx"]] = field(default_factory=list)
    _text_hashes_list: list[TextSampleHash] = field(default_factory=list)
    _activations_map: dict[ActivationSampleHash, int] = field(default_factory=dict)
    _current_idx: int = 0

    # Final combined activations (set during finalize)
    _finalized: bool = False
    _combined_activations: ActivationSampleBatch | None = None

    # activations_map maps ActivationSampleHash to index in combined activations

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
        component_activations_storage: dict[str, list[Float[np.ndarray, " n_ctx"]]],
        component_text_hashes_storage: dict[str, list[TextSampleHash]],
    ) -> None:
        """Build and add a cluster to the dashboard.

        Args:
            cluster_id: ClusterId object
            cluster_components: Component info for this cluster
            criteria: Tracking criteria for top-k samples
            cluster_activations: List of activation arrays for this cluster
            cluster_text_hashes: List of text hashes for cluster activations
            cluster_tokens: List of token strings for cluster activations
            component_activations_storage: Component activations for this cluster
            component_text_hashes_storage: Component text hashes for this cluster
        """
        if self._finalized:
            raise RuntimeError("Cannot add clusters after finalization")

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
        components_info: list[ComponentInfo] = [
            ComponentInfo(module=comp["module"], index=comp["index"])
            for comp in cluster_components
        ]

        # Generate ClusterData with stats and top-k samples
        with SpinnerContext(message="Generating cluster data and top-k samples"):
            cluster_data: ClusterData = ClusterData.generate(
                cluster_id=cluster_id,
                activation_samples=activation_batch,
                criteria=criteria,
                components=components_info,
            )

        # Build component-level data
        with SpinnerContext(message="Building component-level statistics"):
            component_data_dict: dict[str, ComponentActivationData] = {}
            component_labels: list[str] = []

            for comp_info in components_info:
                comp_label: str = comp_info.label
                component_labels.append(comp_label)

                # Get stored component activations
                comp_acts_list: list[Float[np.ndarray, " n_ctx"]] = component_activations_storage.get(
                    comp_label, []
                )

                if not comp_acts_list:
                    continue

                # Stack activations
                comp_acts_array: Float[np.ndarray, "n_samples n_ctx"] = np.stack(comp_acts_list)

                # Compute component statistics
                comp_stats: dict[str, Any] = {
                    "mean": float(np.mean(comp_acts_array)),
                    "max": float(np.max(comp_acts_array)),
                    "min": float(np.min(comp_acts_array)),
                    "median": float(np.median(comp_acts_array)),
                    "n_samples": len(comp_acts_list),
                }

                # Create ComponentActivationData
                # Note: activation_sample_hashes and activation_indices filled below
                component_data_dict[comp_label] = ComponentActivationData(
                    component_label=comp_label,
                    activation_sample_hashes=[],  # Filled below
                    activation_indices=[],  # Filled below
                    stats=comp_stats,
                )

        # Compute component metrics
        with SpinnerContext(message="Computing component coactivations and similarities"):
            from spd.clustering.dashboard.core.compute_helpers import (
                compute_component_metrics_from_storage,
            )

            metrics = compute_component_metrics_from_storage(
                component_labels=component_labels,
                component_activations=component_activations_storage,
            )

        # Update cluster_data with component-level data
        from dataclasses import replace

        cluster_data = replace(
            cluster_data,
            component_activations=component_data_dict if component_data_dict else None,
            component_coactivations=metrics.coactivations if metrics else None,
            component_cosine_similarities=metrics.cosine_similarities if metrics else None,
        )

        # Add cluster-level activations to global storage
        act_hashes: list[ActivationSampleHash] = activation_batch.activation_hashes
        for i, (text_hash, acts) in enumerate(
            zip(activation_batch.text_hashes, activation_batch.activations, strict=True)
        ):
            self._activations_map[act_hashes[i]] = self._current_idx
            self._activations_list.append(acts)
            self._text_hashes_list.append(text_hash)
            self._current_idx += 1

        # Add component-level activations to global storage
        if cluster_data.component_activations:
            for comp_label in cluster_data.component_activations:
                comp_acts_list = component_activations_storage.get(comp_label, [])
                comp_text_hashes = component_text_hashes_storage.get(comp_label, [])

                if not comp_acts_list:
                    continue

                comp_acts_stacked: Float[np.ndarray, "n_samples n_ctx"] = np.stack(comp_acts_list)

                for text_hash, comp_acts in zip(comp_text_hashes, comp_acts_stacked, strict=True):
                    comp_act_hash = ActivationSampleHash(f"{cluster_hash}:{comp_label}:{text_hash}")
                    self._activations_map[comp_act_hash] = self._current_idx
                    self._activations_list.append(comp_acts)
                    self._text_hashes_list.append(text_hash)
                    self._current_idx += 1

        # Store cluster
        self.clusters[cluster_hash] = cluster_data

    def _finalize(self) -> None:
        """Internal method to finalize the dashboard by creating the combined activations array."""
        if self._finalized:
            return

        if not self._activations_list:
            raise ValueError("No activations collected - cannot finalize empty dashboard")

        if not self.clusters:
            raise ValueError("No clusters added - cannot finalize empty dashboard")

        # Stack all activations
        combined_activations: Float[np.ndarray, "total_samples n_ctx"] = np.stack(
            self._activations_list
        )

        # Use first cluster's ID as dummy for combined batch
        first_cluster: ClusterData = next(iter(self.clusters.values()))
        dummy_cluster_id: ClusterId = ClusterId.from_string(first_cluster.cluster_hash)

        self._combined_activations = ActivationSampleBatch(
            cluster_id=dummy_cluster_id,
            text_hashes=self._text_hashes_list,
            activations=combined_activations,
        )

        self._finalized = True

    @property
    def activations(self) -> ActivationSampleBatch:
        """Get combined activations batch. Automatically finalizes on first access."""
        self._finalize()
        assert self._combined_activations is not None
        return self._combined_activations

    @property
    def activations_map(self) -> dict[ActivationSampleHash, int]:
        """Get activations map. Automatically finalizes on first access."""
        self._finalize()
        return self._activations_map

    def save(self, output_dir: str) -> None:
        """Save dashboard data to directory structure for efficient frontend access.

        Automatically finalizes if not already finalized.

        Structure:
        - clusters.json - All cluster data
        - text_samples.json - All text samples by hash
        - activations.npz - Numpy array with all activations (float16, compressed)
        - activations_map.json - Maps activation hashes to indices in activations array
        """
        # Accessing activations_map will trigger finalization if needed
        _ = self.activations_map

        import json
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save all cluster data as JSONL (one cluster per line)
        with open(output_path / "clusters.jsonl", "w") as f:
            for cluster_data in self.clusters.values():
                f.write(json.dumps(cluster_data.serialize()) + "\n")

        # Save text samples as JSONL (one sample per line)
        with open(output_path / "text_samples.jsonl", "w") as f:
            for hash_, sample in self.text_samples.items():
                sample_obj = {
                    "text_hash": str(hash_),
                    "full_text": sample.full_text,
                    "tokens": sample.tokens,
                }
                f.write(json.dumps(sample_obj) + "\n")

        # Collect only activations that are referenced by clusters
        referenced_hashes: set[ActivationSampleHash] = set()
        for cluster_data in self.clusters.values():
            referenced_hashes.update(cluster_data.get_unique_activation_hashes())

        # Build compact activations array and map using full hashes
        compact_activations_list: list[Float[np.ndarray, " n_ctx"]] = []
        compact_map: dict[str, int] = {}

        # Use activations_map which has correct cluster IDs for each activation
        for act_hash, old_idx in self.activations_map.items():
            if act_hash in referenced_hashes:
                new_idx = len(compact_activations_list)

                # Get activation data using old index
                activation = self.activations.activations[old_idx]
                compact_activations_list.append(activation)

                # Use full hash
                compact_map[act_hash] = new_idx

        # Stack and convert to float16 for space savings
        compact_activations: Float[np.ndarray, "n_samples n_ctx"] = np.stack(
            compact_activations_list
        )
        activations_float16 = compact_activations.astype(np.float16)

        # Save as npz (compressed)
        np.savez_compressed(output_path / "activations.npz", activations=activations_float16)

        # Save compact activations map
        with open(output_path / "activations_map.json", "w") as f:
            json.dump(compact_map, f, indent=2)
