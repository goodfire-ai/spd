"""Top-level dashboard data structure."""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

from spd.clustering.dashboard.core.base import (
    ActivationSampleBatch,
    ActivationSampleHash,
    ClusterIdHash,
    TextSample,
    TextSampleHash,
)
from spd.clustering.dashboard.core.cluster_data import ClusterData


@dataclass(frozen=True, slots=True, kw_only=True)
class DashboardData:
    """All data for the dashboard."""

    clusters: dict[ClusterIdHash, ClusterData]
    text_samples: dict[TextSampleHash, TextSample]
    activations_map: dict[ActivationSampleHash, int]
    activations: ActivationSampleBatch

    # activations_map maps ActivationSampleHash to index in `activations`

    def save(self, output_dir: str) -> None:
        """Save dashboard data to directory structure for efficient frontend access.

        Structure:
        - clusters.json - All cluster data
        - text_samples.json - All text samples by hash
        - activations.npz - Numpy array with all activations (float16, compressed)
        - activations_map.json - Maps activation hashes to indices in activations array
        """
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
