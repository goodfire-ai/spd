"""Data saving functionality for dashboard core."""

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Float, Int
from muutils.json_serialize import SerializableDataclass, serializable_field
from zanj import ZANJ

from spd.dashboard.core.acts import Activations, ComponentLabel
from spd.dashboard.core.compute import (
    ComponentEmbeddings,
    FlatActivations,
    _compute_activated_per_token,
    _compute_P_active_given_token,
    _compute_P_token_given_active,
)
from spd.dashboard.core.dashboard_config import ComponentDashboardConfig
from spd.dashboard.core.summary import SubcomponentSummary
from spd.dashboard.core.trees import DecisionTreesData


@dataclass
class IndexSummaries(SerializableDataclass):
    """Lightweight summary for index.html display.

    Contains only the data needed by the main page,
    excluding large fields like full activations.
    """

    summaries: list[SubcomponentSummary] = serializable_field(
        serialization_fn=lambda x: [s.serialize() for s in x],
        deserialize_fn=lambda x: [SubcomponentSummary.load(s) for s in x],
    )

    @classmethod
    def from_activations(
        cls, activations: Activations, config: ComponentDashboardConfig
    ) -> "IndexSummaries":
        # Flatten activations
        acts_flat: FlatActivations = FlatActivations.create(activations)
        assert acts_flat.n_components

        # Compute embeddings
        embeddings: ComponentEmbeddings = ComponentEmbeddings.create(
            flat_acts=acts_flat, embed_dim=config.embed_dim
        )

        # Compute conditional probabilities for ALL components at once
        token_idx: Int[np.ndarray, " n_samples"]
        activated_per_token: Float[np.ndarray, "d_vocab C"]
        token_idx, activated_per_token = _compute_activated_per_token(
            acts_flat, activation_threshold=config.activation_threshold
        )
        p_active_given_token: Float[np.ndarray, "d_vocab C"] = _compute_P_active_given_token(
            activated_per_token, token_idx, acts_flat.token_data.vocab_arr.shape[0]
        )
        p_token_given_active: Float[np.ndarray, "d_vocab C"] = _compute_P_token_given_active(
            activated_per_token, acts_flat.activations > config.activation_threshold
        )

        # Create summary for each component
        summaries: list[SubcomponentSummary] = []
        i: int
        label: ComponentLabel
        for i, label in enumerate(acts_flat.component_labels):
            summary: SubcomponentSummary = SubcomponentSummary.create(
                label=label,
                tokens=acts_flat.token_data,
                activations=acts_flat.activations[:, i],
                embeds=embeddings.embeddings[i],
                config=config,
                p_active_given_token=p_active_given_token[:, i],
                p_token_given_active=p_token_given_active[:, i],
                activated_per_token=activated_per_token[:, i],
            )
            summaries.append(summary)

        return cls(summaries=summaries)


@dataclass
class DashboardData:
    """Container for all dashboard data with serialization and saving logic."""

    config: ComponentDashboardConfig
    metadata: dict[str, Any]
    activations: Activations
    trees: DecisionTreesData

    def serialize(self) -> dict[str, Any]:
        """Serialize all dashboard data by calling serialize() on components.

        Returns dict ready for ZANJ storage with proper structure for dashboard consumption.
        """
        data: dict[str, Any] = {
            "metadata": self.metadata,
            "activations": self.activations.serialize(),
            "index_summaries": self.index_summaries.serialize(),
            "embeddings": self.embeddings.serialize(),
        }

        if self.trees is not None:
            data["trees"] = self.trees.serialize()

        return data

    def save(self, output_path: Path, extract: bool = True) -> None:
        """Save dashboard data to ZANJ file and optionally extract.

        Args:
            output_path: Path to save the ZANJ file (e.g., 'dashboard.zanj')
            extract: Whether to extract the ZANJ file to a directory (default: True)
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize data
        serialized_data: dict[str, Any] = self.serialize()

        # Save to ZANJ with threshold for externalization
        ZANJ(external_list_threshold=64).save(serialized_data, str(output_path))

        print(f"Saved dashboard data to '{output_path}'")

        # Extract if requested
        if extract:
            extract_dir: Path = output_path.parent / f"{output_path.stem}_extracted"
            extract_dir.mkdir(exist_ok=True)

            with zipfile.ZipFile(output_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            print(f"Extracted dashboard data to '{extract_dir}'")
