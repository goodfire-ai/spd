"""Data saving functionality for dashboard core."""

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zanj import ZANJ

from spd.dashboard.core.acts import Activations
from spd.dashboard.core.compute import ComponentEmbeddings
from spd.dashboard.core.summary import IndexSummaries
from spd.dashboard.core.trees import DecisionTreesData


@dataclass
class DashboardData:
    """Container for all dashboard data with serialization and saving logic."""

    metadata: dict[str, Any]
    activations: Activations
    index_summaries: IndexSummaries
    embeddings: ComponentEmbeddings
    trees: DecisionTreesData | None = None

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
