"""Dataset attributions data repository.

Owns SPD_OUT_DIR/dataset_attributions/<run_id>/ and provides read access
to the attribution matrix. No in-memory caching.

Supports two layouts:
- Sub-run layout (current): dataset_attributions/<run_id>/da-YYYYMMDD_HHMMSS/dataset_attributions.pt
- Legacy layout (fallback): dataset_attributions/<run_id>/dataset_attributions.pt
"""

from spd.dataset_attributions.loaders import load_dataset_attributions
from spd.dataset_attributions.storage import DatasetAttributionStorage


class AttributionRepo:
    """Read access to dataset attribution data for a single run."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id

    def has_attributions(self) -> bool:
        return self.get_attributions() is not None

    def get_attributions(self) -> DatasetAttributionStorage | None:
        return load_dataset_attributions(self.run_id)
