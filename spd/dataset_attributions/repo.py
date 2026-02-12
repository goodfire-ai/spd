"""Dataset attributions data repository.

Owns SPD_OUT_DIR/dataset_attributions/<run_id>/ and provides read access
to the attribution matrix.

Use AttributionRepo.open() to construct — returns None if no attribution data exists.

Supports two layouts:
- Sub-run layout (current): dataset_attributions/<run_id>/da-YYYYMMDD_HHMMSS/dataset_attributions.pt
- Legacy layout (fallback): dataset_attributions/<run_id>/dataset_attributions.pt
"""

from spd.dataset_attributions.loaders import load_dataset_attributions
from spd.dataset_attributions.storage import DatasetAttributionStorage


class AttributionRepo:
    """Read access to dataset attribution data for a single run.

    Constructed via AttributionRepo.open(). Storage is loaded eagerly at construction.
    """

    def __init__(self, storage: DatasetAttributionStorage) -> None:
        self._storage = storage

    @classmethod
    def open(cls, run_id: str) -> "AttributionRepo | None":
        """Open attribution data for a run. Returns None if no attribution data exists."""
        storage = load_dataset_attributions(run_id)
        if storage is None:
            return None
        return cls(storage)

    def get_attributions(self) -> DatasetAttributionStorage:
        return self._storage
