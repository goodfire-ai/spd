"""Minimal storage base class for clustering - just path management."""

from pathlib import Path

from spd.utils.run_utils import ExecutionStamp


class StorageBase:
    """Base class for storage - provides ExecutionStamp and base directory.

    Subclasses define path constants (relative to base_dir) and set absolute paths in __init__.
    Caller handles all actual saving and WandB uploading.
    """

    def __init__(self, execution_stamp: ExecutionStamp) -> None:
        """Initialize storage with execution stamp."""
        self.execution_stamp: ExecutionStamp = execution_stamp
        self.base_dir: Path = execution_stamp.out_dir
        self.plots_dir: Path = self.base_dir / "plots"
