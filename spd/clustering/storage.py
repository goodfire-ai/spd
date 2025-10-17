"""Minimal storage base class for clustering - just path management."""

from pathlib import Path

from spd.utils.run_utils import ExecutionStamp


class StorageBase:
    """Base class for storage - provides ExecutionStamp and base directory.

    Subclasses define path constants (relative to base_dir) and properties (absolute paths).
    Caller handles all actual saving and WandB uploading.
    """

    def __init__(self, execution_stamp: ExecutionStamp):
        """Initialize storage with execution stamp.

        Args:
            execution_stamp: Execution stamp with run_id and directory info
        """
        self.execution_stamp: ExecutionStamp = execution_stamp

    @property
    def base_dir(self) -> Path:
        """Get base directory from execution stamp."""
        return self.execution_stamp.out_dir

    @property
    def plots_dir(self) -> Path:
        """Get plots directory."""
        return self.base_dir / "plots"
