"""Utilities for parallel processing with shared memory."""

from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

import numpy as np
from jaxtyping import Float
from numpy import ndarray


@dataclass
class SharedArrayMetadata:
    """Metadata for reconstructing shared numpy arrays in worker processes."""

    shm_name: str
    shape: tuple[int, ...]
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for passing to workers."""
        return {
            "shm_name": self.shm_name,
            "shape": self.shape,
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SharedArrayMetadata":
        """Deserialize from dict."""
        return cls(
            shm_name=data["shm_name"],
            shape=tuple(data["shape"]),
            dtype=data["dtype"],
        )


def create_shared_array(
    array: Float[ndarray, "..."],
) -> tuple[shared_memory.SharedMemory, SharedArrayMetadata]:
    """Create shared memory block from numpy array.

    Args:
        array: Numpy array to share across processes

    Returns:
        Tuple of (SharedMemory object, metadata for reconstruction)
    """
    shm = shared_memory.SharedMemory(create=True, size=array.nbytes)

    # Copy array data to shared memory
    shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    shared_array[:] = array[:]

    metadata = SharedArrayMetadata(
        shm_name=shm.name,
        shape=array.shape,
        dtype=str(array.dtype),
    )

    return shm, metadata


def attach_shared_array(metadata: SharedArrayMetadata) -> ndarray:
    """Reconstruct numpy array from shared memory in worker process.

    Args:
        metadata: Metadata describing the shared array

    Returns:
        Numpy array view backed by shared memory (read-only recommended)
    """
    shm = shared_memory.SharedMemory(name=metadata.shm_name)
    array = np.ndarray(metadata.shape, dtype=np.dtype(metadata.dtype), buffer=shm.buf)

    # Return read-only view for safety
    array.flags.writeable = False

    return array


def cleanup_shared_array(shm: shared_memory.SharedMemory) -> None:
    """Release shared memory resources.

    Args:
        shm: SharedMemory object to cleanup
    """
    shm.close()
    shm.unlink()
