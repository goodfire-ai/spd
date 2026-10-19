"""Packed rows as plain arrays: no dataset reader, so a process that only passes rows along
loads no Arrow. `datasets` reads them from a dataset's shards."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PackedRows:
    """Packed rows, as a dataset stores them: each row's token ids and its tokens'
    document labels, validated contiguous within each row."""

    token_ids: NDArray[np.int32]
    document_ids: NDArray[np.int32]

    def __post_init__(self) -> None:
        assert self.token_ids.dtype == self.document_ids.dtype == np.int32, (
            "token_ids and document_ids must be int32"
        )
        assert self.token_ids.ndim == 2 and self.token_ids.shape[1] > 0, (
            "token_ids must have shape [batch, sequence] with a nonempty sequence"
        )
        assert self.document_ids.shape == self.token_ids.shape, (
            "token_ids and document_ids must have identical shapes"
        )
        assert np.all(self.token_ids >= 0), "token IDs must be nonnegative"
        assert np.all(self.document_ids >= 0), "document IDs must be nonnegative"
        transitions = np.diff(self.document_ids, axis=1)
        assert np.all((transitions == 0) | (transitions == 1)), (
            "document IDs must be contiguous and nondecreasing within each row"
        )

    def validate_vocabulary(self, vocab_size: int) -> None:
        assert vocab_size > 0, "vocab_size must be positive"
        assert np.all(self.token_ids < vocab_size), (
            f"token IDs outside [0, {vocab_size}); dataset and target vocabularies differ"
        )

    @classmethod
    def from_unsegmented_sequences(cls, token_ids: NDArray[np.int32]) -> "PackedRows":
        """Treat each sequence as unpadded, without internal document boundaries."""
        return cls(token_ids, np.zeros_like(token_ids))
