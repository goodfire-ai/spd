import hashlib
import json
from dataclasses import dataclass
from functools import cached_property

import torch
from numpy import ndarray
from torch import Tensor
from transformers import PreTrainedTokenizer

from spd.dashboard.core.tokenization import attach_vocab_arr, simple_batch_decode


@dataclass(frozen=True)
class TextSample:
    """A sequence of tokens with its location in the flattened activation array."""

    tokens: list[str]
    dataset_idx: tuple[int, int]  # (start, end) slice into flattened n_samples array

    def __hash__(self) -> int:
        """Hash based on token tuple for Python dict/set compatibility."""
        return hash(self.sequence_hash)

    @cached_property
    def sequence_hash(self) -> str:
        """SHA256 hash of tokens for stable string keys."""
        return hashlib.sha256(json.dumps(self.tokens).encode()).digest().hex()


@dataclass
class TokenSequenceData:
    """Efficient storage for token sequences using flat array with range indexing."""

    all_tokens: ndarray  # [total_tokens] flat string array with dtype U{max_token_length}
    sequence_ranges: dict[str, tuple[int, int]]  # sequence_hash -> (start, end) into all_tokens
    dataset_indices: dict[str, tuple[int, int]]  # sequence_hash -> (start, end) into n_samples

    def get_sequence_tokens(self, sequence_hash: str) -> TextSample:
        """Get TextSample for a specific sequence by its hash.

        Args:
            sequence_hash: SHA256 hash of the sequence

        Returns:
            TextSample with tokens and dataset_idx
        """
        start, end = self.sequence_ranges[sequence_hash]
        tokens: list[str] = self.all_tokens[start:end].tolist()
        dataset_idx: tuple[int, int] = self.dataset_indices[sequence_hash]
        return TextSample(tokens=tokens, dataset_idx=dataset_idx)

    def get_all_sequences(self) -> list[TextSample]:
        """Get all sequences as TextSample objects.

        Returns:
            List of all TextSample objects
        """
        return [self.get_sequence_tokens(h) for h in self.sequence_ranges]

    @classmethod
    def from_token_batches(
        cls,
        token_batches: list[Tensor],
        tokenizer: PreTrainedTokenizer,
    ) -> "TokenSequenceData":
        """Generate TokenSequenceData from batches of token IDs.

        Args:
            token_batches: List of token ID tensors [batch_size, n_ctx]
            tokenizer: Tokenizer to decode token IDs

        Returns:
            TokenSequenceData with flat array storage
        """
        # Ensure tokenizer has vocab array for fast batch decoding
        if not hasattr(tokenizer, "vocab_arr"):
            attach_vocab_arr(tokenizer)

        # Concatenate batches: [n_sequences, n_ctx]
        tokens_concat: Tensor = torch.cat(token_batches, dim=0)
        n_sequences: int = tokens_concat.shape[0]
        n_ctx: int = tokens_concat.shape[1]

        # Decode all tokens using efficient batch decode
        tokens_decoded: ndarray = simple_batch_decode(
            tokenizer, tokens_concat.numpy()
        )  # [n_sequences, n_ctx]

        # Flatten to single array: [total_tokens]
        all_tokens_flat: ndarray = tokens_decoded.reshape(-1)

        # Build sequence mappings
        sequence_ranges: dict[str, tuple[int, int]] = {}
        dataset_indices: dict[str, tuple[int, int]] = {}

        for seq_idx in range(n_sequences):
            # Extract tokens for this sequence
            seq_tokens: list[str] = tokens_decoded[seq_idx].tolist()

            # Compute stable hash
            seq_hash: str = hashlib.sha256(str(tuple(seq_tokens)).encode()).hexdigest()

            # Token range in all_tokens array
            token_start: int = seq_idx * n_ctx
            token_end: int = (seq_idx + 1) * n_ctx
            sequence_ranges[seq_hash] = (token_start, token_end)

            # Dataset index range in flattened n_samples array
            dataset_indices[seq_hash] = (token_start, token_end)

        return cls(
            all_tokens=all_tokens_flat,
            sequence_ranges=sequence_ranges,
            dataset_indices=dataset_indices,
        )
