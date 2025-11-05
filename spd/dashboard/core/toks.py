from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch
from jaxtyping import Int, Shaped
from torch import Tensor
from transformers import PreTrainedTokenizer

from spd.dashboard.core.tokenization import attach_vocab_arr, simple_batch_decode


@dataclass
class TokenSequenceData:
    """2D token storage aligned with 3D activation arrays."""

    tokens: Shaped[np.ndarray, "n_sequences n_ctx"]  # of type `U{max_token_length}`
    token_ids: Int[np.ndarray, "n_sequences n_ctx"]  # integer token IDs
    vocab_arr: Shaped[np.ndarray, " d_vocab"]  # of type `U{max_token_length}`

    @cached_property
    def token_vocab_idx(self) -> dict[str, int]:
        """Map from token string to its vocabulary index."""
        return {token: idx for idx, token in enumerate(self.vocab_arr)}

    @property
    def d_vocab(self) -> int:
        """Size of the vocabulary."""
        return self.vocab_arr.shape[0]

    @property
    def n_sequences(self) -> int:
        """Number of sequences in the dataset."""
        return self.tokens.shape[0]

    @classmethod
    def from_token_batches(
        cls,
        token_batches: list[Int[Tensor, "batch n_ctx"]],
        tokenizer: PreTrainedTokenizer,
    ) -> "TokenSequenceData":
        """Decode token batches into 2D array."""
        if not hasattr(tokenizer, "vocab_arr"):
            attach_vocab_arr(tokenizer)

        token_ids: Int[np.ndarray, "n_sequences n_ctx"] = (
            torch.cat(token_batches, dim=0).cpu().numpy()
        )
        tokens_decoded: Shaped[np.ndarray, "n_sequences n_ctx"] = simple_batch_decode(
            tokenizer=tokenizer,
            batch=token_ids,
        )

        return cls(
            tokens=tokens_decoded,
            token_ids=token_ids,
            vocab_arr=tokenizer.vocab_arr,  # pyright: ignore[reportAttributeAccessIssue]
        )

    @cached_property
    def token_counts(self) -> dict[str, int]:
        """Count occurrences of each token in the dataset."""
        unique_tokens, counts = np.unique(self.tokens, return_counts=True)
        return {token: int(count) for token, count in zip(unique_tokens, counts, strict=True)}
