from dataclasses import dataclass

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

    @classmethod
    def from_token_batches(
        cls,
        token_batches: list[Int[Tensor, "batch n_ctx"]],
        tokenizer: PreTrainedTokenizer,
    ) -> "TokenSequenceData":
        """Decode token batches into 2D array."""
        if not hasattr(tokenizer, "vocab_arr"):
            attach_vocab_arr(tokenizer)

        tokens_decoded: Shaped[np.ndarray, "n_sequences n_ctx"] = simple_batch_decode(
            tokenizer=tokenizer,
            batch=torch.cat(token_batches, dim=0).cpu().numpy(),
        )

        return cls(tokens=tokens_decoded)
