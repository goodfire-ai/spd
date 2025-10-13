"""Text processing utilities for dashboard data generation."""

import numpy as np
from jaxtyping import Int
from transformers import PreTrainedTokenizer

from spd.clustering.dashboard.core.base import TextSample, TextSampleHash

# TODO: pyright.... hates tokenizers???
# pyright: reportAttributeAccessIssue=false, reportUnknownParameterType=false


def attach_vocab_arr(tokenizer: PreTrainedTokenizer) -> None:
    """Attach a numpy array of token strings to the tokenizer for fast batch decoding.

    Creates a vocab_arr attribute containing all tokens as unicode strings,
    enabling O(1) array indexing instead of repeated convert_ids_to_tokens calls.
    """
    vocab_size: int = tokenizer.vocab_size
    vocab_list: list[str] = [tokenizer.convert_ids_to_tokens(i) for i in range(vocab_size)]
    max_token_length: int = max(len(token) for token in vocab_list)
    print(f"{max_token_length = }")
    vocab_arr: np.ndarray = np.array(
        vocab_list,
        dtype=f"U{max_token_length}",  # Unicode strings, not bytes
    )
    tokenizer.vocab_arr = vocab_arr  # type: ignore[attr-defined]


def simple_batch_decode(
    tokenizer: PreTrainedTokenizer,
    batch: Int[np.ndarray, "batch_size n_ctx"],
) -> np.ndarray:
    """Decode a batch of token IDs to their string representations.

    Args:
        tokenizer: PreTrainedTokenizer with vocab_arr attached
        batch: Token IDs array of shape (batch_size, n_ctx)

    Returns:
        Array of shape (batch_size, n_ctx) containing unicode token strings
    """
    assert hasattr(tokenizer, "vocab_arr"), (
        "Tokenizer missing vocab_arr attribute, call attach_vocab_arr first"
    )
    return tokenizer.vocab_arr[batch]


def tokenize_and_create_text_samples(
    batch: Int[np.ndarray, "batch_size n_ctx"],
    tokenizer: PreTrainedTokenizer,
    text_samples: dict[TextSampleHash, TextSample],
) -> list[TextSample]:
    """Tokenize batch and create TextSample objects.

    Note: This function decodes tokens by converting IDs to token strings and joining with spaces.
    This bypasses the tokenizer's native .decode() logic, which may handle special tokens,
    BPE merge undoing, and special whitespace differently. For display purposes in the dashboard,
    this simplified approach is acceptable and significantly faster.

    Args:
        batch: Input token IDs
        tokenizer: Tokenizer for decoding
        text_samples: Existing text samples dict (for deduplication)

    Returns:
        List of TextSample objects for the batch
    """
    batch_token_strings: list[list[str]] = simple_batch_decode(
        tokenizer, batch
    ).tolist()  # [batch_size, n_ctx] of strings

    # Create text samples for entire batch
    batch_text_samples: list[TextSample] = []
    for token_strings in batch_token_strings:
        text: str = " ".join(token_strings)
        text_sample: TextSample = TextSample(full_text=text, tokens=token_strings)
        text_samples[text_sample.text_hash] = text_sample
        batch_text_samples.append(text_sample)

    return batch_text_samples
