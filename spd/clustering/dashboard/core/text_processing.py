"""Text processing utilities for dashboard data generation."""

from jaxtyping import Int
from torch import Tensor
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from spd.clustering.dashboard.core.base import TextSample, TextSampleHash
from spd.clustering.dashboard.core.tokenization import simple_batch_decode


def tokenize_and_create_text_samples(
    batch: Int[Tensor, "batch_size n_ctx"],
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
    batch_size: int = batch.shape[0]

    batch_token_strings: list[list[str]] = simple_batch_decode(
        tokenizer, batch.cpu().numpy()
    ).tolist()  # [batch_size, n_ctx] of strings

    # Create text samples for entire batch
    batch_text_samples: list[TextSample] = []
    for token_strings in batch_token_strings:
        text: str = " ".join(token_strings)
        text_sample: TextSample = TextSample(full_text=text, tokens=token_strings)
        text_samples[text_sample.text_hash] = text_sample
        batch_text_samples.append(text_sample)

    return batch_text_samples
