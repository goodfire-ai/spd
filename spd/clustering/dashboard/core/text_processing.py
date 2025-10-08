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

    Args:
        batch: Input token IDs
        tokenizer: Tokenizer for decoding
        text_samples: Existing text samples dict (for deduplication)

    Returns:
        List of TextSample objects for the batch
    """
    batch_size: int = batch.shape[0]

    # Convert to numpy and use optimized batch decode
    batch_np = batch.cpu().numpy()
    batch_token_bytes = simple_batch_decode(tokenizer, batch_np)  # [batch_size, n_ctx] of bytes

    # Decode bytes to strings
    batch_token_strings: list[list[str]] = [
        [token.decode("utf-8", errors="replace") for token in seq] for seq in batch_token_bytes
    ]

    # Create text samples for entire batch
    batch_text_samples: list[TextSample] = []
    for token_strings in tqdm(batch_token_strings, total=batch_size, desc="Creating text samples"):
        text: str = " ".join(token_strings)
        text_sample: TextSample = TextSample(full_text=text, tokens=token_strings)
        text_samples[text_sample.text_hash] = text_sample
        batch_text_samples.append(text_sample)

    return batch_text_samples
