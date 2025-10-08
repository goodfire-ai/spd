import numpy as np
from jaxtyping import Int
from transformers import PreTrainedTokenizer


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
