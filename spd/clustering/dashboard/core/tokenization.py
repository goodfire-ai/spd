from transformers import PreTrainedTokenizer
import numpy as np
import torch
from torch import Tensor
from jaxtyping import Int


def attach_vocab_arr(tokenizer: PreTrainedTokenizer) -> None:
    vocab_size: int = tokenizer.vocab_size
    vocab_list: list[str] = [
        tokenizer.convert_ids_to_tokens(i) for i in range(vocab_size)
    ]
    max_token_length: int = max(len(token) for token in vocab_list)
    print(f"{max_token_length = }")
    vocab_arr: np.ndarray = np.array(
        vocab_list,
        dtype=f"S{max_token_length}",
    )
    tokenizer.vocab_arr = vocab_arr  # type: ignore[attr-defined]

def simple_batch_decode(
    tokenizer: PreTrainedTokenizer,
    batch: Int[np.ndarray, "batch_size n_ctx"],
) -> np.ndarray:
    """Decode a batch of token IDs to their string representations
    
    takes an int array of shape (batch_size, n_ctx) and returns an array of shape (batch_size, n_ctx) of bytes
    
    TODO: will probably break with unicode tokens
    """
    assert hasattr(tokenizer, "vocab_arr"), "Tokenizer missing vocab_arr attribute, call attach_vocab_arr first"
    return tokenizer.vocab_arr[batch]