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


"""


================================================================================
TOKENIZATION BENCHMARKING
================================================================================

Small batch: 4 x 10 = 40 tokens
--------------------------------------------------------------------------------
Method 1 (original)           :   0.418 ms/batch (0.042s total for 100 iterations)
Method 3 (vocab_arr)          :   0.013 ms/batch (0.001s total for 100 iterations)
Method 4 (loop convert)       :   0.095 ms/batch (0.010s total for 100 iterations)

Fastest: vocab_arr
Speedup vs original:
  vocab_arr           : 33.22x
  loop_convert        :  4.38x

Medium batch: 16 x 32 = 512 tokens
--------------------------------------------------------------------------------
Method 1 (original)           :   4.817 ms/batch (0.482s total for 100 iterations)
Method 3 (vocab_arr)          :   0.011 ms/batch (0.001s total for 100 iterations)
Method 4 (loop convert)       :   0.955 ms/batch (0.095s total for 100 iterations)

Fastest: vocab_arr
Speedup vs original:
  vocab_arr           : 456.34x
  loop_convert        :  5.05x

Large batch: 32 x 128 = 4096 tokens
--------------------------------------------------------------------------------
Method 1 (original)           :  40.417 ms/batch (4.042s total for 100 iterations)
Method 3 (vocab_arr)          :   0.026 ms/batch (0.003s total for 100 iterations)
Method 4 (loop convert)       :   7.174 ms/batch (0.717s total for 100 iterations)

Fastest: vocab_arr
Speedup vs original:
  vocab_arr           : 1560.87x
  loop_convert        :  5.63x

================================================================================



"""