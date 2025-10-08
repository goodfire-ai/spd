#%%

from transformers import AutoTokenizer
import numpy as np
import torch
from torch import Tensor
from jaxtyping import Int


tokenizer = AutoTokenizer.from_pretrained('SimpleStories/SimpleStories-1.25M')
# tokenizer.vocab

# dir(tokenizer)


# for k in dir(tokenizer):
# 	if not k.startswith('__'):
# 		print(f"{k:<50}\t:\t{type(getattr(tokenizer, k))}")


# keys_types: list[tuple[str,str]] = [
# 	(k, str(type(getattr(tokenizer, k)))) for k in dir(tokenizer) if not k.startswith('__')
# ]


# for k,t in sorted(keys_types, key=lambda x: x[1]):
# 	print(f"{k:<50}\t:\t{t}")


# func = tokenizer._convert_id_to_token
# func = tokenizer._tokenizer.id_to_token
# # find the source code for this function
# func
# import inspect
# print(inspect.getsource(func))
# # print(inspect.getsourcefile(func))
# func(10)

# tokenizer.get_vocab()



def attach_vocab_arr(tokenizer: PreTrainedTokenizer) -> None:
    vocab_size: int = tokenizer.vocab_size
    vocab_list: list[str] = [
        tokenizer.convert_ids_to_tokens(i) for i in range(vocab_size)
    ]
    max_token_length: int = max(len(token) for token in vocab_list)
    print(f"{max_token_length = }")
    vocab_arr: np.ndarray = np.array(
        vocab_list, dtype=f"S{max_token_length}"
		# [token.ljust(max_token_length) for token in vocab_list], dtype=f"S{max_token_length}"
    )
    tokenizer.vocab_arr = vocab_arr  # type: ignore[attr-defined]


attach_vocab_arr(tokenizer)
print(f"{tokenizer.vocab_arr = }")



def simple_batch_decode(
    tokenizer: PreTrainedTokenizer,
    batch,#: Int[Tensor, "batch_size n_ctx"],
) -> list[list[str]]:
    assert hasattr(tokenizer, "vocab_arr"), "Tokenizer missing vocab_arr attribute, call attach_vocab_arr first"
    return tokenizer.vocab_arr[batch]



vocab_size: int = tokenizer.vocab_size

random_batch: Int[Tensor, "batch_size n_ctx"] = (
	torch.randint(low=0, high=vocab_size, size=(4, 10), dtype=torch.int64)
)

print(f"{random_batch = }")

batch_token_strings: list[list[str]] = simple_batch_decode(tokenizer, random_batch)
print(f"{batch_token_strings = }")

torch.tensor(batch_token_strings)


