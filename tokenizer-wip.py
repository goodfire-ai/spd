#%%

from transformers import AutoTokenizer
import numpy as np
import torch
from torch import Tensor
from jaxtyping import Int

from spd.clustering.dashboard.core.tokenization import attach_vocab_arr, simple_batch_decode


tokenizer = AutoTokenizer.from_pretrained('SimpleStories/SimpleStories-1.25M')

attach_vocab_arr(tokenizer)
print(f"{tokenizer.vocab_arr = }")



vocab_size: int = tokenizer.vocab_size

random_batch: Int[Tensor, "batch_size n_ctx"] = (
	torch.randint(low=0, high=vocab_size, size=(4, 10), dtype=torch.int64)
)

print(f"{random_batch = }")

batch_token_strings = simple_batch_decode(tokenizer, random_batch)
print(f"{batch_token_strings = }")

# torch.tensor(batch_token_strings)


#%% Benchmarking different approaches
import time
from typing import Callable


def method_1_original(tokenizer, batch: torch.Tensor) -> list[list[str]]:
	"""Original approach: flatten, batch_decode, reshape"""
	batch_size, n_ctx = batch.shape
	flattened_tokens = batch.reshape(-1, 1)
	all_token_strings: list[str] = tokenizer.batch_decode(flattened_tokens)
	batch_token_strings: list[list[str]] = [
		all_token_strings[i * n_ctx : (i + 1) * n_ctx] for i in range(batch_size)
	]
	return batch_token_strings


# def method_2_batch_convert(tokenizer, batch: torch.Tensor) -> list[list[str]]:
# 	"""New approach: batch_convert_ids_to_tokens"""
# 	return tokenizer.batch_convert_ids_to_tokens(batch)


def method_3_vocab_arr(tokenizer, batch: torch.Tensor) -> np.ndarray:
	"""Custom approach: vocab_arr indexing"""
	return simple_batch_decode(tokenizer, batch)


def method_4_loop_convert(tokenizer, batch: torch.Tensor) -> list[list[str]]:
	"""Loop with convert_ids_to_tokens per sequence"""
	return [tokenizer.convert_ids_to_tokens(seq) for seq in batch]


def benchmark_method(
	name: str,
	method: Callable,
	tokenizer,
	batch: torch.Tensor,
	n_iterations: int = 100,
) -> float:
	"""Benchmark a tokenization method"""
	# Warmup
	for _ in range(5):
		method(tokenizer, batch)

	# Actual timing
	start = time.perf_counter()
	for _ in range(n_iterations):
		result = method(tokenizer, batch)
	end = time.perf_counter()

	elapsed = end - start
	avg_time = elapsed / n_iterations

	print(f"{name:30s}: {avg_time*1000:7.3f} ms/batch ({elapsed:.3f}s total for {n_iterations} iterations)")
	return avg_time


# Test batches of different sizes
test_configs = [
	(4, 10, "Small"),      # 4 sequences, 10 tokens each
	(16, 32, "Medium"),    # 16 sequences, 32 tokens each
	(32, 128, "Large"),    # 32 sequences, 128 tokens each
]

print("\n" + "="*80)
print("TOKENIZATION BENCHMARKING")
print("="*80)

for batch_size, seq_len, size_name in test_configs:
	print(f"\n{size_name} batch: {batch_size} x {seq_len} = {batch_size * seq_len} tokens")
	print("-" * 80)

	test_batch = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), dtype=torch.int64)

	times = {}
	times["original"] = benchmark_method("Method 1 (original)", method_1_original, tokenizer, test_batch)
	# times["batch_convert"] = benchmark_method("Method 2 (batch_convert)", method_2_batch_convert, tokenizer, test_batch)
	times["vocab_arr"] = benchmark_method("Method 3 (vocab_arr)", method_3_vocab_arr, tokenizer, test_batch)
	times["loop_convert"] = benchmark_method("Method 4 (loop convert)", method_4_loop_convert, tokenizer, test_batch)

	# Find fastest
	fastest_name = min(times.items(), key=lambda x: x[1])[0]
	print(f"\nFastest: {fastest_name}")

	# Show speedups vs original
	print(f"Speedup vs original:")
	for name, time_val in times.items():
		if name != "original":
			speedup = times["original"] / time_val
			print(f"  {name:20s}: {speedup:5.2f}x")

print("\n" + "="*80)


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