"""Test the optimized create_data_loader with rank 0 tokenization.

This script tests that:
1. Rank 0 tokenizes first while others wait
2. Other ranks load from HuggingFace cache (should be fast)
3. Total time is roughly: tokenization_time + small_overhead (not tokenization_time * n_ranks)
"""

import time

from spd.data import DatasetConfig, create_data_loader
from spd.utils.distributed_utils import (
    cleanup_distributed,
    get_distributed_state,
    init_distributed,
    sync_across_processes,
)


def log(msg: str, rank: int) -> None:
    """Log with rank prefix and timestamp."""
    elapsed = time.time() - START_TIME
    print(f"[RANK {rank:2d}] [{elapsed:7.2f}s] {msg}", flush=True)


START_TIME = time.time()

DATASET_NAME = "SimpleStories/SimpleStories"
TOKENIZER_NAME = "SimpleStories/test-SimpleStories-gpt2-1.25M"
MAX_SEQ_LEN = 512


def main() -> None:
    print(f"[RANK ??] [{0.0:7.2f}s] Script started", flush=True)

    # Initialize distributed
    t0 = time.time()
    dist_state = init_distributed()
    t1 = time.time()

    rank = dist_state.rank if dist_state else 0
    world_size = dist_state.world_size if dist_state else 1

    log(f"init_distributed completed in {t1 - t0:.2f}s (world_size={world_size})", rank)

    sync_across_processes()
    log("Initial barrier complete", rank)

    # Test create_data_loader with the optimization
    log("=" * 50, rank)
    log("Testing OPTIMIZED create_data_loader", rank)
    log("=" * 50, rank)

    dataset_config = DatasetConfig(
        name=DATASET_NAME,
        hf_tokenizer_path=TOKENIZER_NAME,
        split="train",
        n_ctx=MAX_SEQ_LEN,
        is_tokenized=False,  # Force tokenization
        streaming=False,
        column_name="story",
        shuffle_each_epoch=True,
    )

    sync_across_processes()
    t0 = time.time()

    train_loader, tokenizer = create_data_loader(
        dataset_config=dataset_config,
        batch_size=8,
        buffer_size=1000,
        global_seed=0,
        dist_state=dist_state,
    )

    t1 = time.time()
    log(f"create_data_loader completed in {t1 - t0:.2f}s", rank)

    # Get first batch to verify it works
    log("Getting first batch...", rank)
    t0 = time.time()
    batch = next(iter(train_loader))
    t1 = time.time()
    log(f"First batch retrieved in {t1 - t0:.2f}s, shape: {batch['input_ids'].shape}", rank)

    sync_across_processes()

    # Save rank before cleanup
    rank_saved = rank

    # Cleanup
    cleanup_distributed()

    total_time = time.time() - START_TIME
    if rank_saved == 0:
        print(f"\n{'='*60}", flush=True)
        print(f"TEST COMPLETE: Total time {total_time:.2f}s", flush=True)
        print(f"{'='*60}", flush=True)
        print("\nExpected behavior:", flush=True)
        print("- Rank 0 should tokenize (~95s on 1 GPU)", flush=True)
        print("- Other ranks should load from cache (~1-5s)", flush=True)
        print("- Total time should be ~100s, NOT ~200s+", flush=True)


if __name__ == "__main__":
    main()
