"""Phase 4b: Investigate tokenization scaling with multiple ranks.

This script tests whether tokenization slows down with more ranks due to:
1. Disk I/O contention
2. CPU contention
3. HuggingFace cache locking
4. Memory bandwidth

We'll measure each sub-operation separately to pinpoint the bottleneck.
"""

import os
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from spd.data import tokenize_and_concatenate
from spd.utils.distributed_utils import (
    cleanup_distributed,
    get_distributed_state,
    init_distributed,
    sync_across_processes,
)


def log(msg: str, rank: int | None = None) -> None:
    """Log with rank prefix and timestamp."""
    if rank is None:
        state = get_distributed_state()
        rank = state.rank if state else 0
    elapsed = time.time() - START_TIME
    print(f"[RANK {rank:2d}] [{elapsed:7.2f}s] {msg}", flush=True)


START_TIME = time.time()

DATASET_NAME = "SimpleStories/SimpleStories"
TOKENIZER_NAME = "SimpleStories/test-SimpleStories-gpt2-1.25M"
SPLIT = "train"
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

    # Barrier to synchronize before tests
    sync_across_processes()
    log("Initial barrier complete", rank)

    # =========================================================================
    # Test 1: Measure load_dataset time (disk I/O)
    # =========================================================================
    log("=" * 50, rank)
    log("TEST 1: load_dataset (disk I/O test)", rank)
    log("=" * 50, rank)

    sync_across_processes()
    t0 = time.time()

    dataset = load_dataset(
        DATASET_NAME,
        streaming=False,
        split=SPLIT,
        trust_remote_code=False,
    )

    t1 = time.time()
    log(f"load_dataset completed in {t1 - t0:.2f}s ({len(dataset)} examples)", rank)

    # =========================================================================
    # Test 2: Measure dataset.shuffle time (memory operation)
    # =========================================================================
    log("=" * 50, rank)
    log("TEST 2: dataset.shuffle (memory test)", rank)
    log("=" * 50, rank)

    sync_across_processes()
    t0 = time.time()

    dataset = dataset.shuffle(seed=42)

    t1 = time.time()
    log(f"dataset.shuffle completed in {t1 - t0:.2f}s", rank)

    # =========================================================================
    # Test 3: Measure tokenizer loading
    # =========================================================================
    log("=" * 50, rank)
    log("TEST 3: AutoTokenizer.from_pretrained", rank)
    log("=" * 50, rank)

    sync_across_processes()
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    t1 = time.time()
    log(f"AutoTokenizer.from_pretrained completed in {t1 - t0:.2f}s", rank)

    # =========================================================================
    # Test 4: Measure tokenize_and_concatenate (CPU-bound tokenization)
    # =========================================================================
    log("=" * 50, rank)
    log("TEST 4: tokenize_and_concatenate (CPU-bound test)", rank)
    log("=" * 50, rank)

    sync_across_processes()
    t0 = time.time()

    # Note: tokenize_and_concatenate uses num_proc=10 by default
    torch_dataset = tokenize_and_concatenate(
        dataset,
        tokenizer,
        max_length=MAX_SEQ_LEN,
        column_name="story",
        add_bos_token=False,
        num_proc=10,  # Default value
    )

    t1 = time.time()
    log(f"tokenize_and_concatenate completed in {t1 - t0:.2f}s", rank)

    # =========================================================================
    # Test 5: Measure with reduced parallelism (num_proc=1)
    # =========================================================================
    log("=" * 50, rank)
    log("TEST 5: tokenize_and_concatenate with num_proc=1", rank)
    log("=" * 50, rank)

    # Reload dataset for fresh test
    dataset2 = load_dataset(
        DATASET_NAME,
        streaming=False,
        split=SPLIT,
        trust_remote_code=False,
    )
    dataset2 = dataset2.shuffle(seed=42)

    sync_across_processes()
    t0 = time.time()

    torch_dataset2 = tokenize_and_concatenate(
        dataset2,
        tokenizer,
        max_length=MAX_SEQ_LEN,
        column_name="story",
        add_bos_token=False,
        num_proc=1,  # Single process
    )

    t1 = time.time()
    log(f"tokenize_and_concatenate (num_proc=1) completed in {t1 - t0:.2f}s", rank)

    sync_across_processes()

    # Save rank before cleanup
    rank_saved = rank

    # Cleanup
    cleanup_distributed()

    total_time = time.time() - START_TIME
    if rank_saved == 0:
        print(f"\n{'='*60}", flush=True)
        print(f"PHASE 4b COMPLETE: Total time {total_time:.2f}s", flush=True)
        print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
