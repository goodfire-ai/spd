"""Phase 4: Test dataset and tokenizer loading.

This script tests:
1. load_dataset() from HuggingFace datasets
2. AutoTokenizer.from_pretrained()
3. create_data_loader() function

Uses the SimpleStories dataset from ss_llama_simple_mlp-1L config.
"""

import os
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from spd.data import DatasetConfig, create_data_loader
from spd.utils.distributed_utils import (
    cleanup_distributed,
    ensure_cached_and_call,
    get_device,
    get_distributed_state,
    init_distributed,
    is_distributed,
    is_main_process,
    sync_across_processes,
)


def log(msg: str) -> None:
    """Log with rank prefix and timestamp."""
    state = get_distributed_state()
    rank = state.rank if state else 0
    elapsed = time.time() - START_TIME
    print(f"[RANK {rank}] [{elapsed:7.2f}s] {msg}", flush=True)


START_TIME = time.time()

# Dataset config from ss_llama_simple_mlp-1L.yaml
DATASET_NAME = "SimpleStories/SimpleStories"
TOKENIZER_NAME = "SimpleStories/test-SimpleStories-gpt2-1.25M"
SPLIT = "train"
MAX_SEQ_LEN = 512


def main() -> None:
    print(f"[RANK ?] [{0.0:7.2f}s] Script started", flush=True)

    # Initialize distributed
    print(f"[RANK ?] [{time.time() - START_TIME:7.2f}s] Starting init_distributed...", flush=True)
    t0 = time.time()
    dist_state = init_distributed()
    t1 = time.time()
    log(f"init_distributed completed in {t1 - t0:.2f}s")

    device = get_device()
    log(f"Device: {device}")

    sync_across_processes()
    log("Initial barrier complete")

    # =========================================================================
    # Test 1: Direct load_dataset (all ranks at once)
    # =========================================================================
    log("=" * 50)
    log("TEST 1: Direct load_dataset (all ranks)")
    log("=" * 50)

    t0 = time.time()
    dataset = load_dataset(
        DATASET_NAME,
        streaming=False,
        split=SPLIT,
        trust_remote_code=False,
    )
    t1 = time.time()
    log(f"load_dataset completed in {t1 - t0:.2f}s")
    log(f"Dataset size: {len(dataset)} examples")

    sync_across_processes()

    # =========================================================================
    # Test 2: Direct AutoTokenizer.from_pretrained (all ranks at once)
    # =========================================================================
    log("=" * 50)
    log("TEST 2: Direct AutoTokenizer.from_pretrained (all ranks)")
    log("=" * 50)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    t1 = time.time()
    log(f"AutoTokenizer.from_pretrained completed in {t1 - t0:.2f}s")
    log(f"Vocab size: {tokenizer.vocab_size}")

    sync_across_processes()

    # =========================================================================
    # Test 3: load_dataset with ensure_cached_and_call
    # =========================================================================
    log("=" * 50)
    log("TEST 3: load_dataset with ensure_cached_and_call")
    log("=" * 50)

    t0 = time.time()
    dataset2 = ensure_cached_and_call(
        load_dataset,
        DATASET_NAME,
        streaming=False,
        split=SPLIT,
        trust_remote_code=False,
    )
    t1 = time.time()
    log(f"ensure_cached_and_call(load_dataset) completed in {t1 - t0:.2f}s")

    sync_across_processes()

    # =========================================================================
    # Test 4: Full create_data_loader
    # =========================================================================
    log("=" * 50)
    log("TEST 4: Full create_data_loader")
    log("=" * 50)

    dataset_config = DatasetConfig(
        name=DATASET_NAME,
        hf_tokenizer_path=TOKENIZER_NAME,
        split=SPLIT,
        n_ctx=MAX_SEQ_LEN,
        is_tokenized=False,
        streaming=False,
        column_name="story",
        shuffle_each_epoch=True,
    )

    t0 = time.time()
    train_loader, _ = create_data_loader(
        dataset_config=dataset_config,
        batch_size=8,  # Small batch for testing
        buffer_size=1000,
        global_seed=0,
        dist_state=dist_state,
    )
    t1 = time.time()
    log(f"create_data_loader completed in {t1 - t0:.2f}s")

    sync_across_processes()

    # =========================================================================
    # Test 5: Iterate first batch
    # =========================================================================
    log("=" * 50)
    log("TEST 5: Get first batch")
    log("=" * 50)

    t0 = time.time()
    batch = next(iter(train_loader))
    t1 = time.time()
    log(f"First batch retrieved in {t1 - t0:.2f}s")
    log(f"Batch shape: {batch['input_ids'].shape}")

    sync_across_processes()

    # Save rank before cleanup
    rank = dist_state.rank if dist_state else 0

    # Cleanup
    cleanup_distributed()

    total_time = time.time() - START_TIME
    if rank == 0:
        print(f"\n{'='*60}", flush=True)
        print(f"PHASE 4 COMPLETE: Total time {total_time:.2f}s", flush=True)
        print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
