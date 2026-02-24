"""Shuffle and re-upload pile-uncopyrighted-tok to HuggingFace.

The tokenized dataset has sequences in document order, so large documents produce many
consecutive similar sequences. This script globally shuffles all sequences and uploads
to a new dataset (danbraunai/pile-uncopyrighted-tok-shuffled).

Usage: python scripts/shuffle_and_reupload_dataset.py
"""

import time

from datasets import load_dataset

DATASET_NAME = "danbraunai/pile-uncopyrighted-tok"
NEW_DATASET_NAME = "danbraunai/pile-uncopyrighted-tok-shuffled"
SEED = 42
NUM_PROC = 160
SPLITS = ["train", "val", "test"]
# Match original shard counts so DDP num_shards-based sharding still works
SHARD_COUNTS = {"train": 2021, "val": 12, "test": 2}


def process_split(split: str):
    print(f"\n{'='*60}")
    print(f"Processing split: {split}")
    print(f"{'='*60}")

    t0 = time.time()
    print(f"Loading {split}...", flush=True)
    ds = load_dataset(DATASET_NAME, split=split)
    print(f"  Loaded {len(ds)} rows in {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    print(f"Shuffling {split} (seed={SEED})...", flush=True)
    ds = ds.shuffle(seed=SEED)
    print(f"  Shuffled in {time.time() - t1:.1f}s", flush=True)

    t2 = time.time()
    print(f"Flattening indices (num_proc={NUM_PROC})...", flush=True)
    ds = ds.flatten_indices(num_proc=NUM_PROC)
    print(f"  Flattened in {time.time() - t2:.1f}s", flush=True)

    t3 = time.time()
    num_shards = SHARD_COUNTS[split]
    print(
        f"Pushing {split} to {NEW_DATASET_NAME} ({num_shards} shards, num_proc={NUM_PROC})...",
        flush=True,
    )
    ds.push_to_hub(
        NEW_DATASET_NAME,
        split=split,
        num_shards=num_shards,
    )
    print(f"  Pushed in {time.time() - t3:.1f}s", flush=True)
    print(f"Total for {split}: {time.time() - t0:.1f}s", flush=True)


def main():
    total_start = time.time()
    for split in SPLITS:
        process_split(split)
    print(f"\nAll splits done in {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
