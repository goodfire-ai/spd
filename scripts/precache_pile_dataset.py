"""Pre-cache the Pile dataset: tokenize with maximum CPU parallelism.

Populates the shared HF datasets cache (HF_DATASETS_CACHE) so that future
training runs with streaming=False hit the cache and skip tokenization.

All pile_llama_simple_mlp configs share the same dataset parameters
(n_ctx=512, seed=0, tokenizer=gpt-neox-20b), so one cache run covers all of them.

Usage (request a full node for maximum parallelism):
    srun --cpus-per-task=192 --mem=0 --time=12:00:00 \
        python scripts/precache_pile_dataset.py

Or locally (will use available CPUs):
    python scripts/precache_pile_dataset.py

Requires HF_DATASETS_CACHE to be set (shared across users on the cluster).
"""

import os
import time

from datasets import load_dataset
from transformers import AutoTokenizer

from spd.data import tokenize_and_concatenate

# NUM_PROC = os.cpu_count() or 10
NUM_PROC = 160

# Must match create_data_loader args exactly (except num_proc) for cache compatibility
DATASET_NAME = "danbraunai/pile-uncopyrighted"
SPLITS = ["train", "val", "test"]
SEED = 0
N_CTX = 513  # train.py adds +1 for labels: 512 + 1
COLUMN_NAME = "text"
TOKENIZER_PATH = "EleutherAI/gpt-neox-20b"


def main():
    cache_dir = os.environ.get("HF_DATASETS_CACHE")
    assert cache_dir, "HF_DATASETS_CACHE must be set (shared cluster cache)"
    print(f"CPUs available: {NUM_PROC}")
    print(f"HF_DATASETS_CACHE: {cache_dir}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    to_lower = "SimpleStories" in DATASET_NAME  # False for Pile

    for split in SPLITS:
        print(f"{'=' * 60}")
        print(f"Processing split: {split}")
        print(f"{'=' * 60}")
        print()

        print(f"[1/3] Downloading {DATASET_NAME} split={split}...")
        t0 = time.time()
        dataset = load_dataset(
            DATASET_NAME,
            streaming=False,
            split=split,
            trust_remote_code=False,
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.0f}s. Dataset has {len(dataset)} examples.")
        print(f"  Fingerprint: {dataset._fingerprint}")
        print()

        print(f"[2/3] Shuffling with seed={SEED}...")
        t0 = time.time()
        dataset = dataset.shuffle(seed=SEED)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.0f}s.")
        print(f"  Fingerprint after shuffle: {dataset._fingerprint}")
        print()

        # num_proc doesn't affect the .map() cache fingerprint, so using more CPUs here
        # produces a cache that's compatible with create_data_loader's default num_proc=10
        print(f"[3/3] Tokenizing with num_proc={NUM_PROC} (max_length={N_CTX})...")
        t0 = time.time()
        tokenized = tokenize_and_concatenate(
            dataset,
            tokenizer,
            max_length=N_CTX,
            column_name=COLUMN_NAME,
            add_bos_token=False,
            num_proc=NUM_PROC,
            to_lower=to_lower,
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.0f}s. Produced {len(tokenized)} sequences of length {N_CTX}.")
        print(f"  Fingerprint: {tokenized._fingerprint}")
        print()

    print("All splits cached successfully!")


if __name__ == "__main__":
    main()
