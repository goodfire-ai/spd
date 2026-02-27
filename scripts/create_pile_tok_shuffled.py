"""Create danbraunai/pile-uncopyrighted-tok-shuffled from monology/pile-uncopyrighted.

Unified script combining three stages that were originally run separately:
  1. Re-split: Load single "train" split, carve out val (1M rows) and test (100K rows)
  2. Tokenize: Tokenize with EleutherAI/gpt-neox-20b into 513-token sequences
  3. Shuffle & upload: Global shuffle (seed=42), flatten, push to HuggingFace Hub

Requirements: datasets, transformers, numpy, huggingface_hub (with write access to target repo)

Usage: python scripts/create_pile_tok_shuffled.py
"""

import time

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

SOURCE_REPO = "monology/pile-uncopyrighted"
TARGET_REPO = "danbraunai/pile-uncopyrighted-tok-shuffled"
TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"
N_CTX = 513
VAL_SIZE = 1_000_000
TEST_SIZE = 100_000
SHUFFLE_SEED = 42
TOKENIZE_NUM_PROC = 10
FLATTEN_NUM_PROC = 160
SHARD_COUNTS = {"train": 2021, "val": 12, "test": 2}


# ---------------------------------------------------------------------------
# Stage 1: Load and re-split
# ---------------------------------------------------------------------------


def load_and_split() -> DatasetDict:
    """Load monology/pile-uncopyrighted and split into train/val/test.

    Split boundaries (from the end of the dataset):
      - Last 100K rows  → test
      - Preceding 1M    → val
      - Everything else  → train
    """
    print("Stage 1: Loading source dataset...", flush=True)
    t0 = time.time()
    ds = load_dataset(SOURCE_REPO, split="train")
    n = len(ds)
    print(f"  Loaded {n:,} rows in {time.time() - t0:.1f}s", flush=True)

    assert n > VAL_SIZE + TEST_SIZE, f"Dataset too small: {n}"
    train_end = n - VAL_SIZE - TEST_SIZE

    print(
        f"  Splitting: train={train_end:,}, val={VAL_SIZE:,}, test={TEST_SIZE:,}",
        flush=True,
    )
    return DatasetDict(
        {
            "train": ds.select(range(train_end)),
            "val": ds.select(range(train_end, train_end + VAL_SIZE)),
            "test": ds.select(range(train_end + VAL_SIZE, n)),
        }
    )


# ---------------------------------------------------------------------------
# Stage 2: Tokenize
# ---------------------------------------------------------------------------


def tokenize_and_concatenate(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int,
    column_name: str = "text",
    num_proc: int = TOKENIZE_NUM_PROC,
) -> Dataset:
    """Tokenize text and reshape into fixed-length sequences.

    Joins documents with EOS tokens, tokenizes in parallel chunks, then reshapes
    into (num_sequences, max_length). Adapted from TransformerLens.
    """
    for key in dataset.features:
        if key != column_name:
            dataset = dataset.remove_columns(key)

    def tokenize_fn(
        examples: dict[str, list[str]],
    ) -> dict[str, np.ndarray]:
        full_text = tokenizer.eos_token.join(examples[column_name])

        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [full_text[i * chunk_length : (i + 1) * chunk_length] for i in range(num_chunks)]

        tokens = np.concatenate(
            [tokenizer.encode(chunk, add_special_tokens=False) for chunk in chunks]
        )

        num_batches = len(tokens) // max_length
        tokens = tokens[: max_length * num_batches].reshape((num_batches, max_length))
        return {"input_ids": tokens}

    return dataset.map(tokenize_fn, batched=True, remove_columns=[column_name], num_proc=num_proc)


# ---------------------------------------------------------------------------
# Stage 3: Shuffle and upload
# ---------------------------------------------------------------------------


def shuffle_and_upload(ds: Dataset, split: str) -> None:
    """Globally shuffle sequences and push to HuggingFace Hub."""
    t0 = time.time()
    print(f"  Shuffling (seed={SHUFFLE_SEED})...", flush=True)
    ds = ds.shuffle(seed=SHUFFLE_SEED)
    print(f"  Shuffled in {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    print(f"  Flattening indices (num_proc={FLATTEN_NUM_PROC})...", flush=True)
    ds = ds.flatten_indices(num_proc=FLATTEN_NUM_PROC)
    print(f"  Flattened in {time.time() - t1:.1f}s", flush=True)

    t2 = time.time()
    num_shards = SHARD_COUNTS[split]
    print(f"  Pushing to {TARGET_REPO} ({num_shards} shards)...", flush=True)
    ds.push_to_hub(TARGET_REPO, split=split, num_shards=num_shards)
    print(f"  Pushed in {time.time() - t2:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    total_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    splits = load_and_split()

    for split_name in ["train", "val", "test"]:
        print(f"\n{'=' * 60}", flush=True)
        print(f"Processing split: {split_name}", flush=True)
        print(f"{'=' * 60}", flush=True)

        t0 = time.time()
        print(f"Stage 2: Tokenizing (max_length={N_CTX})...", flush=True)
        tokenized = tokenize_and_concatenate(splits[split_name], tokenizer, max_length=N_CTX)
        print(
            f"  Tokenized {len(tokenized):,} sequences in {time.time() - t0:.1f}s",
            flush=True,
        )

        print("Stage 3: Shuffle and upload", flush=True)
        shuffle_and_upload(tokenized, split_name)

    print(f"\nAll done in {time.time() - total_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
