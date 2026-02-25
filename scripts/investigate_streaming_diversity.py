"""Investigate streaming dataset diversity under DDP sharding.

Compares dp=1 vs dp=8 batch diversity for pile_llama_simple_mlp-4L config.
"""

import itertools
import sys
import time
from collections import Counter

from datasets import load_dataset
from transformers import AutoTokenizer

DATASET_NAME = "danbraunai/pile-uncopyrighted-tok-shuffled"
TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"
SEED = 0
BUFFER_SIZE = 1000
MAX_SEQ_LEN = 512
BATCH_SIZE = 64
WORLD_SIZE = 8
COLUMN_NAME = "input_ids"


def take_sequences(dataset_iter, n: int) -> list[list[int]]:
    """Take n sequences of length MAX_SEQ_LEN from the dataset iterator."""
    seqs = []
    for example in dataset_iter:
        tokens = example[COLUMN_NAME][:MAX_SEQ_LEN]
        if len(tokens) == MAX_SEQ_LEN:
            seqs.append(tokens)
            if len(seqs) == n:
                break
    return seqs


def jaccard_similarity(seq_a: list[int], seq_b: list[int]) -> float:
    set_a = set(seq_a)
    set_b = set(seq_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def avg_pairwise_jaccard(sequences: list[list[int]]) -> float:
    n = len(sequences)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += jaccard_similarity(sequences[i], sequences[j])
            count += 1
    return total / count


def unique_unigram_count(sequences: list[list[int]]) -> int:
    all_tokens: set[int] = set()
    for seq in sequences:
        all_tokens.update(seq)
    return len(all_tokens)


def print_sequences(sequences: list[list[int]], tokenizer, label: str, max_chars: int = 120):
    print(f"\n{'=' * 80}")
    print(f"  {label}")
    print(f"{'=' * 80}")
    for i, seq in enumerate(sequences):
        text = tokenizer.decode(seq[:60], skip_special_tokens=True)
        text = text.replace("\n", " ")[:max_chars]
        print(f"  [{i:2d}] {text}")


def print_diversity_metrics(sequences: list[list[int]], label: str):
    unigrams = unique_unigram_count(sequences)
    avg_jaccard = avg_pairwise_jaccard(sequences)
    print(f"\n--- {label} ---")
    print(f"  Unique unigrams: {unigrams}")
    print(f"  Avg pairwise Jaccard similarity: {avg_jaccard:.4f}")
    return unigrams, avg_jaccard


def load_base_dataset():
    return load_dataset(
        DATASET_NAME,
        streaming=True,
        split="train",
        trust_remote_code=False,
    )


def timed(label: str):
    """Simple timer context manager."""

    class Timer:
        def __enter__(self):
            self.start = time.time()
            print(f"  [{label}] starting...", end="", flush=True)
            return self

        def __exit__(self, *_):
            elapsed = time.time() - self.start
            print(f" done ({elapsed:.1f}s)", flush=True)

    return Timer()


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # =========================================================================
    # 1. dp=1: no sharding, just shuffle
    # =========================================================================
    print("\n" + "#" * 80)
    print("# SECTION 1: dp=1 vs dp=8 batch diversity")
    print("#" * 80)

    with timed("dp=1 load"):
        ds_dp1 = load_base_dataset()
        ds_dp1 = ds_dp1.shuffle(seed=SEED, buffer_size=BUFFER_SIZE)
        seqs_dp1 = take_sequences(iter(ds_dp1), BATCH_SIZE)
    print_sequences(seqs_dp1, tokenizer, "dp=1 batch (64 sequences, no sharding)")

    # =========================================================================
    # 2. dp=8: shard then shuffle, 8 seqs per rank
    # =========================================================================
    seqs_per_rank: dict[int, list[list[int]]] = {}
    for rank in range(WORLD_SIZE):
        with timed(f"dp=8 rank {rank}"):
            ds = load_base_dataset()
            ds = ds.shard(num_shards=WORLD_SIZE, index=rank)
            ds = ds.shuffle(seed=SEED, buffer_size=BUFFER_SIZE)
            seqs_per_rank[rank] = take_sequences(iter(ds), BATCH_SIZE // WORLD_SIZE)

    seqs_dp8 = list(itertools.chain.from_iterable(seqs_per_rank[r] for r in range(WORLD_SIZE)))
    print_sequences(seqs_dp8, tokenizer, "dp=8 batch (8 seqs x 8 ranks = 64 total)")

    # =========================================================================
    # 3. Diversity metrics comparison
    # =========================================================================
    print("\n" + "#" * 80)
    print("# SECTION 2: Diversity metrics")
    print("#" * 80)

    print_diversity_metrics(seqs_dp1, "dp=1 (64 seqs, no sharding)")
    print_diversity_metrics(seqs_dp8, "dp=8 (64 seqs, 8 per rank)")

    # =========================================================================
    # 4. Within-rank clustering (dp=8 only)
    # =========================================================================
    print("\n" + "#" * 80)
    print("# SECTION 3: Within-rank clustering (dp=8)")
    print("#" * 80)

    overall_jaccard = avg_pairwise_jaccard(seqs_dp8)
    print(f"\n  Overall batch Jaccard: {overall_jaccard:.4f}")
    print("  Per-rank Jaccard (higher = more within-rank clustering):")
    for rank in range(WORLD_SIZE):
        rank_jaccard = avg_pairwise_jaccard(seqs_per_rank[rank])
        print(f"    Rank {rank}: {rank_jaccard:.4f}")

    # =========================================================================
    # 5. Long-range clustering check - use dataset.skip() for efficiency
    # =========================================================================
    print("\n" + "#" * 80)
    print("# SECTION 4: Long-range clustering (deeper into stream)")
    print("#" * 80)

    # Use .skip() which is much faster than iterating for HF IterableDataset.
    # Reduced positions to keep runtime reasonable.
    for position in [500, 2000]:
        print(f"\n--- Position ~{position} (skip {position} examples, then take 8) ---")
        for rank in [0, 4, 7]:
            with timed(f"rank {rank} @ {position}"):
                ds = load_base_dataset()
                ds = ds.shard(num_shards=WORLD_SIZE, index=rank)
                ds = ds.shuffle(seed=SEED, buffer_size=BUFFER_SIZE)
                ds = ds.skip(position)
                deep_seqs = take_sequences(iter(ds), 8)
            rank_jaccard = avg_pairwise_jaccard(deep_seqs)
            print(f"  Rank {rank} @ position {position}: Jaccard={rank_jaccard:.4f}")
            for i, seq in enumerate(deep_seqs[:4]):
                text = tokenizer.decode(seq[:60], skip_special_tokens=True)
                text = text.replace("\n", " ")[:120]
                print(f"    [{i}] {text}")

    # =========================================================================
    # 6. Larger within-rank sample for more robust clustering signal
    # =========================================================================
    print("\n" + "#" * 80)
    print("# SECTION 5: Larger within-rank sample (32 seqs per rank)")
    print("#" * 80)

    large_seqs_per_rank: dict[int, list[list[int]]] = {}
    for rank in range(WORLD_SIZE):
        with timed(f"rank {rank} large sample"):
            ds = load_base_dataset()
            ds = ds.shard(num_shards=WORLD_SIZE, index=rank)
            ds = ds.shuffle(seed=SEED, buffer_size=BUFFER_SIZE)
            large_seqs_per_rank[rank] = take_sequences(iter(ds), 32)

    large_seqs_all = list(
        itertools.chain.from_iterable(large_seqs_per_rank[r] for r in range(WORLD_SIZE))
    )

    print_diversity_metrics(large_seqs_all, "dp=8 (32 seqs x 8 ranks = 256 total)")

    overall_jaccard_large = avg_pairwise_jaccard(large_seqs_all)
    print(f"\n  Overall batch Jaccard (256 seqs): {overall_jaccard_large:.4f}")
    print("  Per-rank Jaccard (32 seqs each):")
    for rank in range(WORLD_SIZE):
        rank_jaccard = avg_pairwise_jaccard(large_seqs_per_rank[rank])
        print(f"    Rank {rank}: {rank_jaccard:.4f}")

    # =========================================================================
    # 7. Token frequency overlap between ranks
    # =========================================================================
    print("\n" + "#" * 80)
    print("# SECTION 6: Token frequency analysis across ranks")
    print("#" * 80)

    rank_token_sets: dict[int, set[int]] = {}
    for rank in range(WORLD_SIZE):
        rank_tokens: set[int] = set()
        for seq in seqs_per_rank[rank]:
            rank_tokens.update(seq)
        rank_token_sets[rank] = rank_tokens

    print("\n  Pairwise rank token overlap (Jaccard on vocabulary used):")
    for r1 in range(WORLD_SIZE):
        for r2 in range(r1 + 1, WORLD_SIZE):
            overlap = len(rank_token_sets[r1] & rank_token_sets[r2])
            union = len(rank_token_sets[r1] | rank_token_sets[r2])
            print(f"    Rank {r1} vs Rank {r2}: {overlap}/{union} = {overlap / union:.3f}")

    print("\n  Top-10 non-common tokens per rank (excluding tokens shared by all ranks):")
    shared_tokens = set.intersection(*rank_token_sets.values())
    for rank in range(WORLD_SIZE):
        rank_unique = rank_token_sets[rank] - shared_tokens
        counter = Counter()
        for seq in seqs_per_rank[rank]:
            for tok in seq:
                if tok in rank_unique:
                    counter[tok] += 1
        top_tokens = counter.most_common(10)
        decoded = [tokenizer.decode([t]) for t, _ in top_tokens]
        print(f"    Rank {rank}: {decoded}")

    print("\n" + "#" * 80)
    print("# DONE")
    print("#" * 80)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
