"""Break down streaming overhead into download vs tokenization."""

import time

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

N_BATCHES = 70
BATCH_SIZE = 64
N_CTX = 513  # 512 + 1 for labels

DATASET_NAME = "monology/pile-uncopyrighted"
TOKENIZER_PATH = "EleutherAI/gpt-neox-20b"


def time_raw_streaming_download():
    """Time just downloading raw text from streaming dataset (no tokenization)."""
    dataset = load_dataset(DATASET_NAME, streaming=True, split="train", trust_remote_code=False)
    dataset = dataset.shuffle(seed=0, buffer_size=1000)
    ds_iter = iter(dataset)

    # Fetch enough raw examples to fill N_BATCHES * BATCH_SIZE * N_CTX tokens
    # Each example has variable length text, so we need to fetch more than enough
    target_examples = N_BATCHES * BATCH_SIZE * 5  # generous overcount
    texts = []
    t0 = time.perf_counter()
    for i, example in enumerate(ds_iter):
        texts.append(example["text"])
        if i + 1 >= target_examples:
            break
    t1 = time.perf_counter()
    total_chars = sum(len(t) for t in texts)
    return t1 - t0, len(texts), total_chars, texts


def time_tokenization(texts: list[str]):
    """Time tokenizing pre-downloaded text (no network I/O)."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    seq_len = N_CTX

    # Simulate what tokenize_and_concatenate does
    t0 = time.perf_counter()
    full_text = tokenizer.eos_token.join(texts)

    num_chunks = 20
    chunk_length = (len(full_text) - 1) // num_chunks + 1
    chunks = [full_text[i * chunk_length : (i + 1) * chunk_length] for i in range(num_chunks)]
    tokens = [tokenizer.encode(chunk, add_special_tokens=False) for chunk in chunks]
    tokens = np.concatenate(tokens)

    num_tokens = len(tokens)
    num_batches = num_tokens // seq_len
    tokens = tokens[: seq_len * num_batches]
    tokens = tokens.reshape((num_batches, seq_len))
    t1 = time.perf_counter()

    return t1 - t0, num_batches, num_tokens


def time_combined_streaming():
    """Time the full streaming pipeline (download + tokenize) via the dataloader."""
    from spd.data import DatasetConfig, create_data_loader

    dataset_config = DatasetConfig(
        name=DATASET_NAME,
        is_tokenized=False,
        hf_tokenizer_path=TOKENIZER_PATH,
        streaming=True,
        split="train",
        n_ctx=N_CTX,
        seed=0,
        column_name="text",
    )

    train_loader, _ = create_data_loader(
        dataset_config=dataset_config,
        batch_size=BATCH_SIZE,
        buffer_size=1000,
        global_seed=0,
    )
    train_iter = iter(train_loader)

    t0 = time.perf_counter()
    for _ in range(N_BATCHES):
        _ = next(train_iter)
    t1 = time.perf_counter()

    return t1 - t0


def main():
    print(f"Config: dataset={DATASET_NAME}, batch_size={BATCH_SIZE}, n_ctx={N_CTX}")
    print(f"Fetching {N_BATCHES} batches worth of data")
    print()

    # --- Test 1: Raw download speed ---
    print("=" * 60)
    print("Test 1: Raw streaming download (no tokenization)")
    print("=" * 60)
    dl_time, n_examples, total_chars, texts = time_raw_streaming_download()
    print(f"  Downloaded {n_examples} examples ({total_chars / 1e6:.1f}M chars) in {dl_time:.2f}s")
    print(
        f"  Speed: {n_examples / dl_time:.0f} examples/s, {total_chars / dl_time / 1e6:.1f}M chars/s"
    )
    print()

    # --- Test 2: Tokenization speed ---
    print("=" * 60)
    print("Test 2: Tokenization of pre-downloaded text")
    print("=" * 60)
    tok_time, n_batches, n_tokens = time_tokenization(texts)
    print(f"  Tokenized into {n_batches} batches ({n_tokens / 1e6:.1f}M tokens) in {tok_time:.2f}s")
    print(f"  Speed: {n_tokens / tok_time / 1e6:.1f}M tokens/s")
    print()

    # --- Test 3: Combined streaming pipeline ---
    print("=" * 60)
    print("Test 3: Full streaming pipeline (download + tokenize)")
    print("=" * 60)
    combined_time = time_combined_streaming()
    print(
        f"  {N_BATCHES} batches in {combined_time:.2f}s ({combined_time / N_BATCHES * 1000:.1f}ms/batch)"
    )
    print()

    # --- Summary ---
    print("=" * 60)
    print("BREAKDOWN SUMMARY")
    print("=" * 60)
    print(f"  Raw download:           {dl_time:>8.2f}s")
    print(f"  Tokenization:           {tok_time:>8.2f}s")
    print(f"  Download + Tokenize:    {dl_time + tok_time:>8.2f}s (sum)")
    print(f"  Full streaming pipeline:{combined_time:>8.2f}s (actual)")
    print()

    dl_frac = dl_time / (dl_time + tok_time) * 100
    tok_frac = tok_time / (dl_time + tok_time) * 100
    print(f"  Download fraction:      {dl_frac:.1f}%")
    print(f"  Tokenization fraction:  {tok_frac:.1f}%")


if __name__ == "__main__":
    main()
