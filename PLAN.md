# Plan: Index-based train/val split for streaming datasets

## Context

`monology/pile-uncopyrighted` has only a `train` split. When streaming, we can't use HuggingFace slice notation (`train[:N]`, `train[-N:]`) to create non-overlapping train/val sets. We need a deterministic, zero-overlap split mechanism that works with streaming.

**Approach**: Filter by example index modulo N. Every Nth example is val, the rest is train. This is already a pattern used for DDP sharding in `spd/data.py:207-210`.

## Changes

### 1. `spd/data.py` — Add `index_split` to `DatasetConfig` and implement filtering

Add field to `DatasetConfig`:
```python
index_split: tuple[int, Literal["train", "val"]] | None = None
"""Deterministic index-based train/val split. (N, role) keeps every Nth example
for val and the rest for train. E.g. (100, "train") skips every 100th example,
(100, "val") keeps only every 100th."""
```

In `create_data_loader()`, add filtering right after `load_dataset()` (line ~191), before the streaming/shuffle branching:

```python
if dataset_config.index_split is not None:
    n, role = dataset_config.index_split
    if role == "val":
        dataset = dataset.filter(lambda _ex, idx: idx % n == 0, with_indices=True)
    else:
        dataset = dataset.filter(lambda _ex, idx: idx % n != 0, with_indices=True)
```

### 2. `spd/configs.py` — Add `val_index_split_n` to `LMTaskConfig`

```python
val_index_split_n: int | None = None
"""If set, split train/val by index modulus instead of by HF split name.
Every Nth example is val data."""
```

### 3. `spd/experiments/lm/lm_decomposition.py` — Plumb `index_split` into DatasetConfig construction

When constructing train DatasetConfig (~line 140):
```python
index_split=(task_config.val_index_split_n, "train") if task_config.val_index_split_n else None
```

When constructing eval DatasetConfig (~line 155):
```python
index_split=(task_config.val_index_split_n, "val") if task_config.val_index_split_n else None
```

### 4. `spd/data.py` `train_loader_and_tokenizer()` — Same plumbing

Add `index_split` to the DatasetConfig construction (~line 295):
```python
index_split=(task_config.val_index_split_n, "train") if task_config.val_index_split_n else None
```

### 5. Pile pretrain configs — Use `index_split` for streaming configs

For configs using `streaming: true` (currently `pile_llama_simple_mlp-4L-768.yaml`):

```yaml
train_dataset_config:
  name: monology/pile-uncopyrighted
  streaming: true
  split: train
  index_split: [100, train]
  ...

val_dataset_config:
  name: monology/pile-uncopyrighted
  streaming: true
  split: train
  index_split: [100, val]
  ...
```

Non-streaming configs that use slice notation (`train[:N]`, `train[-N:]`) are left as-is since they already have non-overlapping splits.

## What doesn't change

- Non-streaming configs with HF slice notation — these already work
- SimpleStories configs — these have a real `test` split
- DDP sharding logic — orthogonal (happens after this filter, on the already-split data)
- `tokenize_and_concatenate` — filtering happens before tokenization

## Verification

1. `make check` — type checking and linting
2. `make test` — existing tests pass
3. Manual: verify that a streaming Pile config with `index_split` produces different data for train vs val (e.g. check first few batch indices or log a sample)
