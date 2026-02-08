"""Create train/val/test splits of monology/pile-uncopyrighted and push to HF Hub.

Splits are taken from the end of the dataset:
- test: last 100k rows
- val: preceding 1M rows
- train: everything else
"""

import time

from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi

HF_REPO = "danbraunai/pile-uncopyrighted"
VAL_SIZE = 1_000_000
TEST_SIZE = 100_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


README = f"""\
---
license: mit
dataset_info:
  source: monology/pile-uncopyrighted
---

# Pile Uncopyrighted (with train/val/test splits)

This is [monology/pile-uncopyrighted](https://huggingface.co/datasets/monology/pile-uncopyrighted)
re-split into train, val, and test sets.

The original dataset has a single "train" split. This version takes the last
{TEST_SIZE:,} rows as `test`, the preceding {VAL_SIZE:,} rows as `val`, and
everything else as `train`.

## Creation script

```python
from datasets import DatasetDict, load_dataset

ds = load_dataset("monology/pile-uncopyrighted", split="train")

n = len(ds)
VAL_SIZE = {VAL_SIZE:,}
TEST_SIZE = {TEST_SIZE:,}

result = DatasetDict({{
    "train": ds.select(range(n - VAL_SIZE - TEST_SIZE)),
    "val": ds.select(range(n - VAL_SIZE - TEST_SIZE, n - TEST_SIZE)),
    "test": ds.select(range(n - TEST_SIZE, n)),
}})

result.push_to_hub("{HF_REPO}")
```
"""

log("Loading dataset monology/pile-uncopyrighted")
ds = load_dataset("monology/pile-uncopyrighted", split="train")
log(f"Loaded {len(ds):,} rows")

n = len(ds)
result = DatasetDict(
    {
        "train": ds.select(range(n - VAL_SIZE - TEST_SIZE)),
        "val": ds.select(range(n - VAL_SIZE - TEST_SIZE, n - TEST_SIZE)),
        "test": ds.select(range(n - TEST_SIZE, n)),
    }
)
for split, split_ds in result.items():
    log(f"  {split}: {len(split_ds):,} rows")

for split, split_ds in result.items():
    log(f"Pushing {split} ({len(split_ds):,} rows)...")
    split_ds.push_to_hub(HF_REPO, split=split)
    log(f"Pushed {split}")

log("Uploading README")
api = HfApi()
api.upload_file(
    path_or_fileobj=README.encode(),
    path_in_repo="README.md",
    repo_id=HF_REPO,
    repo_type="dataset",
)
log("Done.")
