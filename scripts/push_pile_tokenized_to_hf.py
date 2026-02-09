"""Tokenize danbraunai/pile-uncopyrighted and push to HF Hub.

Uses the same tokenization approach as spd/data.py:tokenize_and_concatenate:
- Concatenates all text in each batch, separated by EOS tokens
- Tokenizes and reshapes into sequences of length N_CTX
- Produces a dataset with a single "input_ids" column

Tokenizer: EleutherAI/gpt-neox-20b
Sequence length: 513 (512 + 1 for next-token prediction)
"""

import time

from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from spd.data import tokenize_and_concatenate

HF_REPO = "danbraunai/pile-uncopyrighted-tok"
SOURCE_REPO = "danbraunai/pile-uncopyrighted"
TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"
N_CTX = 513
NUM_PROC = 160


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


README = f"""\
---
license: mit
dataset_info:
  source: {SOURCE_REPO}
  tokenizer: {TOKENIZER_NAME}
  sequence_length: {N_CTX}
---

# Pile Uncopyrighted (Tokenized)

This is [{SOURCE_REPO}](https://huggingface.co/datasets/{SOURCE_REPO})
tokenized with [`{TOKENIZER_NAME}`](https://huggingface.co/{TOKENIZER_NAME}).

Each row contains a single `input_ids` column with {N_CTX} token IDs.
Samples are concatenated with EOS tokens between them (following the approach in
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)), then
reshaped into fixed-length sequences.

## Creation script

```python
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

from spd.data import tokenize_and_concatenate

SOURCE_REPO = "{SOURCE_REPO}"
TOKENIZER_NAME = "{TOKENIZER_NAME}"
N_CTX = {N_CTX}

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

result = DatasetDict()
for split in ["train", "val", "test"]:
    ds = load_dataset(SOURCE_REPO, split=split)
    tokenized = tokenize_and_concatenate(
        ds,
        tokenizer,
        column_name="text",
        max_length=N_CTX,
        add_bos_token=False,
        num_proc=10,
        to_lower=False,
    )
    tokenized = tokenized.with_format(None)
    result[split] = tokenized

result.push_to_hub("{HF_REPO}")
```
"""

log(f"Loading tokenizer {TOKENIZER_NAME}")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

result = DatasetDict()
for split in ["train", "val", "test"]:
    log(f"Loading {SOURCE_REPO} split={split}...")
    ds = load_dataset(SOURCE_REPO, split=split)
    log(f"  Loaded {len(ds):,} rows")

    log(f"Tokenizing {split} (max_length={N_CTX})...")
    tokenized = tokenize_and_concatenate(
        ds,
        tokenizer,
        column_name="text",
        max_length=N_CTX,
        add_bos_token=False,
        num_proc=NUM_PROC,
        to_lower=False,
    )
    # Reset torch format so push_to_hub stores arrow-native data
    tokenized = tokenized.with_format(None)
    log(f"  {len(tokenized):,} sequences of length {N_CTX}")
    result[split] = tokenized

for split, split_ds in result.items():
    log(f"Pushing {split} ({len(split_ds):,} sequences)...")
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
