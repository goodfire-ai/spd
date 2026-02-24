"""Verify that streaming+DDP gives different samples on each rank.

Usage: torchrun --nproc_per_node=2 scripts/test_streaming_ddp.py
"""

import hashlib
import os
import sys

import torch.distributed as dist
from datasets import load_dataset

dist.init_process_group(backend="gloo")
rank = dist.get_rank()
world_size = dist.get_world_size()

ds = load_dataset("danbraunai/pile-uncopyrighted-tok", streaming=True, split="train")
num_shards = getattr(ds, "num_shards", None)
print(f"[rank={rank}] num_shards={num_shards}, world_size={world_size}", flush=True)

# Shard like create_data_loader does
assert isinstance(num_shards, int) and num_shards >= world_size
ds = ds.shard(num_shards=world_size, index=rank)

# Small shuffle buffer to start quickly
ds = ds.shuffle(seed=0, buffer_size=100)

# Collect first 20 samples' token hashes
hashes = []
for i, sample in enumerate(ds):
    tokens = sample["input_ids"][:16]
    h = hashlib.md5(str(tokens).encode()).hexdigest()[:8]
    hashes.append(h)
    print(f"[rank={rank}] sample={i} first_16={tokens}", flush=True)
    if i >= 9:
        break

# Gather and compare
all_hashes = [None, None]
dist.all_gather_object(all_hashes, hashes)

if rank == 0:
    set0 = set(all_hashes[0])
    set1 = set(all_hashes[1])
    overlap = set0 & set1
    print(f"\n=== RESULTS ===", flush=True)
    print(f"Rank 0 hashes: {all_hashes[0]}", flush=True)
    print(f"Rank 1 hashes: {all_hashes[1]}", flush=True)
    print(f"Overlap: {len(overlap)} / {len(set0)} samples", flush=True)
    if len(overlap) == 0:
        print("PASS: No overlapping samples between ranks", flush=True)
    else:
        print(f"FAIL: {len(overlap)} overlapping samples!", flush=True)
        sys.exit(1)

dist.destroy_process_group()
