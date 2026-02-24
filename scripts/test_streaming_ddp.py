"""Verify that streaming+DDP gives different samples on each rank.

Usage: torchrun --nproc_per_node=2 scripts/test_streaming_ddp.py
"""

import os

import torch.distributed as dist

from spd.data import DatasetConfig, create_data_loader
from spd.utils.distributed_utils import DistributedState

dist.init_process_group(backend="gloo")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])

dist_state = DistributedState(
    rank=rank, world_size=world_size, local_rank=local_rank, backend="gloo"
)

config = DatasetConfig(
    name="danbraunai/pile-uncopyrighted-tok",
    hf_tokenizer_path="EleutherAI/gpt-neox-20b",
    split="train",
    n_ctx=512,
    is_tokenized=True,
    streaming=True,
    column_name="input_ids",
    shuffle_each_epoch=True,
    seed=None,
)

loader, _tok = create_data_loader(
    dataset_config=config,
    batch_size=4,
    buffer_size=10000,
    global_seed=0,
    dist_state=dist_state,
)

# Grab first 3 batches, print first 8 tokens of each sample
for i, batch in enumerate(loader):
    tokens = batch["input_ids"]  # (batch_size, seq_len)
    for j in range(tokens.shape[0]):
        print(f"[rank={rank}] batch={i} sample={j} first_8_tokens={tokens[j, :8].tolist()}")
    if i >= 2:
        break

dist.destroy_process_group()
