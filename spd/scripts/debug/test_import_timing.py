"""Test import timing to find slow imports."""

import time

print(f"[{time.time():.2f}] Script started", flush=True)

t0 = time.time()
import json
print(f"[{time.time():.2f}] import json: {time.time() - t0:.2f}s", flush=True)

t0 = time.time()
import os
print(f"[{time.time():.2f}] import os: {time.time() - t0:.2f}s", flush=True)

t0 = time.time()
from pathlib import Path
print(f"[{time.time():.2f}] from pathlib import Path: {time.time() - t0:.2f}s", flush=True)

t0 = time.time()
import fire
print(f"[{time.time():.2f}] import fire: {time.time() - t0:.2f}s", flush=True)

t0 = time.time()
import wandb
print(f"[{time.time():.2f}] import wandb: {time.time() - t0:.2f}s", flush=True)

t0 = time.time()
from simple_stories_train.run_info import RunInfo as SSRunInfo
print(f"[{time.time():.2f}] from simple_stories_train.run_info: {time.time() - t0:.2f}s", flush=True)

t0 = time.time()
from spd.configs import Config, LMTaskConfig
print(f"[{time.time():.2f}] from spd.configs: {time.time() - t0:.2f}s", flush=True)

t0 = time.time()
from spd.data import DatasetConfig, create_data_loader
print(f"[{time.time():.2f}] from spd.data: {time.time() - t0:.2f}s", flush=True)

t0 = time.time()
from spd.log import logger
print(f"[{time.time():.2f}] from spd.log: {time.time() - t0:.2f}s", flush=True)

t0 = time.time()
from spd.run_spd import optimize
print(f"[{time.time():.2f}] from spd.run_spd: {time.time() - t0:.2f}s", flush=True)

t0 = time.time()
from spd.utils.distributed_utils import (
    DistributedState,
    call_on_rank0_then_broadcast,
    ensure_cached_and_call,
    get_device,
    init_distributed,
    is_main_process,
    with_distributed_cleanup,
)
print(f"[{time.time():.2f}] from spd.utils.distributed_utils: {time.time() - t0:.2f}s", flush=True)

t0 = time.time()
from spd.utils.general_utils import resolve_class, save_pre_run_info, set_seed
print(f"[{time.time():.2f}] from spd.utils.general_utils: {time.time() - t0:.2f}s", flush=True)

t0 = time.time()
from spd.utils.run_utils import get_output_dir
print(f"[{time.time():.2f}] from spd.utils.run_utils: {time.time() - t0:.2f}s", flush=True)

t0 = time.time()
from spd.utils.wandb_utils import init_wandb
print(f"[{time.time():.2f}] from spd.utils.wandb_utils: {time.time() - t0:.2f}s", flush=True)

print(f"\n[{time.time():.2f}] All imports complete", flush=True)
