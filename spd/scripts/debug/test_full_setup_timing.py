"""Test script to measure full initialization timing.

This script timestamps every step from imports to init_distributed.
"""

import time
import os

START = time.time()

def log(msg: str) -> None:
    rank = os.environ.get("RANK", "?")
    elapsed = time.time() - START
    print(f"[RANK {rank}] [{elapsed:7.2f}s] {msg}", flush=True)

log("Python started - first line of script")

# Measure each import
log("Starting imports...")

t0 = time.time()
import json
from pathlib import Path
log(f"stdlib imports: {time.time() - t0:.2f}s")

t0 = time.time()
import fire
import wandb
log(f"fire+wandb: {time.time() - t0:.2f}s")

t0 = time.time()
from simple_stories_train.run_info import RunInfo as SSRunInfo
log(f"simple_stories_train: {time.time() - t0:.2f}s")

t0 = time.time()
from spd.configs import Config, LMTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.run_spd import optimize
log(f"spd modules: {time.time() - t0:.2f}s")

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
from spd.utils.general_utils import resolve_class, save_pre_run_info, set_seed
from spd.utils.run_utils import get_output_dir
from spd.utils.wandb_utils import init_wandb
log(f"spd.utils: {time.time() - t0:.2f}s")

log("All imports complete")
log("=" * 50)

# init_distributed
log("Calling init_distributed()...")
t0 = time.time()
dist_state = init_distributed()
log(f"init_distributed(): {time.time() - t0:.2f}s")

# Simulate config loading (without actual file)
log("=" * 50)
log(f"TOTAL TIME: {time.time() - START:.2f}s")

# Cleanup
if dist_state:
    from spd.utils.distributed_utils import cleanup_distributed
    cleanup_distributed()
