"""Test import timing in distributed setting."""

import time
import os

START = time.time()

def log(msg: str) -> None:
    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", "?")
    print(f"[RANK {rank} LOCAL {local_rank}] [{time.time() - START:7.2f}s] {msg}", flush=True)

log("Script started - before any imports")

t0 = time.time()
import json
log(f"import json: {time.time() - t0:.2f}s")

t0 = time.time()
from pathlib import Path
log(f"from pathlib import Path: {time.time() - t0:.2f}s")

t0 = time.time()
import fire
log(f"import fire: {time.time() - t0:.2f}s")

t0 = time.time()
import wandb
log(f"import wandb: {time.time() - t0:.2f}s")

t0 = time.time()
from simple_stories_train.run_info import RunInfo as SSRunInfo
log(f"from simple_stories_train.run_info: {time.time() - t0:.2f}s")

t0 = time.time()
from spd.configs import Config, LMTaskConfig
log(f"from spd.configs: {time.time() - t0:.2f}s")

t0 = time.time()
from spd.data import DatasetConfig, create_data_loader
log(f"from spd.data: {time.time() - t0:.2f}s")

t0 = time.time()
from spd.log import logger
log(f"from spd.log: {time.time() - t0:.2f}s")

t0 = time.time()
from spd.run_spd import optimize
log(f"from spd.run_spd: {time.time() - t0:.2f}s")

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
log(f"from spd.utils.distributed_utils: {time.time() - t0:.2f}s")

log("All imports complete")

# Now test init_distributed
log("=" * 50)
log("Calling init_distributed()...")
t0 = time.time()
dist_state = init_distributed()
log(f"init_distributed() completed in {time.time() - t0:.2f}s")

log("=" * 50)
log(f"Total time: {time.time() - START:.2f}s")
