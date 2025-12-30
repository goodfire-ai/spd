"""Test script to verify fast setup with shared venv works correctly."""

import time
import os
import sys

START = time.time()

def log(msg: str) -> None:
    rank = os.environ.get("RANK", "?")
    elapsed = time.time() - START
    print(f"[RANK {rank}] [{elapsed:7.2f}s] {msg}", flush=True)

log("Python started")
log(f"Python executable: {sys.executable}")
log(f"Working directory: {os.getcwd()}")
log(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'not set')}")

# Test that imports work from the cloned workspace
log("Testing imports...")
t0 = time.time()

from spd.configs import Config
from spd.data import create_data_loader
from spd.utils.distributed_utils import init_distributed, cleanup_distributed, is_main_process

log(f"Imports completed in {time.time() - t0:.2f}s")

# Test distributed init
log("Testing init_distributed...")
t0 = time.time()
dist_state = init_distributed()
log(f"init_distributed completed in {time.time() - t0:.2f}s")

if is_main_process():
    log("=" * 50)
    log(f"SUCCESS! Total time: {time.time() - START:.2f}s")
    log("=" * 50)

cleanup_distributed()
