"""Test DDP consistency for SPD runs."""

import os
import pickle
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml


class TestDistributedSPDConsistency:
    """Test that DDP runs produce consistent results."""

    def test_distributed_spd_consistency(self):
        """Test that DDP with 1 and 2 processes produces consistent results."""

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Step 1: Create modified config
            with open("spd/experiments/lm/ss_mlp_config.yaml") as f:
                config = yaml.safe_load(f)

            config.update(
                {
                    "batch_size": 4,
                    "eval_batch_size": 4,
                    "steps": 1,
                    "n_eval_steps": 2,
                    "slow_eval_on_first_step": True,
                    "wandb_project": None,
                    "ddp_backend": "gloo",
                    "seed": 42,
                    "train_log_freq": 1,
                    "eval_freq": 1,
                    "gradient_accumulation_steps": 1,
                    "n_examples_until_dead": 100,
                }
            )

            config_path = tmpdir / "test_config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            # Step 2: Run with dp=1
            dp1_batch_file = tmpdir / "dp1_batches.pkl"
            dp1_metrics_file = tmpdir / "dp1_metrics.pkl"

            self._run_with_capture(
                config_path, n_processes=1, batch_file=dp1_batch_file, metrics_file=dp1_metrics_file
            )

            # Step 3: Run with dp=2
            dp2_batch_file = tmpdir / "dp2_batches.pkl"
            dp2_metrics_file = tmpdir / "dp2_metrics.pkl"

            self._run_with_capture(
                config_path, n_processes=2, batch_file=dp2_batch_file, metrics_file=dp2_metrics_file
            )

            # Step 4: Load and validate results
            # For dp=1, just load the rank 0 file
            with open(dp1_batch_file.parent / "dp1_batches_rank0.pkl", "rb") as f:
                dp1_batches = pickle.load(f)
            with open(dp1_metrics_file.parent / "dp1_metrics_rank0.pkl", "rb") as f:
                dp1_metrics = pickle.load(f)

            # For dp=2, load and combine both rank files
            dp2_batches = []
            dp2_metrics = []
            for rank in range(2):
                batch_file_rank = dp2_batch_file.parent / f"dp2_batches_rank{rank}.pkl"
                metrics_file_rank = dp2_metrics_file.parent / f"dp2_metrics_rank{rank}.pkl"

                if not batch_file_rank.exists():
                    print(f"Warning: {batch_file_rank} does not exist")
                    # List what files do exist
                    print(f"Files in {dp2_batch_file.parent}:")
                    for f in dp2_batch_file.parent.glob("*.pkl"):
                        print(f"  {f.name}")
                    raise FileNotFoundError(f"Expected file {batch_file_rank} not found")

                with open(batch_file_rank, "rb") as f:
                    dp2_batches.extend(pickle.load(f))
                with open(metrics_file_rank, "rb") as f:
                    dp2_metrics.extend(pickle.load(f))

            # Validate batch consistency
            self._validate_batches(dp1_batches, dp2_batches)

            # Only validate metrics if we have any
            if dp1_metrics and dp2_metrics:
                self._validate_metrics(dp1_metrics, dp2_metrics)

    def _run_with_capture(
        self, config_path: Path, n_processes: int, batch_file: Path, metrics_file: Path
    ) -> None:
        """Run the script with mpirun and capture data via monkey-patching."""

        # Create a wrapper script that patches and runs
        wrapper_script = f"""
import sys
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch

# Storage for captured data
captured_batches = []
captured_metrics = []

# Import the real functions before patching
from spd.utils.general_utils import extract_batch_data as real_extract_batch_data
from spd.eval import evaluate as real_evaluate

def capture_batch(batch_item):
    from spd.utils.distributed_utils import get_rank
    # Call the real function
    tensor = real_extract_batch_data(batch_item)
    captured_batches.append({{
        'rank': get_rank(),
        'shape': tuple(tensor.shape),
        'data': tensor.cpu().numpy().copy()
    }})
    return tensor  # Return the tensor, not the batch_item

def capture_metrics(*args, **kwargs):
    from spd.utils.distributed_utils import get_rank
    # Call the real function
    result = real_evaluate(*args, **kwargs)
    captured_metrics.append({{
        'rank': get_rank(),
        'metrics': {{k: v for k, v in result.items() if isinstance(v, (int, float))}}
    }})
    return result

# Storage for rank
my_rank = [None]  # Use list to allow modification in nested function

def save_rank_wrapper(original_cleanup):
    def wrapper():
        # Save rank before cleanup
        from spd.utils.distributed_utils import get_rank
        my_rank[0] = get_rank()
        # Call original cleanup
        return original_cleanup()
    return wrapper

# Patch cleanup_distributed to save rank before it's reset
from spd.utils.distributed_utils import cleanup_distributed as orig_cleanup

# The main function will handle distributed initialization
# Apply patches
with patch('spd.utils.distributed_utils.cleanup_distributed', side_effect=save_rank_wrapper(orig_cleanup)):
    with patch('spd.utils.general_utils.extract_batch_data', side_effect=capture_batch):
        with patch('spd.run_spd.extract_batch_data', side_effect=capture_batch):
            with patch('spd.eval.evaluate', side_effect=capture_metrics):
                # Import and run main (it will initialize distributed internally)
                from spd.experiments.lm.lm_decomposition import main
                main("{config_path}")

# Save captured data using the saved rank
actual_rank = my_rank[0] if my_rank[0] is not None else 0

# Simple approach: each rank saves its own file
batch_file_rank = "{batch_file}".replace('.pkl', f'_rank{{actual_rank}}.pkl')
metrics_file_rank = "{metrics_file}".replace('.pkl', f'_rank{{actual_rank}}.pkl')

with open(batch_file_rank, 'wb') as f:
    pickle.dump(captured_batches, f)
with open(metrics_file_rank, 'wb') as f:
    pickle.dump(captured_metrics, f)
"""

        # Write wrapper script
        wrapper_path = config_path.parent / f"wrapper_{n_processes}.py"
        with open(wrapper_path, "w") as f:
            f.write(wrapper_script)

        # Run with mpirun
        env = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
        cmd = ["mpirun", "-np", str(n_processes), "python", str(wrapper_path)]

        result = subprocess.run(
            cmd, env={**os.environ, **env}, capture_output=True, text=True, timeout=300
        )

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            raise RuntimeError(f"mpirun failed with code {result.returncode}")

    def _validate_batches(
        self, dp1_batches: list[dict[str, Any]], dp2_batches: list[dict[str, Any]]
    ) -> None:
        """Validate that batches are split correctly."""
        # Group dp2 batches by call order
        dp2_by_call = {}
        for i, batch in enumerate(dp2_batches):
            # Count how many calls came before this one from the same rank
            call_idx = len(
                [b for j, b in enumerate(dp2_batches) if b["rank"] == batch["rank"] and j < i]
            )
            if call_idx not in dp2_by_call:
                dp2_by_call[call_idx] = []
            dp2_by_call[call_idx].append(batch)

        # Check each batch call
        for i, dp1_batch in enumerate(dp1_batches):
            dp2_batch_pair = dp2_by_call.get(i, [])
            assert len(dp2_batch_pair) == 2, (
                f"Expected 2 batches for call {i}, got {len(dp2_batch_pair)}"
            )

            # Check shapes
            for dp2_batch in dp2_batch_pair:
                assert dp2_batch["shape"][0] == dp1_batch["shape"][0] // 2
                assert dp2_batch["shape"][1:] == dp1_batch["shape"][1:]

            print(
                f"✓ Batch {i}: dp1 shape={dp1_batch['shape']}, "
                f"dp2 shapes={[b['shape'] for b in dp2_batch_pair]}"
            )

    def _validate_metrics(
        self,
        dp1_metrics: list[dict[str, Any]],
        dp2_metrics: list[dict[str, Any]],
        tolerance: float = 1e-4,
    ) -> None:
        """Validate that metrics are consistent."""
        # Group dp2 metrics by eval call
        dp2_by_call = {}
        for i, metric in enumerate(dp2_metrics):
            # Count how many eval calls came before this one from the same rank
            call_idx = len(
                [m for j, m in enumerate(dp2_metrics) if m["rank"] == metric["rank"] and j < i]
            )
            if call_idx not in dp2_by_call:
                dp2_by_call[call_idx] = []
            dp2_by_call[call_idx].append(metric)

        # Check each eval call
        for i, dp1_metric in enumerate(dp1_metrics):
            dp2_metric_pair = dp2_by_call.get(i, [])
            assert len(dp2_metric_pair) == 2, f"Expected 2 metrics for call {i}"

            # Average dp2 metrics
            for key in dp1_metric["metrics"]:
                val_dp1 = dp1_metric["metrics"][key]
                val_dp2_avg = sum(m["metrics"][key] for m in dp2_metric_pair) / 2

                diff = abs(val_dp1 - val_dp2_avg)
                assert diff < tolerance, (
                    f"Metric '{key}' differs: dp1={val_dp1:.6f}, "
                    f"dp2_avg={val_dp2_avg:.6f}, diff={diff:.2e}"
                )

                print(f"✓ Metric '{key}': dp1={val_dp1:.6f}, dp2_avg={val_dp2_avg:.6f}")
