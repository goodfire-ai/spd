"""Test DDP consistency for SPD runs."""

import os
import pickle
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from spd.settings import REPO_ROOT


@pytest.mark.slow
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
                    "C": 3,
                    "batch_size": 2,
                    "eval_batch_size": 2,
                    "lr": 1e-2,
                    "lr_warmup_pct": 0.0,
                    "steps": 0,  # Only run step 0 (first iteration)
                    "n_eval_steps": 2,
                    "slow_eval_on_first_step": True,
                    "stochastic_recon_coeff": None,  # Otherwise we're non-deterministic with dp>1
                    "stochastic_recon_layerwise_coeff": None,  # Otherwise we're non-deterministic with dp>1
                    "wandb_project": None,
                    "ddp_backend": "gloo",
                    "seed": 42,
                    "train_log_freq": 1,
                    "eval_freq": 1,
                    "gradient_accumulation_steps": 1,
                    "n_examples_until_dead": 100,
                    "task_config": {
                        "task_name": "lm",
                        "max_seq_len": 5,
                        "buffer_size": 100,
                        "dataset_name": "SimpleStories/SimpleStories",
                        "column_name": "story",
                        "train_data_split": "train[:100]",
                        "eval_data_split": "test[:100]",
                    },
                }
            )

            config_path = tmpdir / "test_config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            # Step 2: Run with dp=1
            dp1_batch_file = tmpdir / "dp1_batches.pkl"
            dp1_metrics_file = tmpdir / "dp1_metrics.pkl"

            self._run_with_capture(
                config_path,
                n_processes=1,
                batch_file=dp1_batch_file,
                metrics_file=dp1_metrics_file,
                port=29501,
            )

            # Step 3: Run with dp=2
            dp2_batch_file = tmpdir / "dp2_batches.pkl"
            dp2_metrics_file = tmpdir / "dp2_metrics.pkl"

            self._run_with_capture(
                config_path,
                n_processes=2,
                batch_file=dp2_batch_file,
                metrics_file=dp2_metrics_file,
                port=29502,
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
        self,
        config_path: Path,
        n_processes: int,
        batch_file: Path,
        metrics_file: Path,
        port: int = 29500,
    ) -> None:
        """Run the experiment under mpirun while capturing batch/metric data."""
        script_path = REPO_ROOT / "tests" / "capture_spd_run.py"
        assert script_path.exists(), f"{script_path} not found"

        env = {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MASTER_PORT": str(port),  # Use unique port for each run
        }
        cmd = [
            "mpirun",
            "-np",
            str(n_processes),
            "python",
            str(script_path),
            str(config_path),
            str(batch_file),
            str(metrics_file),
        ]

        result = subprocess.run(
            cmd, env={**os.environ, **env}, capture_output=True, text=True, timeout=300
        )

        if result.returncode != 0:
            print(f"FULL STDOUT: {result.stdout}")
            print(f"FULL STDERR: {result.stderr}")
            raise RuntimeError(f"mpirun failed with code {result.returncode}")

    def _validate_batches(
        self, dp1_batches: list[dict[str, Any]], dp2_batches: list[dict[str, Any]]
    ) -> None:
        """Validate batch calls for dp1 (rank-0) versus dp2 (2 ranks).

        Checks that when concatenating the dp2 batches, we get the same result as the dp1 batch

        With steps=0, the training loop only runs step 0, so we expect:
        - dp=1: 1 batch total (step 0 on rank 0)
        - dp=2: 2 batches total (step 0 on rank 0 and rank 1)
        """

        # With steps=0, we only run step 0, so expect 1 batch for dp=1
        assert len(dp1_batches) == 1, (
            f"Expected 1 batch capture for dp=1 run (step 0 only), got {len(dp1_batches)}"
        )
        # For dp=2, expect 1 batch per rank (1 step × 2 ranks = 2 total)
        assert len(dp2_batches) == 2, (
            f"Expected 2 batch captures for dp=2 run (1 per rank), got {len(dp2_batches)}"
        )

        # There should only be one batch for dp=1
        dp1_batch = dp1_batches[0]

        dp1_data = torch.as_tensor(dp1_batch["data"])
        dp2_data = [torch.as_tensor(b["data"]) for b in dp2_batches]

        # Concatenate the dp=2 tensors in both possible orders because we aren't sure how it
        # will be split
        concat_1 = torch.cat([dp2_data[0], dp2_data[1]], dim=0)
        concat_2 = torch.cat([dp2_data[1], dp2_data[0]], dim=0)
        assert torch.equal(dp1_data, concat_1) or torch.equal(dp1_data, concat_2), (
            "First two batch elements of dp1 should match either of the two dp2 batches"
        )

        print(
            f"✓ Step 0 Batch: dp1 shape={dp1_batch['shape']}, "
            f"dp2 shapes={[b['shape'] for b in dp2_batches]}"
        )

    def _validate_metrics(
        self,
        dp1_metrics: list[dict[str, Any]],
        dp2_metrics: list[dict[str, Any]],
        tolerance: float = 1e-1,
    ) -> None:
        """Validate evaluation metrics across data-parallel settings.

        With steps=0 and eval_freq=1, evaluation only happens at step 0.
        We expect:
        - dp=1: 1 metric capture (eval at step 0 on rank 0)
        - dp=2: 2 metric captures (eval at step 0 on rank 0 and rank 1)
        """

        assert len(dp1_metrics) == 1, (
            f"Expected 1 metrics capture for dp=1 run (eval at step 0), got {len(dp1_metrics)}"
        )
        assert len(dp2_metrics) == 2, (
            f"Expected 2 metrics captures for dp=2 run (1 per rank), got {len(dp2_metrics)}"
        )

        # Validate metrics for step 0
        dp1_metric = dp1_metrics[0]

        # Compute per-key average of the two dp2 metric dicts
        for key, val_dp1 in dp1_metric["metrics"].items():
            val_dp2_avg = sum(m["metrics"][key] for m in dp2_metrics) / 2
            diff = abs(val_dp1 - val_dp2_avg)
            assert diff < tolerance, (
                f"Metric '{key}' differs: dp1={val_dp1:.6f}, "
                f"dp2_avg={val_dp2_avg:.6f}, diff={diff:.2e}"
            )

            print(f"✓ Metric '{key}': dp1={val_dp1:.6f}, dp2_avg={val_dp2_avg:.6f}")
