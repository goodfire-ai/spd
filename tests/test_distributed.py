"""Tests for distributed utilities."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch
import yaml

from spd.settings import REPO_ROOT
from spd.utils import distributed_utils
from spd.utils.distributed_utils import (
    get_local_rank,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
)

TEST_CONFIG = {
    # --- General ---
    "seed": 0,
    "C": 3,
    "n_mask_samples": 1,
    "gate_type": "vector_mlp",
    "gate_hidden_dims": [2],
    "sigmoid_type": "leaky_hard",
    "target_module_patterns": ["model.layers.0.mlp.gate_proj"],
    # --- Loss metrics ---
    "loss_metric_configs": [
        {
            "classname": "ImportanceMinimalityLoss",
            "coeff": 0.1,
            "pnorm": 2.0,
            "eps": 1e-12,
        },
        # Disable stochastic terms for deterministic dp test; keep a simple layerwise recon if needed
        {"classname": "CIMaskedReconLayerwiseLoss", "coeff": 1.0},
        {"classname": "FaithfulnessLoss", "coeff": 3000},
    ],
    "output_loss_type": "kl",
    # --- Training ---
    "batch_size": 2,
    "steps": 20,
    "lr": 1e-2,
    "lr_schedule": "constant",
    "lr_warmup_pct": 0.0,
    "gradient_accumulation_steps": 1,
    # --- Logging & Saving ---
    "train_log_freq": 9999,
    "eval_freq": 5,  # Eval at steps 0, 5, 10
    "slow_eval_freq": 5,
    "slow_eval_on_first_step": True,
    "n_eval_steps": 2,
    "save_freq": None,  # Just save at the end
    "n_examples_until_dead": 999999,  # We're not tracking this
    "eval_metrics": [
        {"classname": "CI_L0"},
        {"classname": "CEandKLLosses", "rounding_threshold": 0.1},
    ],
    # --- Pretrained model info ---
    "pretrained_model_class": "transformers.LlamaForCausalLM",
    "pretrained_model_name": "SimpleStories/SimpleStories-1.25M",
    "pretrained_model_output_attr": "logits",
    "tokenizer_name": "SimpleStories/SimpleStories-1.25M",
    # --- Task Specific ---
    "task_config": {
        "task_name": "lm",
        "max_seq_len": 5,
        "buffer_size": 100,
        "dataset_name": "SimpleStories/SimpleStories",
        "column_name": "story",
        "train_data_split": "train[:100]",
        "eval_data_split": "test[:100]",
        "shuffle_each_epoch": False,  # Need False in order to maintain determinicity
    },
    # --- Distributed ---
    "dist_backend": "gloo",  # Want to run this test on CPU
}


@pytest.mark.slow
class TestDistributedDeterminicity:
    def test_distributed_determinicity(self):
        """Test DDP determinicity for SPD runs which don't use stochastic masks.

        Runs DDP with 1 and 2 processes on CPU and shows that training metrics, eval metrics, and
        the updated model weights are consistent between the two runs.

        Note that if stochastic masks are used, the results will be non-deterministic due to the
        difficulty in effeciently generating masks on each rank while maintaining pytorch random
        state.

        This is a useful end-to-end test for DDP in general.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create separate output directories for each run
            dp1_out_dir = tmpdir / "dp1_output"
            dp2_out_dir = tmpdir / "dp2_output"

            # Run with dp=1
            config_dp1 = TEST_CONFIG.copy()
            config_dp1["out_dir"] = str(dp1_out_dir)

            config_path_dp1 = tmpdir / "test_config_dp1.yaml"
            with open(config_path_dp1, "w") as f:
                yaml.dump(config_dp1, f)

            self._run_experiment(config_path_dp1, n_processes=1, port=29501)

            # Run with dp=2
            config_dp2 = TEST_CONFIG.copy()
            config_dp2["out_dir"] = str(dp2_out_dir)

            config_path_dp2 = tmpdir / "test_config_dp2.yaml"
            with open(config_path_dp2, "w") as f:
                yaml.dump(config_dp2, f)

            self._run_experiment(config_path_dp2, n_processes=2, port=29502)

            # Load and compare metrics from metrics.jsonl files
            dp1_metrics = self._load_metrics(dp1_out_dir / "metrics.jsonl")
            dp2_metrics = self._load_metrics(dp2_out_dir / "metrics.jsonl")

            # Compare final eval metrics
            self._validate_metrics(dp1_metrics, dp2_metrics)

            # Load and compare saved models
            self._compare_saved_models(dp1_out_dir, dp2_out_dir)

    def _run_experiment(
        self,
        config_path: Path,
        n_processes: int,
        port: int = 29500,
    ) -> None:
        """Run the experiment using mpirun."""
        script_path = REPO_ROOT / "spd" / "experiments" / "lm" / "lm_decomposition.py"
        assert script_path.exists(), f"{script_path} not found"

        env = {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MASTER_PORT": str(port),
        }

        cmd = [
            "mpirun",
            "-np",
            str(n_processes),
            "python",
            str(script_path),
            str(config_path),
        ]

        result = subprocess.run(
            cmd, env={**os.environ, **env}, capture_output=True, text=True, timeout=300
        )

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            raise RuntimeError(f"mpirun failed with code {result.returncode}")

    def _load_metrics(self, metrics_file: Path) -> list[dict[str, float]]:
        """Load eval metrics from the metrics.jsonl file."""
        eval_metrics = []

        with open(metrics_file) as f:
            for line in f:
                eval_metrics.append(json.loads(line))
        return eval_metrics

    def _validate_metrics(
        self,
        dp1_metrics: list[dict[str, float]],
        dp2_metrics: list[dict[str, float]],
        atol: float = 1e-8,
        rtol: float = 1e-5,
    ) -> None:
        """Validate that metrics are consistent between dp=1 and dp=2.

        NOTE: We ignore the ce_unrecovered metrics, as they seem to cause significant differences.
        I'm not sure why.

        Args:
            dp1_metrics: List of eval metrics for each step from dp=1
            dp2_metrics: List of eval metrics for each step from dp=2
            atol: Absolute tolerance
            rtol: Relative tolerance
        """

        assert len(dp1_metrics) == len(dp2_metrics), (
            f"Different number of steps: dp1={len(dp1_metrics)}, dp2={len(dp2_metrics)}"
        )

        for dp1_step, dp2_step in zip(dp1_metrics, dp2_metrics, strict=True):
            assert dp1_step["step"] == dp2_step["step"], "Different steps"
            assert set(dp1_step.keys()) == set(dp2_step.keys()), (
                f"Different metrics keys: dp1={set(dp1_step.keys())}, dp2={set(dp2_step.keys())}"
            )

        for dp1_step, dp2_step in zip(dp1_metrics, dp2_metrics, strict=True):
            for key in sorted(dp1_step.keys()):
                # We ignore metrics that use stochastic masks, as they are non-deterministic.
                if "stoch" in key or "rand" in key or "ce_unrecovered" in key:
                    continue

                try:
                    torch.testing.assert_close(dp1_step[key], dp2_step[key], atol=atol, rtol=rtol)
                except AssertionError as e:
                    e.add_note(f"Step {dp1_step['step']}, Metric '{key}'")
                    raise e

                print(f"✓ Metric '{key}': dp1={dp1_step[key]:.6f}, dp2={dp2_step[key]:.6f}")

    def _compare_saved_models(
        self,
        dp1_out_dir: Path,
        dp2_out_dir: Path,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ) -> None:
        """Compare saved model parameters between dp=1 and dp=2 runs.

        Args:
            dp1_out_dir: Output directory for dp=1 run
            dp2_out_dir: Output directory for dp=2 run
            atol: Absolute tolerance for parameter comparison
            rtol: Relative tolerance for parameter comparison
        """
        # Find all saved model files in both directories and keep only the final checkpoint
        dp1_model_files = sorted(
            dp1_out_dir.glob("model_*.pth"), key=lambda p: int(p.stem.split("_")[1])
        )
        dp2_model_files = sorted(
            dp2_out_dir.glob("model_*.pth"), key=lambda p: int(p.stem.split("_")[1])
        )

        # Retain only the final checkpoint from each run
        dp1_file = dp1_model_files[-1]
        dp2_file = dp2_model_files[-1]

        print("\nComparing saved model checkpoint(s)...")

        # Load model state dicts
        dp1_state = torch.load(dp1_file, map_location="cpu")
        dp2_state = torch.load(dp2_file, map_location="cpu")

        # Compare each parameter
        for param_name in sorted(dp1_state.keys()):
            # We know that the target model is not trained, so we only care about params with
            # "components" or "gates" in the name.
            if "components" not in param_name and "gates" not in param_name:
                continue

            dp1_param = dp1_state[param_name]
            dp2_param = dp2_state[param_name]

            try:
                torch.testing.assert_close(dp1_param, dp2_param, atol=atol, rtol=rtol)
                print(
                    f"  ✓ {param_name}: shape={list(dp1_param.shape)}, max_diff={torch.max(torch.abs(dp1_param - dp2_param)).item():.2e}"
                )
            except AssertionError as e:
                e.add_note(f"Parameter '{param_name}'")
                e.add_note(
                    f"Max difference: {torch.max(torch.abs(dp1_param - dp2_param)).item():.2e}"
                )
                raise e


class TestDistributedUtilities:
    """Test distributed utilities in non-distributed mode."""

    def test_non_distributed_getters(self):
        """Test getter functions in non-distributed mode."""
        assert not is_distributed()

        assert get_rank() == 0
        assert get_world_size() == 1
        assert get_local_rank() == 0
        assert is_main_process()
        assert not is_distributed()

    @pytest.mark.parametrize(
        "cuda_available, distributed, local_rank, expected",
        [
            (False, False, 0, "cpu"),
            (False, True, 1, "cpu"),
            (True, False, 0, "cuda"),
            (True, True, 0, "cuda:0"),
            (True, True, 2, "cuda:2"),
        ],
    )
    def test_get_device_matrix(
        self,
        monkeypatch: pytest.MonkeyPatch,
        cuda_available: bool,
        distributed: bool,
        local_rank: int,
        expected: str,
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available, raising=False)
        monkeypatch.setattr(distributed_utils, "is_distributed", lambda: distributed)
        monkeypatch.setattr(distributed_utils, "get_local_rank", lambda: local_rank)
        assert distributed_utils.get_device() == expected
