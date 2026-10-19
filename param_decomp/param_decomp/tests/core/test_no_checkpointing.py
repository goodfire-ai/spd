"""A `cadence.checkpointing: {kind: none}` run writes NO `ckpts/` at all — no periodic
saves, no final-step save, and no SIGTERM save — and its run id is single-entry: with
nothing to resume from, re-entering it refuses instead of silently retraining from step 0.

The two save paths need two real child-process runs (see `test_toy_resume` for why
in-process module-main calls poison the test session): only a COMPLETED run can show the
periodic/final-step saves never fired, and only a preempted one can show the SIGTERM save
didn't. The re-entry refusal is `_make_saver`'s marker alone, so it is pinned in-process;
that an engine assert surfaces as a nonzero toy-entry exit is already `test_toy_resume`'s
horizon-refusal coverage."""

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from param_decomp.core.configs import NoCheckpointing
from param_decomp.core.run import _make_saver

_CONFIG = Path(__file__).parents[2] / "experiments" / "tms" / "configs" / "tms_5-2.yaml"
_RUN_ID = "p-1111abcd"


def _tiny_no_checkpoint_config(path: Path, steps: int, train_log_every: int = 1) -> Path:
    raw = yaml.safe_load(_CONFIG.read_text())
    raw["pd"]["steps"] = steps
    raw["pd"]["batch_size"] = 8
    raw["pd"]["faithfulness_warmup_steps"] = 0
    raw["target"]["pretrain"]["steps"] = 2
    raw["target"]["pretrain"]["batch_size"] = 8
    raw["cadence"] = {"train_log_every": train_log_every, "checkpointing": {"kind": "none"}}
    raw["eval"] = None
    raw["wandb"] = None
    config = path / "tiny_tms_no_ckpt.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False))
    return config


def _command(config: Path, data_root: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "param_decomp.experiments.tms.run",
        str(config),
        "--run-id",
        _RUN_ID,
        "--data-root",
        str(data_root),
    ]


def _env(root: Path) -> dict[str, str]:
    return os.environ | {
        "JAX_PLATFORMS": "cpu",
        # Shared across the child processes so only the first invocation pays the compile.
        "JAX_COMPILATION_CACHE_DIR": str(root / "jax_cache"),
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "1",
    }


@dataclass(frozen=True)
class TrainedRun:
    config: Path
    data_root: Path
    env: dict[str, str]
    stdout: str

    @property
    def run_dir(self) -> Path:
        return self.data_root / "runs" / _RUN_ID


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> TrainedRun:
    """One completed 2-step no-checkpoint run, shared by the completion + re-entry tests."""
    root = tmp_path_factory.mktemp("tms_no_ckpt")
    config = _tiny_no_checkpoint_config(root, steps=2)
    data_root = root / "data"
    result = subprocess.run(
        _command(config, data_root), capture_output=True, text=True, env=_env(root)
    )
    assert result.returncode == 0, result.stderr
    return TrainedRun(config, data_root, _env(root), result.stdout)


def test_no_checkpoint_run_writes_no_ckpts_dir(trained: TrainedRun) -> None:
    assert not (trained.run_dir / "ckpts").exists()
    assert "checkpoint saved" not in trained.stdout
    assert (trained.run_dir / "metrics.jsonl").read_text().count("\n") == 2, (
        "the run must still train (and log) every step"
    )


def test_reentering_a_no_checkpoint_run_dir_refuses(tmp_path: Path) -> None:
    assert _make_saver(NoCheckpointing(), tmp_path, is_main=True) is None
    with pytest.raises(AssertionError, match="nothing to resume from"):
        _make_saver(NoCheckpointing(), tmp_path, is_main=True)


def _preempt_after_first_metric(
    tmp_path: Path, train_log_every: int
) -> tuple[subprocess.Popen[str], str, str, Path]:
    """Start an endless no-checkpoint run, SIGTERM it once its first train metric lands, and
    return the finished child with its streams and metrics path."""
    config = _tiny_no_checkpoint_config(tmp_path, steps=1_000_000, train_log_every=train_log_every)
    data_root = tmp_path / "data"
    metrics = data_root / "runs" / _RUN_ID / "metrics.jsonl"

    child = subprocess.Popen(
        _command(config, data_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_env(tmp_path),
    )
    deadline = time.monotonic() + 300
    while not (metrics.exists() and metrics.read_text().strip()):
        assert child.poll() is None, child.communicate()[1]
        assert time.monotonic() < deadline, "no train metric appeared within the deadline"
        time.sleep(0.2)
    child.send_signal(signal.SIGTERM)
    stdout, stderr = child.communicate(timeout=120)
    return child, stdout, stderr, metrics


def test_sigterm_exits_without_saving(tmp_path: Path) -> None:
    """The SIGTERM handler's save is skipped too: a preempted no-checkpoint run must exit
    cleanly having written nothing, not fall back to the requeue save."""
    child, stdout, stderr, metrics = _preempt_after_first_metric(tmp_path, train_log_every=1)

    assert child.returncode == 0, stderr
    assert "SIGTERM: no checkpoint (cadence.checkpointing: none), exiting" in stdout
    assert "checkpoint saved" not in stdout
    assert not (metrics.parent / "ckpts").exists()


def test_sigterm_is_serviced_at_the_next_train_log_boundary(tmp_path: Path) -> None:
    """The cross-rank SIGTERM consensus is taken only at train-log steps, so the run exits
    at a train-log boundary: the last logged step is a multiple of `train_log_every` and
    every train-log step up to it was logged."""
    train_log_every = 3
    child, stdout, stderr, metrics = _preempt_after_first_metric(tmp_path, train_log_every)

    assert child.returncode == 0, stderr
    assert "SIGTERM: no checkpoint (cadence.checkpointing: none), exiting" in stdout
    steps = [json.loads(line)["step"] for line in metrics.read_text().splitlines()]
    assert steps and steps[-1] % train_log_every == 0, steps
    assert steps == list(range(train_log_every, steps[-1] + 1, train_log_every)), steps
