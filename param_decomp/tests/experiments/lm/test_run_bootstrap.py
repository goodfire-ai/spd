import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from param_decomp.experiments.lm import run


def test_bootstrap_applies_launch_env_before_loading_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "runtime": {
                    "mesh": {"replicate": 1, "fsdp": 1, "tp": 1},
                    "sharding": "ddp",
                    "compilation_cache_dir": "~/.cache/param-decomp/xla",
                    "compiler_options": "tuned-v2",
                    "launch_env": {
                        "xla_python_client_mem_fraction": 0.5,
                        "env": {"PD_BOOTSTRAP_SENTINEL": "present"},
                    },
                }
            }
        )
    )
    # a wrapper's export must survive the bootstrap, composed additively with the
    # config's flags — the harness spelling that realizes simulated devices.
    monkeypatch.setenv("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
    observed: list[tuple[str, str, str]] = []
    training = ModuleType("param_decomp.experiments.lm.training")

    def train_main(
        config: object,
        data_root: object,
        local_device_count: int,
        run_id: object = None,
    ) -> None:
        del config, data_root, local_device_count, run_id
        observed.append(
            (
                os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"],
                os.environ["PD_BOOTSTRAP_SENTINEL"],
                os.environ["XLA_FLAGS"],
            )
        )

    training.__dict__["main"] = train_main
    monkeypatch.setitem(sys.modules, training.__name__, training)

    run.main(config, Path("/tmp/unused-data-root"), 1, "p-00000000")

    assert observed == [
        (
            "0.5",
            "present",
            "--xla_gpu_nccl_termination_timeout_seconds=600"
            " --xla_force_host_platform_device_count=8",
        )
    ]


def test_importing_the_bootstrap_does_not_import_jax() -> None:
    """The knobs `launch_env` carries are read at backend init, so exporting them after JAX
    is imported is a silent no-op. That makes the bootstrap's import closure load-bearing:
    it may reach the `runtime:` schema (`lm/runtime.py`) but nothing that pulls JAX in — the
    reason that schema is its own module rather than part of `lm/config.py`. A subprocess
    because the suite has JAX resident by the time this runs."""
    probe = "import sys, param_decomp.experiments.lm.run as _; print('jax' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", out.stdout
