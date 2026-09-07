"""Every maintained config YAML parses, round-trips, and ships in the built package.

A schema PR that breaks a seat migrates it in the same PR, with an executed in-repo
migration — never a script attached to a PR comment (see CONFIGS.md).
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pytest
import yaml

from param_decomp.core.base_config import BaseConfig
from param_decomp.core.configs import (
    ImportanceMinimalityLossConfig,
    MergedStochasticSubsetPooledPPGDReconLossConfig,
    NonlinearityLocalityLossConfig,
    PGDReconLossConfig,
)
from param_decomp.experiments.lm.config import (
    QWEN36_MOE_MODEL_CLASS,
    LMExperimentConfig,
    LMTargetedExperimentConfig,
    PretrainedTarget,
    assert_placement_claims,
)
from param_decomp.experiments.resid_mlp.config import ResidMLPExperimentConfig
from param_decomp.experiments.tms.config import TMSExperimentConfig
from param_decomp.infra.dataset_store import DatasetDir, NamedDataset, resolve_dataset_ref
from param_decomp.pretrain.config import PretrainConfig

REPO = Path(__file__).resolve().parents[3]

LM_CONFIG_PATHS = sorted(REPO.glob("param_decomp/experiments/lm/configs/*.yaml"))
PRETRAIN_CONFIG_PATHS = sorted(REPO.glob("param_decomp/pretrain/configs/*.yaml"))
PUBLIC_SCHEMA_BY_DIR: dict[str, tuple[type[BaseConfig], type[BaseConfig] | None]] = {
    "param_decomp/experiments/lm/configs": (LMExperimentConfig, LMTargetedExperimentConfig),
    "param_decomp/experiments/tms/configs": (TMSExperimentConfig, None),
    "param_decomp/experiments/resid_mlp/configs": (ResidMLPExperimentConfig, None),
    "param_decomp/pretrain/configs": (PretrainConfig, None),
}


def _is_targeted_seat(path: Path) -> bool:
    """Test-local parametrization label only — production consumers read the deliverable
    projection and never dispatch on run shape; the roots each parse their own shape."""
    return "nontarget" in yaml.safe_load(path.read_text())


def _seat_schema(path: Path, rel_dir: str) -> type[BaseConfig]:
    plain, targeted = PUBLIC_SCHEMA_BY_DIR[rel_dir]
    if not _is_targeted_seat(path):
        return plain
    assert targeted is not None, f"{path} is a targeted seat but {rel_dir} has no targeted shape"
    return targeted


PUBLIC_CONFIG_CASES = sorted(
    (path, _seat_schema(path, rel_dir))
    for rel_dir in PUBLIC_SCHEMA_BY_DIR
    for path in REPO.glob(f"{rel_dir}/*.yaml")
)
PUBLIC_CONFIG_PATHS = [path for path, _ in PUBLIC_CONFIG_CASES]


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def test_pile4l_canonical_recipe() -> None:
    config = LMExperimentConfig.from_file(
        REPO / "param_decomp/experiments/lm/configs/pile_llama_simple_mlp-4L.yaml"
    )
    imp = next(
        term for term in config.pd.loss_metrics if isinstance(term, ImportanceMinimalityLossConfig)
    )
    locality = next(
        term for term in config.pd.loss_metrics if isinstance(term, NonlinearityLocalityLossConfig)
    )
    recon = next(
        term
        for term in config.pd.loss_metrics
        if isinstance(term, MergedStochasticSubsetPooledPPGDReconLossConfig)
    )
    assert config.eval is not None
    eval_pgd = next(
        metric for metric in config.eval.metrics if isinstance(metric, PGDReconLossConfig)
    )

    assert imp.coeff == 2e-4
    assert imp.frequency is not None
    assert imp.frequency.coeff == 6.6e-5
    assert imp.frequency.reference_datapoint_count == 6_553_600
    assert locality.coeff == 3e-5
    assert recon.hidden_acts_reconstruction is None
    assert recon.pool_size == 2064
    assert recon.optimizer.lr_schedule.max_val == 0.02
    assert config.runtime.sharding == "owner"
    assert eval_pgd.hidden_acts_reconstruction is not None
    assert eval_pgd.hidden_acts_reconstruction.coeff == 0.0
    assert eval_pgd.hidden_acts_reconstruction.points == (
        "resid.1",
        "resid.2",
        "resid.3",
        "resid.4",
    )

    schedules = (
        imp.gamma,
        config.pd.components_optimizer.lr_schedule,
        config.pd.ci_fn_optimizer.lr_schedule,
    )
    assert all(schedule.points[-2].at == 0.9 for schedule in schedules)
    assert all(schedule.points[-2].frac == schedule.points[-1].frac for schedule in schedules)


def test_gate_collects_the_seat_registry() -> None:
    """Moved roots must not silently make a domain disappear from the parametrized tests."""
    collected_dirs = {str(path.parent.relative_to(REPO)) for path in PUBLIC_CONFIG_PATHS}
    assert collected_dirs == set(PUBLIC_SCHEMA_BY_DIR), (
        f"config glob collected {collected_dirs or 'nothing'} — did a configs dir move?"
    )
    assert len(LM_CONFIG_PATHS) <= 10, (
        f"{len(LM_CONFIG_PATHS)} LM configs exceed the CONFIGS.md registry cap of 10 — "
        "adding a seat requires an eviction (or this is an uncommitted one-off, see rule 2)"
    )


@pytest.mark.parametrize("path", LM_CONFIG_PATHS, ids=lambda p: str(p.relative_to(REPO)))
def test_lm_config_builds_placement_claims(path: Path) -> None:
    config = (
        LMTargetedExperimentConfig.model_validate(_load(path))
        if _is_targeted_seat(path)
        else LMExperimentConfig.model_validate(_load(path))
    )
    # The placement gate resolves the site set from config + arch, so every maintained
    # config's sharding claim is exercised at its pinned dp. A pretrained target is the
    # enumerated gap: resolving it reads a external pretrain cache.
    if not isinstance(config.target.spec, PretrainedTarget):
        assert_placement_claims(config, Path("out"))


# Seats the CPU trace gate covers: families with an abstract (shape-only) model
# builder — today the qwen36_moe family (`trace_check` enumerates the gap).
TRACE_GATE_PATHS = [
    path
    for path in LM_CONFIG_PATHS
    if _load(path)["target"]["spec"].get("model_class") == QWEN36_MOE_MODEL_CLASS
]


@pytest.mark.parametrize("path", TRACE_GATE_PATHS, ids=lambda p: str(p.relative_to(REPO)))
def test_lm_seat_train_step_traces_at_declared_topology(path: Path) -> None:
    """The placement-claims gate never TRACES the step, and explicit-sharding refusals
    (an ambiguous sharded contraction) fire only at trace time. Lower the seat's real
    train step at its declared topology on
    simulated CPU devices: a subprocess, because the world size must exist as local
    devices before jax initializes. Sequence is never sharded, so the pinned 512 (the
    seats' dataset extent) exercises every sharding rule any extent would. The gate also
    lowers the seat's eval programs on every output edge the family supports, so an
    eval-only streamed-edge trace failure dies here too."""
    config = (
        LMTargetedExperimentConfig.model_validate(_load(path))
        if _is_targeted_seat(path)
        else LMExperimentConfig.model_validate(_load(path))
    )
    env = os.environ | {
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={config.runtime.mesh.world_size}",
        "JAX_PLATFORMS": "cpu",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "param_decomp.experiments.lm.trace_check",
            str(path),
            "--seq-len",
            "512",
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"trace gate failed for {path.name}:\n{result.stderr[-3000:]}"


@pytest.mark.parametrize(
    ("path", "schema"),
    PUBLIC_CONFIG_CASES,
    ids=[str(path.relative_to(REPO)) for path, _ in PUBLIC_CONFIG_CASES],
)
def test_public_config_round_trips(path: Path, schema: type[BaseConfig], tmp_path: Path) -> None:
    """Every maintained public seat survives both BaseConfig persistence formats."""
    config = schema.from_file(path)
    assert schema.model_validate(config.model_dump(mode="json")) == config
    for suffix in (".yaml", ".json"):
        persisted = tmp_path / f"config{suffix}"
        config.to_file(persisted)
        assert schema.from_file(persisted) == config


@pytest.mark.parametrize("path", PRETRAIN_CONFIG_PATHS, ids=lambda p: str(p.relative_to(REPO)))
def test_pretrain_seat_resolves_to_the_dataset_store(path: Path) -> None:
    """Every pretrain seat resolves its dataset name under the caller's data root."""
    data = PretrainConfig.model_validate(_load(path)).data
    data_root = Path("/any/data/root")
    match data:
        case NamedDataset(name=name):
            assert resolve_dataset_ref(data, data_root) == data_root / "datasets" / name
        case DatasetDir(dir=dir):
            pytest.fail(f"{path.name} seats an ad-hoc shard dir ({dir}); a seat carries a name")


def test_wheel_contains_every_public_config(tmp_path: Path) -> None:
    """A source checkout can hide missing package-data: inspect the built wheel itself."""
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=REPO,
        check=True,
    )
    [wheel] = tmp_path.glob("param_decomp-*.whl")
    with ZipFile(wheel) as archive:
        packaged = set(archive.namelist())
    expected = {str(path.relative_to(REPO)) for path in PUBLIC_CONFIG_PATHS}
    assert expected <= packaged, sorted(expected - packaged)


def test_toy_sweep_uses_the_public_module_surface() -> None:
    """The public package has no console scripts or former `param_decomp_lab` package."""
    script = (REPO / "param_decomp/experiments/run_toy_sweep.sh").read_text()
    assert "param_decomp_lab" not in script
    assert "pd-tms" not in script
    assert "pd-resid-mlp" not in script
    assert 'python -m "$runner"' in script
    assert "param_decomp.experiments.tms.run" in script
    assert "param_decomp.experiments.resid_mlp.run" in script


def _absolute_path_leaks(node: object, key: str | None = None, exempt: bool = False) -> list[str]:
    """Strings starting with `/`, except the `dir` value of a tagged `kind: dir` arm."""
    match node:
        case str() if node.startswith("/") and not exempt:
            return [f"{key}: {node}"]
        case dict():
            is_dir_arm = node.get("kind") == "dir"
            return [
                leak
                for k, v in node.items()
                for leak in _absolute_path_leaks(v, k, exempt=is_dir_arm and k == "dir")
            ]
        case list():
            return [leak for item in node for leak in _absolute_path_leaks(item, key)]
        case _:
            return []


@pytest.mark.parametrize("path", PUBLIC_CONFIG_PATHS, ids=lambda p: str(p.relative_to(REPO)))
def test_seats_carry_names_never_locations(path: Path) -> None:
    """Committed seats do not hard-code an absolute machine path outside an escape arm."""
    leaks = _absolute_path_leaks(_load(path))
    assert not leaks, f"absolute paths outside tagged `kind: dir` arms: {leaks}"
