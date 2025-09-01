"""Registry of all experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from spd.settings import REPO_ROOT


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment.

    Attributes:
        task_name: Name of the task the experiment is for.
        decomp_script: Path to the decomposition script
        config_path: Path to the configuration YAML file
        expected_runtime: Expected runtime of the experiment in minutes. Used for SLURM job names.
        canonical_run: Wandb path (i.e. prefixed with "wandb:") to a canonical run of the experiment.
            We test that these runs can be loaded to a ComponentModel in
            `tests/test_wandb_run_loading.py`. If None, no canonical run is available.
    """

    task_name: Literal["tms", "resid_mlp", "lm", "ih"]
    decomp_script: Path
    config_path: Path
    expected_runtime: int
    canonical_run: str | None = None


EXPERIMENT_REGISTRY: dict[str, ExperimentConfig] = {
    "tms_5-2": ExperimentConfig(
        task_name="tms",
        decomp_script=Path("spd/experiments/tms/tms_decomposition.py"),
        config_path=Path("spd/experiments/tms/tms_5-2_config.yaml"),
        expected_runtime=4,
        canonical_run="wandb:goodfire/spd/runs/qndu7oz4",
    ),
    "tms_5-2-id": ExperimentConfig(
        task_name="tms",
        decomp_script=Path("spd/experiments/tms/tms_decomposition.py"),
        config_path=Path("spd/experiments/tms/tms_5-2-id_config.yaml"),
        expected_runtime=4,
        canonical_run="wandb:goodfire/spd/runs/z6wxr9vk",
    ),
    "tms_40-10": ExperimentConfig(
        task_name="tms",
        decomp_script=Path("spd/experiments/tms/tms_decomposition.py"),
        config_path=Path("spd/experiments/tms/tms_40-10_config.yaml"),
        expected_runtime=5,
        canonical_run="wandb:goodfire/spd/runs/pq2r77l5",
    ),
    "tms_40-10-id": ExperimentConfig(
        task_name="tms",
        decomp_script=Path("spd/experiments/tms/tms_decomposition.py"),
        config_path=Path("spd/experiments/tms/tms_40-10-id_config.yaml"),
        expected_runtime=5,
        canonical_run="wandb:goodfire/spd/runs/0sz0scrh",
    ),
    "resid_mlp1": ExperimentConfig(
        task_name="resid_mlp",
        decomp_script=Path("spd/experiments/resid_mlp/resid_mlp_decomposition.py"),
        config_path=Path("spd/experiments/resid_mlp/resid_mlp1_config.yaml"),
        expected_runtime=3,
        canonical_run="wandb:goodfire/spd/runs/glbtwl6g",
    ),
    "resid_mlp2": ExperimentConfig(
        task_name="resid_mlp",
        decomp_script=Path("spd/experiments/resid_mlp/resid_mlp_decomposition.py"),
        config_path=Path("spd/experiments/resid_mlp/resid_mlp2_config.yaml"),
        expected_runtime=11,
        canonical_run="wandb:goodfire/spd/runs/84yirdkb",
    ),
    "resid_mlp3": ExperimentConfig(
        task_name="resid_mlp",
        decomp_script=Path("spd/experiments/resid_mlp/resid_mlp_decomposition.py"),
        config_path=Path("spd/experiments/resid_mlp/resid_mlp3_config.yaml"),
        expected_runtime=60,
        canonical_run="wandb:goodfire/spd/runs/xawvyhq3",  # A few imperfections in layers 0 and 1
    ),
    "ss_llama": ExperimentConfig(
        task_name="lm",
        decomp_script=Path("spd/experiments/lm/lm_decomposition.py"),
        config_path=Path("spd/experiments/lm/ss_llama_config.yaml"),
        expected_runtime=60,
    ),
    "ss_gpt2": ExperimentConfig(
        task_name="lm",
        decomp_script=Path("spd/experiments/lm/lm_decomposition.py"),
        config_path=Path("spd/experiments/lm/ss_gpt2_config.yaml"),
        expected_runtime=60,
    ),
    "gpt2": ExperimentConfig(
        task_name="lm",
        decomp_script=Path("spd/experiments/lm/lm_decomposition.py"),
        config_path=Path("spd/experiments/lm/gpt2_config.yaml"),
        expected_runtime=60,
    ),
    # "ss_emb": ExperimentConfig(
    #     task_name="lm",
    #     decomp_script=Path("spd/experiments/lm/lm_decomposition.py"),
    #     config_path=Path("spd/experiments/lm/ss_emb_config.yaml"),
    #     expected_runtime=60,
    # ),
    # "ts": ExperimentConfig(
    #     experiment_type="lm",
    #     decomp_script=Path("spd/experiments/lm/lm_decomposition.py"),
    #     config_path=Path("spd/experiments/lm/ts_config.yaml"),
    #     expected_runtime=120,
    # ),
}


def get_experiment_config_file_contents(key: str) -> dict[str, Any]:
    """given a key in the `EXPERIMENT_REGISTRY`, return contents of the config file as a dict.

    note that since paths are of the form `Path("spd/experiments/tms/tms_5-2_config.yaml")`,
    we strip the "spd/" prefix to be able to read the file using `importlib`.
    This makes our ability to find the file independent of the current working directory.
    """

    return yaml.safe_load((REPO_ROOT / EXPERIMENT_REGISTRY[key].config_path).read_text())


def get_max_expected_runtime(experiments_list: list[str]) -> str:
    """Get the max expected runtime of a list of experiments in XhYm format.

    Args:
        experiments_list: List of experiment names

    Returns:
        Max expected runtime in XhYm format
    """
    max_expected_runtime = max(
        EXPERIMENT_REGISTRY[experiment].expected_runtime for experiment in experiments_list
    )
    return f"{max_expected_runtime // 60}h{max_expected_runtime % 60}m"
