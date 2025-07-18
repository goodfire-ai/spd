"""Registry of all experiments."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from spd.settings import REPO_ROOT


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment.

    Attributes:
        decomp_script: Path to the decomposition script
        config_path: Path to the configuration YAML file
        expected_runtime: Expected runtime of the experiment in minutes. Used for SLURM job names.
    """

    experiment_type: Literal["tms", "resid_mlp", "lm"]
    decomp_script: Path
    config_path: Path
    expected_runtime: int


EXPERIMENT_REGISTRY: dict[str, ExperimentConfig] = {
    "tms_5-2": ExperimentConfig(
        experiment_type="tms",
        decomp_script=Path("spd/experiments/tms/tms_decomposition.py"),
        config_path=Path("spd/experiments/tms/tms_5-2_config.yaml"),
        expected_runtime=4,
    ),
    "tms_5-2-id": ExperimentConfig(
        experiment_type="tms",
        decomp_script=Path("spd/experiments/tms/tms_decomposition.py"),
        config_path=Path("spd/experiments/tms/tms_5-2-id_config.yaml"),
        expected_runtime=4,
    ),
    "tms_40-10": ExperimentConfig(
        experiment_type="tms",
        decomp_script=Path("spd/experiments/tms/tms_decomposition.py"),
        config_path=Path("spd/experiments/tms/tms_40-10_config.yaml"),
        expected_runtime=5,
    ),
    "tms_40-10-id": ExperimentConfig(
        experiment_type="tms",
        decomp_script=Path("spd/experiments/tms/tms_decomposition.py"),
        config_path=Path("spd/experiments/tms/tms_40-10-id_config.yaml"),
        expected_runtime=5,
    ),
    "resid_mlp1": ExperimentConfig(
        experiment_type="resid_mlp",
        decomp_script=Path("spd/experiments/resid_mlp/resid_mlp_decomposition.py"),
        config_path=Path("spd/experiments/resid_mlp/resid_mlp1_config.yaml"),
        expected_runtime=3,
    ),
    "resid_mlp2": ExperimentConfig(
        experiment_type="resid_mlp",
        decomp_script=Path("spd/experiments/resid_mlp/resid_mlp_decomposition.py"),
        config_path=Path("spd/experiments/resid_mlp/resid_mlp2_config.yaml"),
        expected_runtime=11,
    ),
    "resid_mlp3": ExperimentConfig(
        experiment_type="resid_mlp",
        decomp_script=Path("spd/experiments/resid_mlp/resid_mlp_decomposition.py"),
        config_path=Path("spd/experiments/resid_mlp/resid_mlp3_config.yaml"),
        expected_runtime=60,
    ),
    # "ss_emb": ExperimentConfig(
    #     experiment_type="lm",
    #     decomp_script=Path("spd/experiments/lm/lm_decomposition.py"),
    #     config_path=Path("spd/experiments/lm/ss_emb_config.yaml"),
    #     expected_runtime=60,
    # ),
}


def get_experiment_config_file_contents(key: str) -> dict[str, Any]:
    """given a key in the `EXPERIMENT_REGISTRY`, return contents of the config file as a dict.

    note that since paths are of the form `Path("spd/experiments/tms/tms_5-2_config.yaml")`,
    we strip the "spd/" prefix to be able to read the file using `importlib`.
    This makes our ability to find the file independent of the current working directory.
    """
    import yaml

    return yaml.safe_load((REPO_ROOT / EXPERIMENT_REGISTRY[key].config_path).read_text())
