from typing import NamedTuple

from spd.registry import EXPERIMENT_REGISTRY, ExperimentConfig
from spd.spd_types import TaskName

TypedModelPath = NamedTuple(  # noqa: UP014
    "TypedModelPath",
    [
        ("wandb_path", str),
        ("task_name", TaskName),
    ],
)


def convert_model_path(
    model_path: str,
) -> TypedModelPath:
    """convert a model path to a wandb path and task name

    - if a wandb path is given directly, assume its a language model decomposition
    - if a `model_path` starting with `spd_exp:` is given, look in the `EXPERIMENT_REGISTRY`
      - only
    """
    if model_path.startswith("wandb:"):
        return TypedModelPath(
            wandb_path=model_path,
            task_name="lm",
        )
    elif model_path.startswith("spd_exp:"):
        key: str = model_path.split("spd_exp:")[1]
        if key not in EXPERIMENT_REGISTRY:
            raise ValueError(f"Experiment '{key}' not found in EXPERIMENT_REGISTRY")
        exp_config: ExperimentConfig = EXPERIMENT_REGISTRY[key]
        assert exp_config.canonical_run is not None, (
            f"Experiment '{key}' does not have a canonical run defined!"
        )
        return TypedModelPath(
            wandb_path=exp_config.canonical_run,
            task_name=exp_config.task_name,
        )
    else:
        raise ValueError(f"model_path must start with 'wandb:' or 'spd_exp:', got '{model_path}'")
