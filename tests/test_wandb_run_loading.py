"""Test loading models from wandb runs.

If these tests fail, you should consider making your changes backwards compatible so the tests pass.
If you're willing to make breaking changes, see spd/scripts/run.py for creating new runs with
the canonical configs, and update the registry with your new run(s).
"""

from collections.abc import Callable

import pytest

from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.registry import EXPERIMENT_REGISTRY


def _from_run_info(canonical_run: str) -> ComponentModel:
    run_info = SPDRunInfo.from_path(canonical_run)
    return ComponentModel.from_run_info(run_info)


def _from_pretrained(canonical_run: str) -> ComponentModel:
    return ComponentModel.from_pretrained(canonical_run)


CANONICAL_EXPS = [
    (exp_name, exp_config.canonical_run, from_func)
    for exp_name, exp_config in EXPERIMENT_REGISTRY.items()
    if exp_config.canonical_run is not None
    for from_func in [_from_run_info, _from_pretrained]
]


@pytest.mark.requires_wandb
@pytest.mark.slow
@pytest.mark.parametrize("exp_name, canonical_run, from_func", CANONICAL_EXPS)
def test_loading_from_wandb(
    exp_name: str, canonical_run: str, from_func: Callable[[str], ComponentModel]
) -> None:
    try:
        from_func(canonical_run)
    except Exception as e:
        e.add_note(f"Error loading {exp_name} from {canonical_run}")
        raise e
