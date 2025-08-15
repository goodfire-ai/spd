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


@pytest.mark.requires_wandb
@pytest.mark.slow
@pytest.mark.parametrize("from_func", [_from_run_info, _from_pretrained])
def test_loading_from_wandb(from_func: Callable[[str], ComponentModel]) -> None:
    for exp_name, exp_config in EXPERIMENT_REGISTRY.items():
        if exp_config.canonical_run is None:  # pyright: ignore[reportUnnecessaryComparison]
            # No canonical run for this experiment
            continue
        try:
            from_func(exp_config.canonical_run)
        except Exception as e:
            e.add_note(f"Error loading {exp_name} from {exp_config.canonical_run}")
            raise e
