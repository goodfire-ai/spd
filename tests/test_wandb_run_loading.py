"""Test loading models from wandb runs.

If these tests fail, you should consider making your changes backwards compatible so the tests pass.
If you're willing to make breaking changes, see spd/scripts/run.py for creating new runs with
the canonical configs, and update the registry with your new run(s).
"""

import pytest

from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.registry import EXPERIMENT_REGISTRY


def from_run_info(canonical_run: str) -> ComponentModel:
    run_info = SPDRunInfo.from_path(canonical_run)
    return ComponentModel.from_run_info(run_info)


def from_pretrained(canonical_run: str) -> ComponentModel:
    return ComponentModel.from_pretrained(canonical_run)


CANONICAL_EXPS = [
    (exp_name, exp_config.canonical_run)
    for exp_name, exp_config in EXPERIMENT_REGISTRY.items()
    if exp_config.canonical_run is not None
]


@pytest.mark.requires_wandb
@pytest.mark.slow
@pytest.mark.parametrize("exp_name, canonical_run", CANONICAL_EXPS)
def test_loading_from_wandb(exp_name: str, canonical_run: str) -> None:
    # We put both from_run_info and from_pretrained in the same test to avoid distributed read
    # errors from the same wandb cache
    try:
        from_run_info(canonical_run)
    except Exception as e:
        e.add_note(f"Error with from_run_info for {exp_name} from {canonical_run}")
        raise e
    try:
        from_pretrained(canonical_run)
    except Exception as e:
        e.add_note(f"Error with from_pretrained for {exp_name} from {canonical_run}")
        raise e
