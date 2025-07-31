"""Test loading models from wandb runs.

If these tests fail, you should consider making your changes backwards compatible so these do run.
If you're willing to make breaking changes, see spd/scripts/run.py for creating new runs with
the canonical configs, and update the registry with your new run(s).
"""

import pytest

from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.registry import EXPERIMENT_REGISTRY


@pytest.mark.slow
def test_wandb_loading_run_info():
    for exp_name, exp_config in EXPERIMENT_REGISTRY.items():
        if exp_config.canonical_run is None:
            # No canonical run for this experiment
            continue
        try:
            run_info = SPDRunInfo.from_path(exp_config.canonical_run)
            ComponentModel.from_run_info(run_info)
        except Exception as e:
            e.add_note(f"Error loading {exp_name} from {exp_config.canonical_run}")
            raise e


@pytest.mark.slow
def test_wandb_loading_pretrained_model():
    for exp_name, exp_config in EXPERIMENT_REGISTRY.items():
        if exp_config.canonical_run is None:
            # No canonical run for this experiment
            continue
        try:
            ComponentModel.from_pretrained(exp_config.canonical_run)
        except Exception as e:
            e.add_note(f"Error loading {exp_name} from {exp_config.canonical_run}")
            raise e
