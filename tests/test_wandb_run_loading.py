"""Test loading models from wandb runs.

If the CANONICAL_RUNS needs to be updated, you can do so with `spd-run`. See spd/scripts/run.py
for more details.
"""

import pytest

from spd.models.component_model import ComponentModel, SPDRunInfo

CANONICAL_RUNS: dict[str, str] = {
    "tms_5-2": "wandb:goodfire/spd/runs/u9lslp82",
    "tms_5-2-id": "wandb:goodfire/spd/runs/hm77qg0d",
    "tms_40-10": "wandb:goodfire/spd/runs/pwj1eaj2",
    "tms_40-10-id": "wandb:goodfire/spd/s2yj41ak",
    "resid_mlp1": "wandb:goodfire/spd/runs/pzauyxx8",
    "ss_mlp": "wandb:spd/runs/ioprgffh",
}


@pytest.mark.slow
def test_wandb_loading_run_info():
    for exp_name, model_path in CANONICAL_RUNS.items():
        try:
            run_info = SPDRunInfo.from_path(model_path)
            ComponentModel.from_run_info(run_info)
        except Exception as e:
            e.add_note(f"Error loading {exp_name} from {model_path}")
            raise e


@pytest.mark.slow
def test_wandb_loading_pretrained_model():
    for exp_name, model_path in CANONICAL_RUNS.items():
        try:
            ComponentModel.from_pretrained(model_path)
        except Exception as e:
            e.add_note(f"Error loading {exp_name} from {model_path}")
            raise e
