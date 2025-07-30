# %%

import torch

from spd.models.component_model import ComponentModel, SPDRunInfo

CANONICAL_RUNS: dict[str, str] = {
    "tms_5-2": "wandb:goodfire/spd/runs/u9lslp82",
    "tms_5-2-id": "wandb:goodfire/spd/runs/hm77qg0d",
    "tms_40-10": "wandb:goodfire/spd/runs/pwj1eaj2",
    "tms_40-10-id": "wandb:goodfire/spd/s2yj41ak",
    "resid_mlp1": "wandb:goodfire/spd/runs/pzauyxx8",
    "ss_mlp": "wandb:spd/runs/ioprgffh",
}

for _, model_path in CANONICAL_RUNS.items():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_info = SPDRunInfo.from_path(model_path)
    ComponentModel.from_run_info(run_info)
    ComponentModel.from_pretrained(model_path)
# %%
