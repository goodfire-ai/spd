# %%

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Int
from muutils.dbg import dbg_auto
from torch import Tensor

from spd.clustering.activations import component_activations, process_activations
from spd.clustering.merge import (
    MergeConfig,
    MergeEnsemble,
    merge_iteration_ensemble,
)
from spd.clustering.plotting.merge import plot_dists_distribution
from spd.models.component_model import ComponentModel, SPDRunInfo

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# magic autoreload
# %load_ext autoreload
# %autoreload 2

# %%
model_path: str = "wandb:goodfire/spd/runs/ioprgffh"
dataset_path: str = "../data/split_datasets/batchsize_64/batch_00.npz"

SPD_RUN = SPDRunInfo.from_path(model_path)
component_model: ComponentModel = ComponentModel.from_pretrained(SPD_RUN.checkpoint_path)
component_model.to(DEVICE)
cfg = SPD_RUN.config


# %%

data_batch: Int[Tensor, "batch_size n_ctx"] = torch.tensor(np.load(dataset_path)["input_ids"])

# %%

component_acts: dict[str, Tensor] = component_activations(
    model=component_model,
    batch=data_batch,
    device=DEVICE,
    # threshold=0.1,
    # TODO: where can we find this in the model itself???
    sigmoid_type="hard",
)

dbg_auto(component_acts)
# %%
component_coacts: dict[str, Any] = process_activations(
    component_acts,
    filter_dead_threshold=0.001,
    seq_mode="concat",
    plots=True,  # Plot the processed activations
    # plot_title="Processed Activations",
)
# %%


ENSEMBLE: MergeEnsemble = merge_iteration_ensemble(
    activations=component_coacts["activations"],
    component_labels=component_coacts["labels"],
    merge_config=MergeConfig(
        activation_threshold=0.01,
        alpha=0.01,
        iters=100,
        check_threshold=0.1,
        pop_component_prob=0,
        rank_cost_fn=lambda x: 1.0,
    ),
    ensemble_size=8,
)


# %%
DISTANCES = ENSEMBLE.get_distances()


# %%
plot_dists_distribution(
    distances=DISTANCES,
    mode="points",
    # label="v1"
)
plt.legend()
