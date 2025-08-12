# %%
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from jaxtyping import Int
from muutils.dbg import dbg_auto

from spd.clustering.activations import component_activations, process_activations
from spd.clustering.merge import merge_iteration_ensemble
from spd.clustering.merge_config import MergeConfig
from spd.clustering.merge_history import MergeHistoryEnsemble
from spd.clustering.plotting.activations import plot_activations
from spd.clustering.plotting.merge import plot_dists_distribution
from spd.clustering.scripts.s1_split_dataset import split_dataset_lm
from spd.models.component_model import ComponentModel, SPDRunInfo

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# magic autoreload
# %load_ext autoreload
# %autoreload 2

# %%
# Load model and dataset
# ============================================================
MODEL_PATH: str = "wandb:goodfire/spd/runs/ioprgffh"

_, DATA_CFG = split_dataset_lm(
    model_path=MODEL_PATH,
    n_batches=1,
    batch_size=2,
)
DATASET_PATH: str = DATA_CFG["output_files"][0]

SPD_RUN: SPDRunInfo = SPDRunInfo.from_path(MODEL_PATH)
MODEL: ComponentModel = ComponentModel.from_pretrained(SPD_RUN.checkpoint_path)
MODEL.to(DEVICE)
SPD_CONFIG = SPD_RUN.config


# %%
# Load data batch
# ============================================================
DATA_BATCH: Int[Tensor, "batch_size n_ctx"] = torch.tensor(np.load(DATASET_PATH)["input_ids"])

# %%
# Get component activations
# ============================================================
COMPONENT_ACTS: dict[str, Tensor] = component_activations(
    model=MODEL,
    batch=DATA_BATCH,
    device=DEVICE,
    sigmoid_type="hard",
)

_ = dbg_auto(COMPONENT_ACTS)
# %%
# Process activations
# ============================================================
FILTER_DEAD_THRESHOLD: float = 0.001
FILTER_MODULES: str = "model.layers.0"

PROCESSED_ACTIVATIONS: dict[str, Any] = process_activations(
    activations=COMPONENT_ACTS,
    filter_dead_threshold=FILTER_DEAD_THRESHOLD,
	filter_modules=lambda x: x.startswith(FILTER_MODULES),
    seq_mode="concat",
)

plot_activations(
    activations=PROCESSED_ACTIVATIONS["activations_raw"],
    act_concat=PROCESSED_ACTIVATIONS["activations"],
    coact=PROCESSED_ACTIVATIONS["coactivations"],
    labels=PROCESSED_ACTIVATIONS["labels"],
    save_pdf=False,
)

# %%
# Compute ensemble merge iterations
# ============================================================
MERGE_CFG: MergeConfig = MergeConfig(
    activation_threshold=0.01,
    alpha=0.01,
    iters=100,
    merge_pair_sampling_method="range",
    merge_pair_sampling_kwargs={"threshold": 0.1},
    pop_component_prob=0,
	module_name_filter=FILTER_MODULES,
    filter_dead_threshold=FILTER_DEAD_THRESHOLD,
    rank_cost_fn_name="const_1",
)

ENSEMBLE: MergeHistoryEnsemble = merge_iteration_ensemble(
    activations=PROCESSED_ACTIVATIONS["activations"],
    component_labels=PROCESSED_ACTIVATIONS["labels"],
    merge_config=MERGE_CFG,
    ensemble_size=2,
)


# %%
# Compute and plot distances
# ============================================================
DISTANCES = ENSEMBLE.get_distances()

plot_dists_distribution(
    distances=DISTANCES,
    mode="points",
)
plt.legend()
