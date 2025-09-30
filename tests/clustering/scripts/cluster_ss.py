# %%

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Int
from muutils.dbg import dbg_auto
from torch import Tensor

from spd.clustering.activations import (
    ProcessedActivations,
    component_activations,
    process_activations,
)
from spd.clustering.merge import merge_iteration
from spd.clustering.merge_config import MergeConfig
from spd.clustering.merge_history import MergeHistory, MergeHistoryEnsemble
from spd.clustering.pipeline.s1_split_dataset import split_dataset_lm
from spd.clustering.plotting.activations import plot_activations
from spd.clustering.plotting.merge import plot_dists_distribution
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

PROCESSED_ACTIVATIONS: ProcessedActivations = process_activations(
    activations=COMPONENT_ACTS,
    filter_dead_threshold=FILTER_DEAD_THRESHOLD,
    filter_modules=lambda x: x.startswith(FILTER_MODULES),
    seq_mode="concat",
)

plot_activations(
    processed_activations=PROCESSED_ACTIVATIONS,
    save_pdf=False,
)

# %%
# Compute ensemble merge iterations
# ============================================================
MERGE_CFG: MergeConfig = MergeConfig(
    activation_threshold=0.01,
    alpha=0.01,
    iters=2,
    merge_pair_sampling_method="range",
    merge_pair_sampling_kwargs={"threshold": 0.1},
    pop_component_prob=0,
    module_name_filter=FILTER_MODULES,
    filter_dead_threshold=FILTER_DEAD_THRESHOLD,
)

# Modern approach: run merge_iteration multiple times to create ensemble
ENSEMBLE_SIZE: int = 2
histories: list[MergeHistory] = []
for i in range(ENSEMBLE_SIZE):
    history: MergeHistory = merge_iteration(
        merge_config=MERGE_CFG,
        batch_id=f"batch_{i}",
        activations=PROCESSED_ACTIVATIONS.activations,
        component_labels=PROCESSED_ACTIVATIONS.labels,
        log_callback=None,
    )
    histories.append(history)

ENSEMBLE: MergeHistoryEnsemble = MergeHistoryEnsemble(data=histories)


# %%
# Compute and plot distances
# ============================================================
DISTANCES = ENSEMBLE.get_distances()

plot_dists_distribution(
    distances=DISTANCES,
    mode="points",
)
plt.legend()
