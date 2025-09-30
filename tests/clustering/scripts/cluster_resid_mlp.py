# %%
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
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
from spd.clustering.plotting.activations import plot_activations
from spd.clustering.plotting.merge import (
    plot_dists_distribution,
    plot_merge_iteration,
)
from spd.configs import Config
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.registry import EXPERIMENT_REGISTRY
from spd.utils.data_utils import DatasetGeneratedDataLoader

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
TEMP_DIR: Path = Path(
    "tests/.temp"
)  # save to an actual dir that is gitignored, so users can view plots
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# pyright: reportUnusedParameter=false

# magic autoreload
# %load_ext autoreload
# %autoreload 2

# %%
# Load model
# ============================================================
_CANONICAL_RUN: str | None = EXPERIMENT_REGISTRY["resid_mlp2"].canonical_run
assert _CANONICAL_RUN is not None, "No canonical run found for resid_mlp2 experiment"
SPD_RUN: SPDRunInfo = SPDRunInfo.from_path(_CANONICAL_RUN)
MODEL: ComponentModel = ComponentModel.from_pretrained(SPD_RUN.checkpoint_path)
MODEL.to(DEVICE)
SPD_CONFIG: Config = SPD_RUN.config

# %%
# Setup dataset and dataloader
# ============================================================
N_SAMPLES: int = 128

DATASET: ResidMLPDataset = ResidMLPDataset(
    n_features=MODEL.target_model.config.n_features,  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType],
    feature_probability=SPD_CONFIG.task_config.feature_probability,  # pyright: ignore[reportAttributeAccessIssue]
    device=DEVICE,
    calc_labels=False,
    label_type=None,
    act_fn_name=None,
    label_fn_seed=None,
    label_coeffs=None,
    data_generation_type=SPD_CONFIG.task_config.data_generation_type,  # pyright: ignore[reportAttributeAccessIssue]
)

dbg_auto(
    dict(
        n_features=DATASET.n_features,
        feature_probability=DATASET.feature_probability,
        data_generation_type=DATASET.data_generation_type,
    )
)
DATALOADER = DatasetGeneratedDataLoader(DATASET, batch_size=N_SAMPLES, shuffle=False)

# %%
# Get component activations
# ============================================================
# Get a single batch from the dataloader
BATCH_DATA: tuple[Tensor, Tensor] = next(iter(DATALOADER))
BATCH: Tensor = BATCH_DATA[0]

COMPONENT_ACTS: dict[str, Tensor] = component_activations(
    model=MODEL,
    device=DEVICE,
    batch=BATCH,
    sigmoid_type="hard",
)

dbg_auto(COMPONENT_ACTS)

# %%

FILTER_DEAD_THRESHOLD: float = 0.1

# Process activations
# ============================================================
PROCESSED_ACTIVATIONS: ProcessedActivations = process_activations(
    COMPONENT_ACTS,
    filter_dead_threshold=FILTER_DEAD_THRESHOLD,
)


plot_activations(
    processed_activations=PROCESSED_ACTIVATIONS,
    save_dir=TEMP_DIR,
    wandb_run=None,
)

# %%
# run the merge iteration
# ============================================================

MERGE_CFG: MergeConfig = MergeConfig(
    activation_threshold=0.1,
    alpha=1,
    iters=int(PROCESSED_ACTIVATIONS.n_components_alive * 0.9),
    merge_pair_sampling_method="range",
    merge_pair_sampling_kwargs={"threshold": 0.0},
    pop_component_prob=0,
    filter_dead_threshold=FILTER_DEAD_THRESHOLD,
)


def _plot_func(
    current_coact: torch.Tensor,
    component_labels: list[str],
    current_merge: Any,
    costs: torch.Tensor,
    merge_history: MergeHistory,
    iter_idx: int,
    k_groups: int,
    merge_pair_cost: float,
    mdl_loss: float,
    mdl_loss_norm: float,
    diag_acts: torch.Tensor,
) -> None:
    if (iter_idx % 50 == 0 and iter_idx > 0) or iter_idx == 1:
        plot_merge_iteration(
            current_merge=current_merge,
            current_coact=current_coact,
            costs=costs,
            iteration=iter_idx,
            component_labels=component_labels,
            show=True,  # Show the plot interactively
        )


MERGE_HIST: MergeHistory = merge_iteration(
    merge_config=MERGE_CFG,
    batch_id="batch_0",
    activations=PROCESSED_ACTIVATIONS.activations,
    component_labels=PROCESSED_ACTIVATIONS.labels,
    log_callback=_plot_func,
)

# %%
# Plot merge history
# ============================================================

# plt.hist(mh[270]["merges"].components_per_group, bins=np.linspace(0, 56, 57))
# plt.yscale("log")
# plt.xscale("log")


# %%
# compute and plot distances in an ensemble
# ============================================================

# Modern approach: run merge_iteration multiple times to create ensemble
ENSEMBLE_SIZE: int = 4
HISTORIES: list[MergeHistory] = []
for i in range(ENSEMBLE_SIZE):
    HISTORY: MergeHistory = merge_iteration(
        merge_config=MERGE_CFG,
        batch_id=f"batch_{i}",
        activations=PROCESSED_ACTIVATIONS.activations,
        component_labels=PROCESSED_ACTIVATIONS.labels,
        log_callback=None,
    )
    HISTORIES.append(HISTORY)

ENSEMBLE: MergeHistoryEnsemble = MergeHistoryEnsemble(data=HISTORIES)

DISTANCES = ENSEMBLE.get_distances(method="perm_invariant_hamming")

plot_dists_distribution(
    distances=DISTANCES,
    mode="points",
    # label="v1"
)
plt.legend()
