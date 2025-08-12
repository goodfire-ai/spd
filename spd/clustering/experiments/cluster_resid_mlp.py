# %%
from typing import Any

import matplotlib.pyplot as plt
import torch
from muutils.dbg import dbg_auto
from torch import Tensor

from spd.clustering.activations import component_activations, process_activations
from spd.clustering.merge import merge_iteration, merge_iteration_ensemble
from spd.clustering.merge_config import MergeConfig
from spd.clustering.merge_history import MergeHistory, MergeHistoryEnsemble
from spd.clustering.merge_sweep import sweep_multiple_parameters
from spd.clustering.plotting.activations import plot_activations
from spd.clustering.plotting.merge import (
    plot_dists_distribution,
    plot_merge_history_cluster_sizes,
    plot_merge_history_costs,
    plot_merge_iteration,
)
from spd.configs import Config
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.registry import EXPERIMENT_REGISTRY
from spd.utils.data_utils import DatasetGeneratedDataLoader

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# magic autoreload
# %load_ext autoreload
# %autoreload 2

# %%
# Load model
# ============================================================
SPD_RUN: SPDRunInfo = SPDRunInfo.from_path(EXPERIMENT_REGISTRY["resid_mlp2"].canonical_run)
MODEL: ComponentModel = ComponentModel.from_pretrained(SPD_RUN.checkpoint_path)
MODEL.to(DEVICE)
SPD_CONFIG: Config = SPD_RUN.config

# %%
# Setup dataset and dataloader
# ============================================================
N_SAMPLES: int = 128

DATASET: ResidMLPDataset = ResidMLPDataset(
    n_features=MODEL.patched_model.config.n_features,  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType],
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
COMPONENT_ACTS: dict[str, Tensor] = component_activations(
    model=MODEL,
    device=DEVICE,
    dataloader=DATALOADER,
    sigmoid_type="hard",
)

dbg_auto(COMPONENT_ACTS)

# %%

FILTER_DEAD_THRESHOLD: float = 0.1

# Process activations
# ============================================================
PROCESSED_ACTIVATIONS: dict[str, Any] = process_activations(
    COMPONENT_ACTS,
    filter_dead_threshold=FILTER_DEAD_THRESHOLD,
    sort_components=False,  # Test the new sorting functionality
)


plot_activations(
    activations=PROCESSED_ACTIVATIONS["activations_raw"],
    act_concat=PROCESSED_ACTIVATIONS["activations"],
    coact=PROCESSED_ACTIVATIONS["coactivations"],
    labels=PROCESSED_ACTIVATIONS["labels"],
    save_pdf=False,
)

# %%
# run the merge iteration
# ============================================================

MERGE_CFG: MergeConfig = MergeConfig(
    activation_threshold=0.1,
    alpha=1,
    iters=int(PROCESSED_ACTIVATIONS["coactivations"].shape[0] * 0.9),
    merge_pair_sampling_method="range",
    merge_pair_sampling_kwargs={"threshold": 0.0},
    pop_component_prob=0,
    filter_dead_threshold=FILTER_DEAD_THRESHOLD,
    rank_cost_fn_name="const_1",
)


def _plot_func(
    costs: torch.Tensor,
    merge_history: MergeHistory,
    current_merge: Any,
    current_coact: torch.Tensor,
    current_act_mask: torch.Tensor,
    i: int,
    k_groups: int,
    activation_mask_orig: torch.Tensor,
    component_labels: list[str],
    sweep_params: dict[str, Any],
) -> None:
    if (i % 50 == 0 and i > 0) or i == 1:
        # latest = merge_history.latest()
        # latest['merges'].plot()
        plot_merge_iteration(
            current_merge=current_merge,
            current_coact=current_coact,
            costs=costs,
            pair_cost=merge_history.latest()["costs_stats"]["chosen_pair"],
            iteration=i,
            component_labels=component_labels,
        )


MERGE_HIST: MergeHistory = merge_iteration(
    activations=PROCESSED_ACTIVATIONS["activations"],
    merge_config=MERGE_CFG,
    component_labels=PROCESSED_ACTIVATIONS["labels"],
    plot_function=_plot_func,
)

# %%
# Plot merge history
# ============================================================

# plt.hist(mh[270]["merges"].components_per_group, bins=np.linspace(0, 56, 57))
# plt.yscale("log")
# plt.xscale("log")

plot_merge_history_costs(MERGE_HIST)
plot_merge_history_costs(MERGE_HIST, ylim=(-1, 1))
plot_merge_history_cluster_sizes(MERGE_HIST)


# %%
# compute and plot distances in an ensemble
# ============================================================

ENSEMBLE: MergeHistoryEnsemble = merge_iteration_ensemble(
    activations=PROCESSED_ACTIVATIONS["activations"],
    component_labels=PROCESSED_ACTIVATIONS["labels"],
    merge_config=MERGE_CFG,
    ensemble_size=4,
)

DISTANCES = ENSEMBLE.get_distances(method="perm_invariant_hamming")

plot_dists_distribution(
    distances=DISTANCES,
    mode="points",
    # label="v1"
)
plt.legend()


# %%
# do sweeps
# ============================================================

SWEEP_RESULTS: dict[str, Any] = sweep_multiple_parameters(
    activations=PROCESSED_ACTIVATIONS["activations"],
    parameter_sweeps={
        "alpha": [1, 5],
        # "check_threshold": [0.0001, 0.001, 0.01, 0.1, 0.5],
        # "pop_component_prob": [0.0001, 0.01, 0.5],
    },
    base_config=MERGE_CFG.model_dump(mode="json"),
    component_labels=PROCESSED_ACTIVATIONS["labels"],
    ensemble_size=4,
)

# Show all plots
for param_name, (ensembles, fig, ax) in SWEEP_RESULTS.items():  # noqa: B007
    plt.show()
