# %%
import matplotlib.pyplot as plt
import torch
from muutils.dbg import dbg_auto

from spd.clustering.activations import component_activations, process_activations
from spd.clustering.merge import merge_iteration_ensemble
from spd.clustering.merge_config import MergeConfig
from spd.clustering.merge_history import MergeHistoryEnsemble
from spd.clustering.merge_sweep import sweep_multiple_parameters
from spd.clustering.plotting.merge import plot_dists_distribution
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
SPD_RUN = SPDRunInfo.from_path(EXPERIMENT_REGISTRY["resid_mlp2"].canonical_run)
# SPD_RUN = SPDRunInfo.from_path(EXPERIMENT_REGISTRY["resid_mlp3"].canonical_run)
component_model: ComponentModel = ComponentModel.from_pretrained(SPD_RUN.checkpoint_path)
component_model.to(DEVICE)
cfg = SPD_RUN.config

# %%
# Setup dataset and dataloader
N_SAMPLES: int = 512

dataset = ResidMLPDataset(
    n_features=component_model.patched_model.config.n_features,  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType],
    feature_probability=cfg.task_config.feature_probability,  # pyright: ignore[reportAttributeAccessIssue]
    device=DEVICE,
    calc_labels=False,
    label_type=None,
    act_fn_name=None,
    label_fn_seed=None,
    label_coeffs=None,
    data_generation_type=cfg.task_config.data_generation_type,  # pyright: ignore[reportAttributeAccessIssue]
)

dbg_auto(
    dict(
        n_features=dataset.n_features,
        feature_probability=dataset.feature_probability,
        data_generation_type=dataset.data_generation_type,
    )
)
dataloader = DatasetGeneratedDataLoader(dataset, batch_size=N_SAMPLES, shuffle=False)
# %%

dl_iter = iter(dataloader)

for x in dl_iter:
    dbg_auto(x[0])
    dbg_auto(x[1])  # labels, but we don't need them for clustering
    dbg_auto(x[0] - x[1])
    plt.matshow(x[0].cpu().numpy(), cmap="viridis")
    plt.colorbar()
    plt.matshow((x[0] - x[1]).cpu().numpy(), cmap="viridis")
    plt.colorbar()
    plt.show()
    break

# %%
# Get component activations
ci = component_activations(
    model=component_model,
    device=DEVICE,
    dataloader=dataloader,
    sigmoid_type="hard",
)

dbg_auto(ci)
# %%
# Process activations
coa = process_activations(
    ci,
    filter_dead_threshold=0.001,
)

# %%

ENSEMBLE: MergeHistoryEnsemble = merge_iteration_ensemble(
    activations=coa["activations"],
    component_labels=coa["labels"],
    merge_config=MergeConfig(
        activation_threshold=None,
        alpha=0.01,
        iters=140,
        check_threshold=0.1,
        pop_component_prob=0,
        # rank_cost_fn=lambda x: 1.0,
        # stopping_condition=None,
    ),
    ensemble_size=16,
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


# %%
# Or do all sweeps at once with a single function call
all_results = sweep_multiple_parameters(
    activations=coa["activations"],
    parameter_sweeps={
        "alpha": [0.0001, 1, 10000.0],
        # "check_threshold": [0.0001, 0.001, 0.01, 0.1, 0.5],
        # "pop_component_prob": [0.0001, 0.01, 0.5],
    },
    component_labels=coa["labels"],
    ensemble_size=16,
)

# Show all plots
for param_name, (ensembles, fig, ax) in all_results.items():  # noqa: B007
    plt.show()
