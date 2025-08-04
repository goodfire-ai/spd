# %%
import matplotlib.pyplot as plt
import torch
from muutils.dbg import dbg_auto

from spd.clustering.activations import component_activations, process_activations
from spd.clustering.math.merge_sweep import sweep_merge_parameter, sweep_multiple_parameters
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
from spd.models.component_model import ComponentModel

from spd.utils.data_utils import DatasetGeneratedDataLoader

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# magic autoreload
%load_ext autoreload
%autoreload 2

# %%
# Load model
component_model, cfg, path = ComponentModel.from_pretrained(CANONICAL_RUNS["resid_mlp1"])
component_model.to(DEVICE)

# %%
# Setup dataset and dataloader
N_SAMPLES: int = 512

dataset = ResidMLPDataset(
    n_features=component_model.patched_model.config.n_features,
    feature_probability=cfg.task_config.feature_probability,
    device=DEVICE,
    calc_labels=False,
    label_type=None,
    act_fn_name=None,
    label_fn_seed=None,
    label_coeffs=None,
    data_generation_type=cfg.task_config.data_generation_type,
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
# Get component activations
ci = component_activations(
    component_model,
    dataloader,
    device=DEVICE,
    sigmoid_type="hard",
)

dbg_auto(ci)

# %%
# Process activations
coa = process_activations(
    ci,
    filter_dead_threshold=0.001,
    plots=True,
)

# %%
# Sweep over alpha parameter
alpha_ensembles, alpha_fig, alpha_ax = sweep_merge_parameter(
    activations=coa["activations"],
    parameter_name="alpha",
    parameter_values=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
    component_labels=coa["labels"],
    ensemble_size=16,
)
plt.show()

# %%
# Sweep over check_threshold parameter
ct_ensembles, ct_fig, ct_ax = sweep_merge_parameter(
    activations=coa["activations"],
    parameter_name="check_threshold", 
    parameter_values=[0.0001, 0.5],
    component_labels=coa["labels"],
    ensemble_size=16,
)
plt.show()

# %%
# Sweep over pop_component_prob parameter
pop_ensembles, pop_fig, pop_ax = sweep_merge_parameter(
    activations=coa["activations"],
    parameter_name="pop_component_prob",
    parameter_values=[0.0001, 0.001, 0.01, 0.1, 0.5],
    component_labels=coa["labels"],
    ensemble_size=32,
)
plt.show()

# %%
# Or do all sweeps at once with a single function call
all_results = sweep_multiple_parameters(
    activations=coa["activations"],
    parameter_sweeps={
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        "check_threshold": [0.0001, 0.5],
        "pop_component_prob": [0.0001, 0.001, 0.01, 0.1, 0.5],
    },
    component_labels=coa["labels"],
    ensemble_size=16,
)

# Show all plots
for param_name, (ensembles, fig, ax) in all_results.items():
    plt.show()