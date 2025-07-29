# %%

import matplotlib.pyplot as plt
import torch
from muutils.dbg import dbg_auto

from spd.clustering.activations import component_activations, process_activations
from spd.clustering.merge import (
    MergeConfig,
    MergePlotConfig,
    merge_iteration_ensemble,
    plot_dists_distribution,
	MergeEnsemble,
)
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidualMLPDataset
from spd.models.component_model import ComponentModel
from spd.registry import CANONICAL_RUNS
from spd.utils.data_utils import DatasetGeneratedDataLoader

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# magic autoreload
%load_ext autoreload
%autoreload 2

# %%
component_model, cfg, path = ComponentModel.from_pretrained(CANONICAL_RUNS["resid_mlp1"])
component_model.to(DEVICE);

# %%

N_SAMPLES: int = 512

dataset = ResidualMLPDataset(
    n_features=component_model.patched_model.config.n_features,
    feature_probability=cfg.task_config.feature_probability,
    device=DEVICE,
    calc_labels=False,  # Our labels will be the output of the target model
    label_type=None,
    act_fn_name=None,
    label_fn_seed=None,
    label_coeffs=None,
    data_generation_type=cfg.task_config.data_generation_type,
    # synced_inputs=synced_inputs,
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

ci = component_activations(
    component_model,
    dataloader,
    device=DEVICE,
    # threshold=0.1,
    # TODO: where can we find this in the model itself???
    sigmoid_type="hard",
)

dbg_auto(ci);
# %%
coa = process_activations(
    ci,
    filter_dead_threshold=0.001,
    plots=True,  # Plot the processed activations
    # plot_title="Processed Activations",
);

# %%


ENSEMBLE: MergeEnsemble = merge_iteration_ensemble(
    activations=coa["activations"],
    component_labels=coa["labels"],
    merge_config=MergeConfig(
        activation_threshold=None,
        alpha=0.01,
        iters=140,
        check_threshold=0.1,
        pop_component_prob=0.1,
        rank_cost_fn=lambda x: 1.0,
        stopping_condition=None,
    ),
    plot_config=MergePlotConfig(
        plot_every=999,
        plot_every_min=999,
		# plot_every=5,
        save_pdf=False,
        # pdf_prefix="merge_iteration",
        figsize=(16, 3),
        figsize_final=(10, 6),
        tick_spacing=10,
        plot_final=False,
    ),
	ensemble_size=64,
)
# %%
DISTANCES = ENSEMBLE.get_distances()


# %%
plot_dists_distribution(
	distances=DISTANCES,
	mode="dist",
	label="v1"
)
plt.legend()

# %%
