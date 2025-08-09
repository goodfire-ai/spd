# %%

import matplotlib.pyplot as plt
import torch
from muutils.dbg import dbg_auto

from spd.clustering.activations import component_activations, process_activations
from spd.clustering.merge import (
    MergeConfig,
    MergePlotConfig,
    merge_iteration_ensemble,
	MergeHistoryEnsemble,
)

from spd.clustering.plotting.merge import plot_dists_distribution
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
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

dataset = ResidMLPDataset(
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


ENSEMBLE: MergeHistoryEnsemble = merge_iteration_ensemble(
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

fig, ax = plt.subplots(1, 1, figsize=(16, 10))

for alpha in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]:
	print(f"Alpha: {alpha}")

	ens: MergeHistoryEnsemble = merge_iteration_ensemble(
		activations=coa["activations"],
		component_labels=coa["labels"],
		merge_config=MergeConfig(
			activation_threshold=None,
			alpha=alpha,
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
		ensemble_size=16,
	)
	print(f"got ensemble with {ens.n_iters = }, {ens.n_ensemble = }")
	dists = ens.get_distances()
	print(f"Distances shape: {dists.shape}")

	plot_dists_distribution(
		distances=dists,
		mode="dist",
		label=f"$\\alpha={alpha:.4f}$",
		ax=ax,
	)

plt.legend()
plt.show()

# %%


fig, ax = plt.subplots(1, 1, figsize=(16, 10))

for check_threshold in [0.0001, 0.5]:
	print(f"{check_threshold = }")

	ens: MergeHistoryEnsemble = merge_iteration_ensemble(
		activations=coa["activations"],
		component_labels=coa["labels"],
		merge_config=MergeConfig(
			activation_threshold=None,
			alpha=1.0,
			iters=140,
			check_threshold=check_threshold,
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
		ensemble_size=16,
	)
	print(f"got ensemble with {ens.n_iters = }, {ens.n_ensemble = }")
	dists = ens.get_distances()
	print(f"Distances shape: {dists.shape}")

	plot_dists_distribution(
		distances=dists,
		mode="dist",
		label=f"$c={check_threshold:.4f}$",
		ax=ax,
	)

plt.legend()
plt.show()

# %%


fig, ax = plt.subplots(1, 1, figsize=(16, 10))

for pop_component_prob in [0.0001, 0.001, 0.01, 0.1, 0.5]:
	print(f"{pop_component_prob = }")

	ens: MergeHistoryEnsemble = merge_iteration_ensemble(
		activations=coa["activations"],
		component_labels=coa["labels"],
		merge_config=MergeConfig(
			activation_threshold=None,
			alpha=1.0,
			iters=140,
			check_threshold=0.1,
			pop_component_prob=pop_component_prob,
			rank_cost_fn=lambda x: 1.0,
		),
		plot_config=MergePlotConfig(
			plot_every=999,
			plot_every_min=999,
			save_pdf=False,
			plot_final=False,
		),
		ensemble_size=32,
	)
	print(f"got ensemble with {ens.n_iters = }, {ens.n_ensemble = }")
	dists = ens.get_distances()
	print(f"Distances shape: {dists.shape}")

	plot_dists_distribution(
		distances=dists,
		mode="dist",
		label=f"$p={pop_component_prob:.4f}$",
		ax=ax,
	)

plt.legend()
plt.show()