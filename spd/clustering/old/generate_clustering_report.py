# %%
"""
Generate clustering analysis report with figures.
This script runs the analysis from the dev.ipynb notebook and saves all figures.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from muutils.dbg import dbg_auto

from spd.clustering.activations import component_activations, process_activations
from spd.clustering.merge import compute_merge_costs, merge_iteration
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.sweep import (
    SweepConfig,
    cost_ratio_condition,
    create_smart_heatmap,
    plot_evolution_histories,
    run_hyperparameter_sweep,
)
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
from spd.models.component_model import ComponentModel
from spd.utils.data_utils import DatasetGeneratedDataLoader

# Create output directory
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load pretrained model
print("\n1. Loading pretrained model...")
component_model, cfg, path = ComponentModel.from_pretrained("wandb:goodfire/spd/runs/dcjm9g2n")
component_model.to(DEVICE)

# Create dataset and dataloader
N_SAMPLES = 1024
print(f"\n2. Creating dataset with {N_SAMPLES} samples...")
dataset = ResidMLPDataset(
    n_features=component_model.model.config.n_features,
    feature_probability=cfg.task_config.feature_probability,
    device=DEVICE,
    calc_labels=False,
    label_type=None,
    act_fn_name=None,
    label_fn_seed=None,
    label_coeffs=None,
    data_generation_type=cfg.task_config.data_generation_type,
)
dataloader = DatasetGeneratedDataLoader(dataset, batch_size=N_SAMPLES, shuffle=False)

# Get component activations
print("\n3. Computing component activations...")
ci = component_activations(component_model, dataloader, device=DEVICE)
dbg_auto(ci)

# Process activations and create visualizations
print("\n4. Processing activations and creating visualizations...")
coa = process_activations(
    ci,
    filter_dead_threshold=0.001,
    plots=True,
    save_pdf=True,
    pdf_prefix=str(FIGURES_DIR / "activations"),
)

# Additional activation matrix visualization
plt.figure(figsize=(10, 6))
plt.matshow(coa["activations"].T.cpu(), cmap="viridis", vmin=0, vmax=1)
plt.colorbar()
plt.title("Component Activations Matrix")
plt.xlabel("Samples")
plt.ylabel("Components")
plt.savefig(FIGURES_DIR / "activations_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# Group merge identity plot
print("\n5. Creating group merge visualization...")
gm_ident = GroupMerge.identity(n_components=coa["n_components_alive"])
fig = plt.figure(figsize=(10, 2))
gm_ident.plot(figsize=(10, 2), component_labels=coa["labels"], show=False)
plt.savefig(FIGURES_DIR / "group_merge_identity.pdf", dpi=300, bbox_inches="tight")
plt.close()

# Compute and visualize merge costs
print("\n6. Computing merge costs...")
costs = compute_merge_costs(coact=coa["coactivations"], merges=gm_ident)
plt.figure(figsize=(8, 6))
plt.matshow(costs.cpu(), cmap="viridis")
plt.colorbar()
plt.title("Merge Costs Matrix")
plt.savefig(FIGURES_DIR / "merge_costs_matrix.pdf", dpi=300, bbox_inches="tight")
plt.close()

# Run merge iteration example
print("\n7. Running merge iteration example...")
coact_bool = coa["coactivations"] > 0.01
merge_results = merge_iteration(
    coact=coact_bool.float().T @ coact_bool.float(),
    activation_mask=coact_bool,
    check_threshold=0.0001,
    rank_cost_fn=lambda _: 0.001,
    alpha=0.00001,
    iters=200,
    plot_every=10,
    plot_every_min=0,
    component_labels=coa["labels"],
    save_pdf=True,
    pdf_prefix=str(FIGURES_DIR / "merge_iteration"),
)

# Hyperparameter sweep
print("\n8. Running hyperparameter sweep...")
sweep_config = SweepConfig(
    activation_thresholds=np.logspace(-3, -1, 6).tolist(),
    check_thresholds=np.logspace(-3, 0, 5).tolist(),
    alphas=np.logspace(-3, 3, 5).tolist(),
    rank_cost_funcs={
        "constant_1": lambda _: 1.0,
        "linear": lambda c: c,
        "log": lambda c: np.log(c + 1),
    },
    iters=50,
)

sweep_results = run_hyperparameter_sweep(coa["coactivations"], sweep_config, component_labels=coa["labels"])
print(f"\nCompleted sweep with {len(sweep_results)} configurations")

# Evolution histories for different rank cost functions
print("\n9. Creating evolution history plots...")
for rank_cost_name in ["constant_1", "linear", "log"]:
    plt.figure(figsize=(15, 10))
    plot_evolution_histories(
        sweep_results,
        metric="non_diag_costs_min",
        cols_by="activation_threshold",
        rows_by="alpha",
        lines_by="check_threshold",
        fixed_params={"rank_cost_name": rank_cost_name},
    )
    plt.savefig(
        FIGURES_DIR / f"evolution_history_{rank_cost_name}.pdf", dpi=300, bbox_inches="tight"
    )
    plt.close()

# Smart heatmaps
print("\n10. Creating heatmap visualizations...")
heatmap_configs = [
    ("Final Groups", lambda r: r.final_k_groups, False, False),
    ("Total Iterations", lambda r: r.total_iterations, False, False),
    ("Final Cost", lambda r: r.non_diag_costs_min[-1] if r.non_diag_costs_min else 0, True, True),
]

for stat_name, stat_func, log_scale, normalize in heatmap_configs:
    plt.figure(figsize=(12, 8))
    create_smart_heatmap(
        sweep_results,
        statistic_func=stat_func,
        statistic_name=stat_name,
        log_scale=log_scale,
        normalize=normalize,
    )
    filename = f"heatmap_{stat_name.lower().replace(' ', '_')}"
    if log_scale:
        filename += "_log"
    if normalize:
        filename += "_normalized"
    plt.savefig(FIGURES_DIR / f"{filename}.pdf", dpi=300, bbox_inches="tight")
    plt.close()

# Run with stopping condition example
print("\n11. Running merge with stopping condition...")

stop_at_2x = cost_ratio_condition(2.0, "non_diag_costs_min")
result_with_stop = merge_iteration(
    coact=(coa["coactivations"] > 0.002).float().T @ (coa["coactivations"] > 0.002).float(),
    activation_mask=coa["coactivations"] > 0.002,
    alpha=1.0,
    check_threshold=0.1,
    stopping_condition=stop_at_2x,
    plot_every=None,
    plot_final=True,
    save_pdf=True,
    pdf_prefix=str(FIGURES_DIR / "merge_with_stopping"),
)

print("\nMerge with stopping condition completed:")
print(f"  Total iterations: {result_with_stop['total_iterations']}")
print(f"  Final groups: {result_with_stop['final_k_groups']}")

print(f"\nAll figures saved to: {FIGURES_DIR.absolute()}")
print("\nFigures generated:")
for fig_path in sorted(FIGURES_DIR.glob("*.png")) + sorted(FIGURES_DIR.glob("*.pdf")):
    print(f"  - {fig_path.name}")
