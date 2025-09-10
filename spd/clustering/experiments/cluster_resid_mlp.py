# %%
import argparse
import sys
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

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run SPD clustering on ResidMLP models")
    parser.add_argument(
        "model_key", 
        nargs="?",  # Optional positional argument
        default="resid_mlp2",
        choices=["resid_mlp1", "resid_mlp2", "resid_mlp3"],
        help="Model to use for clustering (default: resid_mlp2)"
    )
    parser.add_argument(
        "--list-models", 
        action="store_true",
        help="List available models and exit"
    )
    
    # Handle Jupyter notebook execution
    if hasattr(__builtins__, '__IPYTHON__'):
        # Running in Jupyter - use default or override with custom args
        return parser.parse_args(["resid_mlp2"])  # Default for notebooks
    else:
        return parser.parse_args()

def list_available_models():
    """List all available models in the experiment registry."""
    print("Available models:")
    print("-" * 50)
    for key in EXPERIMENT_REGISTRY.keys():
        run_path = EXPERIMENT_REGISTRY[key].canonical_run
        if run_path:
            print(f"  {key:15} -> {run_path}")
        else:
            print(f"  {key:15} -> None (not available)")
    print("-" * 50)

# Parse arguments
args = parse_arguments()

# List models if requested
if args.list_models:
    list_available_models()
    sys.exit(0)

# Validate model availability
model_key = args.model_key
if model_key not in EXPERIMENT_REGISTRY:
    print(f"Error: Model '{model_key}' not found in experiment registry")
    list_available_models()
    sys.exit(1)

canonical_run = EXPERIMENT_REGISTRY[model_key].canonical_run
if canonical_run is None:
    print(f"Error: Model '{model_key}' is not available (canonical_run is None)")
    list_available_models()
    sys.exit(1)

print(f"Using model: {model_key}")
print(f"Model path: {canonical_run}")
print("-" * 80)

# %%
# Load model
# ============================================================
_CANONICAL_RUN: str | None = EXPERIMENT_REGISTRY[model_key].canonical_run
assert _CANONICAL_RUN is not None, f"No canonical run found for {model_key} experiment"
SPD_RUN: SPDRunInfo = SPDRunInfo.from_path(_CANONICAL_RUN)
MODEL: ComponentModel = ComponentModel.from_pretrained(SPD_RUN.checkpoint_path)
MODEL.to(DEVICE)
SPD_CONFIG: Config = SPD_RUN.config

print(f"✅ Loaded {model_key} model successfully")
print(f"   Checkpoint: {SPD_RUN.checkpoint_path}")
print(f"   Device: {DEVICE}")

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
        model_key=model_key,
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
PROCESSED_ACTIVATIONS: ProcessedActivations = process_activations(
    COMPONENT_ACTS,
    filter_dead_threshold=FILTER_DEAD_THRESHOLD,
    sort_components=False,  # Test the new sorting functionality
)


plot_activations(
    processed_activations=PROCESSED_ACTIVATIONS,
    save_pdf=False,
)

print(f"📊 Processed {PROCESSED_ACTIVATIONS.n_components_alive} alive components from {model_key}")

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
    costs: torch.Tensor,
    merge_history: MergeHistory,
    current_merge: Any,
    current_coact: torch.Tensor,
    # current_act_mask: torch.Tensor,
    i: int,
    # k_groups: int,
    # activation_mask_orig: torch.Tensor,
    component_labels: list[str],
    # sweep_params: dict[str, Any],
    **kwargs: Any,
) -> None:
    assert kwargs
    if (i % 50 == 0 and i > 0) or i == 1:
        try:
            plot_merge_iteration(
                current_merge=current_merge,
                current_coact=current_coact,
                costs=costs,
                # FIXED: Commented out problematic pair_cost parameter
                #pair_cost=merge_history.latest()["costs_stats"]["chosen_pair"],  # pyright: ignore[reportIndexIssue, reportCallIssue, reportArgumentType],
                iteration=i,
                component_labels=component_labels,
                show=True,  # Show the plot interactively
                plot_config={"save_pdf": True, "pdf_prefix": f"/content/clustering_{model_key}_iter"}
            )
        except Exception as e:
            print(f"Plotting error at iteration {i}: {e}")


print(f"Starting clustering for {model_key} with {PROCESSED_ACTIVATIONS.n_components_alive} components...")

MERGE_HIST: MergeHistory = merge_iteration(
    activations=PROCESSED_ACTIVATIONS.activations,
    merge_config=MERGE_CFG,
    component_labels=PROCESSED_ACTIVATIONS.labels,
    plot_callback=_plot_func,
)

print(f"Clustering completed for {model_key}")

# %%
# Plot merge history
# ============================================================

# FIXED: Commented out broken plotting functions
# plot_merge_history_costs(MERGE_HIST)  # Raises NotImplementedError
# plot_merge_history_costs(MERGE_HIST, ylim=(-1, 1))  # Raises NotImplementedError
plot_merge_history_cluster_sizes(MERGE_HIST)


# %%
# compute and plot distances in an ensemble
# ============================================================

ENSEMBLE: MergeHistoryEnsemble = merge_iteration_ensemble(
    activations=PROCESSED_ACTIVATIONS.activations,
    component_labels=PROCESSED_ACTIVATIONS.labels,
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
    activations=PROCESSED_ACTIVATIONS.activations,
    parameter_sweeps={
        "alpha": [1, 5],
        # "check_threshold": [0.0001, 0.001, 0.01, 0.1, 0.5],
        # "pop_component_prob": [0.0001, 0.01, 0.5],
    },
    base_config=MERGE_CFG.model_dump(mode="json"),  # pyright: ignore[reportArgumentType],
    component_labels=PROCESSED_ACTIVATIONS.labels,
    ensemble_size=4,
)

# Show all plots
for param_name, (ensembles, fig, ax) in SWEEP_RESULTS.items():  # noqa: B007
    plt.show()