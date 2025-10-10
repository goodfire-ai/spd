# spd/clustering/experiments/tms_5_2_clustering.py
"""
Clustering pipeline for TMS 5-2 model (with and without identity matrix).

Usage:
    python tms_5_2_clustering.py tms_5-2
    python tms_5_2_clustering.py tms_5-2-id
"""

import argparse
import sys
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from spd.clustering.activations import ProcessedActivations, process_activations
from spd.clustering.merge import merge_iteration, merge_iteration_ensemble
from spd.clustering.merge_config import MergeConfig
from spd.clustering.merge_history import MergeHistory, MergeHistoryEnsemble
from spd.clustering.merge_sweep import sweep_multiple_parameters
from spd.clustering.plotting.activations import plot_activations
from spd.clustering.plotting.merge import (
    plot_dists_distribution,
    plot_merge_history_cluster_sizes,
    plot_merge_iteration,
)
from spd.configs import Config
from spd.experiments.tms.tms_dataset import TMSDataset
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.registry import EXPERIMENT_REGISTRY

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SPD clustering on TMS 5-2 models"
    )
    parser.add_argument(
        "model_key",
        nargs="?",
        default="tms_5-2",
        choices=["tms_5-2", "tms_5-2-id"],
        help="Model to use for clustering (default: tms_5-2)"
    )
    
    # Handle Jupyter notebook execution
    if hasattr(__builtins__, '__IPYTHON__'):
        return parser.parse_args(["tms_5-2"])
    else:
        return parser.parse_args()


# ============================================================
# Parse arguments and validate
# ============================================================
args = parse_arguments()
MODEL_KEY = args.model_key

if MODEL_KEY not in EXPERIMENT_REGISTRY:
    print(f"Error: Model '{MODEL_KEY}' not found in experiment registry")
    print("Available models:")
    for key in ["tms_5-2", "tms_5-2-id"]:
        run = EXPERIMENT_REGISTRY.get(key)
        if run:
            print(f"  {key}: {run.canonical_run}")
    sys.exit(1)

canonical_run = EXPERIMENT_REGISTRY[MODEL_KEY].canonical_run
if canonical_run is None:
    print(f"Error: Model '{MODEL_KEY}' is not available (canonical_run is None)")
    sys.exit(1)

# ============================================================
# Configuration
# ============================================================
N_FEATURES = 5
N_SAMPLES = 1000
FILTER_DEAD_THRESHOLD = 0.1

print(f"TMS 5-2 Clustering Pipeline ({MODEL_KEY})")
print("=" * 80)
print(f"Model path: {canonical_run}")
print(f"Device: {DEVICE}")
print("=" * 80)

# ============================================================
# Load model
# ============================================================
SPD_RUN: SPDRunInfo = SPDRunInfo.from_path(canonical_run)
MODEL: ComponentModel = ComponentModel.from_pretrained(SPD_RUN.checkpoint_path)
MODEL.to(DEVICE)
SPD_CONFIG: Config = SPD_RUN.config

print(f"✓ Loaded model from {SPD_RUN.checkpoint_path}")

# ============================================================
# Setup dataset
# ============================================================
DATASET: TMSDataset = TMSDataset(
    n_features=N_FEATURES,
    feature_probability=0.05,
    device=DEVICE,
    calc_labels=False,
    data_generation_type='clustering',
    n_samples_per_feature=200,
)

print(f"✓ Created dataset: {len(DATASET)} samples ({DATASET.data_generation_type} type)")

# ============================================================
# Get component activations
# ============================================================
batch = DATASET.data[:N_SAMPLES].to(DEVICE)

with torch.no_grad():
    _, pre_weight_acts = MODEL(
        batch,
        mode="input_cache",
        module_names=list(MODEL.components.keys())
    )
    
    COMPONENT_ACTS, _ = MODEL.calc_causal_importances(
        pre_weight_acts=pre_weight_acts,
        sigmoid_type="hard",
        detach_inputs=False,
        sampling="continuous",
    )

print(f"✓ Computed component activations")

# ============================================================
# Process activations
# ============================================================
PROCESSED_ACTIVATIONS: ProcessedActivations = process_activations(
    COMPONENT_ACTS,
    filter_dead_threshold=FILTER_DEAD_THRESHOLD,
    sort_components=False,
)

plot_activations(
    processed_activations=PROCESSED_ACTIVATIONS,
    save_pdf=False,
)

print(f"✓ Processed {PROCESSED_ACTIVATIONS.n_components_alive} alive components")

# ============================================================
# Run merge iteration
# ============================================================
n_merge_iters = int(PROCESSED_ACTIVATIONS.n_components_alive * 0.5)

MERGE_CFG: MergeConfig = MergeConfig(
    activation_threshold=0.1,
    alpha=1,
    iters=n_merge_iters,
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
    i: int,
    component_labels: list[str],
    **kwargs: Any,
) -> None:
    """Plot callback for merge iterations."""
    if (i % 10 == 0 and i > 0) or i == 1:
        try:
            plot_merge_iteration(
                current_merge=current_merge,
                current_coact=current_coact,
                costs=costs,
                iteration=i,
                component_labels=component_labels,
                show=True,
                plot_config={"save_pdf": True, "pdf_prefix": f"./{MODEL_KEY.replace('-', '_')}_iter"}
            )
        except Exception as e:
            print(f"Plotting error at iteration {i}: {e}")


print(f"\nRunning {n_merge_iters} merge iterations on {PROCESSED_ACTIVATIONS.n_components_alive} components")

MERGE_HIST: MergeHistory = merge_iteration(
    activations=PROCESSED_ACTIVATIONS.activations,
    merge_config=MERGE_CFG,
    component_labels=PROCESSED_ACTIVATIONS.labels,
    plot_callback=_plot_func,
)

print(f"✓ Clustering completed\n")

# ============================================================
# Plot merge history
# ============================================================
plot_merge_history_cluster_sizes(MERGE_HIST)

# ============================================================
# Ensemble analysis
# ============================================================
print("Running ensemble analysis...")
ENSEMBLE: MergeHistoryEnsemble = merge_iteration_ensemble(
    activations=PROCESSED_ACTIVATIONS.activations,
    component_labels=PROCESSED_ACTIVATIONS.labels,
    merge_config=MERGE_CFG,
    ensemble_size=2,
)

DISTANCES = ENSEMBLE.get_distances(method="perm_invariant_hamming")

plot_dists_distribution(
    distances=DISTANCES,
    mode="points",
)
plt.legend()
plt.show()

# ============================================================
# Parameter sweeps
# ============================================================
print("\nRunning parameter sweeps...")
SWEEP_RESULTS: dict[str, Any] = sweep_multiple_parameters(
    activations=PROCESSED_ACTIVATIONS.activations,
    parameter_sweeps={
        "alpha": [1, 5],
    },
    base_config=MERGE_CFG.model_dump(mode="json"),
    component_labels=PROCESSED_ACTIVATIONS.labels,
    ensemble_size=2,
)

for param_name, (ensembles, fig, ax) in SWEEP_RESULTS.items():
    plt.show()

