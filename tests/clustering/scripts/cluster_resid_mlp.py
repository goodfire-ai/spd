# %%
import argparse
import sys
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
from spd.clustering.consts import ComponentLabels
from spd.clustering.merge import merge_iteration
from spd.clustering.merge_config import MergeConfig
from spd.clustering.merge_history import MergeHistory, MergeHistoryEnsemble
from spd.clustering.plotting.activations import plot_activations
from spd.clustering.plotting.merge import (
    plot_dists_distribution,
    plot_merge_history_cluster_sizes,
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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run SPD clustering on ResidMLP models")
    parser.add_argument(
        "model_key",
        nargs="?",  # Optional positional argument
        default="resid_mlp2",
        choices=["resid_mlp1", "resid_mlp2", "resid_mlp3"],
        help="Model to use for clustering (default: resid_mlp2)",
    )
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")

    # Handle Jupyter notebook execution
    if hasattr(__builtins__, "__IPYTHON__"):
        # Running in Jupyter - use default or override with custom args
        return parser.parse_args(["resid_mlp2"])  # Default for notebooks
    else:
        return parser.parse_args()


def list_available_models():
    """List all available models in the experiment registry."""
    print("Available models:")
    print("-" * 50)
    for key in EXPERIMENT_REGISTRY:
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
# magic autoreload
# %load_ext autoreload
# %autoreload 2

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
    n_samples_max=256,
    wandb_run=None,
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
    current_coact: torch.Tensor,
    component_labels: ComponentLabels,
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
    # if (iter_idx % 50 == 0 and iter_idx > 0) or iter_idx == 1:
    #     plot_merge_iteration(
    #         current_merge=current_merge,
    #         current_coact=current_coact,
    #         costs=costs,
    #         iteration=iter_idx,
    #         component_labels=component_labels,
    #         show=True,  # Show the plot interactively
    #     )
    # assert kwargs
    if (iter_idx % 50 == 0 and iter_idx > 0) or iter_idx == 1:
        try:
            plot_merge_iteration(
                current_merge=current_merge,
                current_coact=current_coact,
                costs=costs,
                # FIXED: Commented out problematic pair_cost parameter
                # pair_cost=merge_history.latest()["costs_stats"]["chosen_pair"],  # pyright: ignore[reportIndexIssue, reportCallIssue, reportArgumentType],
                iteration=iter_idx,
                component_labels=component_labels,
                show=True,  # Show the plot interactively
                plot_config={
                    "save_pdf": True,
                    "pdf_prefix": f"/content/clustering_{model_key}_iter",
                },
            )
        except Exception as e:
            print(f"Plotting error at iteration {iter_idx}: {e}")


print(
    f"Starting clustering for {model_key} with {PROCESSED_ACTIVATIONS.n_components_alive} components..."
)

MERGE_HIST: MergeHistory = merge_iteration(
    merge_config=MERGE_CFG,
    batch_id="batch_0",
    activations=PROCESSED_ACTIVATIONS.activations,
    component_labels=PROCESSED_ACTIVATIONS.labels,
    log_callback=_plot_func,
)

print(f"Clustering completed for {model_key}")

# %%
# Plot merge history
# ============================================================

# plt.hist(mh[270]["merges"].components_per_group, bins=np.linspace(0, 56, 57))
# plt.yscale("log")
# plt.xscale("log")
# FIXED: Commented out broken plotting functions
# plot_merge_history_costs(MERGE_HIST)  # Raises NotImplementedError
# plot_merge_history_costs(MERGE_HIST, ylim=(-1, 1))  # Raises NotImplementedError
plot_merge_history_cluster_sizes(MERGE_HIST)


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
