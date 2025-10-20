# %%
"""Main execution script for causal importance decision tree training."""

from typing import Any

import numpy as np
import torch
from jaxtyping import Bool, Float
from torch import Tensor

from spd.clustering.activations import (
    ProcessedActivations,
    component_activations,
    process_activations,
)
from spd.clustering.ci_dt.config import CIDTConfig
from spd.clustering.ci_dt.core import (
    LayerModel,
    build_xy,
    layer_metrics,
    proba_for_layer,
    predict_all,
    train_trees,
)
from spd.clustering.ci_dt.plot import (
    plot_activations,
    plot_covariance,
    plot_layer_metrics,
    plot_selected_trees,
    plot_tree_statistics,
)
from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, SPDRunInfo


# magic autoreload
%load_ext autoreload
%autoreload 2

# %%
# ----------------------- configuration -----------------------

config = CIDTConfig(
    n_samples=64, # batch size 64 -> 16GB vram
    activation_threshold=0.01,
    filter_dead_threshold=0.001,
    max_depth=8,
    random_state=42,
)
device: str = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# ----------------------- load model -----------------------

wandb_run_path: str = "wandb:goodfire/spd/runs/lxs77xye"

spd_run: SPDRunInfo = SPDRunInfo.from_path(wandb_run_path)
model: ComponentModel = ComponentModel.from_pretrained(spd_run.checkpoint_path)
model.to(device)
cfg: Config = spd_run.config

print(f"Loaded model from {wandb_run_path}")

# %%
# ----------------------- load dataset -----------------------

# Create LM dataset and dataloader
assert isinstance(cfg.task_config, LMTaskConfig)
pretrained_model_name = cfg.pretrained_model_name
assert pretrained_model_name is not None

dataset_config = DatasetConfig(
    name=cfg.task_config.dataset_name,
    hf_tokenizer_path=pretrained_model_name,
    split=cfg.task_config.train_data_split,
    n_ctx=cfg.task_config.max_seq_len,
    column_name=cfg.task_config.column_name,
    is_tokenized=False,
    streaming=False,
    seed=0,
)
dataloader, _ = create_data_loader(
    dataset_config=dataset_config,
    batch_size=config.n_samples,
    buffer_size=cfg.task_config.buffer_size,
    global_seed=cfg.seed,
    ddp_rank=0,
    ddp_world_size=1,
)
batch_data = next(iter(dataloader))
batch: Tensor = batch_data["input_ids"]
print(f"Created LM dataset with {cfg.task_config.dataset_name}, batch shape: {batch.shape}")

# %%
# ----------------------- get activations -----------------------

# Get component activations (on device)
print("Computing component activations...")
component_acts: dict[str, Tensor] = component_activations(
    model=model,
    device=device,
    batch=batch,
)

# Process activations (filter dead components, concatenate)
print("Processing activations...")
processed_acts: ProcessedActivations = process_activations(
    component_acts,
    filter_dead_threshold=config.filter_dead_threshold,
    seq_mode="seq_mean",  # LM task needs seq_mean
)

print(f"Total components (before filtering): {processed_acts.n_components_original}")
print(f"Alive components: {processed_acts.n_components_alive}")
print(f"Dead components: {processed_acts.n_components_dead}")
print(f"Module keys: {processed_acts.module_keys}")

# %%
# ----------------------- convert to layers -----------------------

# Move to CPU and convert to numpy for sklearn
# Group by module to create "layers" for decision trees
print("\nConverting to boolean layers...")
layers_true: list[Bool[np.ndarray, "n_samples n_components"]] = []
for module_key in processed_acts.module_keys:
    # Get the activations for this module from activations_raw, move to CPU
    module_acts_cpu: Float[np.ndarray, "n_samples n_components"] = (
        processed_acts.activations_raw[module_key].cpu().numpy()
    )
    module_acts_bool: Bool[np.ndarray, "n_samples n_components"] = (
        module_acts_cpu >= config.activation_threshold
    ).astype(bool)

    # Filter out components that are always dead or always alive
    # (they provide no information for decision trees)
    n_before: int = module_acts_bool.shape[1]
    component_variance: Float[np.ndarray, "n_components"] = module_acts_bool.var(axis=0)
    varying_mask: Bool[np.ndarray, "n_components"] = component_variance > 0

    # Count always-dead and always-alive components for diagnostics
    always_dead_mask: Bool[np.ndarray, "n_components"] = ~module_acts_bool.any(axis=0)
    always_alive_mask: Bool[np.ndarray, "n_components"] = module_acts_bool.all(axis=0)
    n_always_dead: int = always_dead_mask.sum()
    n_always_alive: int = always_alive_mask.sum()

    module_acts_filtered: Bool[np.ndarray, "n_samples n_varying"] = module_acts_bool[:, varying_mask]
    n_after: int = module_acts_filtered.shape[1]

    layers_true.append(module_acts_filtered)
    print(
        f"Layer {len(layers_true) - 1} ({module_key}): {n_after} varying components "
        f"({n_always_dead} always dead, {n_always_alive} always alive removed)"
    )

print(f"\nCreated {len(layers_true)} layers for decision tree training")

# %%
# ----------------------- fit and predict -----------------------

print("\nTraining decision trees...")
models: list[LayerModel] = train_trees(
    layers_true, max_depth=config.max_depth, random_state=config.random_state
)
layers_pred: list[np.ndarray] = predict_all(models, [layers_true[0]])

# %%
# ----------------------- metrics -----------------------

XYs_demo = build_xy(layers_true)
per_layer_stats: list[dict[str, Any]] = []
all_triplets: list[tuple[int, int, float]] = []  # (layer, target_idx, AP)

for lm, (Xk, Yk) in zip(models, XYs_demo, strict=True):
    Pk: np.ndarray = proba_for_layer(lm, Xk)
    Yhat_k: np.ndarray = Pk >= 0.5
    ap, acc, bacc, prev = layer_metrics(Yk, Pk, Yhat_k)
    per_layer_stats.append(
        {
            "ap": ap,
            "acc": acc,
            "bacc": bacc,
            "prev": prev,
            "mean_ap": float(np.nanmean(ap)),
            "mean_acc": float(np.nanmean(acc)),
            "mean_bacc": float(np.nanmean(bacc)),
        }
    )
    for j, apj in enumerate(ap):
        all_triplets.append((lm.layer_index, j, float(apj)))

# identify best and worst trees across all outputs by AP
sorted_triplets = sorted(all_triplets, key=lambda t: (np.isnan(t[2]), t[2]))
worst_list = [t for t in sorted_triplets if not np.isnan(t[2])][:2]
best_list = [t for t in sorted_triplets if not np.isnan(t[2])][-2:]

# %%
# ----------------------- plot: layer metrics -----------------------
# Simplest - just bar charts and scatter plot of summary statistics

plot_layer_metrics(per_layer_stats)
print("Layer metrics plots generated.")

# %%
# ----------------------- plot: tree statistics -----------------------
# Distributions of tree depth, leaf counts, and correlations with accuracy

plot_tree_statistics(models, per_layer_stats)
print("Tree statistics plots generated.")

# %%
# ----------------------- plot: activations -----------------------
# Simple heatmaps of true vs predicted activations

plot_activations(layers_true, layers_pred)
print("Activation plots generated.")

# %%
# ----------------------- plot: covariance -----------------------
# Covariance matrix - can be slow with many components

plot_covariance(layers_true)
print("Covariance plot generated.")

# %%
# ----------------------- plot: worst trees -----------------------
# Decision tree visualization for worst performing trees

plot_selected_trees(worst_list, "Worst", models)
print("Worst trees plots generated.")

# %%
# ----------------------- plot: best trees -----------------------
# Decision tree visualization for best performing trees

plot_selected_trees(best_list, "Best", models)
print("Best trees plots generated.")
