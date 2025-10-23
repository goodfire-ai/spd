# %%
"""Main execution script for causal importance decision tree training."""

from typing import Any

import numpy as np
import torch
from jaxtyping import Bool, Float
from torch import Tensor

from spd.clustering.ci_dt.config import CIDTConfig
from spd.clustering.ci_dt.core import LayerModel, predict_all, train_trees
from spd.clustering.ci_dt.pipeline import (
    compute_activations_multibatch,
    compute_tree_metrics,
    convert_to_boolean_layers,
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
    batch_size=50, # 50 ~~ 16GB VRAM
    n_batches=4,
    activation_threshold=0.01,
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
    batch_size=config.batch_size,
    buffer_size=cfg.task_config.buffer_size,
    global_seed=cfg.seed,
    ddp_rank=0,
    ddp_world_size=1,
)
print(f"Created LM dataset with {cfg.task_config.dataset_name}")

# %%
# ----------------------- get activations -----------------------

component_acts_concat: dict[str, Tensor] = compute_activations_multibatch(
    model=model,
    device=device,
    dataloader=dataloader,
    n_batches=config.n_batches,
)

# %%
# ----------------------- convert to boolean layers -----------------------

layers_true: list[Bool[np.ndarray, "n_samples n_components"]] = convert_to_boolean_layers(
    component_acts=component_acts_concat,
    activation_threshold=config.activation_threshold,
)

# %%
# ----------------------- fit and predict -----------------------

print("\nTraining decision trees...")
models: list[LayerModel] = train_trees(
    layers_true, max_depth=config.max_depth, random_state=config.random_state
)
layers_pred: list[np.ndarray] = predict_all(models, [layers_true[0]])

# %%
# ----------------------- metrics -----------------------

per_layer_stats, worst_list, best_list = compute_tree_metrics(
    models=models,
    layers_true=layers_true,
)

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
# ----------------------- generate feature names -----------------------
# Generate feature names with activation statistics and decoded directions

from spd.clustering.ci_dt.feature_names import generate_feature_names

module_keys = list(component_acts_concat.keys())

feature_names = generate_feature_names(
    component_model=model,
    component_acts=component_acts_concat,
    layers_true=layers_true,
    layers_pred=layers_pred,
    tokenizer=cfg.task_config.tokenizer if hasattr(cfg.task_config, 'tokenizer') else None,
    module_keys=module_keys,
    top_k=3,
)
print("Feature names generated.")

# %%
# ----------------------- plot: worst trees -----------------------
# Decision tree visualization for worst performing trees

plot_selected_trees(worst_list, "Worst", models, feature_names=feature_names)
print("Worst trees plots generated.")

# %%
# ----------------------- plot: best trees -----------------------
# Decision tree visualization for best performing trees

plot_selected_trees(best_list, "Best", models, feature_names=feature_names)
print("Best trees plots generated.")
