# %%
"""Main execution script for causal importance decision tree training."""

import numpy as np
import torch
from jaxtyping import Bool
from torch import Tensor

from spd.clustering.ci_dt.config import CIDTConfig
from spd.clustering.ci_dt.core import LayerModel, predict_all, train_trees
from spd.clustering.ci_dt.pipeline import (
    compute_activations_multibatch,
    compute_tree_metrics,
    convert_to_boolean_layers,
)
from spd.clustering.ci_dt.plot import (
    greedy_sort,
    plot_activations,
    plot_ap_vs_prevalence,
    plot_component_activity_breakdown,
    plot_covariance,
    plot_metric,
    plot_selected_trees,
    plot_tree_statistics,
)
from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, SPDRunInfo

# magic autoreload
# %load_ext autoreload
# %autoreload 2

# %%
# ----------------------- configuration -----------------------

config = CIDTConfig(
    batch_size=32,  # 50 ~~ 16GB VRAM max
    n_batches=8,
    # batch_size=2,
    # n_batches=2,
    n_ctx=16,
    activation_threshold=0.01,
    max_depth=3,
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
    n_ctx=config.n_ctx,
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
# Get module keys for labeling
module_keys: list[str] = list(component_acts_concat.keys())

# %%
# ----------------------- plot: component activity breakdown -----------------------

plot_component_activity_breakdown(
    component_acts_concat,
    module_keys,
    config.activation_threshold,
    logy=False,
)
plot_component_activity_breakdown(
    component_acts_concat,
    module_keys,
    config.activation_threshold,
    logy=True,
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
# ----------------------- compute orderings -----------------------
# Generate sample ordering once for use in multiple plots

# Concatenate true activations for ordering
A_true_concat: np.ndarray = np.concatenate(layers_true, axis=1).astype(float)

# Compute sample ordering by similarity
sample_order: np.ndarray = greedy_sort(A_true_concat, axis=0)
print(f"Computed sample ordering ({len(sample_order)} samples)")

# %%
# ----------------------- plots: metrics -----------------------

plot_metric(per_layer_stats, module_keys, "ap")
plot_metric(per_layer_stats, module_keys, "acc")
plot_metric(per_layer_stats, module_keys, "bacc")
plot_metric(per_layer_stats, module_keys, "prev")
plot_metric(per_layer_stats, module_keys, "tpr")
plot_metric(per_layer_stats, module_keys, "tnr")
plot_metric(per_layer_stats, module_keys, "precision")
plot_metric(per_layer_stats, module_keys, "npv")
plot_metric(per_layer_stats, module_keys, "f1")
plot_ap_vs_prevalence(per_layer_stats, models)


# %%
# ----------------------- plots: tree statistics -----------------------

plot_tree_statistics(models, per_layer_stats)

# %%
# ----------------------- plots: activations -----------------------

plot_activations(
    layers_true=layers_true,
    layers_pred=layers_pred,
    module_keys=module_keys,
    activation_threshold=config.activation_threshold,
    sample_order=None,
)

# plot_activations(
#     layers_true=layers_true,
#     layers_pred=layers_pred,
#     module_keys=module_keys,
#     activation_threshold=config.activation_threshold,
#     sample_order=sample_order,
# )

# %%
# ----------------------- plots: covariance -----------------------

plot_covariance(
    layers_true=layers_true,
    module_keys=module_keys,
    component_order=None,
)

component_order: np.ndarray = greedy_sort(A_true_concat, axis=1)
plot_covariance(
    layers_true=layers_true,
    module_keys=module_keys,
    component_order=component_order,
)

# %%
# ----------------------- plots: decision trees -----------------------

plot_selected_trees(worst_list, "Worst", models)
plot_selected_trees(best_list, "Best", models)
