# %%
"""Minimal single-script version of causal importance decision tree training."""

import json
from typing import Any

import numpy as np
import torch
from jaxtyping import Bool, Float
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from torch import Tensor
from tqdm import tqdm

from spd.configs import Config
from spd.dashboard.core.matshow_sort import sort_by_similarity
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, OutputWithCache, SPDRunInfo

# %% ----------------------- Configuration -----------------------
WANDB_RUN_PATH = "wandb:goodfire/spd/runs/lxs77xye"
BATCH_SIZE = 8
N_BATCHES = 8
N_CTX = 32
ACTIVATION_THRESHOLD = 0.01
MAX_DEPTH = 3
RANDOM_STATE = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

# %% ----------------------- Load Model -----------------------
print(f"Loading model from {WANDB_RUN_PATH}...")
spd_run: SPDRunInfo = SPDRunInfo.from_path(WANDB_RUN_PATH)
model: ComponentModel = ComponentModel.from_pretrained(spd_run.checkpoint_path)
model.to(device)
cfg: Config = spd_run.config

# %% ----------------------- Load Dataset -----------------------
assert isinstance(cfg.task_config, LMTaskConfig)
assert cfg.pretrained_model_name is not None

dataset_config = DatasetConfig(
    name=cfg.task_config.dataset_name,
    hf_tokenizer_path=cfg.pretrained_model_name,
    split=cfg.task_config.train_data_split,
    n_ctx=N_CTX,
    column_name=cfg.task_config.column_name,
    is_tokenized=False,
    streaming=False,
    seed=0,
)
dataloader, _ = create_data_loader(
    dataset_config=dataset_config,
    batch_size=BATCH_SIZE,
    buffer_size=cfg.task_config.buffer_size,
    global_seed=cfg.seed,
    ddp_rank=0,
    ddp_world_size=1,
)

# %% ----------------------- Compute Activations -----------------------
print(f"\nComputing activations for {N_BATCHES} batches...")
all_acts: list[dict[str, Tensor]] = []

for _ in tqdm(range(N_BATCHES), desc="Batches"):
    batch: Tensor = next(iter(dataloader))["input_ids"]
    with torch.no_grad():
        output: OutputWithCache = model(batch.to(device), cache_type="input")
        acts: dict[str, Tensor] = model.calc_causal_importances(
            pre_weight_acts=output.cache,
            sampling="continuous",
            detach_inputs=False,
        ).upper_leaky
    all_acts.append({k: v.cpu() for k, v in acts.items()})

# Concatenate batches
module_keys = list(all_acts[0].keys())
acts_concat: dict[str, Tensor] = {
    k: torch.cat([b[k] for b in all_acts], dim=0) for k in module_keys
}

# %% ----------------------- Convert to Boolean Layers -----------------------
print("\nConverting to boolean and filtering constant components...")
layers: list[Bool[np.ndarray, "n_samples n_components"]] = []
# Track which components are varying (not constant) in each layer
varying_component_indices: list[list[int]] = []

for k in module_keys:
    # Flatten if 3D (batch, seq, components) -> (batch*seq, components)
    acts_tensor = acts_concat[k]
    if acts_tensor.ndim == 3:
        acts_np: Float[np.ndarray, "n_samples n_components"] = acts_tensor.reshape(
            -1, acts_tensor.shape[-1]
        ).numpy()
    else:
        acts_np = acts_tensor.numpy()

    # Threshold to boolean
    acts_bool: Bool[np.ndarray, "n_samples n_components"] = (
        acts_np >= ACTIVATION_THRESHOLD
    ).astype(bool)

    # plt.title(f"{k}")
    # sort by column similarity
    acts_sorted = sort_by_similarity(sort_by_similarity(acts_bool.astype(float), axis=0), axis=1)
    plt.matshow(acts_sorted[:, :600], aspect="auto")
    plt.show()

    # Filter constant components (always 0 or always 1)
    varying_mask: Bool[np.ndarray, " n_components"] = acts_bool.var(axis=0) > 0
    acts_varying = acts_bool[:, varying_mask]
    layers.append(acts_varying)
    # Store which original component indices were kept
    varying_component_indices.append(np.where(varying_mask)[0].tolist())
    print(f"  {k}: {acts_varying.shape[1]} varying components")

# %% ----------------------- Train Decision Trees -----------------------
print("\nTraining decision trees...")
# Build (X, Y) pairs: X_k = concat(layers[:k]), Y_k = layers[k]
models: list[tuple[int, MultiOutputClassifier]] = []

for k in tqdm(range(1, len(layers)), desc="Training"):
    X_prev_layers_cis = (
        np.concatenate(layers[:k], axis=1) if k > 0 else np.zeros((layers[0].shape[0], 0), bool)
    )
    Y_current_layer_cis = layers[k]

    clf = MultiOutputClassifier(
        DecisionTreeClassifier(
            max_depth=MAX_DEPTH,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
        )
    )
    clf.fit(X_prev_layers_cis.astype(np.uint8), Y_current_layer_cis.astype(np.uint8))
    models.append((k, clf))

# %% ----------------------- Compute Metrics -----------------------
print("\nComputing metrics...")


def extract_prob_class_1(proba_list: list[np.ndarray], clf: MultiOutputClassifier) -> np.ndarray:
    """Extract P(y=1) for each output."""
    result: list[np.ndarray] = []
    for i, p in enumerate(proba_list):
        estimator = clf.estimators_[i]  # pyright: ignore[reportIndexIssue]
        assert isinstance(estimator, DecisionTreeClassifier)
        assert len(estimator.classes_) == 2
        result.append(p[:, 1])  # P(y=1)
    return np.stack(result, axis=1)


def tree_to_dict(tree: DecisionTreeClassifier) -> dict[str, Any]:
    """Convert sklearn DecisionTree to JSON-serializable dict."""
    tree_ = tree.tree_
    return {
        "feature": tree_.feature.tolist(),
        "threshold": tree_.threshold.tolist(),
        "children_left": tree_.children_left.tolist(),
        "children_right": tree_.children_right.tolist(),
        "value": tree_.value.tolist(),
        "n_node_samples": tree_.n_node_samples.tolist(),
    }


def build_feature_map(layer_idx: int) -> list[dict[str, Any]]:
    """Build feature map for layer k: maps feature index -> component identity."""
    feature_map: list[dict[str, Any]] = []
    for prev_layer_idx in range(layer_idx):
        for comp_idx in varying_component_indices[prev_layer_idx]:
            module_key = module_keys[prev_layer_idx]
            feature_map.append({
                "layer_idx": prev_layer_idx,
                "module_key": module_key,
                "component_idx": comp_idx,
                "label": f"{module_key}:{comp_idx}",
            })
    return feature_map


# Collect all results for saving
results: list[dict[str, Any]] = []

print("\nPer-layer metrics:")
for layer_idx, clf in models:
    # Prepare X, Y for this layer
    X_prev_layers_cis = np.concatenate(layers[:layer_idx], axis=1)
    Y_current_layer_cis = layers[layer_idx]

    # Predict
    proba_list = clf.predict_proba(X_prev_layers_cis.astype(np.uint8))  # type: ignore
    P = extract_prob_class_1(proba_list, clf)
    Y_pred = P >= 0.5

    # Compute metrics per component
    ap_scores: list[float] = []
    acc_scores: list[float] = []
    bacc_scores: list[float] = []

    for j in range(Y_current_layer_cis.shape[1]):
        y_true = Y_current_layer_cis[:, j].astype(int)
        y_prob = P[:, j]
        y_pred = Y_pred[:, j].astype(int)

        ap_scores.append(average_precision_score(y_true, y_prob))  # pyright: ignore[reportArgumentType]
        acc_scores.append(accuracy_score(y_true, y_pred))
        bacc_scores.append(balanced_accuracy_score(y_true, y_pred))

    # Print summary
    print(f"  Layer {layer_idx} ({module_keys[layer_idx]}):")
    print(f"    Mean AP:   {np.mean(ap_scores):.3f}")
    print(f"    Mean Acc:  {np.mean(acc_scores):.3f}")
    print(f"    Mean BAcc: {np.mean(bacc_scores):.3f}")

    # Store results with tree structures
    trees_data = [tree_to_dict(est) for est in clf.estimators_]  # pyright: ignore[reportArgumentType]

    # Build feature map for this layer
    feature_map = build_feature_map(layer_idx)

    # Build labels for varying components in this layer
    varying_labels = [
        f"{module_keys[layer_idx]}:{idx}"
        for idx in varying_component_indices[layer_idx]
    ]

    results.append(
        {
            "layer_idx": layer_idx,
            "module_key": module_keys[layer_idx],
            "feature_map": feature_map,
            "varying_component_indices": varying_component_indices[layer_idx],
            "varying_component_labels": varying_labels,
            "trees": trees_data,
            "ap_scores": ap_scores,
            "acc_scores": acc_scores,
            "bacc_scores": bacc_scores,
            "mean_ap": float(np.mean(ap_scores)),
            "mean_acc": float(np.mean(acc_scores)),
            "mean_bacc": float(np.mean(bacc_scores)),
        }
    )

# %% ----------------------- Save Trees -----------------------
output_path = "data/trees.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results)} layers with trees and metrics to {output_path}")
print("Done!")
