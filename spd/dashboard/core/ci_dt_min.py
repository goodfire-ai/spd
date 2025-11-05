# %%
"""Minimal single-script version of causal importance decision tree training."""

from dataclasses import dataclass
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

@dataclass
class TreeConfig:
    wandb_run_path: str = "wandb:goodfire/spd/runs/lxs77xye"
    batch_size: int = 8
    n_batches: int = 8
    n_ctx: int = 32
    activation_threshold: float = 0.01
    max_depth: int = 3
    random_state: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CONFIG: TreeConfig = TreeConfig()




# %% ----------------------- Train Decision Trees -----------------------
def train_decision_trees(
    layer_acts: LayerActivations,
    max_depth: int,
    random_state: int,
) -> dict[str, MultiOutputClassifier]:
    """Train decision trees to predict each layer from previous layers."""
    print("\nTraining decision trees...")
    models: dict[str, MultiOutputClassifier] = {}

    # Skip first layer (no previous layers to predict from)
    for module_name in list(layer_acts)[1:]:
        X_prev_layers_cis = layer_acts.get_concat_before(module_name)
        Y_current_layer_cis = layer_acts.data[module_name]

        clf = MultiOutputClassifier(
            DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=1,
                random_state=random_state,
            )
        )
        clf.fit(X_prev_layers_cis.astype(np.uint8), Y_current_layer_cis.astype(np.uint8))
        models[module_name] = clf

    return models


# %% ----------------------- Compute Metrics -----------------------
def compute_metrics_and_save(
    models: dict[str, MultiOutputClassifier],
    layer_acts: LayerActivations,
    output_path: str,
) -> list[dict[str, Any]]:
    """Compute metrics, serialize trees, and save to JSON."""
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

    def build_feature_map(module_name: str) -> list[dict[str, Any]]:
        """Build feature map for module: maps feature index -> component identity."""
        feature_map: list[dict[str, Any]] = []
        module_idx = layer_acts.layer_order.index(module_name)
        for prev_idx in range(module_idx):
            prev_module = layer_acts.layer_order[prev_idx]
            for comp_idx in layer_acts.varying_component_indices[prev_module]:
                feature_map.append({
                    "layer_idx": prev_idx,
                    "module_key": prev_module,
                    "component_idx": comp_idx,
                    "label": f"{prev_module}:{comp_idx}",
                })
        return feature_map

    # Collect all results for saving
    results: list[dict[str, Any]] = []

    print("\nPer-layer metrics:")
    for module_name, clf in models.items():
        # Prepare X, Y for this layer
        X_prev_layers_cis = layer_acts.get_concat_before(module_name)
        Y_current_layer_cis = layer_acts.data[module_name]

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
        module_idx = layer_acts.layer_order.index(module_name)
        print(f"  Layer {module_idx} ({module_name}):")
        print(f"    Mean AP:   {np.mean(ap_scores):.3f}")
        print(f"    Mean Acc:  {np.mean(acc_scores):.3f}")
        print(f"    Mean BAcc: {np.mean(bacc_scores):.3f}")

        # Store results with tree structures
        trees_data = [tree_to_dict(est) for est in clf.estimators_]  # pyright: ignore[reportArgumentType]

        # Build feature map for this layer
        feature_map = build_feature_map(module_name)

        # Build labels for varying components in this layer
        varying_labels = [
            f"{module_name}:{idx}"
            for idx in layer_acts.varying_component_indices[module_name]
        ]

        results.append(
            {
                "layer_idx": module_idx,
                "module_key": module_name,
                "feature_map": feature_map,
                "varying_component_indices": layer_acts.varying_component_indices[module_name],
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

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} layers with trees and metrics to {output_path}")
    print("Done!")

    return results


layer_acts = get_acts(
    wandb_run_path=CONFIG.wandb_run_path,
    n_batches=CONFIG.n_batches,
    n_ctx=CONFIG.n_ctx,
    device=CONFIG.device,
    activation_threshold=CONFIG.activation_threshold,
)

models = train_decision_trees(
    layer_acts=layer_acts,
    max_depth=CONFIG.max_depth,
    random_state=CONFIG.random_state,
)

results = compute_metrics_and_save(
    models=models,
    layer_acts=layer_acts,
    output_path="data/trees.json",
)
