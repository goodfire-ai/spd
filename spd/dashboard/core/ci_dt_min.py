# %%
"""Minimal single-script version of causal importance decision tree training."""

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier

from spd.dashboard.core.ci_dt.acts import LayerActivations

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

ACTIVATIONS: LayerActivations = LayerActivations.generate(
    wandb_run_path=CONFIG.wandb_run_path,
    n_batches=CONFIG.n_batches,
    n_ctx=CONFIG.n_ctx,
    device=CONFIG.device,
    activation_threshold=CONFIG.activation_threshold,
)


# %% ----------------------- Train Decision Trees -----------------------
def train_decision_trees(
    layer_acts: LayerActivations,
    max_depth: int,
    random_state: int,
) -> dict[str, MultiOutputClassifier]:
    """Train decision trees to predict each layer from previous layers."""
    print("\nTraining decision trees...")
    layer_trees: dict[str, MultiOutputClassifier] = {}

    # Skip first layer (no previous layers to predict from)
    for module_name in list(layer_acts)[1:]:
        X_prev_layers_cis = layer_acts.get_concat_before(module_name)
        Y_current_layer_cis = layer_acts.data[module_name]

        clf: MultiOutputClassifier = MultiOutputClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=1,
                random_state=random_state,
            )
        )
        clf.fit(X=X_prev_layers_cis.astype(np.uint8), Y=Y_current_layer_cis.astype(np.uint8))
        layer_trees[module_name] = clf

    return layer_trees


LAYER_TREES: dict[str, MultiOutputClassifier] = train_decision_trees(
    layer_acts=ACTIVATIONS,
    max_depth=CONFIG.max_depth,
    random_state=CONFIG.random_state,
)


# %% ----------------------- Compute Metrics -----------------------


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


results = compute_metrics_and_save(
    models=LAYER_TREES,
    layer_acts=ACTIVATIONS,
    output_path="data/trees.json",
)
