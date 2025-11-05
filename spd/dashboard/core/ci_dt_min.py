# %%
"""Minimal single-script version of causal importance decision tree training."""

from typing import Any

import numpy as np
from jaxtyping import Bool
from numpy import ndarray
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from spd.dashboard.core.acts import Activations
from spd.dashboard.core.compute import FlatActivations
from spd.dashboard.core.dashboard_config import ComponentDashboardConfig

# %% ----------------------- Configuration -----------------------

CONFIG = ComponentDashboardConfig(
    model_path="wandb:goodfire/spd/runs/lxs77xye",
    batch_size=4,
    n_batches=4,
    context_length=16,
)

# %% ----------------------- get activations -----------------------

FLAT_ACTIVATIONS: FlatActivations = FlatActivations.create(Activations.generate(config=CONFIG))


# %% ----------------------- Train Decision Trees -----------------------
def train_decision_trees(
    flat_acts: FlatActivations,
    max_depth: int,
    random_state: int,
) -> dict[str, MultiOutputClassifier]:
    """Train decision trees to predict each layer from previous layers."""
    print("\nTraining decision trees...")
    layer_trees: dict[str, MultiOutputClassifier] = {}

    # Skip first layer (no previous layers to predict from)
    for module_name in tqdm(list(flat_acts.layer_order)[1:]):
        X_prev_layers_cis: Bool[ndarray, "n_samples n_features_before"] = (
            flat_acts.get_concat_before_module(module_name) > CONFIG.activation_threshold
        )
        Y_current_layer_cis: Bool[ndarray, "n_samples n_features_this"] = (
            flat_acts.get_concat_this_module(module_name) > CONFIG.activation_threshold
        )

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
    flat_acts=FLAT_ACTIVATIONS,
    max_depth=CONFIG.max_depth,
    random_state=CONFIG.random_state,
)


# %% ----------------------- Compute Metrics -----------------------


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
