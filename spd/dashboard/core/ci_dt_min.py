# %%
"""Minimal single-script version of causal importance decision tree training."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy import ndarray
from jaxtyping import Float, Int, Bool
import torch
from tqdm import tqdm
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier

from spd.dashboard.core.acts import Activations
from spd.dashboard.core.compute import FlatActivations

# %% ----------------------- Configuration -----------------------


@dataclass
class TreeConfig:
    wandb_run_path: str = "wandb:goodfire/spd/runs/lxs77xye"
    batch_size: int = 4
    n_batches: int = 4
    n_ctx: int = 16
    activation_threshold: float = 0.01
    max_depth: int = 3
    random_state: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CONFIG: TreeConfig = TreeConfig()

# %% ----------------------- get activations -----------------------

FLAT_ACTIVATIONS: FlatActivations = FlatActivations.create(
	Activations.generate(
		wandb_run_path=CONFIG.wandb_run_path,
		n_batches=CONFIG.n_batches,
		n_ctx=CONFIG.n_ctx,
		device=CONFIG.device,
		activation_threshold=CONFIG.activation_threshold,
	)
)


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
        X_prev_layers_cis: Bool[ndarray, "n_samples n_features_before"] = flat_acts.get_concat_before_module(module_name) > CONFIG.activation_threshold
        Y_current_layer_cis: Bool[ndarray, "n_samples n_features_this"] = flat_acts.get_concat_this_module(module_name) > CONFIG.activation_threshold

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
