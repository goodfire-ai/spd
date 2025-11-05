from typing import Any

import numpy as np
from jaxtyping import Bool, Float, Int
from muutils.json_serialize import SerializableDataclass, serializable_dataclass, serializable_field
from numpy import ndarray
from sklearn.metrics import balanced_accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from spd.dashboard.core.acts import Activations, ComponentLabel
from spd.dashboard.core.compute import FlatActivations
from spd.dashboard.core.dashboard_config import ComponentDashboardConfig


@serializable_dataclass  # pyright: ignore[reportUntypedClassDecorator]
class ComponentTreeData(SerializableDataclass):
    """Decision tree data for a single component."""

    component_label: ComponentLabel
    component_index: int
    tree_dict: dict[str, Any]
    balanced_accuracy: float


@serializable_dataclass  # pyright: ignore[reportUntypedClassDecorator]
class LayerTreeData(SerializableDataclass):
    """Decision tree data for a single layer."""

    module_name: str
    layer_index: int
    component_trees: list[ComponentTreeData] = serializable_field(
        serialization_fn=lambda x: [c.serialize() for c in x],
        deserialize_fn=lambda x: [ComponentTreeData.load(c) for c in x],
    )
    mean_balanced_accuracy: float


@serializable_dataclass  # pyright: ignore[reportUntypedClassDecorator]
class DecisionTreesData(SerializableDataclass):
    """Decision tree data for all layers."""

    layers: list[LayerTreeData] = serializable_field(
        serialization_fn=lambda x: [l.serialize() for l in x],
        deserialize_fn=lambda x: [LayerTreeData.load(l) for l in x],
    )
    overall_mean_balanced_accuracy: float

    @classmethod
    def create(
        cls,
        flat_acts: FlatActivations,
        config: ComponentDashboardConfig,
    ) -> "DecisionTreesData":
        """Train decision trees and compute metrics for all layers."""
        raise NotImplementedError("TODO: Implement DecisionTreesData.create()")


def train_decision_trees(
    flat_acts: FlatActivations,
    max_depth: int,
	activation_threshold: float = 0.1,
    random_state: int = 42,
) -> dict[str, MultiOutputClassifier]:
    """Train decision trees to predict each layer from previous layers."""
    print("\nTraining decision trees...")
    layer_trees: dict[str, MultiOutputClassifier] = {}

    # Skip first layer (no previous layers to predict from)
    for module_name in tqdm(list(flat_acts.layer_order)[1:]):
        X_prev_layers_cis: Bool[ndarray, "n_samples n_features_before"] = (
            flat_acts.get_concat_before_module(module_name) > activation_threshold
        )
        Y_current_layer_cis: Bool[ndarray, "n_samples n_features_this"] = (
            flat_acts.get_concat_this_module(module_name) > activation_threshold
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