from dataclasses import dataclass
from typing import Any

import numpy as np
from jaxtyping import Bool, Int
from muutils.json_serialize import SerializableDataclass, serializable_dataclass, serializable_field
from numpy import ndarray
from sklearn.metrics import balanced_accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from spd.dashboard.core.acts import ComponentLabel
from spd.dashboard.core.compute import FlatActivations
from spd.dashboard.core.dashboard_config import ComponentDashboardConfig


@dataclass
class ComponentTreeData:
    """Decision tree data for a single component."""

    component_label: ComponentLabel
    component_index: int
    tree_dict: dict[str, Any]
    balanced_accuracy: float


@dataclass
class LayerTreeData:
    """Decision tree data for a single layer."""

    module_name: str
    layer_index: int
    component_trees: list[ComponentTreeData]
    mean_balanced_accuracy: float


@dataclass
class DecisionTreesData:
    """Decision tree data for all layers."""

    layers: list[LayerTreeData]
    overall_mean_balanced_accuracy: float

    @classmethod
    def create(
        cls,
        flat_acts: FlatActivations,
        config: ComponentDashboardConfig,
    ) -> "DecisionTreesData":
        """Train decision trees and compute metrics for all layers."""
        # Train decision trees and get training data
        layer_results: dict[
            str,
            tuple[
                MultiOutputClassifier,
                Bool[ndarray, "n_samples n_features"],
                Bool[ndarray, "n_samples n_features"],
            ],
        ] = train_decision_trees(
            flat_acts,
            max_depth=config.max_depth,
            activation_threshold=config.activation_threshold,
            random_state=config.random_state,
        )

        layers: list[LayerTreeData] = []
        all_bacc_scores: list[float] = []

        # Process each layer
        layer_idx: int
        module_name: str
        for layer_idx, module_name in enumerate(list(flat_acts.layer_order)[1:], start=1):
            clf: MultiOutputClassifier
            X_bool: Bool[ndarray, "n_samples n_features_before"]
            Y_bool: Bool[ndarray, "n_samples n_features_this"]
            clf, X_bool, Y_bool = layer_results[module_name]

            # Get predictions
            Y_pred: Bool[ndarray, "n_samples n_features_this"] = clf.predict(
                X_bool.astype(np.uint8)
            )

            # Get component labels for this layer
            layer_component_labels: list[ComponentLabel] = [
                c for c in flat_acts.component_labels if c.module == module_name
            ]

            # Serialize all trees for this layer at once
            tree_dicts: list[dict[str, Any]] = moc_to_dicts(clf)

            component_trees: list[ComponentTreeData] = []
            layer_bacc_scores: list[float] = []

            # Process each component
            j: int
            for j in range(Y_bool.shape[1]):
                y_true: Bool[ndarray, " n_samples"] = Y_bool[:, j]
                y_pred: Bool[ndarray, " n_samples"] = Y_pred[:, j]
                bacc: float = float(balanced_accuracy_score(y_true, y_pred))
                layer_bacc_scores.append(bacc)
                all_bacc_scores.append(bacc)

                # Get tree dict for this component
                tree_dict: dict[str, Any] = tree_dicts[j]

                component_trees.append(
                    ComponentTreeData(
                        component_label=layer_component_labels[j],
                        component_index=j,
                        tree_dict=tree_dict,
                        balanced_accuracy=bacc,
                    )
                )

            # Create layer data
            layer_data: LayerTreeData = LayerTreeData(
                module_name=module_name,
                layer_index=layer_idx,
                component_trees=component_trees,
                mean_balanced_accuracy=float(np.mean(layer_bacc_scores)),
            )
            layers.append(layer_data)

        # Compute overall mean
        overall_mean: float = float(np.mean(all_bacc_scores)) if all_bacc_scores else 0.0

        return cls(layers=layers, overall_mean_balanced_accuracy=overall_mean)


def train_decision_trees(
    flat_acts: FlatActivations,
    max_depth: int,
    activation_threshold: float = 0.1,
    random_state: int = 42,
) -> dict[
    str,
    tuple[
        MultiOutputClassifier,
        Bool[ndarray, "n_samples n_features"],
        Bool[ndarray, "n_samples n_features"],
    ],
]:
    """Train decision trees and return models with their training data.

    Returns:
        dict mapping module_name -> (trained_model, X_bool, Y_bool)
    """
    print("\nTraining decision trees...")
    layer_trees: dict[
        str,
        tuple[
            MultiOutputClassifier,
            Bool[ndarray, "n_samples n_features"],
            Bool[ndarray, "n_samples n_features"],
        ],
    ] = {}

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

        # Attach feature labels to the MultiOutputClassifier
        start_idx: int = flat_acts.start_of_module_index(module_name)
        feature_labels: list[str] = [
            c.to_string() for c in flat_acts.component_labels[:start_idx]
        ]
        clf.feature_labels_ = feature_labels  # type: ignore

        # Store model along with the data for later evaluation
        layer_trees[module_name] = (clf, X_prev_layers_cis, Y_current_layer_cis)

    return layer_trees


def moc_to_dicts(clf: MultiOutputClassifier) -> list[dict[str, Any]]:
    """Convert MultiOutputClassifier trees to list of JSON-serializable dicts.

    Raises:
        AssertionError: If feature_labels_ not found on clf
    """
    assert hasattr(clf, "feature_labels_"), (
        "MultiOutputClassifier missing feature_labels_. Ensure train_decision_trees() attached them."
    )

    all_feature_labels: list[str] = clf.feature_labels_  # type: ignore

    tree_dicts: list[dict[str, Any]] = []
    estimator: DecisionTreeClassifier
    for estimator in clf.estimators_:  # type: ignore
        tree_ = estimator.tree_

        # Get unique feature indices actually used in this tree
        # (tree_.feature contains -2 for leaf nodes, filter those out)
        features_array: Int[np.ndarray, " n_nodes"] = tree_.feature
        used_features: set[int] = set(features_array[features_array >= 0].tolist())

        # Create mapping for only the used features
        feature_labels_map: dict[int, str] = {
            idx: all_feature_labels[idx] for idx in used_features
        }

        tree_dicts.append({
            "feature": tree_.feature.tolist(),
            "threshold": tree_.threshold.tolist(),
            "children_left": tree_.children_left.tolist(),
            "children_right": tree_.children_right.tolist(),
            "value": tree_.value.tolist(),
            "n_node_samples": tree_.n_node_samples.tolist(),
            "feature_labels": feature_labels_map,  # dict[int, str] mapping index -> label
        })

    return tree_dicts
