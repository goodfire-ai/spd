import json
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from spd.dashboard.core.ci_dt.acts import Activations


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


def compute_metrics_and_save(
    models: dict[str, MultiOutputClassifier],
    layer_acts: Activations,
    output_path: str,
) -> list[dict[str, Any]]:
    """Compute metrics, serialize trees, and save to JSON."""
    print("\nComputing metrics...")

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
        feature_map = layer_acts.build_feature_map(module_name)

        # Build labels for varying components in this layer
        varying_labels = [
            f"{module_name}:{idx}" for idx in layer_acts.varying_component_indices[module_name]
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


def extract_prob_class_1(proba_list: list[np.ndarray], clf: MultiOutputClassifier) -> np.ndarray:
    """Extract P(y=1) for each output."""
    result: list[np.ndarray] = []
    for i, p in enumerate(proba_list):
        estimator = clf.estimators_[i]  # pyright: ignore[reportIndexIssue]
        assert isinstance(estimator, DecisionTreeClassifier)
        assert len(estimator.classes_) == 2
        result.append(p[:, 1])  # P(y=1)
    return np.stack(result, axis=1)
