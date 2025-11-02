"""Serialization for causal importance decision trees."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from spd.clustering.ci_dt.config import CIDTConfig
from spd.clustering.ci_dt.core import LayerModel


@dataclass
class TreeNode:
    """A single node in the decision tree (nested structure)."""

    is_leaf: bool
    n_samples: int
    value: list[float]  # Prediction probabilities [P(class=0), P(class=1)]

    # Only for non-leaf nodes:
    feature: int | None = None  # Which component to check
    left: "TreeNode | None" = None  # Left child (feature is False/0)
    right: "TreeNode | None" = None  # Right child (feature is True/1)

    def serialize(self) -> dict[str, Any]:
        """Recursively serialize to nested dict."""
        result: dict[str, Any] = {
            "is_leaf": self.is_leaf,
            "n_samples": self.n_samples,
            "value": self.value,
        }

        if not self.is_leaf:
            assert self.feature is not None
            assert self.left is not None
            assert self.right is not None
            result["feature"] = self.feature
            result["left"] = self.left.serialize()
            result["right"] = self.right.serialize()

        return result

    @classmethod
    def from_sklearn(cls, tree: DecisionTreeClassifier, node_id: int = 0) -> "TreeNode":
        """Recursively build nested structure from sklearn tree."""
        sklearn_tree = tree.tree_

        # Extract node info
        n_samples = int(sklearn_tree.n_node_samples[node_id])
        value = sklearn_tree.value[node_id][0].tolist()  # Extract [n_samples, n_classes]

        # Check if leaf
        left_child = int(sklearn_tree.children_left[node_id])
        right_child = int(sklearn_tree.children_right[node_id])
        is_leaf = left_child == right_child  # Both -1 for leaves

        if is_leaf:
            return cls(is_leaf=True, n_samples=n_samples, value=value)

        # Non-leaf: recursively build children
        feature = int(sklearn_tree.feature[node_id])
        return cls(
            is_leaf=False,
            n_samples=n_samples,
            value=value,
            feature=feature,
            left=cls.from_sklearn(tree, left_child),
            right=cls.from_sklearn(tree, right_child),
        )


@dataclass
class SavedTree:
    """A single decision tree with metadata."""

    layer_index: int  # Which layer this tree predicts
    target_index: int  # Which component within that layer
    metrics: dict[str, float]  # Performance metrics and tree stats
    structure: TreeNode

    def serialize(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "layer_index": self.layer_index,
            "target_index": self.target_index,
            "metrics": self.metrics,
            "structure": self.structure.serialize(),
        }

    @classmethod
    def from_sklearn(
        cls,
        layer_index: int,
        target_index: int,
        tree: DecisionTreeClassifier,
        metrics_dict: dict[str, np.ndarray],
    ) -> "SavedTree":
        """Create from sklearn tree and metrics."""
        # Extract metrics for this specific target
        metrics = {
            "ap": float(metrics_dict["ap"][target_index]),
            "acc": float(metrics_dict["acc"][target_index]),
            "bacc": float(metrics_dict["bacc"][target_index]),
            "f1": float(metrics_dict["f1"][target_index]),
            "precision": float(metrics_dict["precision"][target_index]),
            "recall": float(metrics_dict["tpr"][target_index]),  # recall = TPR
            "tpr": float(metrics_dict["tpr"][target_index]),
            "tnr": float(metrics_dict["tnr"][target_index]),
            "npv": float(metrics_dict["npv"][target_index]),
            "prev": float(metrics_dict["prev"][target_index]),
            "depth": int(tree.get_depth()),
            "n_nodes": int(tree.tree_.node_count),
            "n_leaves": int(tree.get_n_leaves()),
        }

        return cls(
            layer_index=layer_index,
            target_index=target_index,
            metrics=metrics,
            structure=TreeNode.from_sklearn(tree),
        )


@dataclass
class TreeCollection:
    """Collection of all trees with configuration and metadata."""

    config: CIDTConfig
    module_keys: list[str]  # Layer names
    training_info: dict[str, Any]  # timestamp, device, n_samples, etc.
    trees: list[SavedTree]

    def serialize(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "config": asdict(self.config),
            "module_keys": self.module_keys,
            "training_info": self.training_info,
            "trees": [tree.serialize() for tree in self.trees],
        }

    def save_json(self, path: str | Path) -> None:
        """Save to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.serialize(), f, indent=2)

    @classmethod
    def from_models(
        cls,
        models: list[LayerModel],
        per_layer_stats: list[dict[str, Any]],
        config: CIDTConfig,
        module_keys: list[str],
        device: str,
        n_samples: int,
    ) -> "TreeCollection":
        """Create from trained models and metrics."""
        trees: list[SavedTree] = []

        for lm, metrics_dict in zip(models, per_layer_stats, strict=True):
            for target_idx in range(lm.target_dim):
                # Get the individual tree for this output
                estimator = lm.model.estimators_[target_idx]  # pyright: ignore[reportIndexIssue]
                assert isinstance(estimator, DecisionTreeClassifier)

                trees.append(
                    SavedTree.from_sklearn(
                        layer_index=lm.layer_index,
                        target_index=target_idx,
                        tree=estimator,
                        metrics_dict=metrics_dict,
                    )
                )

        training_info = {
            "timestamp": datetime.now().isoformat(),
            "device": device,
            "n_samples": n_samples,
            "n_layers": len(models),
            "total_trees": len(trees),
        }

        return cls(
            config=config,
            module_keys=module_keys,
            training_info=training_info,
            trees=trees,
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "TreeCollection":
        """Load from JSON file."""
        path = Path(path)
        with path.open() as f:
            data = json.load(f)

        # Reconstruct config
        config = CIDTConfig(**data["config"])

        # Reconstruct trees
        trees = []
        for tree_data in data["trees"]:
            structure = cls._deserialize_tree_node(tree_data["structure"])
            trees.append(
                SavedTree(
                    layer_index=tree_data["layer_index"],
                    target_index=tree_data["target_index"],
                    metrics=tree_data["metrics"],
                    structure=structure,
                )
            )

        return cls(
            config=config,
            module_keys=data["module_keys"],
            training_info=data["training_info"],
            trees=trees,
        )

    @staticmethod
    def _deserialize_tree_node(data: dict[str, Any]) -> TreeNode:
        """Recursively reconstruct TreeNode from dict."""
        if data["is_leaf"]:
            return TreeNode(
                is_leaf=True,
                n_samples=data["n_samples"],
                value=data["value"],
            )

        return TreeNode(
            is_leaf=False,
            n_samples=data["n_samples"],
            value=data["value"],
            feature=data["feature"],
            left=TreeCollection._deserialize_tree_node(data["left"]),
            right=TreeCollection._deserialize_tree_node(data["right"]),
        )
