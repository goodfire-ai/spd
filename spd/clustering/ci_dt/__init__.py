"""Causal importance decision tree package."""

from spd.clustering.ci_dt.config import CIDTConfig
from spd.clustering.ci_dt.core import (
    LayerModel,
    build_xy,
    concat_cols,
    extract_prob_class_1,
    get_estimator_for,
    layer_metrics,
    predict_all,
    predict_k,
    proba_for_layer,
    train_trees,
)
from spd.clustering.ci_dt.plot import (
    plot_activations,
    plot_covariance,
    plot_layer_metrics,
    plot_selected_trees,
)

__all__ = [
    # Config
    "CIDTConfig",
    # Core
    "LayerModel",
    "concat_cols",
    "build_xy",
    "train_trees",
    "extract_prob_class_1",
    "predict_k",
    "predict_all",
    "layer_metrics",
    "proba_for_layer",
    "get_estimator_for",
    # Plot
    "plot_activations",
    "plot_covariance",
    "plot_layer_metrics",
    "plot_selected_trees",
]
