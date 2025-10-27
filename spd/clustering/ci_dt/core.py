"""Core library functions for causal importance decision trees."""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Bool, Float
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm


@dataclass
class LayerModel:
    """Holds a trained per-layer model."""

    layer_index: int
    model: MultiOutputClassifier
    feature_dim: int
    target_dim: int


def concat_cols(
    Xs: Sequence[Bool[np.ndarray, "n_samples n_features"]],
) -> Bool[np.ndarray, "n_samples n_concat"]:
    """Column-concat a sequence or return empty (n,0)."""
    n_samples: int = Xs[0].shape[0] if len(Xs) else 0
    return np.concatenate(Xs, axis=1) if len(Xs) else np.zeros((n_samples, 0), bool)


def build_xy(
    layers: Sequence[Bool[np.ndarray, "n_samples n_components"]],
) -> list[
    tuple[
        Bool[np.ndarray, "n_samples n_features"],
        Bool[np.ndarray, "n_samples n_targets"],
    ]
]:
    """Return (X_k,Y_k) for k=1..L-1 with X_k=concat(layers[:k])."""
    XYs: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(1, len(layers)):
        X_k: np.ndarray = concat_cols(layers[:k])
        Y_k: np.ndarray = layers[k]
        XYs.append((X_k, Y_k))
    return XYs


def train_trees(
    layers: Sequence[Bool[np.ndarray, "n_samples n_components"]],
    *,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int | None = 0,
) -> list[LayerModel]:
    """Train one decision tree per component per target layer using previous layers as features."""
    XYs = build_xy(layers)
    models: list[LayerModel] = []
    for k, (X_k, Y_k) in tqdm(enumerate(XYs, start=1), total=len(XYs), desc="Training trees"):
        base = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        model = MultiOutputClassifier(base)
        model.fit(X_k.astype(np.uint8), Y_k.astype(np.uint8))
        models.append(LayerModel(k, model, int(X_k.shape[1]), int(Y_k.shape[1])))
    return models


def extract_prob_class_1(
    proba_list: list[np.ndarray],
    model: MultiOutputClassifier,
) -> np.ndarray:
    """Extract P(y=1) for each output.

    Assumes constant components are filtered out, so both classes should always be present.
    """
    result: list[np.ndarray] = []
    for i, p in enumerate(proba_list):
        estimator = model.estimators_[i]
        classes = estimator.classes_
        assert len(classes) == 2, f"Expected 2 classes but got {len(classes)} for output {i}"
        # Extract P(y=1) from second column
        result.append(p[:, 1])
    return np.stack(result, axis=1)


def predict_k(
    models: Sequence[LayerModel],
    prefix_layers: Sequence[Bool[np.ndarray, "n_samples n_components"]],
    k: int,
    *,
    threshold: float = 0.5,
) -> Bool[np.ndarray, "n_samples n_components_k"]:
    """Predict layer k activations from layers[:k]."""
    lm: LayerModel = next(m for m in models if m.layer_index == k)
    X: np.ndarray = concat_cols(prefix_layers)
    # dbg_auto(X)
    proba = lm.model.predict_proba(X.astype(np.uint8))  # type: ignore
    # dbg_auto(proba)
    # dbg_auto(proba[0])
    P: np.ndarray = extract_prob_class_1(proba, lm.model)
    # dbg_auto(P)
    Y_hat: np.ndarray = (threshold <= P).astype(bool)
    # dbg_auto(Y_hat)
    return Y_hat


def predict_all(
    models: Sequence[LayerModel],
    seed_layers: Sequence[Bool[np.ndarray, "n_samples n_components"]],
    *,
    thresholds: Sequence[float] | None = None,
) -> list[Bool[np.ndarray, "n_samples n_components"]]:
    """Sequentially predict layers 1.. using layer 0 as seed."""
    out: list[np.ndarray] = [seed_layers[0].copy()]
    ths: list[float] = list(thresholds) if thresholds is not None else []
    for i, lm in enumerate(sorted(models, key=lambda m: m.layer_index)):
        thr: float = ths[i] if i < len(ths) else 0.5
        out.append(predict_k(models, out, lm.layer_index, threshold=thr))
    return out


MetricKey = Literal["ap", "acc", "bacc", "prev", "tpr", "tnr", "precision", "npv", "f1"]


def layer_metrics(
    Y_true: Bool[np.ndarray, "n t"],
    Y_prob: Float[np.ndarray, "n t"],
    Y_pred: Bool[np.ndarray, "n t"],
) -> dict[MetricKey, np.ndarray]:
    """Return per-target metrics: AP, acc, bacc, prevalence, TPR, TNR, precision, NPV, F1.

    Returns:
        Dictionary with keys:
        - ap: Average precision
        - acc: Accuracy
        - bacc: Balanced accuracy
        - prev: Prevalence (fraction of positive samples)
        - tpr: True Positive Rate (Recall/Sensitivity)
        - tnr: True Negative Rate (Specificity)
        - precision: Precision (when we predict active, how often are we right?)
        - npv: Negative Predictive Value (when we predict inactive, how often are we right?)
        - f1: F1 score

        Each value is an array of length T (number of target components).
    """
    T: int = Y_true.shape[1]

    ap: Float[np.ndarray, " t"] = np.full(T, np.nan)
    acc: Float[np.ndarray, " t"] = np.full(T, np.nan)
    bacc: Float[np.ndarray, " t"] = np.full(T, np.nan)
    prev: Float[np.ndarray, " t"] = np.full(T, np.nan)
    tpr: Float[np.ndarray, " t"] = np.full(T, np.nan)
    tnr: Float[np.ndarray, " t"] = np.full(T, np.nan)
    precision: Float[np.ndarray, " t"] = np.full(T, np.nan)
    npv: Float[np.ndarray, " t"] = np.full(T, np.nan)
    f1: Float[np.ndarray, " t"] = np.full(T, np.nan)

    for j in range(T):
        y: np.ndarray = Y_true[:, j].astype(int)
        p: np.ndarray = Y_prob[:, j]
        yhat: np.ndarray = Y_pred[:, j].astype(int)
        prev[j] = float(y.mean())

        # Compute confusion matrix elements
        tp: int = int(((y == 1) & (yhat == 1)).sum())
        tn: int = int(((y == 0) & (yhat == 0)).sum())
        fp: int = int(((y == 0) & (yhat == 1)).sum())
        fn: int = int(((y == 1) & (yhat == 0)).sum())

        # TPR (Recall/Sensitivity) = TP / (TP + FN)
        tpr[j] = tp / (tp + fn)

        # TNR (Specificity) = TN / (TN + FP)
        tnr[j] = tn / (tn + fp)

        # Precision (PPV) = TP / (TP + FP) - when we predict active, how often are we right?
        if (tp + fp) > 0:
            precision[j] = tp / (tp + fp)
        else:
            precision[j] = np.nan
            warnings.warn(f"Precision failed:  {tp=}, {fp=}, {tp+fp=}", stacklevel=1)

        # Negative Predictive Value = TN / (TN + FN) - when we predict inactive, how often are we right?
        npv[j] = tn / (tn + fn)

        # F1 = 2 * (precision * recall) / (precision + recall)
        f1[j] = 2 * (precision[j] * tpr[j]) / (precision[j] + tpr[j])

        # Sklearn metrics
        ap[j] = average_precision_score(y, p)
        acc[j] = accuracy_score(y, yhat)
        bacc[j] = balanced_accuracy_score(y, yhat)

    return {
        "ap": ap,
        "acc": acc,
        "bacc": bacc,
        "prev": prev,
        "tpr": tpr,
        "tnr": tnr,
        "precision": precision,
        "npv": npv,
        "f1": f1,
    }


def proba_for_layer(lm: LayerModel, X: np.ndarray) -> np.ndarray:
    """Return P(y=1) per target column."""
    proba_list = lm.model.predict_proba(X.astype(np.uint8))  # type: ignore
    return extract_prob_class_1(proba_list, lm.model)


def get_estimator_for(
    models: list[LayerModel], layer_idx: int, target_idx: int
) -> DecisionTreeClassifier:
    """Fetch the per-output estimator for a given layer and column."""
    lm = next(m for m in models if m.layer_index == layer_idx)
    return lm.model.estimators_[target_idx]  # type: ignore
