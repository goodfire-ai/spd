"""Core library functions for causal importance decision trees."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Bool, Float
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from muutils.dbg import dbg_auto, dbg, dbg_tensor


@dataclass
class LayerModel:
    """Holds a trained per-layer model."""

    layer_index: int
    model: ClassifierMixin
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
    strategy: Literal["one_vs_all", "single_tree"] = "one_vs_all",
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int | None = 0,
) -> list[LayerModel]:
    """Train one model per target layer using previous layers as features."""
    XYs = build_xy(layers)
    models: list[LayerModel] = []
    for k, (X_k, Y_k) in enumerate(XYs, start=1):
        base = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        model: ClassifierMixin = MultiOutputClassifier(base) if strategy == "one_vs_all" else base
        _ = model.fit(X_k.astype(np.uint8), Y_k.astype(np.uint8))
        models.append(LayerModel(k, model, int(X_k.shape[1]), int(Y_k.shape[1])))
    return models


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
    dbg_auto(X)
    proba = lm.model.predict_proba(X.astype(np.uint8))  # type: ignore
    dbg_auto(proba)
    dbg_auto(proba[0])

    dbg(Counter(tuple(p.shape) for p in proba))

    P: np.ndarray = np.stack(proba, axis=1)
    dbg_auto(P)
    Y_hat: np.ndarray = (float(threshold) <= P).astype(bool)
    dbg_auto(Y_hat)
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


def layer_metrics(
    Y_true: Bool[np.ndarray, "n t"],
    Y_prob: Float[np.ndarray, "n t"],
    Y_pred: Bool[np.ndarray, "n t"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-target AP, acc, bacc, prevalence."""
    T: int = Y_true.shape[1]
    ap: np.ndarray = np.zeros(T)
    acc: np.ndarray = np.zeros(T)
    bacc: np.ndarray = np.zeros(T)
    prev: np.ndarray = np.zeros(T)
    for j in range(T):
        y: np.ndarray = Y_true[:, j].astype(int)
        p: np.ndarray = Y_prob[:, j]
        yhat: np.ndarray = Y_pred[:, j].astype(int)
        prev[j] = float(y.mean())
        try:
            ap[j] = average_precision_score(y, p)
        except Exception:
            ap[j] = np.nan
        try:
            acc[j] = accuracy_score(y, yhat)
        except Exception:
            acc[j] = np.nan
        try:
            bacc[j] = balanced_accuracy_score(y, yhat)
        except Exception:
            bacc[j] = np.nan
    return ap, acc, bacc, prev


def proba_for_layer(lm: LayerModel, X: np.ndarray) -> np.ndarray:
    """Return P(y=1) per target column."""
    pr = lm.model.predict_proba(X.astype(np.uint8))  # type: ignore
    if isinstance(pr, list):
        return np.stack([p[:, 1] for p in pr], axis=1)
    return pr[..., 1]  # type: ignore


def get_estimator_for(
    models: list[LayerModel], layer_idx: int, target_idx: int
) -> DecisionTreeClassifier:
    """Fetch the per-output estimator for a given layer and column."""
    lm = next(m for m in models if m.layer_index == layer_idx)
    if isinstance(lm.model, MultiOutputClassifier):
        return lm.model.estimators_[target_idx]  # type: ignore
    return lm.model  # type: ignore
