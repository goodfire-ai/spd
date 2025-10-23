# %%

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Bool, Float
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from torch import Tensor

from spd.clustering.activations import (
    ProcessedActivations,
    component_activations,
    process_activations,
)
from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, SPDRunInfo

# ----------------------- config -----------------------


@dataclass
class CIDTConfig:
    """Configuration for causal importance decision tree training."""

    experiment_key: str = "ss_emb"  # Key from EXPERIMENT_REGISTRY
    n_samples: int = 250
    activation_threshold: float = 0.01  # Threshold for boolean conversion
    filter_dead_threshold: float = 0.001  # Threshold for filtering dead components
    max_depth: int = 8  # Maximum depth for decision trees
    random_state: int = 7  # Random state for reproducibility


# ----------------------- library code -----------------------


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
    proba = lm.model.predict_proba(X.astype(np.uint8))  # type: ignore
    if isinstance(proba, list):
        P: np.ndarray = np.stack([p[:, 1] for p in proba], axis=1)
    else:
        P = proba[..., 1]  # type: ignore
    Y_hat: np.ndarray = (float(threshold) <= P).astype(bool)
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


# ----------------------- configuration -----------------------

config = CIDTConfig()
device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------- load model -----------------------

wandb_run_path: str = "wandb:goodfire/spd/runs/lxs77xye"

spd_run: SPDRunInfo = SPDRunInfo.from_path(wandb_run_path)
model: ComponentModel = ComponentModel.from_pretrained(spd_run.checkpoint_path)
model.to(device)
cfg: Config = spd_run.config

print(f"Loaded model from {wandb_run_path}")

# ----------------------- load dataset -----------------------

# Create LM dataset and dataloader
assert isinstance(cfg.task_config, LMTaskConfig)
pretrained_model_name = cfg.pretrained_model_name
assert pretrained_model_name is not None

dataset_config = DatasetConfig(
    name=cfg.task_config.dataset_name,
    hf_tokenizer_path=pretrained_model_name,
    split=cfg.task_config.train_data_split,
    n_ctx=cfg.task_config.max_seq_len,
    column_name=cfg.task_config.column_name,
    is_tokenized=False,
    streaming=False,
    seed=0,
)
dataloader, _ = create_data_loader(
    dataset_config=dataset_config,
    batch_size=config.n_samples,
    buffer_size=cfg.task_config.buffer_size,
    global_seed=cfg.seed,
    ddp_rank=0,
    ddp_world_size=1,
)
batch_data = next(iter(dataloader))
batch: Tensor = batch_data["input_ids"]
print(f"Created LM dataset with {cfg.task_config.dataset_name}, batch shape: {batch.shape}")

# ----------------------- get activations -----------------------

# Get component activations (on device)
print("Computing component activations...")
component_acts: dict[str, Tensor] = component_activations(
    model=model,
    device=device,
    batch=batch,
)

# Process activations (filter dead components, concatenate)
print("Processing activations...")
processed_acts: ProcessedActivations = process_activations(
    component_acts,
    filter_dead_threshold=config.filter_dead_threshold,
    seq_mode="seq_mean",  # LM task needs seq_mean
)

print(f"Total components (before filtering): {processed_acts.n_components_original}")
print(f"Alive components: {processed_acts.n_components_alive}")
print(f"Dead components: {processed_acts.n_components_dead}")
print(f"Module keys: {processed_acts.module_keys}")

# ----------------------- convert to layers -----------------------

# Move to CPU and convert to numpy for sklearn
# Group by module to create "layers" for decision trees
print("\nConverting to boolean layers...")
layers_true: list[np.ndarray] = []
for module_key in processed_acts.module_keys:
    # Get the activations for this module from activations_raw, move to CPU
    module_acts_cpu = processed_acts.activations_raw[module_key].cpu().numpy()
    module_acts_bool = (module_acts_cpu >= config.activation_threshold).astype(bool)
    layers_true.append(module_acts_bool)
    print(f"Layer {len(layers_true) - 1} ({module_key}): {module_acts_bool.shape[1]} components")

print(f"\nCreated {len(layers_true)} layers for decision tree training")

# ----------------------- fit and predict -----------------------

print("\nTraining decision trees...")
models: list[LayerModel] = train_trees(
    layers_true, max_depth=config.max_depth, random_state=config.random_state
)
layers_pred: list[np.ndarray] = predict_all(models, [layers_true[0]])

# ----------------------- metrics -----------------------


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


# get probabilities for each layer
def proba_for_layer(lm: LayerModel, X: np.ndarray) -> np.ndarray:
    """Return P(y=1) per target column."""
    pr = lm.model.predict_proba(X.astype(np.uint8))  # type: ignore
    if isinstance(pr, list):
        return np.stack([p[:, 1] for p in pr], axis=1)
    return pr[..., 1]  # type: ignore


XYs_demo = build_xy(layers_true)
per_layer_stats: list[dict[str, Any]] = []
all_triplets: list[tuple[int, int, float]] = []  # (layer, target_idx, AP)

for lm, (Xk, Yk) in zip(models, XYs_demo, strict=True):
    Pk: np.ndarray = proba_for_layer(lm, Xk)
    Yhat_k: np.ndarray = Pk >= 0.5
    ap, acc, bacc, prev = layer_metrics(Yk, Pk, Yhat_k)
    per_layer_stats.append(
        {
            "ap": ap,
            "acc": acc,
            "bacc": bacc,
            "prev": prev,
            "mean_ap": float(np.nanmean(ap)),
            "mean_acc": float(np.nanmean(acc)),
            "mean_bacc": float(np.nanmean(bacc)),
        }
    )
    for j, apj in enumerate(ap):
        all_triplets.append((lm.layer_index, j, float(apj)))

# identify best and worst trees across all outputs by AP
sorted_triplets = sorted(all_triplets, key=lambda t: (np.isnan(t[2]), t[2]))
worst_list = [t for t in sorted_triplets if not np.isnan(t[2])][:2]
best_list = [t for t in sorted_triplets if not np.isnan(t[2])][-2:]


# pull corresponding estimators (MultiOutputClassifier -> estimators_ list)
def get_estimator_for(
    models: list[LayerModel], layer_idx: int, target_idx: int
) -> DecisionTreeClassifier:
    """Fetch the per-output estimator for a given layer and column."""
    lm = next(m for m in models if m.layer_index == layer_idx)
    if isinstance(lm.model, MultiOutputClassifier):
        return lm.model.estimators_[target_idx]  # type: ignore
    return lm.model  # type: ignore


# ----------------------- plotting -----------------------


# 1) Single fig showing activations across all layers (true vs predicted stacked)
def plot_activations(layers_true: list[np.ndarray], layers_pred: list[np.ndarray]) -> None:
    """Show true and predicted activations as heatmaps."""
    A_true: np.ndarray = np.concatenate(layers_true, axis=1)
    A_pred: np.ndarray = np.concatenate([layers_pred[0]] + layers_pred[1:], axis=1)
    fig1 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(2, 1, 1)
    ax1.set_title("Activations (True)")
    ax1.imshow(A_true, aspect="auto", interpolation="nearest")
    ax1.set_xlabel("components (all layers concatenated)")
    ax1.set_ylabel("samples")
    ax2 = fig1.add_subplot(2, 1, 2)
    ax2.set_title("Activations (Predicted)")
    ax2.imshow(A_pred, aspect="auto", interpolation="nearest")
    ax2.set_xlabel("components (all layers concatenated)")
    ax2.set_ylabel("samples")
    fig1.tight_layout()


# 2) Covariance matrix of all components
def plot_covariance(layers_true: list[np.ndarray]) -> None:
    """Plot covariance between all components across layers."""
    A: np.ndarray = np.concatenate(layers_true, axis=1).astype(float)
    C: np.ndarray = np.cov(A, rowvar=False)
    fig2 = plt.figure(figsize=(6, 6))
    ax = fig2.add_subplot(1, 1, 1)
    ax.set_title("Covariance of components (all layers)")
    _im = ax.imshow(C, aspect="auto", interpolation="nearest")
    ax.set_xlabel("component index")
    ax.set_ylabel("component index")
    fig2.tight_layout()


# 3) Accuracy ideas: bar of mean metrics per layer; scatter of prevalence vs AP
def plot_layer_metrics(per_layer_stats: list[dict[str, Any]]) -> None:
    """Plot summary metrics per layer and per-target AP vs prevalence."""
    L: int = len(per_layer_stats)
    mean_ap: np.ndarray = np.array([d["mean_ap"] for d in per_layer_stats])
    mean_acc: np.ndarray = np.array([d["mean_acc"] for d in per_layer_stats])
    mean_bacc: np.ndarray = np.array([d["mean_bacc"] for d in per_layer_stats])

    # bar: mean AP, ACC, BACC per layer (three separate figures to respect one-plot rule)
    fig3 = plt.figure(figsize=(8, 3))
    ax3 = fig3.add_subplot(1, 1, 1)
    ax3.set_title("Mean Average Precision per layer")
    ax3.bar(np.arange(1, L + 1), mean_ap)
    ax3.set_xlabel("layer index (target)")
    ax3.set_ylabel("mean AP")
    fig3.tight_layout()

    fig4 = plt.figure(figsize=(8, 3))
    ax4 = fig4.add_subplot(1, 1, 1)
    ax4.set_title("Mean Accuracy per layer")
    ax4.bar(np.arange(1, L + 1), mean_acc)
    ax4.set_xlabel("layer index (target)")
    ax4.set_ylabel("mean accuracy")
    fig4.tight_layout()

    fig5 = plt.figure(figsize=(8, 3))
    ax5 = fig5.add_subplot(1, 1, 1)
    ax5.set_title("Mean Balanced Accuracy per layer")
    ax5.bar(np.arange(1, L + 1), mean_bacc)
    ax5.set_xlabel("layer index (target)")
    ax5.set_ylabel("mean balanced accuracy")
    fig5.tight_layout()

    # scatter: prevalence vs AP for all targets across layers
    fig6 = plt.figure(figsize=(6, 5))
    ax6 = fig6.add_subplot(1, 1, 1)
    ax6.set_title("Per-target AP vs prevalence")
    x_list: list[float] = []
    y_list: list[float] = []
    for d in per_layer_stats:
        x_list.extend(list(d["prev"]))
        y_list.extend(list(d["ap"]))
    ax6.scatter(x_list, y_list, alpha=0.6)
    ax6.set_xlabel("prevalence")
    ax6.set_ylabel("average precision")
    fig6.tight_layout()


# 4) Display a couple decision trees (worst and best by AP)
def plot_selected_trees(
    picks: list[tuple[int, int, float]],
    title_prefix: str,
    models: list[LayerModel],
    feature_dims_prefix: list[int],
) -> None:
    """Plot a list of selected trees by (layer, target_idx, score)."""
    for layer_idx, target_idx, score in picks:
        est = get_estimator_for(models, layer_idx, target_idx)
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(f"{title_prefix}: layer {layer_idx}, target {target_idx}, AP={score:.3f}")
        plot_tree(est, ax=ax, filled=False)  # default styling
        fig.tight_layout()


# Run the plots
plot_activations(layers_true, layers_pred)
plot_covariance(layers_true)
plot_layer_metrics(per_layer_stats)
plot_selected_trees(worst_list, "Worst", models, [])
plot_selected_trees(best_list, "Best", models, [])

print("Plots generated.")
