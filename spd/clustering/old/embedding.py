from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float
from muutils.dbg import dbg_tensor
from sklearn.base import TransformerMixin
from sklearn.cluster import (
    DBSCAN,
    OPTICS,
    AgglomerativeClustering,
    KMeans,
    SpectralClustering,
)
from sklearn.manifold import TSNE, Isomap
from umap import UMAP

from spd.clustering.grouping import CoactivationResultsGroup

# TODO: this is ugly
NDArray = np.ndarray[Any, Any]

ReduceMethod = Literal["umap", "isomap", "tsne"]
ReduceMethodParam = int | float | str
ClusteringMethod = Literal[
    "kmeans",
    "agglomerative",
    "spectral",
    "dbscan",
    "optics",
]


@dataclass
class EmbeddingResult:
    method: ReduceMethod
    param_name: str
    param_values: list[ReduceMethodParam]
    embeddings: dict[str, Float[NDArray, "n_components n_features"]]


def get_embedding_model(
    method: ReduceMethod,
    kwargs: dict[str, Any],
    n_components: int = 2,
    random_state: int | None = None,
) -> TransformerMixin:
    """Return a configured embedding model (match-case style)."""
    match method:
        case "umap":
            return UMAP(
                n_components=n_components,
                metric="precomputed",
                random_state=random_state,
                **kwargs,  # type: ignore[arg-type]
            )
        case "isomap":
            return Isomap(n_components=n_components, metric="precomputed", **kwargs)
        case "tsne":
            return TSNE(  # type: ignore[arg-type]
                n_components=n_components,
                metric="precomputed",
                random_state=random_state,
                init="random",
                learning_rate="auto",
                **kwargs,
            )
        case _:
            raise ValueError(f"Unsupported embedding method: {method}")


def get_clustering_model(
    method: ClusteringMethod,
    kwargs: dict[str, Any],
    random_state: int | None = None,
) -> Any:
    """Return a configured clustering model (match-case style)."""
    match method:
        case "kmeans":
            return KMeans(random_state=random_state, **kwargs)
        case "agglomerative":
            return AgglomerativeClustering(**kwargs)
        case "spectral":
            return SpectralClustering(random_state=random_state, **kwargs)
        case "dbscan":
            return DBSCAN(**kwargs)
        case "optics":
            return OPTICS(**kwargs)
        case _:
            raise ValueError(f"Unsupported clustering method: {method}")


def compute_embedding_sweep(
    dist: NDArray,
    method: ReduceMethod,
    param_grid: list[dict[str, Any]],
    n_components: int = 2,
    random_state: int | None = None,
) -> dict[str, NDArray]:
    """Compute embeddings for a grid of parameter dicts."""
    embeddings: dict[str, NDArray] = {}
    for kwargs in param_grid:
        key: str = "_".join(f"{k}={v}" for k, v in kwargs.items())
        model: TransformerMixin = get_embedding_model(method, kwargs, n_components, random_state)
        embeddings[key] = model.fit_transform(dist)
    return embeddings


def sweep_embedding_param(
    dist: NDArray,
    method: ReduceMethod,
    param_name: str,
    param_values: list[ReduceMethodParam],
    n_components: int = 2,
    random_state: int | None = None,
) -> EmbeddingResult:
    """Sweep a single hyperparameter and return results."""

    param_grid: list[dict[str, Any]] = [{param_name: val} for val in param_values]
    embeddings: dict[str, NDArray] = compute_embedding_sweep(
        dist, method, param_grid, n_components, random_state
    )

    return EmbeddingResult(
        method=method, param_name=param_name, param_values=param_values, embeddings=embeddings
    )


def plot_embedding_result(
    result: EmbeddingResult,
    figsize: tuple[int, int] | None = None,
    cluster_labels: dict[str, NDArray] | None = None,
    cmap: str = "tab10",
) -> None:
    """Plot all 2-D embeddings (optionally color by clusters)."""
    n: int = len(result.param_values)
    figsize = figsize or (4 * n, 4)
    axes: Sequence[plt.Axes]
    fig, axes = plt.subplots(1, n, figsize=figsize)  # type: ignore
    for ax, param_val in zip(axes, result.param_values, strict=True):
        key: str = f"{result.param_name}={param_val}"
        emb: NDArray = result.embeddings[key]
        c = None if cluster_labels is None else cluster_labels.get(key, None)
        ax.scatter(emb[:, 0], emb[:, 1], c=c, cmap=cmap)
        ax.set_title(f"{result.method.upper()} {key}")
        ax.grid(True)
    plt.tight_layout()
    plt.show()


def get_comp_dist_mat(
    group: CoactivationResultsGroup,
    verbose: bool = True,
    plots: bool = True,
    epsilon: float = 1,
    normalize_dist: bool = True,
) -> Float[NDArray, "n n"]:
    jac: Float[NDArray, "n n"] = group["jaccard"].cpu()

    if verbose:
        dbg_tensor(jac)

    dist: Float[NDArray, "n n"] = 1 / (jac + epsilon)
    if normalize_dist:
        dist = dist / dist.max()

    if verbose:
        dbg_tensor(dist)

    if plots:
        fig, ax = plt.subplots(1, 4, figsize=(12, 3))
        ax[0].matshow(jac, cmap="viridis")
        ax[0].set_title("Jaccard Matrix")
        ax[1].hist(jac.flatten(), bins=20)
        ax[1].set_yscale("log")
        ax[1].set_title("Jaccard Histogram")
        ax[2].matshow(dist, cmap="viridis")
        ax[2].set_title("Distance Matrix")
        ax[3].hist(dist.flatten(), bins=20)
        ax[3].set_yscale("log")
        ax[3].set_title("Distance Histogram")
        plt.tight_layout()
        plt.show()

    return dist


def cluster_in_embedding(
    dist: Float[NDArray, "n n"],
    *,
    embedding_method: ReduceMethod,
    embedding_params: dict[str, Any] | None = None,
    clustering_method: ClusteringMethod = "kmeans",
    clustering_params: dict[str, Any] | None = None,
    n_components: int = 2,
    random_state: int | None = None,
) -> list[int]:
    """Embed a pre-computed distance matrix **then** cluster it.

    # Parameters:
     - `dist : Float[NDArray, "n n"]`
        Square distance matrix.
     - `embedding_method : ReduceMethod`
        `"umap" | "isomap" | "tsne"`.
     - `embedding_params : dict[str, Any]`
        Extra kwargs for the embedder (defaults to `{}`).
     - `clustering_method : ClusteringMethod`
        `"kmeans" | "agglomerative" | "spectral" | "dbscan" | "optics"`.
     - `clustering_params : dict[str, Any]`
        Extra kwargs for the clusterer (defaults to `{}`).
     - `n_components : int`
        Embedding dimensionality (defaults to `2`).
     - `random_state : int | None`
        RNG seed for reproducibility.

    # Returns:
     - `list[int]`
        Cluster label for each sample.
    """
    embedding_params = embedding_params or {}
    clustering_params = clustering_params or {}

    emb_model: TransformerMixin = get_embedding_model(
        embedding_method,
        embedding_params,
        n_components=n_components,
        random_state=random_state,
    )
    emb: Float[NDArray, "n_components n_features"] = emb_model.fit_transform(dist)

    cls_model = get_clustering_model(clustering_method, clustering_params, random_state)
    labels: NDArray = cls_model.fit_predict(emb)

    return labels.tolist()


def sweep_embeddings_and_clusters(
    dist: Float[NDArray, "n n"],
    *,
    n_clusters: int = 5,
    random_state: int | None = 0,
    cmap: str = "tab10",
) -> None:
    """Plot every (embedding, clustering) combination on a single figure."""
    # --- hyper-param specs -------------------------------------------------- #
    embedding_methods: list[ReduceMethod] = ["umap", "isomap", "tsne"]

    clustering_specs: dict[ClusteringMethod, dict[str, Any]] = {
        "kmeans": {"n_clusters": n_clusters},
        "agglomerative": {"n_clusters": n_clusters},
        "spectral": {"n_clusters": n_clusters},
        "dbscan": {"eps": 0.1},  # tweak as needed
        "optics": {},  # defaults are fine
    }

    # --- canvas ------------------------------------------------------------- #
    n_rows, n_cols = len(embedding_methods), len(clustering_specs)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    # --- sweep -------------------------------------------------------------- #
    for row, emb_method in enumerate(embedding_methods):
        emb_model = get_embedding_model(emb_method, {}, n_components=2, random_state=random_state)
        emb: NDArray = emb_model.fit_transform(dist)

        for col, (cls_method, cls_kwargs) in enumerate(clustering_specs.items()):
            cls_model = get_clustering_model(cls_method, cls_kwargs, random_state=random_state)
            labels: NDArray = cls_model.fit_predict(emb)

            ax = axes[row][col]
            ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap=cmap)
            ax.set_title(f"{emb_method}-{cls_method}")
            ax.grid(True)

    plt.tight_layout()
    plt.show()
