"""Pipeline functions for causal importance decision tree training."""

from typing import Any

import numpy as np
import torch
from jaxtyping import Bool, Float
from muutils.dbg import dbg_tensor
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from spd.clustering.activations import component_activations
from spd.clustering.ci_dt.core import LayerModel, build_xy, layer_metrics, proba_for_layer
from spd.models.component_model import ComponentModel


def compute_activations_multibatch(
    model: ComponentModel,
    device: torch.device | str,
    dataloader: DataLoader,
    n_batches: int,
) -> dict[str, Tensor]:
    """Compute activations over multiple batches, concatenate on CPU.

    For each batch:
    - Run inference on GPU
    - Move activations to CPU immediately
    - Store in list

    After all batches:
    - Concatenate along batch dimension
    - Keep sequence dimension for per-token analysis (no seq_mean)

    Args:
        model: ComponentModel to get activations from
        device: Device to run inference on
        dataloader: DataLoader to get batches from
        n_batches: Number of batches to process

    Returns:
        Dictionary mapping module keys to concatenated activations
        (on CPU, shape: batch, seq_len, n_components)
    """
    print(f"Computing activations for {n_batches} batches...")
    all_component_acts: list[dict[str, Tensor]] = []

    for _batch_idx in tqdm(range(n_batches), desc="Batches", total=n_batches):
        batch_data = next(iter(dataloader))
        batch: Tensor = batch_data["input_ids"]

        # Get activations on GPU
        component_acts_gpu: dict[str, Tensor] = component_activations(
            model=model, device=device, batch=batch
        )

        # Move to CPU immediately and store
        component_acts_cpu: dict[str, Tensor] = {
            key: tensor.cpu() for key, tensor in component_acts_gpu.items()
        }
        all_component_acts.append(component_acts_cpu)

    # Concatenate all batches on CPU
    print("Concatenating batches...")
    module_keys: list[str] = list(all_component_acts[0].keys())
    component_acts_concat: dict[str, Tensor] = {
        key: torch.cat([batch[key] for batch in all_component_acts], dim=0) for key in module_keys
    }

    print("Activation shapes (keeping sequence dimension for per-token analysis):")
    for key in module_keys[:3]:  # Show first 3 for brevity
        print(f"  {key}: {component_acts_concat[key].shape}")

    return component_acts_concat


def convert_to_boolean_layers(
    component_acts: dict[str, Tensor],
    activation_threshold: float,
    verbose: bool = False,
) -> list[Bool[np.ndarray, "n_samples n_components"]]:
    """Convert activations to boolean, filter constant (always dead/alive) components.

    Handles 3D activations (batch, seq_len, n_components) by flattening to 2D (batch*seq_len, n_components).

    Args:
        component_acts: Dictionary of continuous activations per module (on CPU, shape: batch, seq_len, n_components or batch, n_components)
        activation_threshold: Threshold for converting to boolean

    Returns:
        List of boolean numpy arrays, one per module (layer), shape (batch*seq_len, n_varying_components)
    """
    print("\nConverting to boolean and filtering constant components...")
    layers_true: list[Bool[np.ndarray, "n_samples n_components"]] = []
    module_keys: list[str] = list(component_acts.keys())

    for module_key in module_keys:
        # Convert to numpy
        module_acts_tensor: Tensor = component_acts[module_key]

        # Flatten if 3D (batch, seq_len, n_components) -> (batch*seq_len, n_components)
        if module_acts_tensor.ndim == 3:
            batch_size, seq_len, n_components = module_acts_tensor.shape
            module_acts_np: Float[np.ndarray, "n_samples n_components"] = (
                module_acts_tensor.reshape(batch_size * seq_len, n_components).numpy()
            )
        else:
            module_acts_np = module_acts_tensor.numpy()

        module_acts_bool: Bool[np.ndarray, "n_samples n_components"] = (
            module_acts_np >= activation_threshold
        ).astype(bool)

        # Filter out components that are always dead or always alive
        # (they provide no information for decision trees)
        component_variance: Float[np.ndarray, " n_components"] = module_acts_bool.var(axis=0)
        varying_mask: Bool[np.ndarray, " n_components"] = component_variance > 0

        # Count always-dead and always-alive components for diagnostics
        always_dead_mask: Bool[np.ndarray, " n_components"] = ~module_acts_bool.any(axis=0)
        always_alive_mask: Bool[np.ndarray, " n_components"] = module_acts_bool.all(axis=0)
        n_always_dead: int = int(always_dead_mask.sum())
        n_always_alive: int = int(always_alive_mask.sum())

        module_acts_varying: Bool[np.ndarray, "n_samples n_varying"] = module_acts_bool[
            :, varying_mask
        ]

        layers_true.append(module_acts_varying)
        if verbose:
            n_varying: int = module_acts_varying.shape[1]
            n_total: int = module_acts_bool.shape[1]
            print(
                f"  {module_key:30s} {n_varying:5d} varying, {n_always_dead:5d} dead, {n_always_alive:5d} const, {n_total:5d} total",
                flush=True,
            )
            dbg_tensor(module_acts_np)
            dbg_tensor(module_acts_bool)
            dbg_tensor(module_acts_varying)

    n_samples: int = layers_true[0].shape[0] if layers_true else 0
    print(f"Created {len(layers_true)} layers with {n_samples} samples for decision tree training")

    return layers_true


def compute_tree_metrics(
    models: list[LayerModel],
    layers_true: list[Bool[np.ndarray, "n_samples n_components"]],
) -> tuple[list[dict[str, Any]], list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    """Compute per-layer metrics and identify best/worst trees by average precision.

    Args:
        models: Trained LayerModel objects
        layers_true: Ground truth boolean layers

    Returns:
        Tuple of:
        - per_layer_stats: List of dicts with metrics per layer
        - worst_list: List of (layer_idx, target_idx, AP) for worst 2 trees
        - best_list: List of (layer_idx, target_idx, AP) for best 2 trees
    """
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

    return per_layer_stats, worst_list, best_list
