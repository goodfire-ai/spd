from typing import Any, Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader

from spd.clustering.util import ModuleFilterFunc
from spd.models.component_model import ComponentModel
from spd.models.sigmoids import SigmoidTypes
from spd.utils.general_utils import extract_batch_data


def component_activations(
    model: ComponentModel,
    device: torch.device | str,
    dataloader: DataLoader[Int[Tensor, "..."]]
    | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]]
    | None = None,
    batch: Int[Tensor, "batch_size n_ctx"] | None = None,
    sigmoid_type: SigmoidTypes = "normal",
) -> dict[str, Float[Tensor, " n_steps C"]]:
    """Get the component activations over a **single** batch."""
    with torch.no_grad():
        batch_: Tensor
        if batch is None:
            assert dataloader is not None, "provide either a batch or a dataloader, not both"
            batch_ = extract_batch_data(next(iter(dataloader)))
        else:
            assert dataloader is None, "provide either a batch or a dataloader, not both"
            batch_ = batch

        batch_ = batch_.to(device)

        _, pre_weight_acts = model._forward_with_pre_forward_cache_hooks(
            batch_, module_names=model.target_module_paths
        )

        causal_importances, _ = model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            sigmoid_type=sigmoid_type,
            detach_inputs=False,
        )

        return causal_importances


def sort_module_components_by_similarity(
    activations: Float[Tensor, "n_steps C"],
) -> tuple[Float[Tensor, "n_steps C"], Int[Tensor, " C"]]:
    """Sort components within a single module by their similarity using greedy ordering.

    Uses a greedy nearest-neighbor approach: starts with the component most similar
    to all others, then iteratively picks the most similar unvisited component.

    Args:
        activations: Activations for a single module

    Returns:
        Tuple of (sorted_activations, sort_indices)
    """
    n_components = activations.shape[1]

    # If only one component, no sorting needed
    if n_components <= 1:
        return activations, torch.arange(n_components, device=activations.device)

    # Compute coactivation matrix for this module
    coact = activations.T @ activations

    # Convert to similarity matrix (normalize by diagonal)
    diag = torch.diagonal(coact).sqrt()
    # Avoid division by zero
    diag = torch.where(diag > 1e-8, diag, torch.ones_like(diag))
    similarity = coact / (diag.unsqueeze(0) * diag.unsqueeze(1))

    # Greedy ordering: start with component most similar to all others
    # (highest average similarity)
    avg_similarity = similarity.mean(dim=1)
    start_idx = int(torch.argmax(avg_similarity).item())

    # Build ordering greedily
    ordered_indices = [start_idx]
    remaining = set(range(n_components))
    remaining.remove(start_idx)

    # Greedily add the nearest unvisited component
    current_idx = start_idx
    while remaining:
        # Find the unvisited component most similar to current
        best_similarity = -1
        best_idx = -1
        for idx in remaining:
            sim = similarity[current_idx, idx].item()
            if sim > best_similarity:
                best_similarity = sim
                best_idx = idx

        ordered_indices.append(best_idx)
        remaining.remove(best_idx)
        current_idx = best_idx

    # Create sorting tensor
    sort_indices = torch.tensor(ordered_indices, dtype=torch.long, device=activations.device)

    # Apply sorting
    sorted_act = activations[:, sort_indices]

    return sorted_act, sort_indices


def process_activations(
    activations: dict[
        str,  # module name to
        Float[Tensor, " n_steps C"]  # (sample x component gate activations)
        | Float[Tensor, " n_sample n_ctx C"],  # (sample x seq index x component gate activations)
    ],
    filter_dead_threshold: float = 0.01,
    seq_mode: Literal["concat", "seq_mean", None] = None,
    filter_modules: ModuleFilterFunc | None = None,
    sort_components: bool = False,
) -> dict[str, Any]:
    """get back a dict of coactivations, slices, and concated activations

    Args:
        activations: Dictionary of activations by module
        filter_dead_threshold: Threshold for filtering dead components
        seq_mode: How to handle sequence dimension
        filter_modules: Function to filter modules
        sort_components: Whether to sort components by similarity within each module
    """

    activations_: dict[str, Float[Tensor, " n_steps C"]]
    if seq_mode == "concat":
        # Concatenate the sequence dimension into the sample dimension
        activations_ = {
            key: act.reshape(act.shape[0] * act.shape[1], act.shape[2])
            for key, act in activations.items()
        }
    elif seq_mode == "seq_mean":
        # Take the mean over the sequence dimension
        activations_ = {
            key: act.mean(dim=1) if act.ndim == 3 else act for key, act in activations.items()
        }
    else:
        # Use the activations as they are
        activations_ = activations

    # filter activations for only the modules we want
    if filter_modules is not None:
        activations_ = {key: act for key, act in activations_.items() if filter_modules(key)}

    # Sort components within each module if requested
    sort_indices_dict: dict[str, Int[Tensor, " C"]] = {}
    if sort_components:
        sorted_activations = {}
        for key, act in activations_.items():
            sorted_act, sort_idx = sort_module_components_by_similarity(act)
            sorted_activations[key] = sorted_act
            sort_indices_dict[key] = sort_idx
        activations_ = sorted_activations

    # compute the labels and total component count
    total_c: int = 0
    labels: list[str] = list()
    for key, act in activations_.items():
        c = act.shape[-1]
        if sort_components and key in sort_indices_dict:
            # Use sorted indices for labeling
            sort_idx = sort_indices_dict[key]
            labels.extend([f"{key}:{int(sort_idx[i].item())}" for i in range(c)])
        else:
            labels.extend([f"{key}:{i}" for i in range(c)])
        total_c += c

    # concat the activations
    act_concat: Float[Tensor, " n_steps c"] = torch.cat(
        [activations_[key] for key in activations_], dim=-1
    )

    # filter dead components
    dead_components_lst: list[str] | None = None
    if filter_dead_threshold > 0:
        dead_components_lst = list()
        max_act = act_concat.max(dim=0).values
        dead_components = max_act < filter_dead_threshold
        if dead_components.any():
            act_concat = act_concat[:, ~dead_components]
            alive_labels: list[tuple[str, bool]] = [
                (lbl, bool(keep.item()))
                for lbl, keep in zip(labels, ~dead_components, strict=False)
            ]
            labels = [label for label, keep in alive_labels if keep]
            dead_components_lst = [label for label, keep in alive_labels if not keep]
            # logger.values({
            #     "total_components": total_c,
            #     "n_alive_components": len(labels),
            #     "n_dead_components": len(dead_components_lst),
            # })

    # compute coactivations
    # TODO: this is wrong for anything but boolean activations
    coact: Float[Tensor, " c c"] = act_concat.T @ act_concat

    # return the output
    output: dict[str, Any] = dict(
        activations_raw=activations_,
        activations=act_concat,
        labels=labels,
        coactivations=coact,
        dead_components_lst=dead_components_lst,
        n_components_original=total_c,
        n_components_alive=len(labels),
        n_components_dead=len(dead_components_lst) if dead_components_lst else 0,
        sort_indices=sort_indices_dict if sort_components else None,
    )

    return output
