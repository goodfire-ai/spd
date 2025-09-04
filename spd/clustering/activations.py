from dataclasses import dataclass
from functools import cached_property
from typing import Literal, NamedTuple

import torch
from jaxtyping import Bool, Float, Float16, Int
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


def compute_coactivatons(
    activations: Float[Tensor, " n_steps c"] | Bool[Tensor, " n_steps c"],
) -> Float16[Tensor, " c c"]:
    """Compute the coactivations matrix from the activations."""
    # TODO: this works for both boolean and continuous activations,
    # but we could do better by just using OR for boolean activations
    # and maybe even some bitshift hacks. but for now, we convert to float16
    activations = activations.to(torch.float16)
    return activations.T @ activations


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

    # Compute coactivation matrix
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


class FilteredActivations(NamedTuple):
    activations: Float[Tensor, " n_steps c"]
    "activations after filtering dead components"

    labels: list[str]
    "list of length c with labels for each preserved component"

    dead_components_labels: list[str] | None
    "list of labels for dead components, or None if no filtering was applied"

    @property
    def n_alive(self) -> int:
        """Number of alive components after filtering."""
        n_alive: int = len(self.labels)
        assert n_alive == self.activations.shape[1], (
            f"{n_alive = } != {self.activations.shape[1] = }"
        )
        return n_alive

    @property
    def n_dead(self) -> int:
        """Number of dead components after filtering."""
        return len(self.dead_components_labels) if self.dead_components_labels else 0


def filter_dead_components(
    activations: Float[Tensor, " n_steps c"],
    labels: list[str],
    filter_dead_threshold: float = 0.01,
) -> FilteredActivations:
    """Filter out dead components based on a threshold

    if `filter_dead_threshold` is 0, no filtering is applied.
    activations and labels are returned as is, `dead_components_labels` is `None`.

    otherwise, components whose **maximum** activations across all samples is below the threshold
    are considered dead and filtered out. The labels of these components are returned in `dead_components_labels`.
    `dead_components_labels` will also be `None` if no components were below the threshold.
    """
    dead_components_lst: list[str] | None = None
    if filter_dead_threshold > 0:
        dead_components_lst = list()
        max_act: Float[Tensor, " c"] = activations.max(dim=0).values
        dead_components: Bool[Tensor, " n_steps c"] = max_act < filter_dead_threshold

        if dead_components.any():
            activations = activations[:, ~dead_components]
            alive_labels: list[tuple[str, bool]] = [
                (lbl, bool(keep.item()))
                for lbl, keep in zip(labels, ~dead_components, strict=False)
            ]
            # re-assign labels only if we are filtering
            labels = [label for label, keep in alive_labels if keep]
            dead_components_lst = [label for label, keep in alive_labels if not keep]

    return FilteredActivations(
        activations=activations,
        labels=labels,
        dead_components_labels=dead_components_lst,
    )


@dataclass(frozen=True)
class ProcessedActivations:
    """Processed activations after filtering and concatenation"""

    activations_raw: dict[str, Float[Tensor, " n_steps C"]]
    "activations after filtering, but prior to concatenation"

    activations: Float[Tensor, " n_steps c"]
    "activations after filtering and concatenation"

    labels: list[str]
    "list of length c with labels for each preserved component, format `{module_name}:{component_index}`"

    dead_components_lst: list[str] | None
    "list of labels for dead components, or None if no filtering was applied"

    def validate(self) -> None:
        """Validate the processed activations"""
        # getting this property will also perform a variety of other checks
        assert self.n_components_alive > 0

    @property
    def n_components_original(self) -> int:
        """Total number of components before filtering. equal to the sum of all components in `activations_raw`, or to `n_components_alive + n_components_dead`"""
        return sum(act.shape[1] for act in self.activations_raw.values())

    @property
    def n_components_alive(self) -> int:
        """Number of alive components after filtering. equal to the length of `labels`"""
        n_alive: int = len(self.labels)
        assert n_alive + self.n_components_dead == self.n_components_original, (
            f"({n_alive = }) + ({self.n_components_dead = }) != ({self.n_components_original = })"
        )
        assert n_alive == self.activations.shape[1], (
            f"{n_alive = } != {self.activations.shape[1] = }"
        )

        return n_alive

    @property
    def n_components_dead(self) -> int:
        """Number of dead components after filtering. equal to the length of `dead_components_lst` if it is not None, or 0 otherwise"""
        return len(self.dead_components_lst) if self.dead_components_lst else 0

    @cached_property
    def label_index(self) -> dict[str, int | None]:
        """Create a mapping from label to alive index (`None` if dead)"""
        return {
            **{label: i for i, label in enumerate(self.labels)},
            **(
                {label: None for label in self.dead_components_lst}
                if self.dead_components_lst
                else {}
            ),
        }

    def get_label_index(self, label: str) -> int | None:
        """Get the index of a label in the activations, or None if it is dead"""
        return self.label_index[label]

    def get_label_index_alive(self, label: str) -> int:
        """Get the index of a label in the activations, or raise if it is dead"""
        idx: int | None = self.get_label_index(label)
        if idx is None:
            raise ValueError(f"Label '{label}' is dead and has no index in the activations.")
        return idx

    @property
    def module_keys(self) -> list[str]:
        """Get the module keys from the activations_raw"""
        return list(self.activations_raw.keys())

    def get_module_indices(self, module_key: str) -> list[int | None]:
        """given a module key, return a list len "num components in that moduel", with int index in alive components, or None if dead"""
        return [
            self.label_index[f"{module_key}:{i}"]
            for i in range(self.activations_raw[module_key].shape[1])
        ]


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
) -> ProcessedActivations:
    """get back a dict of coactivations, slices, and concated activations

    Args:
        activations: Dictionary of activations by module
        filter_dead_threshold: Threshold for filtering dead components
        seq_mode: How to handle sequence dimension
        filter_modules: Function to filter modules
        sort_components: Whether to sort components by similarity within each module
    """

    # reshape -- special cases for llms
    # ============================================================
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

    # put the labelled activations into one big matrix and filter them
    # ============================================================

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
    filtered_components: FilteredActivations = filter_dead_components(
        activations=act_concat,
        labels=labels,
        filter_dead_threshold=filter_dead_threshold,
    )

    assert filtered_components.n_alive + filtered_components.n_dead == total_c, (
        f"({filtered_components.n_alive = }) + ({filtered_components.n_dead = }) != ({total_c = })"
    )

    # logger.values({
    #     "total_components": total_c,
    #     "n_alive_components": len(labels),
    #     "n_dead_components": len(dead_components_lst),
    # })

    # return
    # ============================================================
    return ProcessedActivations(
        activations_raw=activations_,
        activations=filtered_components.activations,
        labels=filtered_components.labels,
        dead_components_lst=filtered_components.dead_components_labels,
    )
