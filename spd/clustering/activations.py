from typing import Any, Literal

import torch
from jaxtyping import Float, Int
from muutils.dbg import dbg, dbg_auto
from torch import Tensor
from torch.utils.data import DataLoader

from spd.models.component_model import ComponentModel
from spd.models.sigmoids import SigmoidTypes
from spd.utils.general_utils import extract_batch_data


@torch.no_grad()
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
    batch_: Tensor
    if batch is None:
        assert dataloader is not None, "provide either a batch or a dataloader, not both"
        batch_ = extract_batch_data(next(iter(dataloader)))
    else:
        assert dataloader is None, "provide either a batch or a dataloader, not both"
        batch_ = batch

    batch_ = batch_.to(device)

    _, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
        batch_, module_names=model.target_module_paths
    )

    causal_importances, _ = model.calc_causal_importances(
        pre_weight_acts=pre_weight_acts,
        sigmoid_type=sigmoid_type,
        detach_inputs=False,
    )

    return causal_importances


def process_activations(
    activations: dict[
        str,  # module name to
        Float[Tensor, " n_steps C"]  # (sample x component gate activations)
        | Float[Tensor, " n_sample n_ctx C"],  # (sample x seq index x component gate activations)
    ],
    filter_dead_threshold: float = 0.01,
    seq_mode: Literal["concat", "seq_mean", None] = None,
    plots: bool = False,
    save_pdf: bool = False,
    pdf_prefix: str = "activations",
    figsize_raw: tuple[int, int] = (12, 4),
    figsize_concat: tuple[int, int] = (12, 2),
    figsize_coact: tuple[int, int] = (8, 6),
    hist_scales: tuple[str, str] = ("lin", "log"),
    hist_bins: int = 100,
) -> dict[str, Any]:
    """get back a dict of coactivations, slices, and concated activations"""

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

    # compute the labels and total component count
    total_c: int = 0
    labels: list[str] = list()
    for key, act in activations_.items():
        c = act.shape[-1]
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
                (lbl, keep.item()) for lbl, keep in zip(labels, ~dead_components, strict=False)
            ]
            labels = [label for label, keep in alive_labels if keep]
            dead_components_lst = [label for label, keep in alive_labels if not keep]
            dbg((len(dead_components_lst), len(labels)))

    # compute coactivations
    coact: Float[Tensor, " c c"] = act_concat.T @ act_concat

    # return the output
    output: dict[str, Any] = dict(
        activations=act_concat,
        labels=labels,
        coactivations=coact,
        dead_components_lst=dead_components_lst,
        n_components_original=total_c,
        n_components_alive=len(labels),
        n_components_dead=len(dead_components_lst) if dead_components_lst else 0,
    )

    dbg_auto(output)

    if plots:
        # Import plotting function only when needed
        from spd.clustering.plotting import plot_activations

        # Use original activations for raw plots, but filtered data for concat/coact/histograms
        plot_activations(
            activations=activations_,  # Original unfiltered for raw activations
            act_concat=act_concat,  # Filtered concatenated activations
            coact=coact,  # Coactivations from filtered data
            labels=labels,  # Labels matching filtered data
            save_pdf=save_pdf,
            pdf_prefix=pdf_prefix,
            figsize_raw=figsize_raw,
            figsize_concat=figsize_concat,
            figsize_coact=figsize_coact,
            hist_scales=hist_scales,
            hist_bins=hist_bins,
        )

    return output
