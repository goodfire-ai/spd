from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float, Int
from muutils.dbg import dbg, dbg_auto
from torch import Tensor
from torch.utils.data import DataLoader

from spd.models.component_model import ComponentModel
from spd.models.components import EmbeddingComponent, GateMLP, LinearComponent, VectorGateMLP
from spd.utils.component_utils import calc_causal_importances
from spd.utils.general_utils import extract_batch_data


def plot_activations(
	activations: dict[str, Float[Tensor, " n_steps C"]],
	act_concat: Float[Tensor, " n_steps c"],
	coact: Float[Tensor, " c c"],
	labels: list[str],
	save_pdf: bool = False,
	pdf_prefix: str = "activations",
	figsize_raw: tuple[int, int] = (12, 4),
	figsize_concat: tuple[int, int] = (12, 2),
	figsize_coact: tuple[int, int] = (8, 6),
	hist_scales: tuple[str, str] = ("lin", "log"),
	hist_bins: int = 100,
) -> None:
	"""Plot activation visualizations including raw, concatenated, sorted, and coactivations.
	
	Args:
		activations: Dictionary of raw activations by module
		act_concat: Concatenated activations tensor
		coact: Coactivation matrix
		labels: Component labels
		save_pdf: Whether to save plots as PDFs
		pdf_prefix: Prefix for PDF filenames
		figsize_raw: Figure size for raw activations
		figsize_concat: Figure size for concatenated activations
		figsize_coact: Figure size for coactivations
		hist_scales: Tuple of (x_scale, y_scale) where each is "lin" or "log"
		hist_bins: Number of bins for histograms
	"""
	# Raw activations
	fig1, axs_act = plt.subplots(len(activations), 1, figsize=figsize_raw)
	if len(activations) == 1:
		axs_act = [axs_act]
	for i, (key, act) in enumerate(activations.items()):
		act_raw_data: np.ndarray = act.T.cpu().numpy()
		axs_act[i].matshow(act_raw_data, aspect="auto", vmin=act_raw_data.min(), vmax=act_raw_data.max())
		axs_act[i].set_ylabel(f"components\n{key}")
		axs_act[i].set_title(f"Raw Activations: {key} (shape: {act_raw_data.shape})")

	# Concatenated activations
	fig2, ax2 = plt.subplots(figsize=figsize_concat)
	act_data: np.ndarray = act_concat.T.cpu().numpy()
	im2 = ax2.matshow(act_data, aspect="auto", vmin=act_data.min(), vmax=act_data.max())
	ax2.set_title("Concatenated Activations")
	
	# Add component labeling on y-axis
	add_component_labeling(ax2, labels, axis='y')
	
	plt.colorbar(im2)
	
	if save_pdf:
		fig2.savefig(f"{pdf_prefix}_concatenated.pdf", bbox_inches='tight', dpi=300)

	# Concatenated activations, sorted samples
	fig3, ax3 = plt.subplots(figsize=figsize_concat)
	
	# Compute gram matrix (sample similarity) and sort samples
	gram_matrix: Float[Tensor, " n_steps n_steps"] = act_concat @ act_concat.T
	
	# Use hierarchical clustering approach for better sorting
	similarity_scores: Float[Tensor, " n_steps"] = gram_matrix.sum(dim=1)
	sorted_indices: torch.Tensor = torch.argsort(similarity_scores, descending=True)
	
	act_concat_sorted: Float[Tensor, " n_steps c"] = act_concat[sorted_indices]
	
	# Handle log10 properly - add small epsilon to avoid log(0)
	act_sorted_data: np.ndarray = act_concat_sorted.T.cpu().numpy()
	act_sorted_log: np.ndarray = np.log10(act_sorted_data + 1e-10)
	im3 = ax3.matshow(act_sorted_log, aspect="auto", vmin=act_sorted_log.min(), vmax=act_sorted_log.max())
	ax3.set_title("Concatenated Activations $\\log_{10}$, Sorted Samples")
	
	# Add component labeling on y-axis
	add_component_labeling(ax3, labels, axis='y')
	
	plt.colorbar(im3)
	
	if save_pdf:
		fig3.savefig(f"{pdf_prefix}_concatenated_sorted.pdf", bbox_inches='tight', dpi=300)

	# Coactivations
	fig4, ax4 = plt.subplots(figsize=figsize_coact)
	coact_data: np.ndarray = coact.cpu().numpy()
	im4 = ax4.matshow(coact_data, aspect="auto", vmin=coact_data.min(), vmax=coact_data.max())
	ax4.set_title("Coactivations")
	
	# Add component labeling on both axes
	add_component_labeling(ax4, labels, axis='x')
	add_component_labeling(ax4, labels, axis='y')
	
	plt.colorbar(im4)
	
	if save_pdf:
		fig4.savefig(f"{pdf_prefix}_coactivations.pdf", bbox_inches='tight', dpi=300)

	# Activation histograms
	fig5, (ax5a, ax5b, ax5c) = plt.subplots(1, 3, figsize=(15, 4))
	
	x_scale, y_scale = hist_scales
	
	# Histogram 1: All activations
	all_activations: Float[Tensor, ""] = act_concat.flatten()
	all_vals: np.ndarray = all_activations.cpu().numpy()
	hist_counts, bin_edges = np.histogram(all_vals, bins=hist_bins)
	bin_centers: np.ndarray = (bin_edges[:-1] + bin_edges[1:]) / 2
	ax5a.plot(bin_centers, hist_counts, color='blue', linewidth=2)
	ax5a.set_title("All Activations")
	ax5a.set_xlabel("Activation Value")
	ax5a.set_ylabel("Count")
	if x_scale == "log":
		ax5a.set_xscale("log")
	if y_scale == "log":
		ax5a.set_yscale("log")
	ax5a.grid(True, alpha=0.3)
	
	# Histogram 2: Activations per component
	n_components: int = act_concat.shape[1]
	n_samples: int = act_concat.shape[0]
	
	# Common bin edges for all component histograms
	all_min: float = float(all_vals.min())
	all_max: float = float(all_vals.max())
	common_bins: np.ndarray = np.linspace(all_min, all_max, hist_bins)
	common_centers: np.ndarray = (common_bins[:-1] + common_bins[1:]) / 2
	
	# Get unique label prefixes and assign colors
	import matplotlib.cm as cm
	label_prefixes: list[str] = [label.split(':')[0] for label in labels]
	unique_prefixes: list[str] = list(dict.fromkeys(label_prefixes))  # Preserve order
	colors = cm.tab10(np.linspace(0, 1, len(unique_prefixes)))
	prefix_colors: dict[str, tuple] = {prefix: colors[i] for i, prefix in enumerate(unique_prefixes)}
	
	for comp_idx in range(n_components):
		component_activations: Float[Tensor, " n_samples"] = act_concat[:, comp_idx]
		comp_vals: np.ndarray = component_activations.cpu().numpy()
		hist_counts, _ = np.histogram(comp_vals, bins=common_bins, density=True)
		
		# Get color based on label prefix
		prefix: str = label_prefixes[comp_idx]
		color: tuple = prefix_colors[prefix]
		
		ax5b.plot(common_centers, hist_counts, color=color, alpha=0.1, linewidth=1)
	
	ax5b.set_title(f"Per Component ({n_components} components)")
	ax5b.set_xlabel("Activation Value")
	ax5b.set_ylabel("Density")
	if x_scale == "log":
		ax5b.set_xscale("log")
	if y_scale == "log":
		ax5b.set_yscale("log")
	ax5b.grid(True, alpha=0.3)
	
	# Histogram 3: Activations per sample
	for sample_idx in range(n_samples):
		sample_activations: Float[Tensor, " n_components"] = act_concat[sample_idx, :]
		sample_vals: np.ndarray = sample_activations.cpu().numpy()
		hist_counts, _ = np.histogram(sample_vals, bins=common_bins, density=True)
		ax5c.plot(common_centers, hist_counts, color='blue', alpha=0.1, linewidth=1)
	
	ax5c.set_title(f"Per Sample ({n_samples} samples)")
	ax5c.set_xlabel("Activation Value")
	ax5c.set_ylabel("Density")
	if x_scale == "log":
		ax5c.set_xscale("log")
	if y_scale == "log":
		ax5c.set_yscale("log")
	ax5c.grid(True, alpha=0.3)
	
	plt.tight_layout()
	
	if save_pdf:
		fig5.savefig(f"{pdf_prefix}_histograms.pdf", bbox_inches='tight', dpi=300)


def add_component_labeling(ax: plt.Axes, component_labels: list[str], axis: str = 'x') -> None:
	"""Add component labeling using major/minor ticks to show module boundaries.
	
	Args:
		ax: Matplotlib axis to modify
		component_labels: List of component labels in format "module:index"
		axis: Which axis to label ('x' or 'y')
	"""
	if not component_labels:
		return
		
	# Extract module information
	module_changes: list[int] = []
	current_module: str = component_labels[0].split(':')[0]
	module_labels: list[str] = []
	
	for i, label in enumerate(component_labels):
		module: str = label.split(':')[0]
		if module != current_module:
			module_changes.append(i)
			module_labels.append(current_module)
			current_module = module
	module_labels.append(current_module)
	
	# Set up major and minor ticks
	# Minor ticks: every 10 components
	minor_ticks: list[int] = list(range(0, len(component_labels), 10))
	
	# Major ticks: module boundaries (start of each module)
	major_ticks: list[int] = [0] + module_changes
	major_labels: list[str] = module_labels
	
	if axis == 'x':
		ax.set_xticks(minor_ticks, minor=True)
		ax.set_xticks(major_ticks)
		ax.set_xticklabels(major_labels)
		ax.set_xlim(-0.5, len(component_labels) - 0.5)
		# Style the ticks
		ax.tick_params(axis='x', which='minor', length=2, width=0.5)
		ax.tick_params(axis='x', which='major', length=6, width=1.5)
	else:
		ax.set_yticks(minor_ticks, minor=True)
		ax.set_yticks(major_ticks)
		ax.set_yticklabels(major_labels)
		ax.set_ylim(-0.5, len(component_labels) - 0.5)
		# Style the ticks
		ax.tick_params(axis='y', which='minor', length=2, width=0.5)
		ax.tick_params(axis='y', which='major', length=6, width=1.5)


@torch.no_grad()
def component_activations(
    model: ComponentModel,
    dataloader: DataLoader[Int[Tensor, "..."]]
    | DataLoader[tuple[Float[Tensor, "..."], Float[Tensor, "..."]]],
    device: torch.device,
) -> dict[str, Float[Tensor, " n_steps C"]]:
    """Get the number and strength of the masks over the full dataset."""
    # We used "-" instead of "." as module names can't have "." in them
    gates: dict[str, GateMLP | VectorGateMLP] = {
        k.removeprefix("gates.").replace("-", "."): cast(GateMLP | VectorGateMLP, v)
        for k, v in model.gates.items()
    }
    components: dict[str, LinearComponent | EmbeddingComponent] = {
        k.removeprefix("components.").replace("-", "."): cast(
            LinearComponent | EmbeddingComponent, v
        )
        for k, v in model.components.items()
    }

    # --- Get Batch --- #
    batch = extract_batch_data(next(iter(dataloader)))
    batch = batch.to(device)

    _, pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
        batch, module_names=list(components.keys())
    )
    Vs = {module_name: v.V for module_name, v in components.items()}

    causal_importances, _ = calc_causal_importances(
        pre_weight_acts=pre_weight_acts,
        Vs=Vs,
        gates=gates,
        detach_inputs=False,
    )

    return causal_importances




def process_activations(
	activations: dict[str, Float[Tensor, " n_steps C"]], # module name to sample x component gate activations
	filter_dead_threshold: float = 0.01,
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
	
	# compute the labels and total component count
	total_c: int = 0
	labels: list[str] = list()
	for key, act in activations.items():
		c = act.shape[-1]
		labels.extend([f"{key}:{i}" for i in range(c)])		
		total_c += c

	dbg(total_c)

	# concat the activations
	act_concat: Float[Tensor, " n_steps c"] = torch.cat(
		[activations[key] for key in activations], dim=-1
	)

	# filter dead components
	dead_components_lst: list[str]| None = None
	if filter_dead_threshold > 0:
		dead_components_lst = list()
		max_act = act_concat.max(dim=0).values
		dead_components = max_act < filter_dead_threshold
		if dead_components.any():
			act_concat = act_concat[:, ~dead_components]
			alive_labels: list[tuple[str, bool]] = [
				(lbl, keep.item())
				for lbl, keep in zip(labels, ~dead_components, strict=False)
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
		# Use original activations for raw plots, but filtered data for concat/coact/histograms
		plot_activations(
			activations=activations,  # Original unfiltered for raw activations
			act_concat=act_concat,    # Filtered concatenated activations
			coact=coact,              # Coactivations from filtered data
			labels=labels,            # Labels matching filtered data
			save_pdf=save_pdf,
			pdf_prefix=pdf_prefix,
			figsize_raw=figsize_raw,
			figsize_concat=figsize_concat,
			figsize_coact=figsize_coact,
			hist_scales=hist_scales,
			hist_bins=hist_bins,
		)

	return output
