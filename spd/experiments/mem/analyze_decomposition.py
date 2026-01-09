"""Analysis script for SPD decompositions of mem models.

Shows which memorized inputs each component activates on with high causal importance.
"""

from pathlib import Path

import fire
import torch
from jaxtyping import Float
from torch import Tensor

from spd.experiments.mem.mem_dataset import MemDataset
from spd.experiments.mem.models import MemTargetRunInfo, MemTransformer
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device


def analyze_decomposition(
    spd_run_path: str,
    ci_threshold: float = 0.1,
    max_facts_to_show: int = 20,
    output_file: str | None = None,
    device: str | None = None,
    print_component_idx: int | None = None,
) -> None:
    """Analyze which memorized inputs each component activates on.

    Args:
        spd_run_path: Path to the SPD decomposition run (wandb path or local path)
        ci_threshold: Causal importance threshold for considering a component "active"
        max_facts_to_show: Maximum number of facts to show per component
        output_file: Path to write output to (if None, prints to stdout)
        device: Device to use (defaults to auto-detection)
        print_component_idx: If specified, print the U and V matrices for this component index
    """
    if device is None:
        device = get_device()

    # Set up output: either file or stdout
    lines: list[str] = []

    def out(s: str = "") -> None:
        lines.append(s)

    out(f"Loading SPD decomposition from: {spd_run_path}")
    out(f"Using device: {device}")
    out(f"Causal importance threshold: {ci_threshold}")
    out()

    # Load the SPD run info and component model
    print("Loading SPD decomposition...")
    spd_run_info = SPDRunInfo.from_path(spd_run_path)
    spd_config = spd_run_info.config
    comp_model = ComponentModel.from_run_info(spd_run_info)
    comp_model.to(device)
    comp_model.eval()

    # Get the original target model path and load its training info
    assert spd_config.pretrained_model_path, "SPD config must have pretrained_model_path"
    target_run_info = MemTargetRunInfo.from_path(spd_config.pretrained_model_path)
    target_model = comp_model.target_model
    assert isinstance(target_model, MemTransformer), "Target model must be MemTransformer"

    # Create the dataset with the same parameters as training
    dataset = MemDataset(
        n_facts=target_run_info.n_facts,
        vocab_size=target_model.config.vocab_size,
        seq_len=target_model.config.seq_len,
        device=device,
        seed=target_run_info.config.seed,
    )

    # Get C from module_info (use first module's C value as representative)
    n_components = spd_config.module_info[0].C

    out(f"Loaded dataset with {dataset.n_facts} facts")
    out(f"Vocab size: {target_model.config.vocab_size}")
    out(f"Sequence length: {target_model.config.seq_len}")
    out(f"Number of components (C): {n_components}")
    out()

    # Print U and V matrices for a specific component if requested
    if print_component_idx is not None:
        if print_component_idx < 0 or print_component_idx >= n_components:
            raise ValueError(
                f"print_component_idx must be in [0, {n_components - 1}], got {print_component_idx}"
            )
        out("=" * 80)
        out(f"U AND V MATRICES FOR COMPONENT {print_component_idx}")
        out("=" * 80)
        for module_name, component in comp_model.components.items():
            out(f"\nModule: {module_name}")
            out(f"  V shape: {list(component.V.shape)} (v_dim x C)")
            out(f"  U shape: {list(component.U.shape)} (C x u_dim)")
            out()
            v_col = component.V[:, print_component_idx]
            u_row = component.U[print_component_idx, :]
            out(f"  V[:, {print_component_idx}] (v_dim={v_col.shape[0]}):")
            out(f"    {v_col.detach().cpu().numpy()}")
            out()
            out(f"  U[{print_component_idx}, :] (u_dim={u_row.shape[0]}):")
            out(f"    {u_row.detach().cpu().numpy()}")
        out()

    # Get all facts
    all_inputs, all_labels = dataset.get_all_facts()

    # Compute causal importances for all facts
    print("Computing causal importances for all facts...")
    with torch.no_grad():
        # Run the model to get input activations
        pre_weight_acts = comp_model(all_inputs, cache_type="input").cache

        # Calculate causal importances
        ci_outputs = comp_model.calc_causal_importances(
            pre_weight_acts=pre_weight_acts,
            detach_inputs=True,
            sampling=spd_config.sampling,
        )

    # Get the lower leaky sigmoid causal importances (more interpretable)
    importances_by_module = ci_outputs.lower_leaky

    out()
    out("=" * 80)
    out("COMPONENT ACTIVATION ANALYSIS")
    out("=" * 80)

    # For each module, analyze each component
    for module_name, ci_values in importances_by_module.items():
        # ci_values shape: [n_facts, seq_len, C]
        out()
        out(f"\n{'=' * 80}")
        out(f"MODULE: {module_name}")
        out(f"{'=' * 80}")

        _, _, _ = ci_values.shape  # n_facts, seq_len, n_components

        # We care about the final sequence position for the mem task
        # ci_values[:, -1, :] has shape [n_facts, C]
        ci_final_pos: Float[Tensor, "n_facts C"] = ci_values[:, -1, :]  # noqa: F821

        # Compute mean CI per component and sort by it (descending)
        mean_ci_per_comp = ci_final_pos.mean(dim=0)  # [C]
        sorted_comp_indices = torch.argsort(mean_ci_per_comp, descending=True)

        for rank, comp_idx in enumerate(sorted_comp_indices.tolist()):
            comp_ci = ci_final_pos[:, comp_idx]  # [n_facts]

            # Find facts where this component has CI above threshold
            active_mask = comp_ci > ci_threshold
            active_indices = torch.where(active_mask)[0]
            n_active = len(active_indices)

            mean_ci = mean_ci_per_comp[comp_idx].item()

            if n_active == 0:
                out(
                    f"\n  [Rank {rank + 1}] Component {comp_idx} (mean CI={mean_ci:.3f}): No facts above threshold"
                )
                continue

            # Sort by causal importance (descending)
            active_cis = comp_ci[active_indices]
            sorted_order = torch.argsort(active_cis, descending=True)
            active_indices = active_indices[sorted_order]
            active_cis = active_cis[sorted_order]

            out(
                f"\n  [Rank {rank + 1}] Component {comp_idx} (mean CI={mean_ci:.3f}): {n_active} facts above threshold"
            )
            out(f"  {'─' * 60}")

            # Show up to max_facts_to_show
            n_to_show = min(n_active, max_facts_to_show)
            for i in range(n_to_show):
                fact_idx = int(active_indices[i].item())
                ci_val = active_cis[i].item()
                input_tokens = all_inputs[fact_idx].tolist()
                label_token = int(all_labels[fact_idx].item())

                out(
                    f"    Fact {fact_idx:4d}: input={input_tokens} → label={label_token}  (CI={ci_val:.3f})"
                )

            if n_active > max_facts_to_show:
                out(f"    ... and {n_active - max_facts_to_show} more facts")

    # Summary statistics
    out()
    out("=" * 80)
    out("SUMMARY STATISTICS")
    out("=" * 80)

    for module_name, ci_values in importances_by_module.items():
        ci_final_pos = ci_values[:, -1, :]  # [n_facts, C]

        out(f"\n{module_name}:")

        # Compute mean CI per component and sort by it (descending)
        mean_ci_per_comp = ci_final_pos.mean(dim=0)  # [C]
        sorted_comp_indices = torch.argsort(mean_ci_per_comp, descending=True)

        # For each component (sorted by mean CI), count how many facts it activates on
        for rank, comp_idx in enumerate(sorted_comp_indices.tolist()):
            comp_ci = ci_final_pos[:, comp_idx]
            n_active = (comp_ci > ci_threshold).sum().item()
            mean_ci = mean_ci_per_comp[comp_idx].item()
            max_ci = comp_ci.max().item()
            out(
                f"  [Rank {rank + 1:2d}] Component {comp_idx:3d}: "
                f"active on {n_active:4d}/{dataset.n_facts} facts, "
                f"mean CI={mean_ci:.3f}, max CI={max_ci:.3f}"
            )

    # Per-fact analysis: which components activate on each fact
    out()
    out("=" * 80)
    out("PER-FACT COMPONENT ANALYSIS")
    out("=" * 80)

    for module_name, ci_values in importances_by_module.items():
        ci_final_pos = ci_values[:, -1, :]  # [n_facts, C]

        out(f"\n{'=' * 80}")
        out(f"MODULE: {module_name}")
        out(f"{'=' * 80}")

        for fact_idx in range(dataset.n_facts):
            fact_ci = ci_final_pos[fact_idx, :]  # [C]

            # Find components that activate above threshold on this fact
            active_mask = fact_ci > ci_threshold
            active_comp_indices = torch.where(active_mask)[0]
            n_active = len(active_comp_indices)

            input_tokens = all_inputs[fact_idx].tolist()
            label_token = int(all_labels[fact_idx].item())

            if n_active == 0:
                out(f"\n  Fact {fact_idx:4d}: input={input_tokens} → label={label_token}")
                out("    No components above threshold")
                continue

            # Sort by causal importance (descending)
            active_cis = fact_ci[active_comp_indices]
            sorted_order = torch.argsort(active_cis, descending=True)
            active_comp_indices = active_comp_indices[sorted_order]
            active_cis = active_cis[sorted_order]

            out(f"\n  Fact {fact_idx:4d}: input={input_tokens} → label={label_token}")
            out(f"    {n_active} components above threshold:")

            # List all active components
            comp_strs = [
                f"C{int(active_comp_indices[i].item())}({active_cis[i].item():.3f})"
                for i in range(n_active)
            ]
            # Print in rows of ~8 components each for readability
            for row_start in range(0, len(comp_strs), 8):
                row_end = min(row_start + 8, len(comp_strs))
                out(f"      {', '.join(comp_strs[row_start:row_end])}")

    # Write output
    output_text = "\n".join(lines)

    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text)
        print(f"Output written to: {output_path}")
    else:
        print(output_text)


if __name__ == "__main__":
    fire.Fire(analyze_decomposition)
