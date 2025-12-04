"""Analysis script for SPD decompositions of mem models.

Shows which memorized inputs each component activates on with high causal importance.
"""

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
    max_facts_to_show: int = 10,
    device: str | None = None,
) -> None:
    """Analyze which memorized inputs each component activates on.

    Args:
        spd_run_path: Path to the SPD decomposition run (wandb path or local path)
        ci_threshold: Causal importance threshold for considering a component "active"
        max_facts_to_show: Maximum number of facts to show per component
        device: Device to use (defaults to auto-detection)
    """
    if device is None:
        device = get_device()

    print(f"Loading SPD decomposition from: {spd_run_path}")
    print(f"Using device: {device}")
    print(f"Causal importance threshold: {ci_threshold}")
    print()

    # Load the SPD run info and component model
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

    print(f"Loaded dataset with {dataset.n_facts} facts")
    print(f"Vocab size: {target_model.config.vocab_size}")
    print(f"Sequence length: {target_model.config.seq_len}")
    print(f"Number of components (C): {spd_config.C}")
    print()

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

    print()
    print("=" * 80)
    print("COMPONENT ACTIVATION ANALYSIS")
    print("=" * 80)

    # For each module, analyze each component
    for module_name, ci_values in importances_by_module.items():
        # ci_values shape: [n_facts, seq_len, C]
        print()
        print(f"\n{'='*80}")
        print(f"MODULE: {module_name}")
        print(f"{'='*80}")

        n_facts, seq_len, n_components = ci_values.shape

        # We care about the final sequence position for the mem task
        # ci_values[:, -1, :] has shape [n_facts, C]
        ci_final_pos: Float[Tensor, "n_facts C"] = ci_values[:, -1, :]  # noqa: F821

        for comp_idx in range(n_components):
            comp_ci = ci_final_pos[:, comp_idx]  # [n_facts]

            # Find facts where this component has CI above threshold
            active_mask = comp_ci > ci_threshold
            active_indices = torch.where(active_mask)[0]
            n_active = len(active_indices)

            if n_active == 0:
                print(f"\n  Component {comp_idx}: No facts above threshold")
                continue

            # Sort by causal importance (descending)
            active_cis = comp_ci[active_indices]
            sorted_order = torch.argsort(active_cis, descending=True)
            active_indices = active_indices[sorted_order]
            active_cis = active_cis[sorted_order]

            print(f"\n  Component {comp_idx}: {n_active} facts above threshold")
            print(f"  {'─'*60}")

            # Show up to max_facts_to_show
            n_to_show = min(n_active, max_facts_to_show)
            for i in range(n_to_show):
                fact_idx = active_indices[i].item()
                ci_val = active_cis[i].item()
                input_tokens = all_inputs[fact_idx].tolist()
                label_token = all_labels[fact_idx].item()

                print(f"    Fact {fact_idx:4d}: input={input_tokens} → label={label_token}  (CI={ci_val:.3f})")

            if n_active > max_facts_to_show:
                print(f"    ... and {n_active - max_facts_to_show} more facts")

    # Summary statistics
    print()
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for module_name, ci_values in importances_by_module.items():
        ci_final_pos = ci_values[:, -1, :]  # [n_facts, C]
        n_components = ci_final_pos.shape[1]

        print(f"\n{module_name}:")

        # For each component, count how many facts it activates on
        for comp_idx in range(n_components):
            comp_ci = ci_final_pos[:, comp_idx]
            n_active = (comp_ci > ci_threshold).sum().item()
            mean_ci = comp_ci.mean().item()
            max_ci = comp_ci.max().item()
            print(
                f"  Component {comp_idx}: "
                f"active on {n_active:4d}/{dataset.n_facts} facts, "
                f"mean CI={mean_ci:.3f}, max CI={max_ci:.3f}"
            )


if __name__ == "__main__":
    fire.Fire(analyze_decomposition)

