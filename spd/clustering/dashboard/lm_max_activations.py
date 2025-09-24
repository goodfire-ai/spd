"""Compute max-activating text samples for language model component clusters."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from spd.clustering.activations import (
    ProcessedActivations,
    component_activations,
    process_activations,
)
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.merge_history import MergeHistory
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.general_utils import extract_batch_data, get_module_device


@dataclass
class TextSample:
    """A text sample with activation information."""

    full_text: str  # Original full context
    dataset_index: int  # Index from original dataset
    tokens: list[str]  # Token strings
    activations: list[float]  # Activation for each token
    mean_activation: float
    median_activation: float
    max_activation: float
    max_position: int  # Position of max activation

    def serialize(self) -> dict[str, Any]:
        """Serialize the TextSample to a dictionary."""
        return asdict(self)


def compute_max_activations(
    model: ComponentModel,
    tokenizer: PreTrainedTokenizer,
    dataloader: DataLoader[Any],
    merge_history: MergeHistory,
    iteration: int,
    n_samples: int,
    n_batches: int,
) -> dict[int, dict[str, list[dict[str, Any]]]]:
    device: torch.device = get_module_device(model)

    # Get the merge at specified iteration
    if iteration < 0:
        iteration = merge_history.n_iters_current + iteration
    assert 0 <= iteration < merge_history.n_iters_current, (
        f"Invalid iteration: {iteration = }, {merge_history.n_iters_current = }"
    )
    merge: GroupMerge = merge_history.merges[iteration]

    # Get unique cluster IDs
    unique_clusters: list[int] = torch.unique(merge.group_idxs).tolist()

    # Initialize tracking
    max_acts: dict[int, Float[Tensor, " n_samples"]] = {
        cid: torch.full((n_samples,), -1e10, device=device) for cid in unique_clusters
    }
    max_texts: dict[int, list[TextSample | None]] = {
        cid: [None] * n_samples for cid in unique_clusters
    }
    dataset_idx_counter: int = 0

    # Map clusters to component labels as dicts
    cluster_components: dict[int, list[dict[str, Any]]] = {}
    for cluster_id in unique_clusters:
        component_indices: list[int] = (
            (merge.group_idxs == cluster_id).nonzero().squeeze(-1).tolist()
        )
        cluster_components[cluster_id] = []
        for comp_idx in component_indices:
            label: str = merge_history.labels[comp_idx]
            module: str
            idx_str: str
            module, idx_str = label.rsplit(":", 1)
            cluster_components[cluster_id].append(
                {"module": module, "index": int(idx_str), "label": label}
            )

    for batch_idx, batch_data in enumerate(tqdm(dataloader, total=n_batches)):
        if batch_idx >= n_batches:
            break

        batch: Int[Tensor, "batch_size n_ctx"] = extract_batch_data(batch_data).to(device)
        batch_size: int
        seq_len: int
        batch_size, seq_len = batch.shape

        # Get activations
        activations: dict[str, Float[Tensor, "n_steps C"]] = component_activations(
            model, device, batch=batch
        )
        processed: ProcessedActivations = process_activations(activations, seq_mode="concat")

        for cluster_id in unique_clusters:
            # Get indices for components in this cluster
            comp_indices: list[int] = []
            for component_info in cluster_components[cluster_id]:
                label: str = component_info["label"]
                comp_idx: int | None = processed.get_label_index(label)
                if comp_idx is not None:
                    comp_indices.append(comp_idx)

            if not comp_indices:
                continue

            # Average activations across cluster components
            cluster_acts: Float[Tensor, " n_steps"] = processed.activations[:, comp_indices].mean(
                dim=1
            )
            acts_2d: Float[Tensor, "batch_size seq_len"] = cluster_acts.view(batch_size, seq_len)

            # Find top activations across all positions
            flat_acts: Float[Tensor, " batch_size*seq_len"] = acts_2d.flatten()
            k: int = min(n_samples, len(flat_acts))
            top_vals: Float[Tensor, " k"]
            top_idx: Int[Tensor, " k"]
            top_vals, top_idx = torch.topk(flat_acts, k)

            for val, idx in zip(top_vals, top_idx, strict=False):
                batch_idx_i: int = int(idx // seq_len)
                pos_idx: int = int(idx % seq_len)

                # Insert in sorted order
                for j in range(n_samples):
                    if val > max_acts[cluster_id][j]:
                        # Shift and insert
                        if j < n_samples - 1:
                            max_acts[cluster_id][j + 1 :] = max_acts[cluster_id][j:-1].clone()
                            max_texts[cluster_id][j + 1 :] = max_texts[cluster_id][j:-1]

                        max_acts[cluster_id][j] = val

                        # Extract full sequence data
                        tokens: Int[Tensor, " n_ctx"] = batch[batch_idx_i].cpu()
                        tokens_list: list[int] = tokens.tolist()
                        text: str = tokenizer.decode(tokens)

                        # Convert token IDs to token strings
                        token_strings: list[str] = [tokenizer.decode([tid]) for tid in tokens_list]

                        # Get all activations for this sequence
                        sequence_acts: Float[Tensor, " seq_len"] = acts_2d[batch_idx_i]
                        activations_list: list[float] = sequence_acts.cpu().tolist()

                        # Compute statistics
                        mean_act: float = float(sequence_acts.mean().item())
                        median_act: float = float(sequence_acts.median().item())
                        max_act: float = float(val)

                        max_texts[cluster_id][j] = TextSample(
                            full_text=text,
                            dataset_index=dataset_idx_counter + batch_idx_i,
                            tokens=token_strings,
                            activations=activations_list,
                            mean_activation=mean_act,
                            median_activation=median_act,
                            max_activation=max_act,
                            max_position=pos_idx,
                        )
                        break

        dataset_idx_counter += batch_size

    # Format output
    result: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for cluster_id in unique_clusters:
        samples: list[TextSample] = [s for s in max_texts[cluster_id] if s is not None]
        result[cluster_id] = {
            "components": cluster_components[cluster_id],
            "samples": [s.serialize() for s in samples],
        }

    return result


def generate_model_info(
    model: ComponentModel,
    merge_history: MergeHistory,
    merge: GroupMerge,
    iteration: int,
    model_path: str,
    tokenizer_name: str,
    config_dict: dict[str, Any] | None = None,
    wandb_run_path: str | None = None,
) -> dict[str, Any]:
    """Generate model information dictionary.

    Args:
        model: The ComponentModel instance
        merge_history: MergeHistory containing component labels
        merge: GroupMerge for the current iteration
        iteration: Current iteration number
        model_path: Path to the model
        tokenizer_name: Name of the tokenizer
        config_dict: Optional config dictionary
        wandb_run_path: Optional wandb run path

    Returns:
        Dictionary containing model information
    """
    # Count unique modules from all components in the merge history
    unique_modules: set[str] = set()
    total_components: int = len(merge_history.labels)

    for label in merge_history.labels:
        module, _ = label.rsplit(":", 1)
        unique_modules.add(module)

    # Count parameters in the model
    total_params: int = sum(p.numel() for p in model.parameters())
    trainable_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Create model info dictionary
    model_info: dict[str, Any] = {
        "total_modules": len(unique_modules),
        "total_components": total_components,
        "total_clusters": len(torch.unique(merge.group_idxs)),
        "iteration": iteration,
        "model_path": model_path,
        "tokenizer_name": tokenizer_name,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "component_size": getattr(model, "C", None),
        "module_list": sorted(list(unique_modules)),
    }

    # Add config information if available
    if config_dict is not None:
        model_info["config"] = config_dict

    # Add wandb run information if available
    if wandb_run_path is not None:
        model_info["wandb_run"] = wandb_run_path

    return model_info


def main() -> None:
    logger.info("parsing args")
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Compute max-activating text samples for language model component clusters."
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to model file or wandb:project/entity/run_id format",
    )
    parser.add_argument(
        "merge_history_path",
        type=Path,
        help="Path to merge history file",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="Merge iteration to analyze (negative indexes from end, default: -1 for latest)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=4,
        help="Number of top-activating samples to collect per cluster (default: 10)",
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=4,
        help="Number of data batches to process (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for data loading (default: 4)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=64,
        help="Context length for tokenization (default: 64)",
    )
    args: argparse.Namespace = parser.parse_args()

    # Load model and merge history
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("getting tokenizer and model")
    tokenizer_name: str
    wandb_run_path: str | None = None
    config_dict: dict[str, Any] | None = None

    if args.model_path.startswith("wandb:"):
        spd_run: SPDRunInfo = SPDRunInfo.from_path(args.model_path)
        model: ComponentModel = ComponentModel.from_run_info(spd_run)
        model.to(device)
        model.eval()
        config: Any = spd_run.config
        tokenizer_name = config.tokenizer_name
        wandb_run_path = args.model_path
        # Convert config to dict for JSON serialization
        config_dict = vars(config) if hasattr(config, "__dict__") else dict(config)
    else:
        model = torch.load(args.model_path)
        model.to(device)
        model.eval()
        tokenizer_name = "gpt2"  # fallback

    logger.info(f"{tokenizer_name = }")

    merge_history: MergeHistory = MergeHistory.read(args.merge_history_path)
    logger.info(f"Loaded: {merge_history = }")

    # Get the merge at specified iteration for component counts
    if args.iteration < 0:
        iteration = merge_history.n_iters_current + args.iteration
    else:
        iteration = args.iteration
    merge: GroupMerge = merge_history.merges[iteration]

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    logger.info(f"Loaded: {tokenizer = }")

    # TODO: read this from batches_config.json
    dataset_config: DatasetConfig = DatasetConfig(
        name="SimpleStories/SimpleStories",
        hf_tokenizer_path=tokenizer_name,
        split="train",
        n_ctx=args.context_length,
        is_tokenized=False,  # Text dataset
        streaming=False,
        column_name="story",
    )
    logger.info(f"Using {dataset_config = }")

    dataloader: DataLoader[Any]
    dataloader, _ = create_data_loader(
        dataset_config=dataset_config,
        batch_size=args.batch_size,
        buffer_size=4,
        ddp_rank=0,
        ddp_world_size=1,
    )
    logger.info(f"Created {dataloader = }")

    # Compute max activations
    logger.info("computing max activations")
    result: dict[int, dict[str, list[dict[str, Any]]]] = compute_max_activations(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        merge_history=merge_history,
        iteration=args.iteration,
        n_samples=args.n_samples,
        n_batches=args.n_batches,
    )
    logger.info(f"computed max activations: {len(result) = }")

    # Generate model information
    model_info: dict[str, Any] = generate_model_info(
        model=model,
        merge_history=merge_history,
        merge=merge,
        iteration=iteration,
        model_path=args.model_path,
        tokenizer_name=tokenizer_name,
        config_dict=config_dict,
        wandb_run_path=wandb_run_path,
    )

    # Save to output directory with reasonable name
    output_dir: Path = args.merge_history_path.parent
    output_filename: str = f"max_activations_iter{args.iteration}_n{args.n_samples}.json"
    output_path: Path = output_dir / output_filename

    # Save model info
    model_info_path: Path = output_dir / "model_info.json"
    with open(model_info_path, "w") as f:
        json.dump(model_info, f, indent=2)
    logger.info(f"Model info saved to: {model_info_path}")

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
