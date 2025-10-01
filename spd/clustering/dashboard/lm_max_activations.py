"""Compute max-activating text samples for language model component clusters."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import wandb
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from wandb.apis.public import Run

from spd.clustering.activations import (
    ProcessedActivations,
    component_activations,
    process_activations,
)
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.merge_history import MergeHistory
from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.sigmoids import SigmoidTypes
from spd.settings import REPO_ROOT
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


class ClusterMaxTracker:
    """Tracks top-k max-activating samples per cluster."""

    def __init__(self, cluster_ids: list[int], n_samples: int, device: torch.device):
        self.n_samples: int = n_samples
        self.device: torch.device = device

        # Initialize tracking structures
        self.max_acts: dict[int, Float[Tensor, " n_samples"]] = {
            cid: torch.full((n_samples,), -1e10, device=device) for cid in cluster_ids
        }
        self.max_texts: dict[int, list[TextSample | None]] = {
            cid: [None] * n_samples for cid in cluster_ids
        }
        self.used_dataset_indices: dict[int, set[int]] = {cid: set() for cid in cluster_ids}

    def try_insert_batch(
        self,
        cluster_id: int,
        vals: Float[Tensor, " k"],
        text_samples: list[TextSample],
    ) -> int:
        """Try to insert multiple text samples if they're in the top-k for the cluster.

        Args:
            cluster_id: Cluster ID
            vals: Activation values (length k)
            text_samples: TextSamples to insert (length k)

        Returns:
            Number of samples successfully inserted
        """
        assert len(vals) == len(text_samples), "vals and text_samples must have same length"

        n_inserted: int = 0
        for val, text_sample in zip(vals, text_samples, strict=True):
            # Skip if we've already used this dataset index for this cluster
            if text_sample.dataset_index in self.used_dataset_indices[cluster_id]:
                continue

            # Find insertion point
            for j in range(self.n_samples):
                if val > self.max_acts[cluster_id][j]:
                    # Shift and insert
                    if j < self.n_samples - 1:
                        self.max_acts[cluster_id][j + 1 :] = self.max_acts[cluster_id][
                            j:-1
                        ].clone()
                        self.max_texts[cluster_id][j + 1 :] = self.max_texts[cluster_id][j:-1]

                    self.max_acts[cluster_id][j] = val
                    self.max_texts[cluster_id][j] = text_sample
                    self.used_dataset_indices[cluster_id].add(text_sample.dataset_index)
                    n_inserted += 1
                    break

        return n_inserted

    def to_result_dict(
        self, cluster_components: dict[int, list[dict[str, Any]]]
    ) -> dict[int, dict[str, list[dict[str, Any]]]]:
        """Convert tracking state to final result dictionary.

        Args:
            cluster_components: Mapping from cluster_id to component info dicts

        Returns:
            Dict mapping cluster_id to dict with keys "components" and "samples"
        """
        result: dict[int, dict[str, list[dict[str, Any]]]] = {}
        for cluster_id in self.max_texts:
            samples: list[TextSample] = [s for s in self.max_texts[cluster_id] if s is not None]
            result[cluster_id] = {
                "components": cluster_components[cluster_id],
                "samples": [s.serialize() for s in samples],
            }
        return result


def _create_text_sample(
    batch: Int[Tensor, "batch_size n_ctx"],
    batch_idx: int,
    pos_idx: int,
    sequence_acts: Float[Tensor, " seq_len"],
    val: float,
    dataset_index: int,
    tokenizer: PreTrainedTokenizer,
) -> TextSample:
    """Create a TextSample from batch data and activations.

    Args:
        batch: Input token batch
        batch_idx: Index within batch
        pos_idx: Position of max activation
        sequence_acts: Activations for entire sequence
        val: Max activation value
        dataset_index: Index in original dataset
        tokenizer: Tokenizer for decoding

    Returns:
        TextSample instance
    """
    # Extract full sequence data
    tokens: Int[Tensor, " n_ctx"] = batch[batch_idx].cpu()
    tokens_list: list[int] = tokens.tolist()
    text: str = tokenizer.decode(tokens)  # pyright: ignore[reportAttributeAccessIssue]

    # Convert token IDs to token strings
    token_strings: list[str] = [
        tokenizer.decode([tid])  # pyright: ignore[reportAttributeAccessIssue]
        for tid in tokens_list
    ]

    # Get all activations for this sequence
    activations_list: list[float] = sequence_acts.cpu().tolist()

    # Compute statistics
    mean_act: float = float(sequence_acts.mean().item())
    median_act: float = float(sequence_acts.median().item())
    max_act: float = float(val)

    return TextSample(
        full_text=text,
        dataset_index=dataset_index,
        tokens=token_strings,
        activations=activations_list,
        mean_activation=mean_act,
        median_activation=median_act,
        max_activation=max_act,
        max_position=pos_idx,
    )


def _compute_cluster_activations(
    processed: ProcessedActivations,
    cluster_components: list[dict[str, Any]],
    batch_size: int,
    seq_len: int,
) -> Float[Tensor, "batch_size seq_len"]:
    """Compute average activations for a cluster across its components.

    Args:
        processed: ProcessedActivations containing all component activations
        cluster_components: List of component info dicts for this cluster
        batch_size: Batch size
        seq_len: Sequence length

    Returns:
        2D tensor of cluster activations (batch_size x seq_len)
    """
    # Get indices for components in this cluster
    comp_indices: list[int] = []
    for component_info in cluster_components:
        label: str = component_info["label"]
        comp_idx: int | None = processed.get_label_index(label)
        if comp_idx is not None:
            comp_indices.append(comp_idx)

    if not comp_indices:
        # Return zeros if no valid components
        return torch.zeros((batch_size, seq_len), device=processed.activations.device)

    # Average activations across cluster components
    cluster_acts: Float[Tensor, " n_steps"] = processed.activations[:, comp_indices].mean(dim=1)
    return cluster_acts.view(batch_size, seq_len)


def load_wandb_artifacts(wandb_path: str) -> tuple[MergeHistory, dict[str, Any]]:
    """Download and load WandB artifacts.

    Args:
        wandb_path: WandB run path (e.g., entity/project/run_id)

    Returns:
        Tuple of (MergeHistory, run_config_dict)
    """
    api: wandb.Api = wandb.Api()
    run: Run = api.run(wandb_path)
    logger.info(f"Loaded WandB run: {run.name} ({run.id})")

    # Download merge history artifact
    logger.info("Downloading merge history artifact...")
    artifacts: list[Any] = [a for a in run.logged_artifacts() if a.type == "merge_history"]
    if not artifacts:
        raise ValueError(f"No merge_history artifacts found in run {wandb_path}")
    artifact: Any = artifacts[0]
    logger.info(f"Found artifact: {artifact.name}")

    artifact_dir: str = artifact.download()
    merge_history_path: Path = Path(artifact_dir) / "merge_history.zip"
    merge_history: MergeHistory = MergeHistory.read(merge_history_path)
    logger.info(f"Loaded merge history: {merge_history}")

    return merge_history, run.config


def setup_model_and_data(
    run_config: dict[str, Any],
    context_length: int,
    batch_size: int,
) -> tuple[ComponentModel, PreTrainedTokenizer, DataLoader[Any], Config]:
    """Set up model, tokenizer, and dataloader.

    Args:
        run_config: WandB run config dictionary
        context_length: Context length for tokenization
        batch_size: Batch size for data loading

    Returns:
        Tuple of (model, tokenizer, dataloader, spd_config)
    """
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_path: str = run_config["model_path"]
    logger.info(f"Loading model from: {model_path}")
    spd_run: SPDRunInfo = SPDRunInfo.from_path(model_path)
    model: ComponentModel = ComponentModel.from_run_info(spd_run)
    model.to(device)
    model.eval()
    config: Config = spd_run.config
    tokenizer_name: str = config.tokenizer_name  # pyright: ignore[reportAssignmentType]
    logger.info(f"{tokenizer_name = }")

    # Load tokenizer
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    logger.info(f"Loaded: {tokenizer = }")

    # Create dataloader
    # TODO: read this from batches_config.json
    dataset_config: DatasetConfig = DatasetConfig(
        name="SimpleStories/SimpleStories",
        hf_tokenizer_path=tokenizer_name,
        split="train",
        n_ctx=context_length,
        is_tokenized=False,  # Text dataset
        streaming=False,
        column_name="story",
    )
    logger.info(f"Using {dataset_config = }")

    dataloader: DataLoader[Any]
    dataloader, _ = create_data_loader(
        dataset_config=dataset_config,
        batch_size=batch_size,
        buffer_size=4,
        ddp_rank=0,
        ddp_world_size=1,
    )
    logger.info(f"Created {dataloader = }")

    return model, tokenizer, dataloader, config


def compute_max_activations(
    model: ComponentModel,
    sigmoid_type: SigmoidTypes,
    tokenizer: PreTrainedTokenizer,
    dataloader: DataLoader[Any],
    merge_history: MergeHistory,
    iteration: int,
    n_samples: int,
    n_batches: int,
) -> dict[int, dict[str, list[dict[str, Any]]]]:
    device: torch.device = get_module_device(model)

    # Get unique clusters and component info using MergeHistory methods
    unique_clusters: list[int] = merge_history.get_unique_clusters(iteration)
    cluster_components: dict[int, list[dict[str, Any]]] = {
        cid: merge_history.get_cluster_components_info(iteration, cid)
        for cid in unique_clusters
    }

    # Initialize tracker
    tracker: ClusterMaxTracker = ClusterMaxTracker(unique_clusters, n_samples, device)
    dataset_idx_counter: int = 0

    for batch_idx, batch_data in enumerate(tqdm(dataloader, total=n_batches)):
        if batch_idx >= n_batches:
            break

        batch: Int[Tensor, "batch_size n_ctx"] = extract_batch_data(batch_data).to(device)
        batch_size: int
        seq_len: int
        batch_size, seq_len = batch.shape

        # Get activations
        activations: dict[str, Float[Tensor, "n_steps C"]] = component_activations(
            model,
            device,
            batch=batch,
            sigmoid_type=sigmoid_type,
        )
        processed: ProcessedActivations = process_activations(activations, seq_mode="concat")

        for cluster_id in unique_clusters:
            # Compute cluster activations
            acts_2d: Float[Tensor, "batch_size seq_len"] = _compute_cluster_activations(
                processed, cluster_components[cluster_id], batch_size, seq_len
            )

            if acts_2d.abs().max() == 0:
                continue

            # Find top activations across all positions
            flat_acts: Float[Tensor, " batch_size*seq_len"] = acts_2d.flatten()
            k: int = min(n_samples, len(flat_acts))
            top_vals: Float[Tensor, " k"]
            top_idx: Int[Tensor, " k"]
            top_vals, top_idx = torch.topk(flat_acts, k)

            # Create TextSamples for batch insertion
            text_samples: list[TextSample] = []
            for val, idx in zip(top_vals, top_idx, strict=False):
                batch_idx_i: int = int(idx // seq_len)
                pos_idx: int = int(idx % seq_len)
                current_dataset_idx: int = dataset_idx_counter + batch_idx_i

                text_sample: TextSample = _create_text_sample(
                    batch=batch,
                    batch_idx=batch_idx_i,
                    pos_idx=pos_idx,
                    sequence_acts=acts_2d[batch_idx_i],
                    val=float(val),
                    dataset_index=current_dataset_idx,
                    tokenizer=tokenizer,
                )
                text_samples.append(text_sample)

            # Batch insert into tracker
            tracker.try_insert_batch(cluster_id, top_vals, text_samples)

        dataset_idx_counter += batch_size

    return tracker.to_result_dict(cluster_components)


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


def main(
    wandb_run: str,
    output_dir: Path | None,
    iteration: int,
    n_samples: int,
    n_batches: int,
    batch_size: int,
    context_length: int,
) -> None:
    """Compute max-activating text samples for language model component clusters.

    Args:
        wandb_run: WandB clustering run path (e.g., entity/project/run_id)
        output_dir: Output directory (default: REPO_ROOT/spd/clustering/dashboard/data/{run_id})
        iteration: Merge iteration to analyze (negative indexes from end)
        n_samples: Number of top-activating samples to collect per cluster
        n_batches: Number of data batches to process
        batch_size: Batch size for data loading
        context_length: Context length for tokenization
    """
    # Parse wandb run path
    wandb_path: str = wandb_run.removeprefix("wandb:")
    logger.info(f"Loading WandB run: {wandb_path}")

    # Load artifacts from WandB
    merge_history: MergeHistory
    run_config: dict[str, Any]
    merge_history, run_config = load_wandb_artifacts(wandb_path)

    # Extract run_id for output directory
    api: wandb.Api = wandb.Api()
    run: Run = api.run(wandb_path)
    run_id: str = run.id

    # Set up output directory
    final_output_dir: Path = output_dir or (
        REPO_ROOT / "spd" / "clustering" / "dashboard" / "data" / run_id
    )
    final_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {final_output_dir}")

    # Setup model and data
    model: ComponentModel
    tokenizer: PreTrainedTokenizer
    dataloader: DataLoader[Any]
    config: Config
    model, tokenizer, dataloader, config = setup_model_and_data(
        run_config, context_length, batch_size
    )

    # Compute max activations
    logger.info("computing max activations")
    result: dict[int, dict[str, list[dict[str, Any]]]] = compute_max_activations(
        model=model,
        sigmoid_type=config.sigmoid_type,
        tokenizer=tokenizer,
        dataloader=dataloader,
        merge_history=merge_history,
        iteration=iteration,
        n_samples=n_samples,
        n_batches=n_batches,
    )
    logger.info(f"computed max activations: {len(result) = }")

    # Get iteration for model info
    actual_iteration: int = iteration if iteration >= 0 else merge_history.n_iters_current + iteration
    merge: GroupMerge = merge_history.merges[actual_iteration]

    # Generate model information
    logger.info("Generating model information")
    model_info: dict[str, Any] = generate_model_info(
        model=model,
        merge_history=merge_history,
        merge=merge,
        iteration=actual_iteration,
        model_path=run_config["model_path"],
        tokenizer_name=config.tokenizer_name,  # pyright: ignore[reportArgumentType]
        config_dict=config.model_dump(mode="json"),
        wandb_run_path=run_config["model_path"],
    )

    # Save results
    max_act_filename: str = f"max_activations_iter{iteration}_n{n_samples}.json"
    max_act_path: Path = final_output_dir / max_act_filename
    max_act_path.write_text(json.dumps(result, indent=2))
    logger.info(f"Max activations saved to: {max_act_path}")

    model_info_path: Path = final_output_dir / "model_info.json"
    model_info_path.write_text(json.dumps(model_info, indent=2))
    logger.info(f"Model info saved to: {model_info_path}")


def cli() -> None:
    """CLI entry point with argument parsing."""
    logger.info("parsing args")
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Compute max-activating text samples for language model component clusters."
    )
    parser.add_argument(
        "--wandb-run",
        "-w",
        type=str,
        help="WandB clustering run path (e.g., entity/project/run_id or wandb:entity/project/run_id)",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="Output directory (default: REPO_ROOT/spd/clustering/dashboard/data/{run_id})",
        default=None,
    )
    parser.add_argument(
        "--iteration",
        "-i",
        type=int,
        default=-1,
        help="Merge iteration to analyze (negative indexes from end, default: -1 for latest)",
    )
    parser.add_argument(
        "--n-samples",
        "-n",
        type=int,
        default=16,
        help="Number of top-activating samples to collect per cluster",
    )
    parser.add_argument(
        "--n-batches",
        "-s",
        type=int,
        default=4,
        help="Number of data batches to process",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=64,
        help="Batch size for data loading",
    )
    parser.add_argument(
        "--context-length",
        "-c",
        type=int,
        default=64,
        help="Context length for tokenization (default: 64)",
    )
    args: argparse.Namespace = parser.parse_args()

    main(
        wandb_run=args.wandb_run,
        output_dir=args.output_dir,
        iteration=args.iteration,
        n_samples=args.n_samples,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        context_length=args.context_length,
    )


if __name__ == "__main__":
    cli()
