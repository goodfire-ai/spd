"""Data structures for tracking max-activating text samples."""

import hashlib
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal, NewType

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from spd.clustering.activations import (
    ProcessedActivations,
    component_activations,
    process_activations,
)
from spd.clustering.merge_history import MergeHistory
from spd.models.component_model import ComponentModel
from spd.models.sigmoids import SigmoidTypes
from spd.utils.general_utils import extract_batch_data, get_module_device

# Type aliases
TextHash = NewType("TextHash", str)
ActivationHash = NewType("ActivationHash", str)
ClusterHash = NewType("ClusterHash", str)
CriterionHash = NewType("CriterionHash", str)
Direction = Literal["max", "min"]

_SEPARATOR_1: str = "-"
_SEPARATOR_2: str = ":"


@dataclass(frozen=True, slots=True, kw_only=True)
class ClusterLabel:
    """Label for a cluster component."""

    module_name: str
    original_index: int

    def __str__(self) -> str:
        """Format as 'module_name:original_index'."""
        return f"{self.module_name}:{self.original_index}"

    def as_tuple(self) -> tuple[str, int]:
        """Return as a tuple (module_name, original_index)."""
        return (self.module_name, self.original_index)

    def string(self) -> str:
        """Return as a string 'module_name-original_index'."""
        return f"{self.module_name}{_SEPARATOR_1}{self.original_index}"

    @staticmethod
    def from_string(s: str) -> "ClusterLabel":
        """Create ClusterLabel from its string representation."""
        parts: list[str] = s.split(_SEPARATOR_1)
        if len(parts) != 2:
            raise ValueError(f"Invalid ClusterLabel string: {s}")
        return ClusterLabel(module_name=parts[0], original_index=int(parts[1]))


@dataclass(frozen=True, slots=True, kw_only=True)
class ClusterId:
    """Unique identifier for a cluster. This should uniquely identify a cluster *globally*"""

    spd_run: str  # SPD run identifier
    clustering_run: str  # Clustering run identifier
    iteration: int  # Merge iteration number
    cluster_label: ClusterLabel  # Component label

    def to_list(self) -> list[Any]:
        """Return as a list of identifying components."""
        return [
            self.spd_run,
            self.clustering_run,
            self.iteration,
            self.cluster_label.module_name,
            self.cluster_label.original_index,
        ]

    def to_string(self) -> ClusterHash:
        """Hash uniquely identifying this cluster."""
        # Use all identifying information to create unique hash
        transformed_lst: list[str] = [str(part) for part in self.to_list()]
        assert all(_SEPARATOR_1 not in part for part in transformed_lst), (
            "Parts cannot contain separator"
        )
        assert all(_SEPARATOR_2 not in part for part in transformed_lst), (
            "Parts cannot contain separator"
        )

        return ClusterHash(_SEPARATOR_1.join(transformed_lst))

    @classmethod
    def from_string(cls, s: ClusterHash) -> "ClusterId":
        """Create ClusterId from its string representation."""
        parts: list[str] = s.split(_SEPARATOR_1)
        if len(parts) != 5:
            raise ValueError(f"Invalid ClusterId string: {s}")
        return cls(
            spd_run=parts[0],
            clustering_run=parts[1],
            iteration=int(parts[2]),
            cluster_label=ClusterLabel(module_name=parts[3], original_index=int(parts[4])),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TextSample:
    """Text content, with a reference to the dataset. depends on tokenizer used."""

    full_text: str
    tokens: list[str]

    @cached_property
    def text_hash(self) -> TextHash:
        """Hash of full_text for deduplication."""
        return TextHash(hashlib.sha256(self.full_text.encode()).hexdigest()[:8])

    def length(self) -> int:
        """Return the number of tokens."""
        return len(self.tokens)


@dataclass(frozen=True, slots=True, kw_only=True)
class ActivationSample:
    """Activations for a text sample in a specific cluster."""

    cluster_id: ClusterId
    text_hash: TextHash  # Reference to TextSample
    activations: np.ndarray  # Shape: (seq_len,)

    @cached_property
    def activation_hash(self) -> ActivationHash:
        """Hash uniquely identifying this activation sample (cluster_hash + text_hash)."""
        # Combine cluster_hash with text_hash to create unique identifier
        return ActivationHash(f"{self.cluster_id.to_string()}:{self.text_hash}")

    @property
    def mean_activation(self) -> float:
        """Compute mean activation across all tokens."""
        return float(np.mean(self.activations))

    @property
    def min_activation(self) -> float:
        """Compute minimum activation across all tokens."""
        return float(np.min(self.activations))

    @property
    def median_activation(self) -> float:
        """Compute median activation across all tokens."""
        return float(np.median(self.activations))

    @property
    def max_activation(self) -> float:
        """Compute maximum activation across all tokens."""
        return float(np.max(self.activations))

    @property
    def max_position(self) -> int:
        """Find position of maximum activation."""
        return int(np.argmax(self.activations))

    def length(self) -> int:
        """Return the number of activation values (sequence length)."""
        return len(self.activations)


@dataclass(frozen=True, slots=True, kw_only=True)
class TrackingCriterion:
    """Defines what statistics to track."""

    property_name: str
    "max_activation, mean_activation, etc. - must be a property on ActivationSample"

    direction: Direction

    n_samples: int

    def to_string(self) -> CriterionHash:
        """Convert to hash string representation."""
        parts: list[str] = [self.property_name, self.direction, str(self.n_samples)]
        assert all(_SEPARATOR_2 not in part for part in parts), "Parts cannot contain separator"
        return CriterionHash(_SEPARATOR_2.join(parts))

    @classmethod
    def from_string(cls, s: CriterionHash) -> "TrackingCriterion":
        """Create TrackingCriterion from its string representation."""
        parts: list[str] = s.split(_SEPARATOR_2)
        if len(parts) != 3:
            raise ValueError(f"Invalid TrackingCriterion string: {s}")
        direction = parts[1]
        if direction not in ("max", "min"):
            raise ValueError(f"Invalid direction: {direction}")
        return cls(
            property_name=parts[0],
            direction=direction,  # pyright: ignore[reportArgumentType]
            n_samples=int(parts[2]),
        )


@dataclass
class ClusterTopSamplesTracker:
    """Tracks top-k text samples per cluster across multiple criteria.

    This can work with multiple clusters and be parallelized.
    Stores only references (text hashes) and can be joined with text/activation tables later.

    Table:
    - top_samples: {ClusterHash: {CriterionHash: [(TextHash, stat_value)]}}
    """

    # Table: ClusterHash -> {CriterionHash -> [(TextHash, stat_value)]}
    # Store tuples of (text_hash, stat_value) for efficient comparison during insertion
    top_samples: dict[ClusterHash, dict[CriterionHash, list[tuple[TextHash, float]]]]

    # Metadata
    criteria: list[TrackingCriterion]

    @classmethod
    def create_empty(
        cls, cluster_hashes: list[ClusterHash], criteria: list[TrackingCriterion]
    ) -> "ClusterTopSamplesTracker":
        """Create an empty tracker for multiple clusters."""
        top_samples: dict[ClusterHash, dict[CriterionHash, list[tuple[TextHash, float]]]] = {}
        for cluster_hash in cluster_hashes:
            top_samples[cluster_hash] = {}
            for criterion in criteria:
                crit_hash = criterion.to_string()
                top_samples[cluster_hash][crit_hash] = []

        return cls(
            top_samples=top_samples,
            criteria=criteria,
        )

    def try_insert(
        self,
        cluster_hash: ClusterHash,
        text_hash: TextHash,
        stat_values: dict[CriterionHash, float],
    ) -> dict[CriterionHash, bool]:
        """Try to insert a text sample for all criteria in a specific cluster.

        Args:
            cluster_hash: Hash of the cluster to insert into
            text_hash: Hash of the text sample
            stat_values: Dict mapping CriterionHash to its computed statistic value

        Returns:
            Dict mapping CriterionHash to True if inserted, False otherwise
        """
        inserted: dict[CriterionHash, bool] = {}

        for criterion in self.criteria:
            crit_hash = criterion.to_string()
            stat_value: float = stat_values[crit_hash]

            top_list: list[tuple[TextHash, float]] = self.top_samples[cluster_hash][crit_hash]

            # Check if text_hash already in list
            if any(th == text_hash for th, _ in top_list):
                inserted[crit_hash] = False
                continue

            # Find insertion point based on direction
            insert_pos: int | None = None

            for j, (_, existing_value) in enumerate(top_list):
                if (criterion.direction == "max" and stat_value > existing_value) or (
                    criterion.direction == "min" and stat_value < existing_value
                ):
                    insert_pos = j
                    break

            # Insert if we found a position or list not full
            if insert_pos is not None:
                top_list.insert(insert_pos, (text_hash, stat_value))
                # Keep only top n_samples
                if len(top_list) > criterion.n_samples:
                    top_list.pop()
                inserted[crit_hash] = True
            elif len(top_list) < criterion.n_samples:
                top_list.append((text_hash, stat_value))
                inserted[crit_hash] = True
            else:
                inserted[crit_hash] = False

        return inserted

    def get_cluster_top_samples(self) -> dict[ClusterHash, dict[CriterionHash, list[TextHash]]]:
        """Get top text hashes for all clusters and criteria (without stat values)."""
        return {
            cluster_hash: {
                crit_hash: [th for th, _ in self.top_samples[cluster_hash][crit_hash]]
                for crit_hash in self.top_samples[cluster_hash]
            }
            for cluster_hash in self.top_samples
        }


@dataclass
class ClusterActivationDatabase:
    """Database for storing text samples and activations per cluster.

    Database tables:
    - text_samples: {TextHash: TextSample}
    - cluster_top_samples: {ClusterHash: {CriterionHash: [TextHash]}}
    - activation_samples: {(TextHash, ClusterHash): ActivationSample}
    """

    # Table: TextHash -> TextSample
    text_samples: dict[TextHash, TextSample]

    # Table: ClusterHash -> {CriterionHash -> [TextHash]}
    cluster_top_samples: dict[ClusterHash, dict[CriterionHash, list[TextHash]]]

    # Table: (TextHash, ClusterHash) -> ActivationSample
    activation_samples: dict[tuple[TextHash, ClusterHash], ActivationSample]

    @classmethod
    def from_tracker(
        cls,
        tracker: ClusterTopSamplesTracker,
        text_samples: dict[TextHash, TextSample],
        activation_samples: dict[tuple[TextHash, ClusterHash], ActivationSample],
    ) -> "ClusterActivationDatabase":
        """Create database from a tracker and sample tables.

        Args:
            tracker: ClusterTopSamplesTracker with all clusters
            text_samples: Complete text samples table
            activation_samples: Complete activation samples table

        Returns:
            Complete ClusterActivationDatabase
        """
        cluster_top_samples = tracker.get_cluster_top_samples()

        return cls(
            text_samples=text_samples,
            cluster_top_samples=cluster_top_samples,
            activation_samples=activation_samples,
        )

    @classmethod
    def generate(
        cls,
        model: ComponentModel,
        sigmoid_type: SigmoidTypes,
        tokenizer: PreTrainedTokenizer,
        dataloader: DataLoader[Any],
        merge_history: MergeHistory,
        iteration: int,
        criteria: list[TrackingCriterion],
        spd_run: str,
        clustering_run: str,
        n_batches: int,
    ) -> "ClusterActivationDatabase":
        """Generate database by computing activations from model and data.

        Args:
            model: ComponentModel to get activations from
            sigmoid_type: Sigmoid type for activation computation
            tokenizer: Tokenizer for decoding text
            dataloader: DataLoader providing batches
            merge_history: MergeHistory containing cluster information
            iteration: Merge iteration to analyze
            criteria: List of tracking criteria
            spd_run: SPD run identifier
            clustering_run: Clustering run identifier
            n_batches: Number of batches to process

        Returns:
            Populated ClusterActivationDatabase
        """
        device: torch.device = get_module_device(model)

        # Get unique cluster indices and component info
        unique_cluster_indices: list[int] = merge_history.get_unique_clusters(iteration)
        cluster_components: dict[int, list[dict[str, Any]]] = {
            cid: merge_history.get_cluster_components_info(iteration, cid)
            for cid in unique_cluster_indices
        }

        # Create ClusterId objects for each cluster
        cluster_ids: list[ClusterId] = []
        for idx in unique_cluster_indices:
            # Get cluster label from first component
            components = cluster_components[idx]
            if components:
                first_label = components[0]["label"]
                module_name, original_index_str = first_label.rsplit(":", 1)
                cluster_label = ClusterLabel(
                    module_name=module_name, original_index=int(original_index_str)
                )
            else:
                cluster_label = ClusterLabel(module_name="unknown", original_index=idx)

            cluster_ids.append(
                ClusterId(
                    spd_run=spd_run,
                    clustering_run=clustering_run,
                    iteration=iteration,
                    cluster_label=cluster_label,
                )
            )

        # Create mapping from cluster_index to ClusterId
        cluster_id_map: dict[int, ClusterId] = {
            unique_cluster_indices[i]: cluster_ids[i] for i in range(len(cluster_ids))
        }

        # Initialize empty database
        db = cls.create_empty(cluster_ids=cluster_ids, criteria=criteria)

        # Process batches
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

            for cluster_idx in unique_cluster_indices:
                # Compute cluster activations
                acts_2d: Float[Tensor, "batch_size seq_len"] = _compute_cluster_activations(
                    processed, cluster_components[cluster_idx], batch_size, seq_len
                )

                if acts_2d.abs().max() == 0:
                    continue

                # Find top activations across all positions
                flat_acts: Float[Tensor, " batch_size*seq_len"] = acts_2d.flatten()
                # Get top k per criterion (max of n_samples across all criteria)
                k: int = min(max(c.n_samples for c in criteria), len(flat_acts))
                top_idx: Int[Tensor, " k"]
                _, top_idx = torch.topk(flat_acts, k)

                # Get ClusterId for this cluster
                cluster_id: ClusterId = cluster_id_map[cluster_idx]

                # Create and insert samples
                for idx in top_idx:
                    batch_idx_i: int = int(idx // seq_len)

                    text_sample, act_sample = _create_samples(
                        cluster_id=cluster_id,
                        batch=batch,
                        batch_idx=batch_idx_i,
                        sequence_acts=acts_2d[batch_idx_i],
                        tokenizer=tokenizer,
                    )

                    # Try to insert into database
                    db._try_insert_activation(act_sample, text_sample)

        return db


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


def _create_samples(
    cluster_id: ClusterId,
    batch: Int[Tensor, "batch_size n_ctx"],
    batch_idx: int,
    sequence_acts: Float[Tensor, " seq_len"],
    tokenizer: PreTrainedTokenizer,
) -> tuple[TextSample, ActivationSample]:
    """Create TextSample and ActivationSample from batch data.

    Args:
        cluster_id: ClusterId this sample belongs to
        batch: Input token batch
        batch_idx: Index within batch
        sequence_acts: Activations for entire sequence
        tokenizer: Tokenizer for decoding

    Returns:
        Tuple of (TextSample, ActivationSample)
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

    # Create TextSample
    text_sample = TextSample(
        full_text=text,
        tokens=token_strings,
    )

    # Create ActivationSample (with numpy array)
    activations_np: np.ndarray = sequence_acts.cpu().numpy()
    activation_sample = ActivationSample(
        cluster_id=cluster_id,
        text_hash=text_sample.text_hash,
        activations=activations_np,
    )

    return text_sample, activation_sample
