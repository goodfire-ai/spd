"""Configuration for causal importance decision tree training."""

from dataclasses import dataclass


@dataclass
class CIDTConfig:
    """Configuration for causal importance decision tree training."""

    batch_size: int = 10  # Number of samples per batch for GPU inference
    n_batches: int = 25  # Number of batches to process (total samples = batch_size * n_batches)
    activation_threshold: float = 0.01  # Threshold for boolean conversion
    max_depth: int = 8  # Maximum depth for decision trees
    random_state: int = 7  # Random state for reproducibility
