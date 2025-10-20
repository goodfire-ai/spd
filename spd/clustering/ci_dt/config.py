"""Configuration for causal importance decision tree training."""

from dataclasses import dataclass


@dataclass
class CIDTConfig:
    """Configuration for causal importance decision tree training."""

    n_samples: int = 250
    activation_threshold: float = 0.01  # Threshold for boolean conversion
    filter_dead_threshold: float = 0.001  # Threshold for filtering dead components
    max_depth: int = 8  # Maximum depth for decision trees
    random_state: int = 7  # Random state for reproducibility
