# spd/experiments/tms/tms_dataset.py
"""Dataset for Toy Model of Superposition (TMS) experiments."""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Literal


class TMSDataset(Dataset):
    """
    Dataset for Toy Model of Superposition (TMS) experiments.
    Compatible with the same interface as ResidMLPDataset.

    Generates data according to the TMS specification from Elhage et al. 2022
    and the SPD paper: sparse sums of one-hot features with random scaling.
    """

    def __init__(
        self,
        n_features: int = 5,
        feature_probability: float = 0.05,
        device: str = 'cpu',
        calc_labels: bool = False,
        label_type: Optional[str] = None,
        act_fn_name: Optional[str] = None,
        label_fn_seed: Optional[int] = None,
        label_coeffs: Optional[torch.Tensor] = None,
        data_generation_type: Literal['standard', 'clustering'] = 'standard',
        n_samples_per_feature: int = 200,
        n_total_samples: int = 10000,
        seed: Optional[int] = None
    ):
        """
        Initialize TMS dataset.

        Args:
            n_features: Number of input features (5 for TMS 5-2)
            feature_probability: Probability each feature is active (0.05 in paper)
            device: Device to store tensors on
            calc_labels: Whether to calculate labels (unused for TMS, kept for compatibility)
            label_type: Type of labels (unused for TMS, kept for compatibility)
            act_fn_name: Activation function name (unused for TMS, kept for compatibility)
            label_fn_seed: Seed for label function (unused for TMS, kept for compatibility)
            label_coeffs: Label coefficients (unused for TMS, kept for compatibility)
            data_generation_type: 'standard' for normal TMS data, 'clustering' for clustering analysis
            n_samples_per_feature: Number of samples per feature (for clustering type)
            n_total_samples: Total number of samples (for standard type)
            seed: Random seed for reproducibility
        """
        self.n_features = n_features
        self.feature_probability = feature_probability
        self.device = device
        self.calc_labels = calc_labels
        self.label_type = label_type
        self.act_fn_name = act_fn_name
        self.label_fn_seed = label_fn_seed
        self.label_coeffs = label_coeffs
        self.data_generation_type = data_generation_type
        self.n_samples_per_feature = n_samples_per_feature
        self.n_total_samples = n_total_samples

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Generate the dataset based on type
        if data_generation_type == 'standard':
            self.data = self._generate_standard_tms_data()
        elif data_generation_type == 'clustering':
            self.data = self._generate_clustering_tms_data()
        else:
            raise ValueError(f"Unknown data_generation_type: {data_generation_type}")

        # Generate labels if requested (though typically not used for TMS)
        if calc_labels:
            self.labels = self._generate_labels()
        else:
            self.labels = None

    def _generate_standard_tms_data(self) -> torch.Tensor:
        """
        Generate standard TMS data according to the paper specifications.

        Each feature is activated with probability `feature_probability`
        and scaled by a value sampled from [0, 1] uniform distribution.
        """
        # Generate binary mask for feature activation
        feature_mask = np.random.binomial(
            1, self.feature_probability,
            size=(self.n_total_samples, self.n_features)
        ).astype(np.float32)

        # Generate random scaling factors for active features
        feature_scales = np.random.uniform(
            0.0, 1.0,
            size=(self.n_total_samples, self.n_features)
        ).astype(np.float32)

        # Combine mask and scales to get final input
        data = feature_mask * feature_scales

        return torch.tensor(data, device=self.device)

    def _generate_clustering_tms_data(self) -> torch.Tensor:
        """
        Generate clustering-specific TMS data where each feature
        is predominantly active in separate sample groups.
        """
        total_samples = self.n_features * self.n_samples_per_feature
        data = np.zeros((total_samples, self.n_features), dtype=np.float32)

        for i in range(self.n_features):
            start_idx = i * self.n_samples_per_feature
            end_idx = (i + 1) * self.n_samples_per_feature

            # Set the target feature to be high (0.8-1.0)
            data[start_idx:end_idx, i] = np.random.uniform(
                0.8, 1.0, self.n_samples_per_feature
            )

            # Set other features to be low (0.0-0.1)
            for j in range(self.n_features):
                if i != j:
                    data[start_idx:end_idx, j] = np.random.uniform(
                        0.0, 0.1, self.n_samples_per_feature
                    )

        return torch.tensor(data, device=self.device)

    def _generate_labels(self) -> torch.Tensor:
        """
        Generate labels for the dataset (if calc_labels=True).
        For TMS, this could be the reconstruction target (same as input).
        """
        return self.data.clone()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.calc_labels:
            return self.data[idx], self.labels[idx]
        return self.data[idx]

    def get_batch(self, batch_size: int) -> torch.Tensor:
        """Get a random batch of data."""
        indices = torch.randperm(len(self.data))[:batch_size]
        return self.data[indices]

    def generate_batch(self, batch_size: int) -> torch.Tensor:
        """
        Generate a batch of data. Required by DatasetGeneratedDataLoader.

        For TMS datasets, this returns the same as get_batch since we're using
        pre-generated data rather than generating on-the-fly.
        """
        if batch_size >= len(self.data):
            return self.data

        if self.data_generation_type == 'clustering':
            # Return first batch_size samples to include samples from each group
            return self.data[:batch_size]
        else:
            # For standard TMS data, return random samples
            indices = torch.randperm(len(self.data))[:batch_size]
            return self.data[indices]

    @property
    def input_size(self) -> int:
        """Return the input size (number of features)."""
        return self.n_features

    @property
    def output_size(self) -> int:
        """Return the output size (same as input for TMS autoencoder)."""
        return self.n_features