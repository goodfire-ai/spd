from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch
from jaxtyping import Bool, Float, Int, Shaped
from numpy import ndarray
from sklearn.manifold import Isomap
from torch import Tensor

from spd.dashboard.core.acts import Activations, ComponentLabel
from spd.dashboard.core.toks import TokenSequenceData


@dataclass(kw_only=True)
class FlatActivations:
    module_order: list[str]
    token_data: TokenSequenceData
    component_labels: list[ComponentLabel]
    activations: Float[np.ndarray, "n_samples C"]

    @cached_property
    def tokens_flat(self) -> Shaped[np.ndarray, " n_samples"]:
        """Flattened tokens aligned with activations."""
        return self.token_data.tokens.reshape(-1)

    @property
    def n_components(self) -> int:
        """Number of components."""
        return len(self.component_labels)

    @cached_property
    def component_label_inverse(self) -> dict[ComponentLabel, int]:
        """Map from component label string to its index in activations."""
        return {label: idx for idx, label in enumerate(self.component_labels)}

    @cached_property
    def coactivations(self) -> Float[np.ndarray, "n_components n_components"]:
        """Binary coactivation matrix (how often components are active together)."""
        binary: Float[np.ndarray, "n_samples n_components"] = (self.activations > 0).astype(
            np.float32
        )
        return binary.T @ binary

    def get_component_activations(
        self,
        label: ComponentLabel,
    ) -> Float[np.ndarray, " n_samples"]:
        """Get activations for a specific component by label."""
        idx: int = self.component_label_inverse[label]
        return self.activations[:, idx]

    @classmethod
    def create(
        cls,
        activations: Activations,
    ) -> "FlatActivations":
        flattened: Float[Tensor, "n_samples n_components_total"] = torch.cat(
            [torch.from_numpy(activations.data_batch_concat[k]) for k in activations.module_order],
            dim=1,
        ).float()

        component_labels: list[ComponentLabel] = activations.component_labels_concat

        assert flattened.shape[1] == len(component_labels)

        return FlatActivations(
            component_labels=component_labels,
            activations=flattened.numpy(),
            token_data=activations.token_data,
            module_order=activations.module_order,
        )

    def start_of_module_index(self, module_name: str) -> int:
        """Get the starting index of the specified module in the flattened activations."""
        for idx, c in enumerate(self.component_labels):
            if c.module == module_name:
                return idx

        raise ValueError(f"Module {module_name} not found in component labels.")

    def get_concat_before_module(self, module_name: str) -> Float[ndarray, "n_samples n_features"]:
        """Get concatenated activations of all layers before the specified module."""
        idx: int = self.start_of_module_index(module_name)
        if idx == 0:
            raise ValueError(
                f"No previous layers to concatenate before the first layer: {module_name=},{self.module_order=}"
            )

        return self.activations[:, :idx]

    def get_concat_this_module(self, module_name: str) -> Float[ndarray, "n_samples n_features"]:
        """Get concatenated activations of the specified module."""
        start_idx: int = self.start_of_module_index(module_name)
        end_idx: int
        try:
            next_module: int = self.module_order.index(module_name) + 1
            end_idx = self.start_of_module_index(self.module_order[next_module])
        except (IndexError, ValueError):
            end_idx = self.n_components

        # check all component labels are correct
        for idx, c_label in enumerate(self.component_labels):
            if start_idx <= idx < end_idx:
                assert c_label.module == module_name, (
                    f"Component label mismatch at index {idx}: "
                    f"expected module {module_name}, got {c_label.module}"
                )
            else:
                assert c_label.module != module_name, (
                    f"Component label mismatch at index {idx}: "
                    f"expected **NOT** module {module_name}, got {c_label.module}"
                )

        return self.activations[:, start_idx:end_idx]


def _compute_activated_per_token(
    flat_acts: FlatActivations,
    activation_threshold: float,
) -> tuple[
    Int[np.ndarray, " n_samples"],  # token ids
    Float[np.ndarray, "d_vocab C"],  # number of activations of this component per token
]:
    """Compute co-occurrence matrix of tokens and activations.

    Returns:
        token_idx: Token indices for each sample
        activated_per_token: Count matrix of activations per token (d_vocab, C)
    """
    activations: Bool[np.ndarray, " n_samples C"] = flat_acts.activations > activation_threshold
    n_samples: int = activations.shape[0]
    C_components: int = activations.shape[1]
    d_vocab: int = flat_acts.token_data.vocab_arr.shape[0]

    token_idx: Int[np.ndarray, " n_samples"] = flat_acts.token_data.token_ids.reshape(-1)
    assert token_idx.shape[0] == n_samples, (
        f"Tokens and activations must have same length: "
        f"{token_idx.shape[0]} != {activations.shape[0]}"
    )

    assert np.all((token_idx >= 0) & (token_idx < d_vocab)), (
        f"Token indices out of bounds: min={token_idx.min()}, max={token_idx.max()}, d_vocab={d_vocab}"
    )

    activated_per_token: Float[np.ndarray, "d_vocab C"] = np.zeros(
        (d_vocab, C_components), dtype=float
    )
    np.add.at(activated_per_token, token_idx, activations)

    return token_idx, activated_per_token


def _compute_P_active_given_token(
    activated_per_token: Float[np.ndarray, "d_vocab C"],
    token_idx: Int[np.ndarray, " n_samples"],
    d_vocab: int,
) -> Float[np.ndarray, "d_vocab C"]:
    """Compute P(activation | token)."""
    denom_act_given: Float[np.ndarray, "d_vocab 1"] = np.bincount(token_idx, minlength=d_vocab)[
        :, None
    ]
    return activated_per_token / denom_act_given


def _compute_P_token_given_active(
    activated_per_token: Float[np.ndarray, "d_vocab C"],
    activations: Bool[np.ndarray, "n_samples C"],
) -> Float[np.ndarray, "d_vocab C"]:
    """Compute P(token | activation)."""
    total_acts: Float[np.ndarray, " C"] = activations.sum(axis=0)
    return activated_per_token / total_acts[None, :]


@dataclass
class ComponentEmbeddings:
    """Component embeddings for visualization."""

    embeddings: Float[np.ndarray, "n_components embed_dim"]
    component_labels: list[ComponentLabel]

    @property
    def n_components(self) -> int:
        return len(self.component_labels)

    @classmethod
    def create(
        cls,
        flat_acts: FlatActivations,
        embed_dim: int = 3,
    ) -> "ComponentEmbeddings":
        """Compute component embeddings from activations using Isomap.

        Computes embeddings based on correlation distance.

        Args:
            flat_acts: Flattened activations data
            embed_dim: Target dimensionality for embeddings (typically 2 for visualization)

        Returns:
            ComponentEmbeddings with embeddings and component labels
        """
        # Compute Pearson correlations
        correlations: Float[np.ndarray, "n_components n_components"] = np.corrcoef(
            flat_acts.activations.T
        )

        # Convert correlation to distance
        distance_matrix: Float[np.ndarray, "n_components n_components"] = np.abs(correlations)
        max_distance: float = float(distance_matrix.max())
        distance: Float[np.ndarray, "n_components n_components"] = max_distance - distance_matrix
        distance = (distance + distance.T) / 2.0

        isomap: Isomap = Isomap(n_components=embed_dim, metric="precomputed")
        embeddings: Float[np.ndarray, "n_components embed_dim"] = isomap.fit_transform(distance)

        return cls(
            embeddings=embeddings,
            component_labels=flat_acts.component_labels,
        )
