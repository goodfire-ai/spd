from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch
from jaxtyping import Bool, Float, Int, Shaped
from numpy import ndarray
from torch import Tensor

from spd.dashboard.core.acts import Activations, ComponentLabel
from spd.dashboard.core.toks import TokenSequenceData


@dataclass
class FlatActivations:
    component_labels: list[ComponentLabel]
    activations: Float[np.ndarray, "n_samples C"]
    token_data: TokenSequenceData
    layer_order: list[str]

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
            [torch.from_numpy(activations.data_batch_concat[k]) for k in activations.layer_order],
            dim=1,
        ).float()

        component_labels: list[ComponentLabel] = activations.component_labels_concat

        assert flattened.shape[1] == len(component_labels)

        return FlatActivations(
            component_labels=component_labels,
            activations=flattened.numpy(),
            token_data=activations.token_data,
            layer_order=activations.layer_order,
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
                f"No previous layers to concatenate before the first layer: {module_name=},{self.layer_order=}"
            )

        return self.activations[:, :idx]

    def get_concat_this_module(self, module_name: str) -> Float[ndarray, "n_samples n_features"]:
        """Get concatenated activations of the specified module."""
        start_idx: int = self.start_of_module_index(module_name)
        end_idx: int = self.start_of_module_index(
            self.layer_order[self.layer_order.index(module_name) + 1]
        )

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


def conditional_matrices(
    tsd: TokenSequenceData,
    activations: Bool[np.ndarray, "n_samples C"],
    laplace: float = 0.0,
) -> tuple[Float[np.ndarray, "d_vocab C"], Float[np.ndarray, "d_vocab C"]]:
    """compute P(activation | token) and P(token | activation) as (d_vocab, C) arrays

    # Returns:
     - `tuple[Float[np.ndarray, "d_vocab C"], Float[np.ndarray, "d_vocab C"]]`
        (p_activation_given_token, p_token_given_activation)
    """
    n_samples: int = activations.shape[0]
    C_components: int = activations.shape[1]
    d_vocab: int = tsd.vocab_arr.shape[0]

    tokens_flat: Int[np.ndarray, " n_samples"] = tsd.tokens.reshape(-1)
    assert tokens_flat.shape[0] == n_samples, (
        f"Tokens and activations must have same length: "
        f"{tokens_flat.shape[0]} != {activations.shape[0]}"
    )

    # TODO: optimize -- we are going back to token idxs from strings here
    token_idx: Int[np.ndarray, " n_samples"] = np.array(
        [tsd.token_vocab_idx[t] for t in tokens_flat], dtype=np.int64
    )

    assert np.all((token_idx >= 0) & (token_idx < d_vocab)), (
        f"Token indices out of bounds: min={token_idx.min()}, max={token_idx.max()}, d_vocab={d_vocab}"
    )

    activated_per_token: Float[np.ndarray, "d_vocab C"] = np.zeros(
        (d_vocab, C_components), dtype=float
    )
    np.add.at(activated_per_token, token_idx, activations)

    denom_act_given: Float[np.ndarray, "d_vocab 1"] = np.bincount(token_idx, minlength=d_vocab)[
        :, None
    ]
    num_act_given: Float[np.ndarray, "d_vocab C"] = activated_per_token.copy()
    if laplace:
        num_act_given = num_act_given + laplace
        denom_act_given = denom_act_given + 2.0 * laplace

    with np.errstate(divide="ignore", invalid="ignore"):
        p_activation_given_token: Float[np.ndarray, "d_vocab C"] = np.divide(
            num_act_given,
            denom_act_given,
            out=np.zeros_like(num_act_given, dtype=float),
            where=denom_act_given > 0,
        )

    total_acts: Float[np.ndarray, " C"] = activations.sum(axis=0)
    safe_den: Float[np.ndarray, "1 C"] = np.where(total_acts > 0, total_acts, 1.0)[None, :]
    p_token_given_activation: Float[np.ndarray, "d_vocab C"] = activated_per_token / safe_den
    zero_cols: Bool[np.ndarray, " C"] = total_acts == 0
    if np.any(zero_cols):
        p_token_given_activation[:, zero_cols] = 0.0

    return p_activation_given_token, p_token_given_activation
