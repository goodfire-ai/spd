"""Minimal single-script version of causal importance decision tree training."""

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from numpy import ndarray
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer

from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, OutputWithCache, SPDRunInfo


@dataclass
class TokenSequenceData:
    """Token and sequence tracking data parallel to flattened activations."""

    token_strs: ndarray  # [n_samples] - Decoded token strings (object dtype)
    sequence_ids: Int[ndarray, " n_samples"]  # Which sequence (0 to n_sequences-1)
    position_ids: Int[ndarray, " n_samples"]  # Position within sequence (0 to n_ctx-1)
    token_hashes: ndarray  # [n_samples] - SHA256 hashes of tokens (object dtype, hex strings)

    # Index: hash -> list of sample indices
    _hash_index: dict[str, list[int]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Build hash index for fast lookup by token hash."""
        for idx, token_hash in enumerate(self.token_hashes):
            if token_hash not in self._hash_index:
                self._hash_index[token_hash] = []
            self._hash_index[token_hash].append(idx)

    def get_samples_for_token(self, token: str) -> Bool[ndarray, " n_samples"]:
        """Return boolean mask of samples matching the given token string."""
        return self.token_strs == token

    def get_samples_for_hash(self, token_hash: str) -> Int[ndarray, " n_matches"]:
        """Return array of sample indices matching the given token hash."""
        return np.array(self._hash_index.get(token_hash, []), dtype=int)

    def get_samples_for_position(self, position: int | range) -> Bool[ndarray, " n_samples"]:
        """Return boolean mask of samples at the given position(s) in sequence."""
        if isinstance(position, int):
            return self.position_ids == position
        # Handle range
        mask: Bool[ndarray, " n_samples"] = self.position_ids >= position.start
        if position.stop is not None:
            mask = mask & (self.position_ids < position.stop)
        return mask

    def get_sequence(self, sequence_id: int) -> dict[str, Any]:
        """Return all token data for a specific sequence.

        Returns:
            Dict with keys: 'tokens', 'positions', 'hashes'
        """
        mask: Bool[ndarray, " n_samples"] = self.sequence_ids == sequence_id
        return {
            "tokens": self.token_strs[mask],
            "positions": self.position_ids[mask],
            "hashes": self.token_hashes[mask],
        }

    @staticmethod
    def compute_token_hash(token: str) -> str:
        """Compute SHA256 hash of a token string."""
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @classmethod
    def from_token_batches(
        cls,
        token_batches: list[Tensor],
        tokenizer: Any,
    ) -> "TokenSequenceData":
        """Generate TokenSequenceData from batches of token IDs.

        Args:
            token_batches: List of token ID tensors [batch_size, n_ctx]
            tokenizer: Tokenizer to decode token IDs

        Returns:
            TokenSequenceData with flattened parallel arrays
        """
        # Concatenate batches: [n_sequences, n_ctx]
        tokens_concat: Tensor = torch.cat(token_batches, dim=0)
        n_sequences: int = tokens_concat.shape[0]
        n_ctx: int = tokens_concat.shape[1]

        # Flatten to match activation samples: [n_samples]
        tokens_flat: Int[ndarray, " n_samples"] = tokens_concat.reshape(-1).numpy()

        # Create sequence and position IDs
        sequence_ids: Int[ndarray, " n_samples"] = np.repeat(np.arange(n_sequences), n_ctx)
        position_ids: Int[ndarray, " n_samples"] = np.tile(np.arange(n_ctx), n_sequences)

        # Decode tokens and compute hashes
        token_strs: ndarray = np.array(
            [tokenizer.decode([int(token_id)]) for token_id in tokens_flat], dtype=object
        )
        token_hashes: ndarray = np.array(
            [cls.compute_token_hash(token_str) for token_str in token_strs], dtype=object
        )

        return cls(
            token_strs=token_strs,
            sequence_ids=sequence_ids,
            position_ids=position_ids,
            token_hashes=token_hashes,
        )


@dataclass
class LayerActivations:
    """Container for layer-wise activations with ordered access."""

    data: dict[str, Bool[np.ndarray, "n_samples n_components"]]
    layer_order: list[str]
    varying_component_indices: dict[str, list[int]]
    token_data: TokenSequenceData

    def get_concat_before(self, module_name: str) -> Bool[ndarray, "n_samples n_features"]:
        """Get concatenated activations of all layers before the specified module."""
        idx: int = self.layer_order.index(module_name)
        if idx == 0:
            # No previous layers, return empty array with correct number of samples
            n_samples: int = list(self.data.values())[0].shape[0]
            return np.zeros((n_samples, 0), dtype=bool)
        prev_layers: list[Bool[ndarray, "n_samples n_features"]] = [
            self.data[self.layer_order[i]] for i in range(idx)
        ]
        return np.concatenate(prev_layers, axis=1)

    def __len__(self) -> int:
        return len(self.layer_order)

    def __iter__(self) -> Iterator[str]:
        return iter(self.layer_order)

    def build_feature_map(self, module_name: str) -> list[dict[str, Any]]:
        """Build feature map for module: maps feature index -> component identity."""
        feature_map: list[dict[str, Any]] = []
        module_idx: int = self.layer_order.index(module_name)
        for prev_idx in range(module_idx):
            prev_module: str = self.layer_order[prev_idx]
            for comp_idx in self.varying_component_indices[prev_module]:
                feature_map.append(
                    {
                        "layer_idx": prev_idx,
                        "module_key": prev_module,
                        "component_idx": comp_idx,
                        "label": f"{prev_module}:{comp_idx}",
                    }
                )
        return feature_map

    @classmethod
    def generate(
        cls,
        wandb_run_path: str,
        n_batches: int,
        n_ctx: int,
        device: str,
        activation_threshold: float,
    ) -> "LayerActivations":
        # get model
        print(f"Loading model from {wandb_run_path}...")
        spd_run: SPDRunInfo = SPDRunInfo.from_path(path=wandb_run_path)
        model: ComponentModel = ComponentModel.from_pretrained(path=spd_run.checkpoint_path)
        model.to(device=device)
        spd_cfg: Config = spd_run.config

        # get dataloader and tokenizer
        assert isinstance(spd_cfg.task_config, LMTaskConfig)
        assert spd_cfg.pretrained_model_name is not None

        dataset_config: DatasetConfig = DatasetConfig(
            name=spd_cfg.task_config.dataset_name,
            hf_tokenizer_path=spd_cfg.pretrained_model_name,
            split=spd_cfg.task_config.train_data_split,
            n_ctx=n_ctx,
            column_name=spd_cfg.task_config.column_name,
            is_tokenized=False,
            streaming=False,
            seed=0,
        )
        dataloader, _ = create_data_loader(
            dataset_config=dataset_config,
            batch_size=spd_cfg.batch_size,
            buffer_size=spd_cfg.task_config.buffer_size,
            global_seed=spd_cfg.seed,
            ddp_rank=0,
            ddp_world_size=1,
        )

        # compute acts and collect tokens
        print(f"\nComputing activations for {n_batches} batches...")
        all_acts: list[dict[str, Tensor]] = []
        all_tokens: list[Tensor] = []

        for _ in tqdm(range(n_batches), desc="Batches"):
            batch: Tensor = next(iter(dataloader))["input_ids"]
            all_tokens.append(batch.cpu())
            with torch.no_grad():
                output: OutputWithCache = model(batch.to(device), cache_type="input")
                acts: dict[str, Tensor] = model.calc_causal_importances(
                    pre_weight_acts=output.cache,
                    sampling="continuous",
                    detach_inputs=False,
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                ).upper_leaky  # TODO: is this right?
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            all_acts.append({k: v.cpu() for k, v in acts.items()})

        # Generate token sequence data
        tokenizer = AutoTokenizer.from_pretrained(spd_cfg.pretrained_model_name)
        print("\nProcessing token sequences...")
        token_data: TokenSequenceData = TokenSequenceData.from_token_batches(
            token_batches=all_tokens,
            tokenizer=tokenizer,
        )

        # Concatenate batches
        module_keys: list[str] = list(all_acts[0].keys())
        acts_concat: dict[str, Tensor] = {
            k: torch.cat([b[k] for b in all_acts], dim=0) for k in module_keys
        }

        # Convert to boolean and filter constant components
        print("\nConverting to boolean and filtering constant components...")
        layers: dict[str, Bool[np.ndarray, "n_samples n_components"]] = {}
        varying_component_indices: dict[str, list[int]] = {}

        for module_name, acts_tensor in acts_concat.items():
            # Flatten if 3D (batch, seq, components) -> (batch*seq, components)
            acts_np: Float[np.ndarray, "n_samples n_components"] = acts_tensor.reshape(
                -1, acts_tensor.shape[-1]
            ).numpy()

            # Threshold to boolean
            acts_bool: Bool[np.ndarray, "n_samples n_components"] = (
                acts_np >= activation_threshold
            ).astype(bool)

            # Filter constant components (always 0 or always 1)
            varying_mask: Bool[np.ndarray, " n_components"] = acts_bool.var(axis=0) > 0
            acts_varying = acts_bool[:, varying_mask]
            layers[module_name] = acts_varying
            # Store which original component indices were kept
            varying_component_indices[module_name] = np.where(varying_mask)[0].tolist()
            print(f"  {module_name}: {acts_varying.shape[1]} varying components")

        return LayerActivations(
            data=layers,
            layer_order=module_keys,
            varying_component_indices=varying_component_indices,
            token_data=token_data,
        )
