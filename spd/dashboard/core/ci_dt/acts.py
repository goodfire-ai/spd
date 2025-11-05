"""Minimal single-script version of causal importance decision tree training."""

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from numpy import ndarray
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
import json

from spd.configs import Config
from spd.dashboard.core.tokenization import attach_vocab_arr, simple_batch_decode
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, OutputWithCache, SPDRunInfo



@dataclass(frozen=True)
class TextSample:
    """A sequence of tokens with its location in the flattened activation array."""

    tokens: list[str]
    dataset_idx: tuple[int, int]  # (start, end) slice into flattened n_samples array

    def __hash__(self) -> int:
        """Hash based on token tuple for Python dict/set compatibility."""
        return hash(self.sequence_hash)

    @cached_property
    def sequence_hash(self) -> str:
        """SHA256 hash of tokens for stable string keys."""
        return hashlib.sha256(
            json.dumps(self.tokens).encode()
        ).digest().hex()


@dataclass
class TokenSequenceData:
    """Efficient storage for token sequences using flat array with range indexing."""

    all_tokens: ndarray  # [total_tokens] flat string array with dtype U{max_token_length}
    sequence_ranges: dict[str, tuple[int, int]]  # sequence_hash -> (start, end) into all_tokens
    dataset_indices: dict[str, tuple[int, int]]  # sequence_hash -> (start, end) into n_samples

    def get_sequence_tokens(self, sequence_hash: str) -> TextSample:
        """Get TextSample for a specific sequence by its hash.

        Args:
            sequence_hash: SHA256 hash of the sequence

        Returns:
            TextSample with tokens and dataset_idx
        """
        start, end = self.sequence_ranges[sequence_hash]
        tokens: list[str] = self.all_tokens[start:end].tolist()
        dataset_idx: tuple[int, int] = self.dataset_indices[sequence_hash]
        return TextSample(tokens=tokens, dataset_idx=dataset_idx)

    def get_all_sequences(self) -> list[TextSample]:
        """Get all sequences as TextSample objects.

        Returns:
            List of all TextSample objects
        """
        return [self.get_sequence_tokens(h) for h in self.sequence_ranges]

    @classmethod
    def from_token_batches(
        cls,
        token_batches: list[Tensor],
        tokenizer: PreTrainedTokenizer,
    ) -> "TokenSequenceData":
        """Generate TokenSequenceData from batches of token IDs.

        Args:
            token_batches: List of token ID tensors [batch_size, n_ctx]
            tokenizer: Tokenizer to decode token IDs

        Returns:
            TokenSequenceData with flat array storage
        """
        # Ensure tokenizer has vocab array for fast batch decoding
        if not hasattr(tokenizer, "vocab_arr"):
            attach_vocab_arr(tokenizer)

        # Concatenate batches: [n_sequences, n_ctx]
        tokens_concat: Tensor = torch.cat(token_batches, dim=0)
        n_sequences: int = tokens_concat.shape[0]
        n_ctx: int = tokens_concat.shape[1]

        # Decode all tokens using efficient batch decode
        tokens_decoded: ndarray = simple_batch_decode(
            tokenizer, tokens_concat.numpy()
        )  # [n_sequences, n_ctx]

        # Flatten to single array: [total_tokens]
        all_tokens_flat: ndarray = tokens_decoded.reshape(-1)

        # Build sequence mappings
        sequence_ranges: dict[str, tuple[int, int]] = {}
        dataset_indices: dict[str, tuple[int, int]] = {}

        for seq_idx in range(n_sequences):
            # Extract tokens for this sequence
            seq_tokens: list[str] = tokens_decoded[seq_idx].tolist()

            # Compute stable hash
            seq_hash: str = hashlib.sha256(str(tuple(seq_tokens)).encode()).hexdigest()

            # Token range in all_tokens array
            token_start: int = seq_idx * n_ctx
            token_end: int = (seq_idx + 1) * n_ctx
            sequence_ranges[seq_hash] = (token_start, token_end)

            # Dataset index range in flattened n_samples array
            dataset_indices[seq_hash] = (token_start, token_end)

        return cls(
            all_tokens=all_tokens_flat,
            sequence_ranges=sequence_ranges,
            dataset_indices=dataset_indices,
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
