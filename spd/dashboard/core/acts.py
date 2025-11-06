"""Minimal single-script version of causal importance decision tree training."""

import itertools
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Final

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer

from spd.configs import Config
from spd.dashboard.core.dashboard_config import ComponentDashboardConfig
from spd.dashboard.core.toks import TokenSequenceData
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, OutputWithCache, SPDRunInfo

_COMPONENT_LABEL_SEP: Final[str] = ":"


@dataclass(frozen=True)
class ComponentLabel:
    module: str
    index: int

    def as_str(self) -> str:
        return f"{self.module}{_COMPONENT_LABEL_SEP}{self.index}"

    def serialize(self) -> str:
        """Serialize to string format for JSON compatibility."""
        return self.as_str()

    def __hash__(self) -> int:
        return hash((self.module, self.index))

    @classmethod
    def from_str(cls, label_str: str) -> "ComponentLabel":
        """Deserialize from string format."""
        module, index_str = label_str.split(_COMPONENT_LABEL_SEP)
        return cls(module=module, index=int(index_str))


@dataclass(kw_only=True)
class Activations:
    """Container for layer-wise activations with ordered access."""

    module_order: list[str]
    token_data: TokenSequenceData
    data: dict[str, Float[np.ndarray, "n_sequences n_ctx n_components"]]
    varying_component_indices: dict[str, list[int]]

    @property
    def n_components_total(self) -> int:
        """Total number of varying components across all layers."""
        return sum(acts.shape[2] for acts in self.data.values())

    @cached_property
    def component_labels(self) -> dict[str, list[ComponentLabel]]:
        """Map module name to list of ComponentLabels for its varying components."""
        return {
            module: [
                ComponentLabel(module=module, index=comp_idx)
                for comp_idx in self.varying_component_indices[module]
            ]
            for module in self.module_order
        }

    @cached_property
    def component_labels_concat(self) -> list[ComponentLabel]:
        """Concatenated list of all ComponentLabels in layer order."""
        return list(
            itertools.chain.from_iterable(
                self.component_labels[module] for module in self.module_order
            )
        )

    @cached_property
    def data_batch_concat(self) -> dict[str, Float[np.ndarray, "n_samples n_components"]]:
        """Flattened version of data: (n_sequences * n_ctx, n_components)."""
        return {k: v.reshape(-1, v.shape[-1]) for k, v in self.data.items()}

    def serialize(self) -> dict[str, Any]:
        """Serialize activations for ZANJ storage.

        Returns dict with module-grouped data for efficient lazy loading.
        """
        return {
            "layer_order": self.module_order,
            "token_data": self.token_data.serialize(),
            # dict[module_name, ndarray] - ZANJ will externalize
            # TODO: this is horribly broken. we serialize without externalizing if we call json_serialize which blows up the size, but if we dont call it then some arrays get serialized as strings? like literally str(array) and then we cant read it :(
            # "data": json_serialize(self.data),
            "data": self.data,
            "varying_component_indices": self.varying_component_indices,
            "n_components_total": self.n_components_total,
            "component_labels": {
                module: [label.as_str() for label in labels]
                for module, labels in self.component_labels.items()
            },
            "component_labels_concat": [label.as_str() for label in self.component_labels_concat],
        }

    @classmethod
    def generate(
        cls,
        config: ComponentDashboardConfig,
    ) -> "Activations":
        # get model
        print(f"Loading model from {config.model_path}...")
        spd_run: SPDRunInfo = SPDRunInfo.from_path(path=config.model_path)
        model: ComponentModel = ComponentModel.from_pretrained(path=spd_run.checkpoint_path)
        model.to(device=config.device)
        spd_cfg: Config = spd_run.config

        # get dataloader and tokenizer
        assert isinstance(spd_cfg.task_config, LMTaskConfig)
        assert spd_cfg.pretrained_model_name is not None

        dataset_config: DatasetConfig = DatasetConfig(
            name=config.dataset_name,
            hf_tokenizer_path=spd_cfg.pretrained_model_name,
            split=config.dataset_split,
            n_ctx=config.context_length,
            column_name=config.dataset_column,
            is_tokenized=False,
            streaming=config.dataset_streaming,
            seed=0,
        )
        dataloader, _ = create_data_loader(
            dataset_config=dataset_config,
            batch_size=config.batch_size,
            buffer_size=spd_cfg.task_config.buffer_size,
            global_seed=spd_cfg.seed,
            ddp_rank=0,
            ddp_world_size=1,
        )

        # compute acts and collect tokens
        print(f"\nComputing activations for {config.n_batches} batches...")
        all_acts: list[dict[str, Float[Tensor, "batch n_ctx C"]]] = []
        all_tokens: list[Int[Tensor, "batch n_ctx"]] = []

        for _ in tqdm(range(config.n_batches), desc="Batches"):
            batch: Tensor = next(iter(dataloader))["input_ids"]
            all_tokens.append(batch.cpu())
            with torch.no_grad():
                output: OutputWithCache = model(batch.to(config.device), cache_type="input")
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
        layers: dict[str, Float[np.ndarray, "n_batches n_ctx n_components"]] = {}
        varying_component_indices: dict[str, list[int]] = {}

        for module_name, acts_raw in acts_concat.items():
            # Keep as 3D: (n_batches, n_ctx, n_components)
            acts_3d: Float[np.ndarray, "n_batches n_ctx n_components"] = acts_raw.numpy()

            # Flatten temporarily for filtering: (n_batches * n_ctx, n_components)
            acts_flat: Float[np.ndarray, "n_samples n_components"] = acts_3d.reshape(
                -1, acts_3d.shape[-1]
            )

            # Threshold to boolean for filtering only
            acts_bool: Bool[np.ndarray, "n_samples n_components"] = (
                acts_flat >= config.activation_threshold
            ).astype(bool)

            # Filter constant components (always 0 or always 1)
            varying_mask: Bool[np.ndarray, " n_components"] = acts_bool.var(axis=0) > 0
            acts_varying_3d = acts_3d[:, :, varying_mask]  # Keep 3D structure, filter components
            layers[module_name] = acts_varying_3d

            # Store which original component indices were kept
            varying_component_indices[module_name] = np.where(varying_mask)[0].tolist()

            print(
                f"  {module_name}: {acts_varying_3d.shape[0]} batches, "
                f"{acts_varying_3d.shape[1]} ctx, {acts_varying_3d.shape[2]} varying components"
            )

        return Activations(
            data=layers,
            module_order=module_keys,
            varying_component_indices=varying_component_indices,
            token_data=token_data,
        )
