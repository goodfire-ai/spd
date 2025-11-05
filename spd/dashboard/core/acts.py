"""Minimal single-script version of causal importance decision tree training."""

import itertools
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import numpy as np
import torch
from jaxtyping import Bool, Float, Int, Shaped
from numpy import ndarray
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer

from spd.configs import Config
from spd.dashboard.core.ci_dt.toks import TokenSequenceData
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, OutputWithCache, SPDRunInfo

ComponentLabel = NamedTuple(  # noqa: UP014
    "ComponentLabel",
    [
        ("module", str),
        ("index", int),
    ],
)


@dataclass
class FlatActivations:
    component_labels: list[ComponentLabel]
    activations: Float[np.ndarray, "n_samples n_components_total"]
    tokens: Shaped[np.ndarray, " n_samples"]  # of string type `U{max_token_length}`


@dataclass
class Activations:
    """Container for layer-wise activations with ordered access."""

    data: dict[str, Float[np.ndarray, "n_sequences n_ctx n_components"]]
    layer_order: list[str]
    varying_component_indices: dict[str, list[int]]
    token_data: TokenSequenceData

    @cached_property
    def component_labels(self) -> dict[str, list[ComponentLabel]]:
        """Map module name to list of ComponentLabels for its varying components."""
        return {
            module: [
                ComponentLabel(module=module, index=comp_idx)
                for comp_idx in self.varying_component_indices[module]
            ]
            for module in self.layer_order
        }

    @cached_property
    def data_batch_concat(self) -> dict[str, Float[np.ndarray, "n_samples n_components"]]:
        """Flattened version of data: (n_sequences * n_ctx, n_components)."""
        return {k: v.reshape(-1, v.shape[-1]) for k, v in self.data.items()}

    @cached_property
    def as_flat(
        self,
    ) -> FlatActivations:
        flattened: Float[Tensor, "n_samples n_components_total"] = torch.cat(
            [torch.from_numpy(self.data_batch_concat[k]) for k in self.layer_order],
            dim=1,
        ).float()

        component_labels: list[ComponentLabel] = list(
            itertools.chain.from_iterable(
                self.component_labels[module] for module in self.layer_order
            )
        )

        tokens_flat: Shaped[np.ndarray, " n_samples"] = self.token_data.tokens.reshape(-1)

        assert flattened.shape[1] == len(component_labels)
        assert flattened.shape[0] == tokens_flat.shape[0]

        return FlatActivations(
            component_labels=component_labels,
            activations=flattened.numpy(),
            tokens=tokens_flat,
        )

    def get_concat_before(self, module_name: str) -> Float[ndarray, "n_samples n_features"]:
        """Get concatenated activations of all layers before the specified module."""
        idx: int = self.layer_order.index(module_name)
        if idx == 0:
            # No previous layers, return empty array with correct number of samples
            n_samples: int = list(self.data_batch_concat.values())[0].shape[0]
            return np.zeros((n_samples, 0), dtype=float)
        prev_layers: list[Float[ndarray, "n_samples n_features"]] = [
            self.data_batch_concat[self.layer_order[i]] for i in range(idx)
        ]
        return np.concatenate(prev_layers, axis=1)

    @classmethod
    def generate(
        cls,
        wandb_run_path: str,
        n_batches: int,
        n_ctx: int,
        device: str,
        activation_threshold: float,
    ) -> "Activations":
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
        all_acts: list[dict[str, Float[Tensor, "batch n_ctx C"]]] = []
        all_tokens: list[Int[Tensor, "batch n_ctx"]] = []

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
                acts_flat >= activation_threshold
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
            layer_order=module_keys,
            varying_component_indices=varying_component_indices,
            token_data=token_data,
        )
