"""Minimal single-script version of causal importance decision tree training."""

from dataclasses import dataclass
import json
from typing import Any

import numpy as np
import torch
from jaxtyping import Bool, Float
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from torch import Tensor
from tqdm import tqdm

from spd.configs import Config
from spd.dashboard.core.matshow_sort import sort_by_similarity
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, OutputWithCache, SPDRunInfo

@dataclass
class LayerActivations:
    """Container for layer-wise activations with ordered access."""
    data: dict[str, Bool[np.ndarray, "n_samples n_components"]]
    layer_order: list[str]
    varying_component_indices: dict[str, list[int]]

    def get_concat_before(self, module_name: str) -> Bool[np.ndarray, "n_samples n_features"]:
        """Get concatenated activations of all layers before the specified module."""
        idx = self.layer_order.index(module_name)
        if idx == 0:
            # No previous layers, return empty array with correct number of samples
            n_samples = list(self.data.values())[0].shape[0]
            return np.zeros((n_samples, 0), dtype=bool)
        prev_layers = [self.data[self.layer_order[i]] for i in range(idx)]
        return np.concatenate(prev_layers, axis=1)

    def __len__(self) -> int:
        return len(self.layer_order)

    def __iter__(self):
        return iter(self.layer_order)

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
        spd_run: SPDRunInfo = SPDRunInfo.from_path(wandb_run_path)
        model: ComponentModel = ComponentModel.from_pretrained(spd_run.checkpoint_path)
        model.to(device)
        spd_cfg: Config = spd_run.config

        # get dataloader
        assert isinstance(spd_cfg.task_config, LMTaskConfig)
        assert spd_cfg.pretrained_model_name is not None

        dataset_config = DatasetConfig(
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

        # compute acts
        print(f"\nComputing activations for {n_batches} batches...")
        all_acts: list[dict[str, Tensor]] = []

        for _ in tqdm(range(n_batches), desc="Batches"):
            batch: Tensor = next(iter(dataloader))["input_ids"]
            with torch.no_grad():
                output: OutputWithCache = model(batch.to(device), cache_type="input")
                acts: dict[str, Tensor] = model.calc_causal_importances(
                    pre_weight_acts=output.cache,
                    sampling="continuous",
                    detach_inputs=False,
                ).upper_leaky # TODO
            all_acts.append({k: v.cpu() for k, v in acts.items()})

        # Concatenate batches
        module_keys = list(all_acts[0].keys())
        acts_concat: dict[str, Tensor] = {
            k: torch.cat([b[k] for b in all_acts], dim=0) for k in module_keys
        }

        # Convert to boolean and filter constant components
        print("\nConverting to boolean and filtering constant components...")
        layers: dict[str, Bool[np.ndarray, "n_samples n_components"]] = {}
        varying_component_indices: dict[str, list[int]] = {}

        for module_name, acts_tensor in acts_concat.items():
            # Flatten if 3D (batch, seq, components) -> (batch*seq, components)
            if acts_tensor.ndim == 3:
                acts_np: Float[np.ndarray, "n_samples n_components"] = acts_tensor.reshape(
                    -1, acts_tensor.shape[-1]
                ).numpy()
            else:
                acts_np = acts_tensor.numpy()

            # Threshold to boolean
            acts_bool: Bool[np.ndarray, "n_samples n_components"] = (
                acts_np >= activation_threshold
            ).astype(bool)

            # plt.title(f"{module_name}")
            # sort by column similarity
            acts_sorted = sort_by_similarity(sort_by_similarity(acts_bool.astype(float), axis=0), axis=1)
            plt.matshow(acts_sorted[:, :600], aspect="auto")
            plt.show()

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
        )