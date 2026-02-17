from collections.abc import Callable
from functools import cached_property
from typing import override

import torch
from jaxtyping import Float
from torch.utils.data import DataLoader

from spd.autointerp.schemas import ArchitectureInfo
from spd.configs import LMTaskConfig
from spd.data import train_loader_and_tokenizer
from spd.decomposition import Decomposition
from spd.decomposition.configs import SPDDecompositionConfig
from spd.harvest.schemas import HarvestBatch
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.topology import TransformerTopology
from spd.utils.general_utils import extract_batch_data
from spd.utils.wandb_utils import parse_wandb_run_path


class SPDDecomposition(Decomposition):
    def __init__(self, wandb_path: str, activation_threshold: float):
        self.wandb_path = wandb_path
        self._activation_threshold = activation_threshold

        self._spd_run_info = SPDRunInfo.from_path(self.wandb_path)
        self._spd_config = self._spd_run_info.config
        self._component_model = ComponentModel.from_run_info(self._spd_run_info)

    @property
    @override
    def id(self) -> str:
        _, _, run_id = parse_wandb_run_path(self.wandb_path)
        return run_id

    @property
    @override
    def tokenizer_name(self) -> str:
        tok = self._spd_run_info.config.tokenizer_name
        assert tok is not None
        return tok

    @cached_property
    def _topology(self) -> TransformerTopology:
        return TransformerTopology(self._component_model.target_model)

    @property
    @override
    def architecture_info(self) -> ArchitectureInfo:
        task_config = self._spd_run_info.config.task_config
        assert isinstance(task_config, LMTaskConfig)
        return ArchitectureInfo(
            vocab_size=self._topology.unembed_module.out_features,
            n_blocks=self._topology.n_blocks,
            c_per_layer=self._component_model.module_to_c,
            model_class=self._spd_run_info.config.pretrained_model_class,
            dataset_name=task_config.dataset_name,
            tokenizer_name=self.tokenizer_name,
            layer_descriptions={
                path: self._topology.target_to_canon(path)
                for path in self._component_model.target_module_paths
            },
        )

    @override
    def dataloader(self, batch_size: int) -> DataLoader[torch.Tensor]:
        return train_loader_and_tokenizer(self._spd_config, batch_size)[0]

    @staticmethod
    def _compute_u_norms(
        model: ComponentModel,
        device: torch.device,
    ) -> dict[str, Float[torch.Tensor, " C"]]:
        """Compute ||U[c,:]|| for each component c in each layer.

        Component activations (v_i^T @ a) have a scale invariance: scaling V by alpha and U by 1/alpha
        leaves the weight matrix unchanged but scales component activations by alpha. To make component
        activations reflect actual output contribution, we multiply by the U row norms.
        """
        return {
            layer_name: component.U.norm(dim=1).to(device)
            for layer_name, component in model.components.items()
        }

    @override
    def make_harvest_fn(
        self,
        device: torch.device,
    ) -> Callable[[torch.Tensor], HarvestBatch]:
        model = self._component_model

        # TODO(oli): this is gross having the .eval() here.
        model.eval()

        u_norms = self._compute_u_norms(model, device)

        def spd_harvest_fn(batch_item: torch.Tensor) -> HarvestBatch:
            batch = extract_batch_data(batch_item).to(device)

            out = model(batch, cache_type="input")
            probs = torch.softmax(out.output, dim=-1)

            ci_dict = model.calc_causal_importances(
                pre_weight_acts=out.cache,
                detach_inputs=True,
                sampling=self._spd_config.sampling,
            ).lower_leaky

            per_layer_acts = model.get_all_component_acts(out.cache)

            firings = {
                layer: ci_dict[layer] > self._activation_threshold for layer in model.components
            }

            activations = {
                layer: {
                    "causal_importance": ci_dict[layer],
                    "component_activation": per_layer_acts[layer] * u_norms[layer],
                }
                for layer in model.target_module_paths
            }

            return HarvestBatch(
                tokens=batch,
                firings=firings,
                activations=activations,
                output_probs=probs,
            )

        return spd_harvest_fn

    @staticmethod
    def from_config(config: SPDDecompositionConfig) -> "SPDDecomposition":
        return SPDDecomposition(config.wandb_path, config.activation_threshold)
