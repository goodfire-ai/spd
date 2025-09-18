"""Model comparison script for geometric similarity analysis.

This script compares two SPD models by computing geometric similarities between
their learned subcomponents. It's designed for post-hoc analysis of completed runs.

Usage:
    python spd/scripts/compare_models.py --config spd/scripts/compare_models_config.yaml
"""

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import extract_batch_data, load_config
from spd.utils.run_utils import save_file


class ModelComparator:
    """Compare two SPD models for geometric similarity between subcomponents."""

    def __init__(
        self,
        current_model_path: str,
        reference_model_path: str,
        density_threshold: float = 0.0,
        device: str = "auto",
        comparison_config: dict[str, Any] | None = None,
    ):
        """Initialize the model comparator.

        Args:
            current_model_path: Path to current model (wandb: or local path)
            reference_model_path: Path to reference model (wandb: or local path)
            density_threshold: Minimum activation density for components to be included
            device: Device to run comparison on ("auto", "cuda", "cpu")
            comparison_config: Full comparison configuration dict
        """
        self.current_model_path = current_model_path
        self.reference_model_path = reference_model_path
        self.density_threshold = density_threshold
        self.comparison_config = comparison_config or {}

        if device == "auto":
            self.device = get_device()
        else:
            self.device = device

        logger.info(f"Loading current model from: {current_model_path}")
        self.current_model, self.current_config = self._load_model_and_config(current_model_path)

        logger.info(f"Loading reference model from: {reference_model_path}")
        self.reference_model, self.reference_config = self._load_model_and_config(
            reference_model_path
        )

    def _load_model_and_config(self, model_path: str) -> tuple[ComponentModel, dict[str, Any]]:
        """Load model and config using the standard pattern from existing codebase."""
        run_info = SPDRunInfo.from_path(model_path)
        model = ComponentModel.from_run_info(run_info)
        model.to(self.device)
        model.eval()
        model.requires_grad_(False)

        config_dict = run_info.config.model_dump()

        return model, config_dict

    def create_eval_data_loader(self, config: dict[str, Any]) -> Iterator[Any]:
        """Create evaluation data loader using exact same patterns as decomposition scripts."""
        task_config = config.get("task_config", {})
        task_name = task_config.get("task_name")

        if not task_name:
            raise ValueError("task_config.task_name must be set")

        if task_name == "tms":
            from spd.experiments.tms.models import TMSTargetRunInfo
            from spd.utils.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset

            if "pretrained_model_path" not in config or not config["pretrained_model_path"]:
                raise ValueError("pretrained_model_path must be set for TMS models")

            target_run_info = TMSTargetRunInfo.from_path(config["pretrained_model_path"])
            n_features = target_run_info.config.tms_model_config.n_features
            synced_inputs = target_run_info.config.synced_inputs

            dataset = SparseFeatureDataset(
                n_features=n_features,
                feature_probability=task_config["feature_probability"],
                device=self.device,
                data_generation_type=task_config["data_generation_type"],
                value_range=(0.0, 1.0),
                synced_inputs=synced_inputs,
            )
            return iter(
                DatasetGeneratedDataLoader(
                    dataset,
                    batch_size=self.comparison_config.get(
                        "eval_batch_size", 1
                    ),  # TODO get rid of 'get' pattern
                    shuffle=self.comparison_config.get("shuffle_data", False),
                )
            )

        elif task_name == "resid_mlp":
            from spd.experiments.resid_mlp.models import ResidMLPTargetRunInfo
            from spd.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
            from spd.utils.data_utils import DatasetGeneratedDataLoader

            if "pretrained_model_path" not in config or not config["pretrained_model_path"]:
                raise ValueError("pretrained_model_path must be set for ResidMLP models")

            target_run_info = ResidMLPTargetRunInfo.from_path(config["pretrained_model_path"])
            n_features = target_run_info.config.resid_mlp_model_config.n_features
            synced_inputs = target_run_info.config.synced_inputs

            dataset = ResidMLPDataset(
                n_features=n_features,
                feature_probability=task_config["feature_probability"],
                device=self.device,
                calc_labels=False,
                label_type=None,
                act_fn_name=None,
                label_fn_seed=None,
                synced_inputs=synced_inputs,
            )
            return iter(
                DatasetGeneratedDataLoader(
                    dataset,
                    batch_size=self.comparison_config.get("eval_batch_size", 1),
                    shuffle=self.comparison_config.get("shuffle_data", False),
                )
            )

        elif task_name == "lm":
            from spd.data import DatasetConfig, create_data_loader

            if "tokenizer_name" not in config or not config["tokenizer_name"]:
                raise ValueError("tokenizer_name must be set for language models")

            dataset_config = DatasetConfig(
                name=task_config["dataset_name"],
                hf_tokenizer_path=config["tokenizer_name"],
                split=task_config["eval_data_split"],
                n_ctx=task_config["max_seq_len"],
                is_tokenized=task_config["is_tokenized"],
                streaming=task_config["streaming"],
                column_name=task_config["column_name"],
                shuffle_each_epoch=task_config["shuffle_each_epoch"],
                seed=None,
            )
            loader, _ = create_data_loader(
                dataset_config=dataset_config,
                batch_size=self.comparison_config.get("eval_batch_size", 1),
                buffer_size=task_config["buffer_size"],
                global_seed=config["seed"] + 1,
                ddp_rank=0,
                ddp_world_size=1,
            )
            return iter(loader)

        elif task_name == "ih":
            from spd.experiments.ih.model import InductionModelTargetRunInfo
            from spd.utils.data_utils import DatasetGeneratedDataLoader, InductionDataset

            if "pretrained_model_path" not in config or not config["pretrained_model_path"]:
                raise ValueError("pretrained_model_path must be set for Induction Heads models")

            target_run_info = InductionModelTargetRunInfo.from_path(config["pretrained_model_path"])
            vocab_size = target_run_info.config.ih_model_config.vocab_size
            seq_len = target_run_info.config.ih_model_config.seq_len
            prefix_window = task_config.get("prefix_window") or seq_len - 3

            dataset = InductionDataset(
                vocab_size=vocab_size,
                seq_len=seq_len,
                prefix_window=prefix_window,
                device=self.device,
            )
            return iter(
                DatasetGeneratedDataLoader(
                    dataset,
                    batch_size=self.comparison_config.get("eval_batch_size", 1),
                    shuffle=self.comparison_config.get("shuffle_data", False),
                )
            )

        raise ValueError(
            f"Unsupported task type: {task_name}. Supported types: tms, lm, resid_mlp, ih"
        )

    def compute_activation_densities(
        self, model: ComponentModel, eval_iterator: Iterator[Any], n_steps: int = 5
    ) -> dict[str, Float[Tensor, " C"]]:
        """Compute activation densities using same logic as ComponentActivationDensity."""
        # Get config for this model
        config_dict = self.current_config if model is self.current_model else self.reference_config
        ci_alive_threshold = self.comparison_config.get("ci_alive_threshold", 0.0)

        device = next(iter(model.parameters())).device
        n_tokens = 0
        component_activation_counts: dict[str, Float[Tensor, " C"]] = {
            module_name: torch.zeros(model.C, device=device) for module_name in model.components
        }

        model.eval()
        with torch.no_grad():
            for _step in range(n_steps):
                batch = extract_batch_data(next(eval_iterator))
                batch = batch.to(self.device)
                _, pre_weight_acts = model(
                    batch, mode="pre_forward_cache", module_names=list(model.components.keys())
                )
                ci, _ci_upper_leaky = model.calc_causal_importances(
                    pre_weight_acts,
                    sigmoid_type=config_dict["sigmoid_type"],
                    sampling=config_dict["sampling"],
                )

                n_tokens_batch = next(iter(ci.values())).shape[:-1].numel()
                n_tokens += n_tokens_batch

                for module_name, ci_vals in ci.items():
                    active_components = ci_vals > ci_alive_threshold
                    n_activations_per_component = einops.reduce(
                        active_components, "... C -> C", "sum"
                    )
                    component_activation_counts[module_name] += n_activations_per_component

        densities = {
            module_name: component_activation_counts[module_name] / n_tokens
            for module_name in model.components
        }

        return densities

    def compute_geometric_similarities(
        self, activation_densities: dict[str, Float[Tensor, " C"]]
    ) -> dict[str, float]:
        """Compute geometric similarities between subcomponents."""
        similarities = {}

        for layer_name in self.current_model.components:
            if layer_name not in self.reference_model.components:
                logger.warning(f"Layer {layer_name} not found in reference model, skipping")
                continue

            current_components = self.current_model.components[layer_name]
            reference_components = self.reference_model.components[layer_name]

            if current_components.C != reference_components.C:
                logger.warning(
                    f"Component count mismatch for {layer_name}: {current_components.C} vs {reference_components.C}"
                )
                continue

            # Extract U and V matrices
            C = current_components.C
            current_U = current_components.U  # Shape: [C, d_out]
            current_V = current_components.V  # Shape: [d_in, C]
            ref_U = reference_components.U
            ref_V = reference_components.V

            # Filter out components that aren't active enough in the current model
            C_alive = sum(activation_densities[layer_name] > self.density_threshold)
            if C_alive == 0:
                logger.warning(
                    f"No components are active enough in {layer_name} for density threshold {self.density_threshold}. Skipping."
                )
                continue

            current_U_alive = current_U[activation_densities[layer_name] > self.density_threshold]
            current_V_alive = current_V[
                :, activation_densities[layer_name] > self.density_threshold
            ]

            # Compute rank-one matrices: V @ U for each component
            current_rank_one = einops.einsum(
                current_V_alive,
                current_U_alive,
                "d_in C_alive, C_alive d_out -> C_alive d_in d_out",
            )
            ref_rank_one = einops.einsum(ref_V, ref_U, "d_in C, C d_out -> C d_in d_out")

            # Compute cosine similarities between all pairs
            current_flat = current_rank_one.reshape(int(C_alive.item()), -1)
            ref_flat = ref_rank_one.reshape(C, -1)

            current_norm = F.normalize(current_flat, p=2, dim=1)
            ref_norm = F.normalize(ref_flat, p=2, dim=1)

            cosine_sim_matrix = einops.einsum(
                current_norm, ref_norm, "C_alive d_in_d_out, C_ref d_in_d_out -> C_alive C_ref"
            )
            cosine_sim_matrix = cosine_sim_matrix.abs()

            max_similarities = cosine_sim_matrix.max(dim=1).values
            similarities[f"mean_max_abs_cosine_sim/{layer_name}"] = max_similarities.mean().item()
            similarities[f"max_abs_cosine_sim_std/{layer_name}"] = max_similarities.std().item()
            similarities[f"max_abs_cosine_sim_min/{layer_name}"] = max_similarities.min().item()
            similarities[f"max_abs_cosine_sim_max/{layer_name}"] = max_similarities.max().item()

        metric_names = [
            "mean_max_abs_cosine_sim",
            "max_abs_cosine_sim_std",
            "max_abs_cosine_sim_min",
            "max_abs_cosine_sim_max",
        ]

        for metric_name in metric_names:
            values = [
                similarities[f"{metric_name}/{layer_name}"]
                for layer_name in self.current_model.components
                if f"{metric_name}/{layer_name}" in similarities
            ]
            if values:
                similarities[f"{metric_name}/all_layers"] = sum(values) / len(values)

        return similarities

    def run_comparison(
        self, eval_iterator: Iterator[Any], n_steps: int | None = None
    ) -> dict[str, float]:
        """Run the full comparison pipeline."""
        if n_steps is None:
            n_steps = self.comparison_config.get("n_eval_steps", 5)
        assert isinstance(n_steps, int)  # Ensure n_steps is an int for type checking

        logger.info("Computing activation densities for current model...")
        activation_densities = self.compute_activation_densities(
            self.current_model, eval_iterator, n_steps
        )

        logger.info("Computing geometric similarities...")
        similarities = self.compute_geometric_similarities(activation_densities)

        return similarities


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Compare two SPD models for geometric similarity")
    parser.add_argument(
        "--config",
        type=str,
        default="spd/scripts/compare_models_config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config, dict)
    current_model_path = config["current_model_path"]
    reference_model_path = config["reference_model_path"]
    density_threshold = config.get("density_threshold", 0.0)
    device = config.get("device", "auto")
    output_dir = Path(config.get("output_dir", "./comparison_results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    comparator = ModelComparator(
        current_model_path=current_model_path,
        reference_model_path=reference_model_path,
        density_threshold=density_threshold,
        device=device,
        comparison_config=config,
    )

    logger.info("Setting up evaluation data...")
    eval_iterator = comparator.create_eval_data_loader(comparator.current_config)

    logger.info("Starting model comparison...")
    similarities = comparator.run_comparison(eval_iterator)

    results_file = output_dir / "similarity_results.json"
    save_file(similarities, results_file)

    logger.info(f"Comparison complete! Results saved to {results_file}")
    logger.info("Similarity metrics:")
    for key, value in similarities.items():
        logger.info(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
