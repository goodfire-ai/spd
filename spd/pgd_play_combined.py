import warnings
from pathlib import Path
from typing import Any, Literal

import matplotlib.cm as cm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from spd.configs import MaskScope, PGDReconLossConfig
from spd.experiments.resid_mlp.configs import ResidMLPTaskConfig
from spd.experiments.resid_mlp.models import ResidMLP, ResidMLPTargetRunInfo
from spd.experiments.resid_mlp.resid_mlp_dataset import ResidMLPDataset
from spd.experiments.tms.configs import TMSTaskConfig
from spd.experiments.tms.models import TMSModel, TMSTargetRunInfo
from spd.metrics.pgd_masked_recon_loss import PGDReconLoss
from spd.models.component_model import ComponentModel, OutputWithCache, SPDRunInfo
from spd.utils.data_utils import DatasetGeneratedDataLoader, SparseFeatureDataset
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import extract_batch_data, set_seed

ModelType = Literal["tms_40_10", "tms_40_10_id", "resid_mlp1", "resid_mlp2"]


class ModelSetup:
    """Container for loaded model and dataset."""

    def __init__(
        self,
        model: ComponentModel,
        dataset: SparseFeatureDataset | ResidMLPDataset,
        config: Any,
        device: Any,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device


def load_model_and_dataset(model_type: ModelType, run_info_path: str) -> ModelSetup:
    """Load model and create dataset. Call once per model type."""
    device = get_device()

    # Determine base model type
    is_tms = model_type.startswith("tms")

    # Load run info and validate config
    run_info = SPDRunInfo.from_path(run_info_path)
    config = run_info.config
    task_config = config.task_config

    if is_tms:
        assert isinstance(task_config, TMSTaskConfig), "task_config not TMSTaskConfig"
    else:
        assert isinstance(task_config, ResidMLPTaskConfig), "task_config not ResidMLPTaskConfig"

    # Load model
    model = ComponentModel.from_run_info(run_info)
    model.to(device)
    target_model = model.target_model

    if is_tms:
        assert isinstance(target_model, TMSModel), "target_model not TMSModel"
    else:
        assert isinstance(target_model, ResidMLP), "target_model not ResidMLP"

    target_model.requires_grad_(False)

    # Load target run info and create dataset
    assert config.pretrained_model_path, "pretrained_model_path must be set"
    set_seed(0)

    if is_tms:
        target_run_info = TMSTargetRunInfo.from_path(config.pretrained_model_path)
        synced_inputs = target_run_info.config.synced_inputs
        dataset = SparseFeatureDataset(
            n_features=target_model.config.n_features,
            feature_probability=task_config.feature_probability,
            device=device,
            data_generation_type=task_config.data_generation_type,
            value_range=(0.0, 1.0),
            synced_inputs=synced_inputs,
        )
    else:
        target_run_info = ResidMLPTargetRunInfo.from_path(config.pretrained_model_path)
        synced_inputs = target_run_info.config.synced_inputs
        dataset = ResidMLPDataset(
            n_features=target_model.config.n_features,
            feature_probability=task_config.feature_probability,
            device=device,
            calc_labels=False,
            label_type=None,
            act_fn_name=None,
            label_fn_seed=None,
            label_coeffs=None,
            data_generation_type=task_config.data_generation_type,
            synced_inputs=synced_inputs,
        )

    return ModelSetup(model, dataset, config, device)


def run_experiment(
    model_type: ModelType,
    model_setup: ModelSetup,
    seed: int,
    mask_scope: MaskScope,
    n_batches: int,
    batch_sizes: list[int],
    step_size: float,
    n_steps: int,
) -> dict[str, float]:
    """Run PGD reconstruction loss experiment for a single seed using pre-loaded model."""
    model = model_setup.model
    dataset = model_setup.dataset
    config = model_setup.config
    device = model_setup.device

    # Run experiments for each batch size
    results: dict[str, float] = {}

    for batch_size in batch_sizes:
        set_seed(seed)
        data_loader = DatasetGeneratedDataLoader(dataset, batch_size=batch_size, shuffle=False)

        pgd_config = PGDReconLossConfig(
            init="random",
            step_size=step_size,
            n_steps=n_steps,
            mask_scope=mask_scope,
        )

        pgd_recon = PGDReconLoss(
            model=model,
            device=device,
            output_loss_type=config.output_loss_type,
            pgd_config=pgd_config,
            use_delta_component=config.use_delta_component,
        )

        data_loader_iter = iter(data_loader)
        weight_deltas = model.calc_weight_deltas()

        for _ in range(n_batches):
            batch = extract_batch_data(next(data_loader_iter)).to(device)
            target_model_output: OutputWithCache = model(batch, cache_type="input")
            ci = model.calc_causal_importances(
                pre_weight_acts=target_model_output.cache,
                detach_inputs=False,
                sampling=config.sampling,
            )
            pgd_recon.update(
                batch=batch,
                target_out=target_model_output.output,
                ci=ci,
                weight_deltas=weight_deltas,
            )

        loss = pgd_recon.compute().item()
        results[f"batch_size_{batch_size}"] = loss
        print(
            f"{model_type}, {mask_scope}, seed={seed}, n_batches={n_batches}, batch_size={batch_size}, loss={loss}"
        )

    return results


def plot_results(
    results: dict[ModelType, dict[str, dict[str, float]]],
    output_path: Path,
    log_scale: bool = True,
) -> None:
    """Plot results with subplots for each model."""
    _, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    model_names = ["tms_40_10", "tms_40_10_id", "resid_mlp1", "resid_mlp2"]
    colormap = cm.get_cmap("tab10")
    colors = [colormap(i / 6.0) for i in range(6)]  # 6 seeds

    for idx, model_name in enumerate(model_names):
        if model_name not in results:
            continue

        ax = axes[idx]
        model_data = results[model_name]

        for seed in range(6):
            seed_key = f"seed_{seed}"
            if seed_key not in model_data:
                continue

            seed_data = model_data[seed_key]
            # Extract batch sizes and losses
            batch_sizes = sorted([int(k.split("_")[-1]) for k in seed_data])
            losses = [seed_data[f"batch_size_{bs}"] for bs in batch_sizes]

            ax.plot(
                batch_sizes,
                losses,
                label=f"Seed {seed}",
                color=colors[seed],
                alpha=0.7,
                marker="o",
                markersize=4,
            )

        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Loss")
        ax.set_title(model_name)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        if log_scale:
            # ax.set_yscale("log")
            ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def main() -> None:
    """Run experiments for all models and seeds, generate report."""
    # Model configurations
    model_configs: dict[ModelType, str] = {
        "tms_40_10": "wandb:goodfire/spd/runs/amyo423r",  # tms_40-10
        "tms_40_10_id": "wandb:goodfire/spd/runs/kyywp29j",  # tms_40-10-id
        "resid_mlp1": "wandb:goodfire/spd/runs/uz5swum7",  # resid_mlp1
        "resid_mlp2": "wandb:goodfire/spd/runs/grevt2h2",  # resid_mlp2
    }

    # mask_scope = "unique_per_datapoint"
    mask_scope = "shared_across_batch"
    n_batches = 1
    step_size = 0.1
    n_steps = 100
    seeds = list(range(6))  # 0-5
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    # Collect results
    results: dict[ModelType, dict[str, dict[str, float]]] = {
        "tms_40_10": {},
        "tms_40_10_id": {},
        "resid_mlp1": {},
        "resid_mlp2": {},
    }

    for model_type, run_info_path in model_configs.items():
        print(f"\n{'=' * 60}")
        print(f"Loading model for {model_type.upper()}")
        print(f"{'=' * 60}\n")

        # Load model once per model type
        model_setup = load_model_and_dataset(model_type, run_info_path)

        print(f"Running experiments for {model_type.upper()}\n")

        for seed in seeds:
            print(f"Processing seed {seed}...")
            seed_results = run_experiment(
                model_type=model_type,
                model_setup=model_setup,
                seed=seed,
                mask_scope=mask_scope,
                n_batches=n_batches,
                batch_sizes=batch_sizes,
                step_size=step_size,
                n_steps=n_steps,
            )
            results[model_type][f"seed_{seed}"] = seed_results

    # Generate plot
    output_path = Path(f"pgd_results_{mask_scope}_{n_steps}_{step_size}.png")
    plot_results(results, output_path, log_scale=True)
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()
