import warnings
from pathlib import Path
from typing import Any, Literal

import matplotlib.cm as cm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from spd.configs import Config, MaskScope, PGDReconLossConfig
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.metrics.pgd_masked_recon_loss import PGDReconLoss
from spd.models.component_model import ComponentModel, OutputWithCache, SPDRunInfo
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import extract_batch_data, replace_pydantic_model, set_seed

ModelType = Literal["lm1", "lm2", "lm3", "lm4"]


class ModelSetup:
    """Container for loaded model and config."""

    def __init__(
        self,
        model: ComponentModel,
        config: Any,
        device: Any,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device


def load_model(model_type: ModelType, run_info_path: str) -> ModelSetup:
    """Load model. Call once per model type."""
    device = get_device()

    # Load run info and validate config
    run_info = SPDRunInfo.from_path(run_info_path)
    config = run_info.config
    task_config = config.task_config

    assert isinstance(task_config, LMTaskConfig), "task_config not LMTaskConfig"

    # Load model
    model = ComponentModel.from_run_info(run_info)
    model.to(device)
    model.target_model.requires_grad_(False)

    return ModelSetup(model, config, device)


def run_experiment(
    model_type: ModelType,
    model_setup: ModelSetup,
    seed: int,
    mask_scope: MaskScope,
    n_batches: int,
    batch_sizes: list[int],
    step_size: float,
    n_steps: int,
    max_seq_len: int,
) -> dict[str, float]:
    """Run PGD reconstruction loss experiment for a single seed using pre-loaded model."""
    model = model_setup.model
    config: Config = model_setup.config
    device = model_setup.device

    task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig), "task_config not LMTaskConfig"

    # Run experiments for each batch size
    results: dict[str, float] = {}

    for batch_size in batch_sizes:
        set_seed(seed)

        # Update task config with desired max_seq_len
        updated_task_config = replace_pydantic_model(
            task_config, {"max_seq_len": max_seq_len, "train_data_split": "train"}
        )

        train_data_config = DatasetConfig(
            name=updated_task_config.dataset_name,
            hf_tokenizer_path=config.tokenizer_name,
            split=updated_task_config.train_data_split,
            n_ctx=updated_task_config.max_seq_len,
            is_tokenized=updated_task_config.is_tokenized,
            streaming=updated_task_config.streaming,
            column_name=updated_task_config.column_name,
            shuffle_each_epoch=updated_task_config.shuffle_each_epoch,
            seed=None,
        )

        set_seed(0)
        data_loader, _tokenizer = create_data_loader(
            dataset_config=train_data_config,
            batch_size=batch_size,
            buffer_size=updated_task_config.buffer_size,
            global_seed=config.seed,
            ddp_rank=0,
            ddp_world_size=1,
        )

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
            f"{model_type}, {mask_scope}, seed={seed}, n_batches={n_batches}, batch_size={batch_size}, seq_len={max_seq_len}, loss={loss}"
        )

    return results


def plot_results(
    results: dict[ModelType, dict[str, dict[str, float]]],
    output_path: Path,
    log_scale: bool = True,
) -> None:
    """Plot results with subplots for each model."""
    n_models = len(results)
    if n_models == 0:
        print("No results to plot")
        return

    # Determine grid size
    n_cols = 2
    n_rows = (n_models + 1) // 2

    _, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    model_names = list(results.keys())
    colormap = cm.get_cmap("tab10")
    colors = [colormap(i / 6.0) for i in range(6)]  # 6 seeds

    for idx, model_name in enumerate(model_names):
        if idx >= len(axes):
            break

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
            ax.set_xscale("log")

    # Hide unused subplots
    for idx in range(len(model_names), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def main() -> None:
    """Run experiments for all models and seeds, generate report."""
    # Model configurations - update these with your LM run paths
    model_configs: dict[ModelType, str] = {
        "lm1": "wandb:goodfire/spd/runs/lxs77xye",  # Example: update with actual LM runs
        # "lm2": "wandb:goodfire/spd/runs/...",
        # "lm3": "wandb:goodfire/spd/runs/...",
        # "lm4": "wandb:goodfire/spd/runs/...",
    }

    mask_scope: MaskScope = "shared_across_batch"
    n_batches = 1
    step_size = 0.001
    n_steps = 100
    seeds = list(range(6))  # 0-5
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    max_seq_len = 8  # Adjust as needed

    # Collect results
    results: dict[ModelType, dict[str, dict[str, float]]] = {}

    for model_type, run_info_path in model_configs.items():
        print(f"\n{'=' * 60}")
        print(f"Loading model for {model_type.upper()}")
        print(f"{'=' * 60}\n")

        # Load model once per model type
        model_setup = load_model(model_type, run_info_path)

        # Initialize results dict for this model
        results[model_type] = {}

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
                max_seq_len=max_seq_len,
            )
            results[model_type][f"seed_{seed}"] = seed_results

    # Generate plot
    output_path = Path(f"pgd_results_lm_{mask_scope}_{n_steps}_{step_size}.png")
    plot_results(results, output_path, log_scale=True)
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()
