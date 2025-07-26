"""Train a single layer MLP to memorize key-value pairs."""

from pathlib import Path
from typing import Literal

import fire
import torch
import wandb
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm

from spd.experiments.memorization.memorization_dataset import KeyValueMemorizationDataset
from spd.experiments.memorization.models import MemorizationConfig, SingleLayerMemorizationMLP
from spd.log import logger
from spd.utils.data_utils import DatasetGeneratedDataLoader
from spd.utils.general_utils import get_device, load_config, set_seed
from spd.utils.run_utils import get_output_dir, save_file
from spd.utils.wandb_utils import init_wandb


def evaluate_memorization(
    model: SingleLayerMemorizationMLP,
    keys: Float[Tensor, "n_pairs d_model"],
    values: Float[Tensor, "n_pairs d_model"],
) -> dict[str, float | Tensor]:
    """Evaluate memorization performance on all key-value pairs.

    Returns:
        Dictionary with metrics:
        - avg_mse: Average MSE across all pairs
        - worst_mse: Maximum MSE among all pairs
        - worst_idx: Index of the worst-performing pair
        - frac_memorized_1e-1: Fraction of pairs with MSE < 1e-1
        - frac_memorized_1e-2: Fraction of pairs with MSE < 1e-2
        - frac_memorized_1e-3: Fraction of pairs with MSE < 1e-3
        - frac_memorized_1e-4: Fraction of pairs with MSE < 1e-4
        - per_pair_mse: Tensor of MSE values for histogram
    """
    model.eval()
    with torch.no_grad():
        predicted_values = model(keys)
        per_pair_mse = ((predicted_values - values) ** 2).sum(dim=1)

        avg_mse = per_pair_mse.mean().item()
        worst_mse = per_pair_mse.max().item()
        worst_idx = per_pair_mse.argmax().item()

        # Calculate fraction memorized at different thresholds
        frac_memorized_1e1 = (per_pair_mse < 1e-1).float().mean().item()
        frac_memorized_1e2 = (per_pair_mse < 1e-2).float().mean().item()
        frac_memorized_1e3 = (per_pair_mse < 1e-3).float().mean().item()
        frac_memorized_1e4 = (per_pair_mse < 1e-4).float().mean().item()

    return {
        "avg_mse": avg_mse,
        "worst_mse": worst_mse,
        "worst_idx": worst_idx,
        "frac_memorized_1e-1": frac_memorized_1e1,
        "frac_memorized_1e-2": frac_memorized_1e2,
        "frac_memorized_1e-3": frac_memorized_1e3,
        "frac_memorized_1e-4": frac_memorized_1e4,
        "per_pair_mse": per_pair_mse.cpu(),
    }


class MemorizationTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    wandb_project: str | None = None
    seed: int = 0
    # Memorization model parameters (flattened from MemorizationConfig)
    n_pairs: PositiveInt | None = None
    n_pairs_multiplier: PositiveFloat | None = (
        None  # If set, n_pairs = n_pairs_multiplier * d_hidden
    )
    d_model: PositiveInt
    d_hidden: PositiveInt
    act_fn_name: str = "relu"
    use_bias: bool = True
    # Training parameters
    dataset_seed: int = 0
    batch_size: PositiveInt
    steps: PositiveInt
    print_freq: PositiveInt
    lr: PositiveFloat
    lr_schedule: Literal["constant", "cosine"] = "constant"
    lr_warmup_steps: int = 0  # Number of warmup steps for cosine schedule


def train(
    config: MemorizationTrainConfig,
    model: SingleLayerMemorizationMLP,
    dataloader: DatasetGeneratedDataLoader[
        tuple[
            Float[Tensor, "batch d_model"],
            Float[Tensor, "batch d_model"],
        ]
    ],
    device: str,
    out_dir: Path,
    run_name: str,
) -> Float[Tensor, ""]:
    if config.wandb_project:
        tags = ["memorization-train"]
        config = init_wandb(config, config.wandb_project, name=run_name, tags=tags)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = out_dir / "memorization_train_config.yaml"
    save_file(config.model_dump(mode="json"), config_path)
    logger.info(f"Saved config to {config_path}")
    if config.wandb_project:
        wandb.save(str(config_path), base_path=out_dir, policy="now")

    # Save the key-value pairs
    assert isinstance(dataloader.dataset, KeyValueMemorizationDataset)
    dataloader.dataset.save_key_value_pairs(out_dir / "key_value_pairs.json")
    logger.info(f"Saved key-value pairs to {out_dir / 'key_value_pairs.json'}")
    if config.wandb_project:
        wandb.save(str(out_dir / "key_value_pairs.json"), base_path=out_dir, policy="now")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # Setup learning rate scheduler
    if config.lr_schedule == "cosine":
        # For cosine schedule with warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.steps - config.lr_warmup_steps, eta_min=0.0
        )
    else:
        scheduler = None

    step = 0
    pbar = tqdm(total=config.steps)
    while step < config.steps:
        keys, values = dataloader.dataset.generate_batch(config.batch_size)

        optimizer.zero_grad()
        keys: Float[Tensor, "batch d_model"] = keys.to(device)
        values: Float[Tensor, "batch d_model"] = values.to(device)

        predicted_values = model(keys)
        loss = F.mse_loss(predicted_values, values)

        loss.backward()
        optimizer.step()

        # Handle learning rate schedule
        if config.lr_schedule == "cosine":
            if step < config.lr_warmup_steps:
                # Linear warmup
                warmup_lr = config.lr * (step + 1) / config.lr_warmup_steps
                for param_group in optimizer.param_groups:
                    param_group["lr"] = warmup_lr
            else:
                # Cosine annealing after warmup
                scheduler.step()

        if step % config.print_freq == 0:
            # Evaluate on all pairs
            metrics = evaluate_memorization(
                model, dataloader.dataset.keys, dataloader.dataset.values
            )
            tqdm.write(
                f"step {step}: loss={loss.item():.2e}, "
                f"avg_mse={metrics['avg_mse']:.2e}, "
                f"worst_mse={metrics['worst_mse']:.2e}, "
                f"frac_mem(1e-3)={metrics['frac_memorized_1e-3']:.3f}"
            )
            if config.wandb_project:
                log_dict = {
                    "loss": loss.item(),
                    "avg_mse": metrics["avg_mse"],
                    "worst_mse": metrics["worst_mse"],
                    "frac_memorized_1e-1": metrics["frac_memorized_1e-1"],
                    "frac_memorized_1e-2": metrics["frac_memorized_1e-2"],
                    "frac_memorized_1e-3": metrics["frac_memorized_1e-3"],
                    "frac_memorized_1e-4": metrics["frac_memorized_1e-4"],
                    "lr": optimizer.param_groups[0]["lr"],
                }
                # Log histogram every 5000 steps to avoid too much data
                if step % 5000 == 0:
                    log_dict["mse_histogram"] = wandb.Histogram(metrics["per_pair_mse"])
                wandb.log(log_dict, step=step)
            model.train()  # Set back to training mode

        step += 1
        pbar.update(1)

    model_path = out_dir / "memorization.pth"
    save_file(model.state_dict(), model_path)
    if config.wandb_project:
        wandb.save(str(model_path), base_path=out_dir, policy="now")
    print(f"Saved model to {model_path}")

    # Calculate final losses by testing each key-value pair
    final_metrics = evaluate_memorization(model, dataloader.dataset.keys, dataloader.dataset.values)

    print("Final metrics:")
    print(f"  Average MSE: {final_metrics['avg_mse']:.6f}")
    print(f"  Worst MSE: {final_metrics['worst_mse']:.6f} (pair {final_metrics['worst_idx']})")
    print(f"  Fraction memorized (MSE < 1e-1): {final_metrics['frac_memorized_1e-1']:.3f}")
    print(f"  Fraction memorized (MSE < 1e-2): {final_metrics['frac_memorized_1e-2']:.3f}")
    print(f"  Fraction memorized (MSE < 1e-3): {final_metrics['frac_memorized_1e-3']:.3f}")
    print(f"  Fraction memorized (MSE < 1e-4): {final_metrics['frac_memorized_1e-4']:.3f}")

    if config.wandb_project:
        wandb.log(
            {
                "final_avg_mse": final_metrics["avg_mse"],
                "final_worst_mse": final_metrics["worst_mse"],
                "final_frac_memorized_1e-1": final_metrics["frac_memorized_1e-1"],
                "final_frac_memorized_1e-2": final_metrics["frac_memorized_1e-2"],
                "final_frac_memorized_1e-3": final_metrics["frac_memorized_1e-3"],
                "final_frac_memorized_1e-4": final_metrics["frac_memorized_1e-4"],
            }
        )

    return torch.tensor(final_metrics["avg_mse"])


def run_train(config: MemorizationTrainConfig, device: str) -> Float[Tensor, ""]:
    # Calculate n_pairs from multiplier if provided
    if config.n_pairs_multiplier is not None:
        n_pairs = int(config.n_pairs_multiplier * config.d_hidden)
        logger.info(
            f"Calculated n_pairs={n_pairs} from n_pairs_multiplier={config.n_pairs_multiplier} * d_hidden={config.d_hidden}"
        )
    else:
        n_pairs = config.n_pairs

    # Log the actual n_pairs to wandb
    if config.wandb_project and wandb.run is not None:
        wandb.config.update({"n_pairs": n_pairs}, allow_val_change=True)

    # Create MemorizationConfig from flattened parameters
    model_cfg = MemorizationConfig(
        n_pairs=n_pairs,
        d_model=config.d_model,
        d_hidden=config.d_hidden,
        act_fn_name=config.act_fn_name,
        use_bias=config.use_bias,
    )
    run_name = (
        f"memorization_n{model_cfg.n_pairs}_d{model_cfg.d_model}_"
        f"dh{model_cfg.d_hidden}_"
        f"{model_cfg.act_fn_name}_seed{config.seed}"
    )
    out_dir = get_output_dir()

    model = SingleLayerMemorizationMLP(config=model_cfg).to(device)

    dataset = KeyValueMemorizationDataset(
        n_pairs=model_cfg.n_pairs,
        d_model=model_cfg.d_model,
        device=device,
        dataset_seed=config.dataset_seed,
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    final_losses = train(
        config=config,
        model=model,
        dataloader=dataloader,
        device=device,
        out_dir=out_dir,
        run_name=run_name,
    )
    return final_losses


def main(config_path: Path | str | MemorizationTrainConfig) -> None:
    """Main entry point for training memorization model.

    Args:
        config_path: Path to YAML config file or config object
    """
    device = get_device()
    logger.info(f"Using device: {device}")

    # Load config from file or object
    config = load_config(config_path, config_model=MemorizationTrainConfig)

    # Initialize wandb and merge sweep parameters
    if config.wandb_project:
        config = init_wandb(config, config.wandb_project, tags=["memorization-train"])

    set_seed(config.seed)
    logger.info(config)

    # Run training
    run_train(config, device)

    if config.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
