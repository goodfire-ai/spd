"""Trains a MemTransformer model on a memorization task."""

import torch
import wandb
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm import tqdm

from spd.configs import ScheduleConfig
from spd.experiments.mem.configs import MemModelConfig, MemTrainConfig
from spd.experiments.mem.mem_dataset import MemDataset
from spd.experiments.mem.models import MemTransformer
from spd.log import logger
from spd.settings import DEFAULT_PROJECT_NAME
from spd.utils.data_utils import DatasetGeneratedDataLoader
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import get_scheduled_value, set_seed
from spd.utils.run_utils import ExecutionStamp, save_file
from spd.utils.wandb_utils import init_wandb


def loss_function(
    logits: Float[Tensor, "batch seq_len vocab_size"],  # noqa: F821
    labels: Float[Tensor, "batch"],  # noqa: F821
) -> Float[Tensor, ""]:
    """Compute cross-entropy loss at the final sequence position.

    Args:
        logits: Model output logits [batch, seq_len, vocab_size]
        labels: Target labels [batch]

    Returns:
        Cross-entropy loss (scalar)
    """
    # Get logits at the final position
    final_logits = logits[:, -1, :]  # [batch, vocab_size]
    loss = F.cross_entropy(final_logits, labels)
    return loss


def compute_accuracy(
    logits: Float[Tensor, "batch seq_len vocab_size"],  # noqa: F821
    labels: Float[Tensor, "batch"],  # noqa: F821
) -> float:
    """Compute accuracy at the final sequence position.

    Args:
        logits: Model output logits [batch, seq_len, vocab_size]
        labels: Target labels [batch]

    Returns:
        Accuracy (fraction correct)
    """
    final_logits = logits[:, -1, :]  # [batch, vocab_size]
    predictions = final_logits.argmax(dim=-1)  # [batch]
    accuracy = (predictions == labels).float().mean().item()
    return accuracy


def train(
    config: MemTrainConfig,
    model: MemTransformer,
    trainable_params: list[nn.Parameter],
    dataloader: DatasetGeneratedDataLoader[
        tuple[
            Float[Tensor, "batch seq_len"],  # noqa: F821
            Float[Tensor, "batch"],  # noqa: F821
        ]
    ],
    device: str,
    run_name: str,
) -> Float[Tensor, ""]:
    """Train the MemTransformer model.

    Args:
        config: Training configuration
        model: The model to train
        trainable_params: List of parameters to optimize
        dataloader: DataLoader for the memorization dataset
        device: Device to train on
        run_name: Name for the run

    Returns:
        Final loss value
    """
    execution_stamp = ExecutionStamp.create(run_type="train", create_snapshot=False)
    out_dir = execution_stamp.out_dir
    logger.info(f"Run ID: {execution_stamp.run_id}")
    logger.info(f"Output directory: {out_dir}")

    if config.wandb_project:
        tags = ["mem-train"]
        init_wandb(
            config, config.wandb_project, run_id=execution_stamp.run_id, name=run_name, tags=tags
        )

    # Save config
    config_path = out_dir / "mem_train_config.yaml"
    save_file(config.model_dump(mode="json"), config_path)
    logger.info(f"Saved config to {config_path}")
    if config.wandb_project:
        wandb.save(str(config_path), base_path=out_dir, policy="now")

    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=0.01)

    # Create a ScheduleConfig from the old-style lr config for backward compatibility
    lr_schedule_config = ScheduleConfig(
        start_val=config.lr,
        fn_type=config.lr_schedule,
        final_val_frac=0.0 if config.lr_schedule == "cosine" else 1.0,
    )

    pbar = tqdm(range(config.steps), total=config.steps)
    for step, (batch, labels) in zip(pbar, dataloader, strict=False):
        if step >= config.steps:
            break

        # Update the learning rate
        current_lr = get_scheduled_value(step, config.steps, lr_schedule_config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        optimizer.zero_grad()
        batch = batch.to(device)
        labels = labels.to(device)

        logits = model(batch)
        loss = loss_function(logits, labels)

        loss.backward()
        optimizer.step()

        if step % config.print_freq == 0:
            accuracy = compute_accuracy(logits, labels)
            tqdm.write(
                f"step {step}: loss={loss.item():.4f}, acc={accuracy:.4f}, lr={current_lr:.2e}"
            )
            if config.wandb_project:
                wandb.log({"loss": loss.item(), "accuracy": accuracy, "lr": current_lr}, step=step)

    # Save model
    model_path = out_dir / "mem.pth"
    save_file(model.state_dict(), model_path)
    if config.wandb_project:
        wandb.save(str(model_path), base_path=out_dir, policy="now")
    logger.info(f"Saved model to {model_path}")

    # Calculate final metrics by evaluating on all facts
    model.eval()
    with torch.no_grad():
        assert isinstance(dataloader.dataset, MemDataset)
        all_inputs, all_labels = dataloader.dataset.get_all_facts()
        all_logits = model(all_inputs)
        final_loss = loss_function(all_logits, all_labels)
        final_accuracy = compute_accuracy(all_logits, all_labels)

    logger.info(f"Final loss: {final_loss.item():.4f}")
    logger.info(f"Final accuracy: {final_accuracy:.4f}")

    if config.wandb_project:
        wandb.log({"final_loss": final_loss.item(), "final_accuracy": final_accuracy})

    return final_loss


def run_train(config: MemTrainConfig, device: str) -> Float[Tensor, ""]:
    """Run the training process.

    Args:
        config: Training configuration
        device: Device to train on

    Returns:
        Final loss value
    """
    model_cfg = config.mem_model_config
    run_name = (
        f"mem_v{model_cfg.vocab_size}_d{model_cfg.d_model}_"
        f"mlp{model_cfg.d_mlp}_h{model_cfg.n_heads}_"
        f"facts{config.n_facts}_seed{config.seed}"
    )

    model = MemTransformer(config=model_cfg).to(device)

    dataset = MemDataset(
        n_facts=config.n_facts,
        vocab_size=model_cfg.vocab_size,
        seq_len=model_cfg.seq_len,
        device=device,
        seed=config.seed,
    )
    dataloader = DatasetGeneratedDataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    final_loss = train(
        config=config,
        model=model,
        trainable_params=[p for p in model.parameters() if p.requires_grad],
        dataloader=dataloader,
        device=device,
        run_name=run_name,
    )
    return final_loss


if __name__ == "__main__":
    device = get_device()

    config = MemTrainConfig(
        wandb_project=DEFAULT_PROJECT_NAME,
        seed=0,
        mem_model_config=MemModelConfig(
            vocab_size=64,
            d_model=64,
            d_mlp=64,
            n_heads=1,
            seq_len=3,
            use_layer_norm=True,
        ),
        n_facts=2048,
        batch_size=500,
        steps=10000,
        print_freq=1000,
        lr=1e-3,
        lr_schedule="cosine",
    )

    set_seed(config.seed)
    run_train(config, device)
