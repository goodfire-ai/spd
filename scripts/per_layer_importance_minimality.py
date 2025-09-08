import argparse
from typing import cast

import torch

from spd.configs import Config
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.utils.general_utils import extract_batch_data


def load_model_from_wandb(wandb_path: str) -> tuple[ComponentModel, Config]:
    """Load a trained ComponentModel and its Config from a W&B run path."""
    # Accept both bare "entity/project/run_id" and prefixed "wandb:entity/project/run_id"
    if not wandb_path.startswith("wandb:"):
        wandb_path = f"wandb:{wandb_path}"

    run_info = SPDRunInfo.from_path(wandb_path)

    # Reconstruct target model and wrap into ComponentModel
    model = ComponentModel.from_run_info(run_info)
    model.eval()
    model.requires_grad_(False)
    return model, run_info.config


@torch.inference_mode()
def compute_layerwise_importance_minimality(
    model: ComponentModel,
    config: Config,
    device: str | torch.device,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Compute importance-minimality loss per layer over up to one training batch.

    Uses the same dataset settings as training. For LM tasks, builds a dataloader
    from the task config and pulls a single batch.
    """
    # Build a dataloader equivalent to training setup (LM task expected here)
    assert getattr(config.task_config, "task_name", None) == "lm", (
        "This helper currently expects an LM decomposition config (ss_llama_config.yaml)."
    )
    lm_task = cast(LMTaskConfig, config.task_config)

    train_data_config = DatasetConfig(
        name=lm_task.dataset_name,
        hf_tokenizer_path=config.tokenizer_name,
        split=lm_task.train_data_split,
        n_ctx=lm_task.max_seq_len,
        is_tokenized=lm_task.is_tokenized,
        streaming=lm_task.streaming,
        column_name=lm_task.column_name,
        shuffle_each_epoch=lm_task.shuffle_each_epoch,
        seed=config.seed,
    )

    # Use microbatch size to match shapes expected during training
    microbatch_size = config.microbatch_size

    train_loader, _ = create_data_loader(
        dataset_config=train_data_config,
        batch_size=microbatch_size,
        buffer_size=lm_task.buffer_size,
        global_seed=config.seed,
        ddp_rank=0,
        ddp_world_size=1,
    )

    data_iter = iter(train_loader)
    batch = extract_batch_data(next(data_iter)).to(device)

    # Forward once to get target outputs and pre-weight activations
    _target_out, pre_weight_acts = model(
        batch,
        mode="pre_forward_cache",
        module_names=model.target_module_paths,
    )

    # Compute causal importances with the same sigmoid type used in training
    _, ci_upper_leaky = model.calc_causal_importances(
        pre_weight_acts=pre_weight_acts,
        sigmoid_type=config.sigmoid_type,
        detach_inputs=False,
    )

    # For each layer (key), compute (ci + eps)^p summed over components C, mean over other dims
    p = 0.1
    eps = 1e-12
    layer_to_loss: dict[str, float] = {}
    for layer_name, ci in ci_upper_leaky.items():
        # ci shape: [batch, (pos), C]; sum over last dim (C), mean over the rest
        layer_loss = ((ci + eps) ** p).sum(dim=-1).mean()
        layer_to_loss[layer_name] = float(layer_loss.item())

    return layer_to_loss, ci_upper_leaky


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute importance-minimality loss per layer for a W&B SPD run over one training batch"
        )
    )
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="W&B run path, e.g. 'goodfire/spd-play/ggsc7d3q' or 'wandb:goodfire/spd-play/ggsc7d3q'",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available else cpu)",
    )

    args = parser.parse_args()
    device: str = args.device

    model, config = load_model_from_wandb(args.run)
    model.to(device)

    layer_to_loss, ci_upper_leaky = compute_layerwise_importance_minimality(
        model, config, device=device
    )

    # Print concise results
    print(f"Per-layer importance-minimality loss (p={0.1:.3f}):")
    for layer_name, loss_val in layer_to_loss.items():
        print(f"{layer_name}: {loss_val:.6f}")

    total = sum(layer_to_loss.values())
    print(f"Total (sum across layers): {total:.6f}")

    # Also print ci_upper_leaky for a specific layer at the 10th sequence position (same batch)
    target_layer = "model.layers.0.self_attn.q_proj"
    pos_index = 9  # 0-based index for the 10th token position
    try:
        if target_layer not in ci_upper_leaky:
            raise KeyError(
                f"Layer key '{target_layer}' not found. Available: {list(ci_upper_leaky.keys())}"
            )
        ci = ci_upper_leaky[target_layer]
        if ci.ndim == 3:
            s = ci.shape[1]
            idx = pos_index if pos_index < s else s - 1
            ci_vec = ci[0, idx, :].detach().cpu()
        elif ci.ndim == 2:
            ci_vec = ci[0, :].detach().cpu()
        else:
            raise ValueError(f"Unexpected ci tensor ndim={ci.ndim} for layer '{target_layer}'")
        ci_str = ", ".join(f"{v:.6f}" for v in ci_vec.tolist())
        print(f"ci_upper_leaky[{target_layer}][prompt=0, pos={pos_index}] = [" + ci_str + "]")

        # Diagnostics: explain large importance-minimality for this layer on this batch
        p = 0.1
        eps = 1e-12
        C_dim = ci.shape[-1]
        eps_p = eps**p
        # Sum over C for each token position
        pos_sums = ((ci + eps) ** p).sum(dim=-1)
        idx2 = 0
        if pos_sums.ndim == 2:
            s2 = pos_sums.shape[1]
            idx2 = pos_index if pos_index < s2 else s2 - 1
            pos_sum_pos9 = pos_sums[0, idx2].item()
            mean_pos_sum = pos_sums.mean().item()
            min_pos_sum = pos_sums.min().item()
            max_pos_sum = pos_sums.max().item()
        else:
            pos_sum_pos9 = pos_sums[0].item()
            mean_pos_sum = pos_sums.mean().item()
            min_pos_sum = pos_sums.min().item()
            max_pos_sum = pos_sums.max().item()

        max_ci = ci.max().item()
        nonzero_count = int((ci > 0).sum().item())

        pos_label = idx2 if pos_sums.ndim == 2 else 0
        print(
            "Diagnostics for layer importance-minimality (same batch):\n"
            f"- C (number of components): {C_dim}\n"
            f"- eps^p: {eps_p:.6f} (baseline per component when ci≈0)\n"
            f"- Baseline sum per position (C * eps^p): {C_dim * eps_p:.3f}\n"
            f"- Sum((ci+eps)^p) at pos={pos_label}: {pos_sum_pos9:.3f}\n"
            f"- Per-position sum stats (mean/min/max): {mean_pos_sum:.3f} / {min_pos_sum:.3f} / {max_pos_sum:.3f}\n"
            f"- Max ci value in batch: {max_ci:.6f}\n"
            f"- Count(ci>0) in batch: {nonzero_count}"
        )
    except Exception as e:
        print(f"Could not fetch ci_upper_leaky for {target_layer} at pos {pos_index}: {e}")


if __name__ == "__main__":
    main()
