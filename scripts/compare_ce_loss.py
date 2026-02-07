"""Compare cross-entropy loss of pile-trained models, GPT-2 small, and Pythia models on both the
Pile and OpenWebText validation sets.

Models using different tokenizers are evaluated in separate groups (each on its own tokenized data).
Each model is loaded once and evaluated on both datasets.

Usage:
    python scripts/compare_ce_loss.py
    python scripts/compare_ce_loss.py --val-max-steps 50
    python scripts/compare_ce_loss.py --device cpu
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from spd.data import DatasetConfig, create_data_loader
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.pretrain.run_info import PretrainRunInfo

BATCH_SIZE = 64

CUSTOM_MODELS: dict[str, str] = {
    "pile_12L-768": "wandb:goodfire/spd/runs/t-a35bfd1a",
    "pile_4L-768": "wandb:goodfire/spd/runs/t-5541e2ae",
}

PYTHIA_MODELS: list[str] = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
]

DATASETS: dict[str, tuple[str, str]] = {
    "Pile": ("monology/pile-uncopyrighted", "text"),
    "OWT": ("Skylion007/openwebtext", "text"),
}


def make_val_config(hf_tokenizer_path: str, dataset_name: str, column_name: str) -> DatasetConfig:
    return DatasetConfig(
        name=dataset_name,
        is_tokenized=False,
        hf_tokenizer_path=hf_tokenizer_path,
        split="train[-100000:]",
        streaming=False,
        n_ctx=513,  # n_ctx + 1 for next-token prediction
        seed=0,
        column_name=column_name,
        shuffle_each_epoch=False,  # Deterministic ordering for fair comparison
    )


def get_logits(model: nn.Module, x: torch.Tensor, *, is_custom: bool) -> torch.Tensor:
    if is_custom:
        logits, _ = model(x)
        assert logits is not None
        return logits
    else:
        return model(x).logits


def non_emb_params(model: nn.Module, emb_attr: str) -> int:
    parts = emb_attr.split(".")
    module = model
    for part in parts:
        module = getattr(module, part)
    return sum(p.numel() for p in model.parameters()) - module.weight.numel()


@torch.no_grad()
def eval_all_models(
    models: dict[str, tuple[nn.Module, bool]],
    val_loader: DataLoader,
    val_max_steps: int,
    device: str,
) -> dict[str, float]:
    """Evaluate all models on the exact same batches in a single loop."""
    for model, _ in models.values():
        model.eval()

    totals = {name: 0.0 for name in models}
    n_steps = 0

    for batch in val_loader:
        if n_steps >= val_max_steps:
            break
        tokens = batch["input_ids"].to(torch.long)
        x = tokens[:, :-1].to(device)
        y = tokens[:, 1:].to(device)

        for name, (model, is_custom) in models.items():
            logits = get_logits(model, x, is_custom=is_custom)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
            totals[name] += loss.item()

        n_steps += 1

    assert n_steps > 0
    return {name: total / n_steps for name, total in totals.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Compare CE loss on Pile and OpenWebText validation sets"
    )
    parser.add_argument(
        "--val-max-steps", type=int, default=100, help="Number of validation batches (default: 100)"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto-detect)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Val max steps: {args.val_max_steps}, Batch size: {args.batch_size}")

    # ce_losses[model_name][dataset_label] = loss
    ce_losses: dict[str, dict[str, float]] = {}
    param_counts: dict[str, int] = {}
    tokenizer_labels: dict[str, str] = {}

    # --- Load all models (once) ---

    # GPT-2 tokenizer group: custom models + gpt2-small
    # is_custom=True means LlamaSimpleMLP forward signature, False means HF model
    gpt2_group: dict[str, tuple[nn.Module, bool]] = {}

    for name, wandb_path in CUSTOM_MODELS.items():
        print(f"Loading {name}...")
        run_info = PretrainRunInfo.from_path(wandb_path)
        model = LlamaSimpleMLP.from_run_info(run_info)
        model.to(device)
        gpt2_group[name] = (model, True)
        param_counts[name] = non_emb_params(model, "wte")
        tokenizer_labels[name] = "gpt2"

    print("Loading gpt2-small...")
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    gpt2_group["gpt2-small"] = (gpt2_model, False)
    param_counts["gpt2-small"] = non_emb_params(gpt2_model, "transformer.wte")
    tokenizer_labels["gpt2-small"] = "gpt2"

    # Pythia tokenizer group (all share the GPT-NeoX tokenizer)
    pythia_group: dict[str, tuple[nn.Module, bool]] = {}
    for hf_name in PYTHIA_MODELS:
        display_name = hf_name.split("/")[1]  # e.g. "pythia-70m"
        print(f"Loading {display_name}...")
        model = AutoModelForCausalLM.from_pretrained(hf_name).to(device)
        pythia_group[display_name] = (model, False)
        param_counts[display_name] = non_emb_params(model, "gpt_neox.embed_in")
        tokenizer_labels[display_name] = "neox"

    # --- Evaluate each group on each dataset ---
    groups: list[tuple[str, dict[str, tuple[nn.Module, bool]]]] = [
        ("gpt2", gpt2_group),
        ("EleutherAI/pythia-70m", pythia_group),
    ]

    for dataset_label, (dataset_name, column_name) in DATASETS.items():
        for tokenizer_path, group in groups:
            if not group:
                continue
            group_desc = "GPT-2 tok" if tokenizer_path == "gpt2" else "NeoX tok"
            print(f"\nLoading {dataset_label} validation data ({group_desc})...")
            loader, _ = create_data_loader(
                dataset_config=make_val_config(tokenizer_path, dataset_name, column_name),
                batch_size=args.batch_size,
                buffer_size=1000,
                global_seed=0,
                dist_state=None,
            )

            print(
                f"Evaluating {group_desc} models on {dataset_label} ({args.val_max_steps} batches)..."
            )
            losses = eval_all_models(group, loader, args.val_max_steps, device)
            for name, loss in losses.items():
                ce_losses.setdefault(name, {})[dataset_label] = loss

    # Print summary table
    dataset_labels = list(DATASETS.keys())
    loss_cols = "".join(f" {label + ' CE':>10}" for label in dataset_labels)
    header = f"{'Model':<25}{loss_cols} {'Non-emb Params':>18} {'Tokenizer':>12}"
    width = len(header)

    print("\n" + "=" * width)
    print(header)
    print("-" * width)

    # Sort by average CE loss across datasets
    all_names = list(ce_losses.keys())
    all_names.sort(key=lambda n: sum(ce_losses[n].values()) / len(ce_losses[n]))

    for name in all_names:
        loss_vals = "".join(f" {ce_losses[name][label]:>10.4f}" for label in dataset_labels)
        print(f"{name:<25}{loss_vals} {param_counts[name]:>18,} {tokenizer_labels[name]:>12}")

    print("=" * width)
    print("Note: CE losses across tokenizer groups are not directly comparable.")


if __name__ == "__main__":
    main()
