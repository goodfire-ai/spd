"""Compare cross-entropy loss of pile-trained models, GPT-2 small, and Pythia models on the
OpenWebText validation set.

Uses the same validation split and data loading as spd/pretrain/train.py.
Models using different tokenizers are evaluated in separate groups (each on its own tokenized data).

Usage:
    python scripts/compare_owt_ce_loss.py
    python scripts/compare_owt_ce_loss.py --val-max-steps 50
    python scripts/compare_owt_ce_loss.py --device cpu
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


def make_val_config(hf_tokenizer_path: str) -> DatasetConfig:
    """Matches the validation config in owt_llama_simple_mlp-12L-768.yaml."""
    return DatasetConfig(
        name="Skylion007/openwebtext",
        is_tokenized=False,
        hf_tokenizer_path=hf_tokenizer_path,
        split="train[-100000:]",
        streaming=False,
        n_ctx=513,  # n_ctx + 1 for next-token prediction
        seed=0,
        column_name="text",
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
    parser = argparse.ArgumentParser(description="Compare CE loss on OpenWebText validation set")
    parser.add_argument(
        "--val-max-steps", type=int, default=100, help="Number of validation batches (default: 100)"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto-detect)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Val max steps: {args.val_max_steps}, Batch size: {args.batch_size}")

    ce_losses: dict[str, float] = {}
    param_counts: dict[str, int] = {}
    tokenizer_labels: dict[str, str] = {}

    # --- GPT-2 tokenizer group: custom models + gpt2-small ---
    print("\nLoading validation data (GPT-2 tokenizer)...")
    gpt2_loader, _ = create_data_loader(
        dataset_config=make_val_config("gpt2"),
        batch_size=args.batch_size,
        buffer_size=1000,
        global_seed=0,
        dist_state=None,
    )

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

    print(f"\nEvaluating GPT-2 tokenizer group on {args.val_max_steps} batches...")
    ce_losses.update(eval_all_models(gpt2_group, gpt2_loader, args.val_max_steps, device))

    # --- Pythia tokenizer group (all share the GPT-NeoX tokenizer) ---
    print("\nLoading validation data (Pythia/NeoX tokenizer)...")
    pythia_loader, _ = create_data_loader(
        dataset_config=make_val_config("EleutherAI/pythia-70m"),
        batch_size=args.batch_size,
        buffer_size=1000,
        global_seed=0,
        dist_state=None,
    )

    pythia_group: dict[str, tuple[nn.Module, bool]] = {}
    for hf_name in PYTHIA_MODELS:
        display_name = hf_name.split("/")[1]  # e.g. "pythia-70m"
        print(f"Loading {display_name}...")
        model = AutoModelForCausalLM.from_pretrained(hf_name).to(device)
        pythia_group[display_name] = (model, False)
        param_counts[display_name] = non_emb_params(model, "gpt_neox.embed_in")
        tokenizer_labels[display_name] = "neox"

    print(f"\nEvaluating Pythia tokenizer group on {args.val_max_steps} batches...")
    ce_losses.update(eval_all_models(pythia_group, pythia_loader, args.val_max_steps, device))

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Model':<25} {'CE Loss':>10} {'Non-emb Params':>18} {'Tokenizer':>12}")
    print("-" * 70)
    for name in sorted(ce_losses, key=lambda n: ce_losses[n]):
        print(
            f"{name:<25} {ce_losses[name]:>10.4f}"
            f" {param_counts[name]:>18,} {tokenizer_labels[name]:>12}"
        )
    print("=" * 70)
    print("Note: CE losses across tokenizer groups are not directly comparable.")


if __name__ == "__main__":
    main()
