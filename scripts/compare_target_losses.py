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
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from spd.data import DatasetConfig, create_data_loader
from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.pretrain.run_info import PretrainRunInfo

BATCH_SIZE = 64

CUSTOM_MODELS: dict[str, str] = {
    "pile_12L-768": "wandb:goodfire/spd/runs/t-a35bfd1a",
    "pile_4L-768": "wandb:goodfire/spd/runs/t-5541e2ae",
    "owt_12L-768": "wandb:goodfire/spd/runs/t-686f1c2e",
    "pile_12L-768-ctx2048": "wandb:goodfire/spd/runs/t-b0982592",
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


TOKENIZER_PATHS: dict[str, str] = {
    "gpt2": "gpt2",
    "neox": "EleutherAI/pythia-70m",
}

BPB_N_SAMPLES = 10_000


def compute_tokens_per_byte(
    tokenizer: Any,
    dataset: Dataset,
    column_name: str,
) -> float:
    """Compute average tokens per UTF-8 byte for a tokenizer on a dataset sample."""
    total_tokens = 0
    total_bytes = 0

    for example in dataset:
        text: str = example[column_name]  # pyright: ignore[reportCallIssue, reportArgumentType]
        if not text:
            continue
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(tokenizer.encode(text))

    assert total_bytes > 0
    return total_tokens / total_bytes


def compute_bpb_ratios() -> dict[str, dict[str, float]]:
    """Compute tokens/byte for each (tokenizer, dataset) pair.

    Returns: ratios[tokenizer_label][dataset_label] = tokens_per_byte
    """
    tokenizers = {
        name: AutoTokenizer.from_pretrained(path) for name, path in TOKENIZER_PATHS.items()
    }
    ratios: dict[str, dict[str, float]] = {}

    for ds_label, (ds_name, col_name) in DATASETS.items():
        print(f"Computing tokens/byte on {ds_label} ({BPB_N_SAMPLES} docs)...")
        ds = load_dataset(ds_name, split=f"train[-{BPB_N_SAMPLES}:]")
        assert isinstance(ds, Dataset)

        for tok_name, tokenizer in tokenizers.items():
            ratio = compute_tokens_per_byte(tokenizer, ds, col_name)
            ratios.setdefault(tok_name, {})[ds_label] = ratio
            print(f"  {tok_name}: {ratio:.4f} tokens/byte")

    return ratios


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


def non_emb_params(model: nn.Module, *emb_attrs: str) -> int:
    total = sum(p.numel() for p in model.parameters())
    for emb_attr in emb_attrs:
        module: Any = model
        for part in emb_attr.split("."):
            module = getattr(module, part)
        total -= module.weight.numel()
    return total


@torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
def eval_all_models(
    models: dict[str, tuple[nn.Module, bool]],
    val_loader: DataLoader[dict[str, Any]],
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

    # --- Compute tokens/byte ratios for BPB conversion ---
    bpb_ratios = compute_bpb_ratios()

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
        param_counts[display_name] = non_emb_params(model, "gpt_neox.embed_in", "embed_out")
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

    # Compute BPB for each model
    dataset_labels = list(DATASETS.keys())
    bpb_losses: dict[str, dict[str, float]] = {}
    for name, ces in ce_losses.items():
        tok = tokenizer_labels[name]
        bpb_losses[name] = {ds: ce * bpb_ratios[tok][ds] / math.log(2) for ds, ce in ces.items()}

    # Sort by average BPB (comparable across tokenizers)
    all_names = sorted(bpb_losses, key=lambda n: sum(bpb_losses[n].values()) / len(bpb_losses[n]))

    # Print summary table
    bpb_cols = "".join(f" {label + ' BPB':>10}" for label in dataset_labels)
    ce_cols = "".join(f" {label + ' CE':>10}" for label in dataset_labels)
    header = f"{'Model':<25}{bpb_cols}{ce_cols} {'Non-emb Params':>18} {'Tok':>5}"
    width = len(header)

    print("\n" + "=" * width)
    print(header)
    print("-" * width)

    for name in all_names:
        bpb_vals = "".join(f" {bpb_losses[name][label]:>10.4f}" for label in dataset_labels)
        ce_vals = "".join(f" {ce_losses[name][label]:>10.4f}" for label in dataset_labels)
        print(
            f"{name:<25}{bpb_vals}{ce_vals} {param_counts[name]:>18,} {tokenizer_labels[name]:>5}"
        )

    print("=" * width)
    print("BPB = CE × (tokens/byte) / ln(2). Lower BPB is better. Comparable across tokenizers.")


if __name__ == "__main__":
    main()
