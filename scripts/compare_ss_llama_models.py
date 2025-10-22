#!/usr/bin/env python
"""Compare outputs of two SimpleStories LLaMA models.

This script loads two versions of the SimpleStories LLaMA model:
1. The simple_stories_train format model
2. The HuggingFace format model

It then runs the same data through both and compares their outputs.
"""

import torch
import yaml
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from simple_stories_train.run_info import RunInfo as SSRunInfo

from spd.utils.general_utils import resolve_class


def load_config(config_path: Path) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model_from_config(config: dict):
    """Load a model using the config."""
    pretrained_model_class = resolve_class(config["pretrained_model_class"])
    assert hasattr(pretrained_model_class, "from_pretrained"), (
        f"Model class {pretrained_model_class} should have a `from_pretrained` method"
    )

    if config["pretrained_model_class"].startswith("simple_stories_train"):
        # Load from simple_stories_train format using run_info
        run_info = SSRunInfo.from_path(config["pretrained_model_name"])
        assert hasattr(pretrained_model_class, "from_run_info")
        model = pretrained_model_class.from_run_info(run_info)
    else:
        # Load from HuggingFace format
        model = pretrained_model_class.from_pretrained(config["pretrained_model_name"])

    model.eval()
    return model


def get_output_from_model(model, input_ids: torch.Tensor, output_attr: str):
    """Get output from model using the specified output attribute."""
    with torch.no_grad():
        outputs = model(input_ids)
        if output_attr == "logits":
            return outputs.logits
        elif output_attr == "idx_0":
            # For simple_stories_train format, the output is a tuple
            return outputs[0]
        else:
            return getattr(outputs, output_attr)


def main():
    # Load configs
    config_dir = Path(__file__).parent.parent / "spd" / "experiments" / "lm"
    config1 = load_config(config_dir / "ss_llama_simple_config.yaml")
    config2 = load_config(config_dir / "ss_llama_config.yaml")

    print("=" * 80)
    print("Loading models...")
    print("=" * 80)

    # Load models
    print("\n1. Loading simple_stories_train model from WandB...")
    print(f"   Class: {config1['pretrained_model_class']}")
    print(f"   Name: {config1['pretrained_model_name']}")
    model1 = load_model_from_config(config1)

    print("\n2. Loading HuggingFace model...")
    print(f"   Class: {config2['pretrained_model_class']}")
    print(f"   Name: {config2['pretrained_model_name']}")
    model2 = load_model_from_config(config2)

    # Load tokenizers
    print("\n" + "=" * 80)
    print("Loading tokenizers...")
    print("=" * 80)
    print(f"Tokenizer 1: {config1['tokenizer_name']}")
    tokenizer1 = AutoTokenizer.from_pretrained(config1["tokenizer_name"])
    if tokenizer1.pad_token is None:
        tokenizer1.pad_token = tokenizer1.eos_token

    print(f"Tokenizer 2: {config2['tokenizer_name']}")
    tokenizer2 = AutoTokenizer.from_pretrained(config2["tokenizer_name"])
    if tokenizer2.pad_token is None:
        tokenizer2.pad_token = tokenizer2.eos_token

    print(f"\nTokenizer 1 vocab size: {tokenizer1.vocab_size}")
    print(f"Tokenizer 2 vocab size: {tokenizer2.vocab_size}")

    # Load dataset
    print("\n" + "=" * 80)
    print("Loading dataset...")
    print("=" * 80)
    dataset_name = config1["task_config"]["dataset_name"]
    column_name = config1["task_config"]["column_name"]
    split = config1["task_config"]["eval_data_split"]

    print(f"Dataset: {dataset_name}")
    print(f"Column: {column_name}")
    print(f"Split: {split}")

    dataset = load_dataset(dataset_name, split=split)

    # Get a few samples
    n_samples = 10
    max_seq_len = config1["task_config"]["max_seq_len"]

    print(f"\nProcessing {n_samples} samples with max_seq_len={max_seq_len}...")

    all_outputs1 = []
    all_outputs2 = []
    all_texts = []

    for i in range(n_samples):
        text = dataset[i][column_name]
        all_texts.append(text)

        # Tokenize with each model's respective tokenizer
        inputs1 = tokenizer1(
            text,
            return_tensors="pt",
            max_length=max_seq_len,
            truncation=True,
            padding="max_length"
        )
        input_ids1 = inputs1["input_ids"]

        inputs2 = tokenizer2(
            text,
            return_tensors="pt",
            max_length=max_seq_len,
            truncation=True,
            padding="max_length"
        )
        input_ids2 = inputs2["input_ids"]

        # Get outputs from both models
        output1 = get_output_from_model(model1, input_ids1, config1["pretrained_model_output_attr"])
        output2 = get_output_from_model(model2, input_ids2, config2["pretrained_model_output_attr"])

        all_outputs1.append(output1)
        all_outputs2.append(output2)

        if i == 0:
            print(f"\nSample {i} text (first 100 chars): {text[:100]}...")
            print(f"Input1 shape: {input_ids1.shape}")
            print(f"Input2 shape: {input_ids2.shape}")
            print(f"Output1 shape: {output1.shape}")
            print(f"Output2 shape: {output2.shape}")

    # Stack all outputs
    outputs1 = torch.cat(all_outputs1, dim=0)
    outputs2 = torch.cat(all_outputs2, dim=0)

    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    print(f"\nStacked output shapes:")
    print(f"  Model 1: {outputs1.shape}")
    print(f"  Model 2: {outputs2.shape}")

    # Compare only the overlapping vocabulary
    min_vocab = min(outputs1.shape[-1], outputs2.shape[-1])
    outputs1_overlap = outputs1[..., :min_vocab]
    outputs2_overlap = outputs2[..., :min_vocab]

    print(f"\nComparing overlapping vocabulary (first {min_vocab} tokens)")

    # Compute statistics
    diff = (outputs1_overlap - outputs2_overlap).numpy()

    print(f"\nAbsolute difference statistics:")
    abs_diff = np.abs(diff)
    print(f"  Mean: {abs_diff.mean():.6f}")
    print(f"  Median: {np.median(abs_diff):.6f}")
    print(f"  Max: {abs_diff.max():.6f}")
    print(f"  Min: {abs_diff.min():.6f}")
    print(f"  Std: {abs_diff.std():.6f}")

    print(f"\nRelative difference statistics:")
    # Avoid division by zero
    relative_diff = np.abs(diff) / (np.abs(outputs1_overlap.numpy()) + 1e-10)
    print(f"  Mean: {relative_diff.mean():.6f}")
    print(f"  Median: {np.median(relative_diff):.6f}")
    print(f"  Max: {relative_diff.max():.6f}")

    print(f"\nCorrelation:")
    flat1 = outputs1_overlap.numpy().flatten()
    flat2 = outputs2_overlap.numpy().flatten()
    correlation = np.corrcoef(flat1, flat2)[0, 1]
    print(f"  Pearson correlation: {correlation:.6f}")

    print(f"\nCosine similarity:")
    dot_product = (flat1 * flat2).sum()
    norm1 = np.linalg.norm(flat1)
    norm2 = np.linalg.norm(flat2)
    cosine_sim = dot_product / (norm1 * norm2)
    print(f"  Cosine similarity: {cosine_sim:.6f}")

    # Check if outputs are identical
    if np.allclose(outputs1_overlap.numpy(), outputs2_overlap.numpy(), atol=1e-5):
        print("\n✓ Outputs are nearly identical (within 1e-5 tolerance)")
    elif np.allclose(outputs1_overlap.numpy(), outputs2_overlap.numpy(), atol=1e-3):
        print("\n✓ Outputs are very similar (within 1e-3 tolerance)")
    else:
        print("\n✗ Outputs have significant differences")

    # Show a few sample predictions
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS (first example, first 10 tokens)")
    print("=" * 80)

    sample_idx = 0
    n_tokens = 10

    print(f"\nText: {all_texts[0][:200]}...\n")

    for token_idx in range(n_tokens):
        logits1 = outputs1[sample_idx, token_idx]
        logits2 = outputs2[sample_idx, token_idx]

        pred1 = torch.argmax(logits1).item()
        pred2 = torch.argmax(logits2).item()

        token1 = tokenizer1.decode([pred1])
        token2 = tokenizer2.decode([pred2])

        print(f"Position {token_idx}:")
        print(f"  Model 1 predicts: '{token1}' (id={pred1}, logit={logits1[pred1]:.4f})")
        print(f"  Model 2 predicts: '{token2}' (id={pred2}, logit={logits2[pred2]:.4f})")

        # Only compare if both predictions are within overlapping vocab
        if pred1 < min_vocab and pred2 < min_vocab:
            match = "✓" if pred1 == pred2 else "✗"
            print(f"  Match: {match}")

            if pred1 != pred2:
                # Show what each model thinks of the other's prediction
                if pred2 < outputs1.shape[-1]:
                    print(f"  Model 1's logit for token {pred2}: {logits1[pred2]:.4f}")
                if pred1 < outputs2.shape[-1]:
                    print(f"  Model 2's logit for token {pred1}: {logits2[pred1]:.4f}")
        print()


if __name__ == "__main__":
    main()
