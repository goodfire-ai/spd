#!/usr/bin/env python3
"""
Simple SFT script for correcting model reasoning.

This performs full fine-tuning (no LoRA) to modify actual model weights.
Designed for training on 1-2 corrective texts to fix reasoning flaws
identified via attribution graph analysis.

Usage:
    python scripts/sft_correct.py --config configs/sft_rome_correction.yaml
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


class CorrectionDataset(Dataset):
    """Simple dataset for corrective texts."""

    def __init__(self, texts: list[str], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []

        for text in texts:
            # Tokenize with labels for causal LM
            encoding = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            # For causal LM, labels = input_ids (shifted internally by the model)
            encoding["labels"] = encoding["input_ids"].clone()
            # Mask padding tokens in labels
            encoding["labels"][encoding["attention_mask"] == 0] = -100
            self.encodings.append(encoding)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return {k: v.squeeze(0) for k, v in self.encodings[idx].items()}


def format_chat(text: str, tokenizer, system_prompt: str = "") -> str:
    """Format text using the model's chat template."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": ""})
    messages.append({"role": "assistant", "content": text})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(config: dict):
    """Run SFT training."""

    # Extract config values
    model_name = config.get("model_name", "meta-llama/Llama-3.1-8B-Instruct")
    output_dir = config.get("output_dir", "checkpoints/sft_correction")
    texts = config.get("texts", [])
    use_chat_template = config.get("use_chat_template", True)
    system_prompt = config.get("system_prompt", "")

    # Training hyperparameters (explicit type conversion for YAML values)
    epochs = int(config.get("epochs", 3))
    learning_rate = float(config.get("learning_rate", 1e-5))
    batch_size = int(config.get("batch_size", 1))
    gradient_accumulation_steps = int(config.get("gradient_accumulation_steps", 4))
    warmup_ratio = float(config.get("warmup_ratio", 0.1))
    max_length = int(config.get("max_length", 512))
    weight_decay = float(config.get("weight_decay", 0.01))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))

    # Device settings
    device = config.get("device", "cuda")
    dtype_str = config.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_str)

    print(f"Loading model: {model_name}")
    print(f"Training on {len(texts)} text(s)")
    print(f"Output directory: {output_dir}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model - full precision for training, then cast
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # Prepare texts
    if use_chat_template:
        formatted_texts = [format_chat(t, tokenizer, system_prompt) for t in texts]
    else:
        formatted_texts = texts

    print("\n--- Training texts ---")
    for i, text in enumerate(formatted_texts):
        print(f"\nText {i+1} ({len(tokenizer.encode(text))} tokens):")
        print(text[:500] + "..." if len(text) > 500 else text)
    print("---\n")

    # Create dataset and dataloader
    dataset = CorrectionDataset(formatted_texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer - only train parameters that require grad
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Learning rate scheduler
    total_steps = len(dataloader) * epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Training loop
    print(f"Starting training for {epochs} epochs...")
    print(f"  Total optimization steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")

    model.train()
    global_step = 0
    accumulated_loss = 0

    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(model.device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps

            # Backward pass
            loss.backward()
            accumulated_loss += loss.item()
            epoch_loss += loss.item() * gradient_accumulation_steps

            # Optimizer step
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                progress_bar.set_postfix({
                    "loss": f"{accumulated_loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })
                accumulated_loss = 0

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")

    # Save model
    print(f"\nSaving model to {output_dir}...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save the full model (not adapters)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training config for reference
    config_save_path = Path(output_dir) / "training_config.json"
    with open(config_save_path, "w") as f:
        json.dump({
            "base_model": model_name,
            "texts": texts,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    print("Training complete!")
    print(f"Model saved to: {output_dir}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="SFT for reasoning correction")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
