#!/usr/bin/env python3
"""
Pythia Model Demo Script

This script demonstrates how to use the Pythia 70M model for text generation.
"""

import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM


def main():
    """Run the Pythia model demo."""
    print("Loading Pythia 70M model...")

    try:
        # Load tokenizer and model
        model_path = "EleutherAI/pythia-70m"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = GPTNeoXForCausalLM.from_pretrained(model_path)

        # Set pad token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Use CUDA if available, otherwise CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

        print(f"Model loaded on {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Default prompt for Pythia
        prompt = "Once upon a time"
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        eos_token_id = tokenizer.eos_token_id

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=400,
                temperature=0.7,
                do_sample=True,
                eos_token_id=eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\nGenerated text:\n{output_text}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
