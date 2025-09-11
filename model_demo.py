#!/usr/bin/env python3
"""
Unified Model Demo Script

This script supports both SimpleStories and Pythia models for text generation.
It automatically detects the model type and uses the appropriate model class.
"""

import argparse
import sys

import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM, LlamaForCausalLM


def check_cuda_availability():
    """Check if CUDA is available and provide helpful information."""
    if torch.cuda.is_available():
        print(f"CUDA is available! Using device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        return True
    else:
        print("CUDA is not available. Using CPU (this will be slower).")
        return False


def detect_model_type(model_path):
    """Detect the model type based on the model path."""
    if "SimpleStories" in model_path:
        return "simplestories"
    elif "pythia" in model_path.lower():
        return "pythia"
    else:
        # Try to detect from the model config
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path)
        if hasattr(config, "model_type"):
            if config.model_type == "llama":
                return "simplestories"
            elif config.model_type == "gpt_neox":
                return "pythia"
        return "unknown"


def load_model_and_tokenizer(model_path, model_type=None):
    """Load the model and tokenizer based on the detected type."""
    if model_type is None:
        model_type = detect_model_type(model_path)

    print(f"Loading {model_type} model from: {model_path}")
    print("This may take a moment...")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✓ Tokenizer loaded successfully")

        # Set pad token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("✓ Pad token set to EOS token")

        # Load model based on type
        if model_type == "simplestories":
            model = LlamaForCausalLM.from_pretrained(model_path)
        elif model_type == "pythia":
            model = GPTNeoXForCausalLM.from_pretrained(model_path)
        else:
            # Try to auto-detect the model class
            try:
                model = LlamaForCausalLM.from_pretrained(model_path)
                model_type = "simplestories"
                print("✓ Auto-detected as LlamaForCausalLM (SimpleStories)")
            except Exception:
                try:
                    model = GPTNeoXForCausalLM.from_pretrained(model_path)
                    model_type = "pythia"
                    print("✓ Auto-detected as GPTNeoXForCausalLM (Pythia)")
                except Exception as e:
                    print(f"Error: Could not load model. {e}")
                    sys.exit(1)

        print("✓ Model loaded successfully")

        # Move model to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        print(f"✓ Model moved to {device}")

        return model, tokenizer, device, model_type

    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have an internet connection to download the model")
        print("2. Check if the model path is correct")
        print("3. Ensure you have enough disk space for the model")
        sys.exit(1)


def generate_text(model, tokenizer, device, prompt, max_new_tokens=400, temperature=0.7):
    """Generate text using the loaded model."""
    print(f"\nGenerating text with prompt: '{prompt}'")
    print("Generating...")

    try:
        # Handle empty prompt case
        if not prompt or prompt.strip() == "":
            # For empty prompts, start with just the BOS token or EOS token if BOS is not available
            start_token_id = (
                tokenizer.bos_token_id
                if tokenizer.bos_token_id is not None
                else tokenizer.eos_token_id
            )
            input_ids = torch.tensor([[start_token_id]], device=device)
            attention_mask = torch.tensor([[1]], device=device)
        else:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)

        # Set EOS token ID
        eos_token_id = tokenizer.eos_token_id

        # Generate text
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                eos_token_id=eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode the generated text
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return output_text

    except Exception as e:
        print(f"Error during generation: {e}")
        return None


def main():
    """Main function to run the model demo."""
    parser = argparse.ArgumentParser(description="Demo script for SimpleStories and Pythia models")
    parser.add_argument(
        "--model",
        "-m",
        choices=["simplestories", "pythia"],
        default="simplestories",
        help="Model to use (default: simplestories)",
    )
    parser.add_argument(
        "--size", "-s", help="Model size (e.g., '1.25M' for SimpleStories, '70m' for Pythia)"
    )
    parser.add_argument("--prompt", "-p", help="Custom prompt (default: model-specific)")
    parser.add_argument(
        "--max-tokens",
        "-t",
        type=int,
        default=400,
        help="Maximum new tokens to generate (default: 400)",
    )
    parser.add_argument(
        "--temperature",
        "-temp",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")

    args = parser.parse_args()

    print("=" * 60)
    print("Unified Model Demo - SimpleStories & Pythia")
    print("=" * 60)

    # Set up model path and default prompt based on model type
    if args.model == "simplestories":
        model_size = args.size or "1.25M"
        model_path = f"SimpleStories/SimpleStories-{model_size}"
        default_prompt = ""
    elif args.model == "pythia":
        model_size = args.size or "70m"
        model_path = f"EleutherAI/pythia-{model_size}"
        default_prompt = ""

    # Load model and tokenizer
    model, tokenizer, device, detected_type = load_model_and_tokenizer(model_path)

    # Use custom prompt or default
    prompt = args.prompt or default_prompt

    print(f"\nModel loaded successfully on {device}")
    print(f"Model type: {detected_type}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Generate text with the prompt
    print("\n" + "=" * 60)
    print("GENERATION EXAMPLE")
    print("=" * 60)

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    if generated_text:
        print(f"\nGenerated text:\n{generated_text}")
    else:
        print("Failed to generate text.")
        return

    # Interactive mode
    if args.interactive:
        print("\n" + "=" * 60)
        print("INTERACTIVE MODE")
        print("=" * 60)
        print("Enter your own prompts (or 'quit' to exit):")

        while True:
            try:
                user_prompt = input("\nPrompt: ").strip()

                if user_prompt.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                if not user_prompt:
                    user_prompt = default_prompt
                    print(f"Using default prompt: '{user_prompt}'")

                generated_text = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    prompt=user_prompt,
                    max_new_tokens=200,  # Shorter for interactive mode
                    temperature=args.temperature,
                )

                if generated_text:
                    print(f"\nGenerated text:\n{generated_text}")
                else:
                    print("Failed to generate text.")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
