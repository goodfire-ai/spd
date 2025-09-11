#!/usr/bin/env python3
"""
SimpleStories Model Demo Script

This script demonstrates how to use the SimpleStories model for text generation.
It loads a pre-trained SimpleStories model and generates text based on a given prompt.
"""

import sys

import torch
from transformers import AutoTokenizer, LlamaForCausalLM


def check_cuda_availability():
    """Check if CUDA is available and provide helpful information."""
    if torch.cuda.is_available():
        print(f"CUDA is available! Using device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        return True
    else:
        print("CUDA is not available. Using CPU (this will be slower).")
        return False


def load_model_and_tokenizer(model_size="1.25M"):
    """Load the SimpleStories model and tokenizer."""
    model_path = f"SimpleStories/SimpleStories-{model_size}"

    print(f"Loading model from: {model_path}")
    print("This may take a moment...")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✓ Tokenizer loaded successfully")

        # Set pad token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("✓ Pad token set to EOS token")

        # Load model
        model = LlamaForCausalLM.from_pretrained(model_path)
        print("✓ Model loaded successfully")

        # Move model to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        print(f"✓ Model moved to {device}")

        return model, tokenizer, device

    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have an internet connection to download the model")
        print("2. Check if the model path is correct")
        print("3. Ensure you have enough disk space for the model")
        sys.exit(1)


def generate_text(model, tokenizer, device, prompt, max_new_tokens=400, temperature=0.7):
    """Generate text using the SimpleStories model."""
    print(f"\nGenerating text with prompt: '{prompt}'")
    print("Generating...")

    try:
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
    """Main function to run the SimpleStories demo."""
    print("=" * 60)
    print("SimpleStories Model Demo")
    print("=" * 60)

    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(model_size="5M")

    # Default prompt
    default_prompt = "the curious cat looked at the"

    print(f"\nModel loaded successfully on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Generate text with default prompt
    print("\n" + "=" * 60)
    print("GENERATION EXAMPLE")
    print("=" * 60)

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=default_prompt,
        max_new_tokens=400,
        temperature=0.7,
    )

    if generated_text:
        print(f"\nGenerated text:\n{generated_text}")
    else:
        print("Failed to generate text.")
        return

    # Interactive mode
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
                temperature=0.7,
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
