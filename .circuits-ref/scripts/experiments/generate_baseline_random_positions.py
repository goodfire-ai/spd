#!/usr/bin/env python3
"""Generate baseline attribution graphs from FineWeb with random target positions.

For each FineWeb sample:
1. Use it as a user prompt
2. Generate a completion from the model
3. Randomly select a position within the assistant's response
4. Compute attribution graph from that position

This creates a diverse baseline that captures neuron behavior across different
positions within completions, not just at the final token.

Usage:
    python scripts/generate_baseline_random_positions.py \
        --n-samples 500 \
        --output-dir graphs/fineweb_baseline_random \
        --seed 42
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from circuits.relp import RelPAttributor, RelPConfig

# Llama 3.1 chat template
CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def sample_fineweb_sentences(n_samples: int, seed: int = 42) -> list[str]:
    """Sample random sentences from FineWeb dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets not installed. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    random.seed(seed)

    print("Loading FineWeb samples...", file=sys.stderr)

    try:
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )

        sentences = []
        buffer = []
        for i, example in enumerate(dataset):
            if i >= n_samples * 10:
                break
            text = example.get("text", "")
            if text:
                # Extract first sentence-like chunk (20-200 chars)
                chunk = text[:500].split(".")[0].strip()
                if 20 < len(chunk) < 200:
                    buffer.append(chunk)

        if buffer:
            sentences = random.sample(buffer, min(n_samples, len(buffer)))

        print(f"Sampled {len(sentences)} FineWeb sentences", file=sys.stderr)
        return sentences

    except Exception as e:
        print(f"Error loading FineWeb: {e}", file=sys.stderr)
        sys.exit(1)


def find_assistant_response_range(
    tokenizer: AutoTokenizer,
    full_text: str,
    prompt: str
) -> tuple[int, int]:
    """Find the token range of the assistant's response.

    Returns (start_pos, end_pos) where start_pos is the first token of the
    assistant's response and end_pos is the last token.
    """
    # Tokenize full text
    tokens = tokenizer.encode(full_text, add_special_tokens=False)

    # Find where assistant response starts (after the assistant header)
    # The header ends with: <|end_header_id|>\n\n
    full_token_strs = tokenizer.convert_ids_to_tokens(tokens)

    # Find the second occurrence of <|end_header_id|> (first is system, second is user, third is assistant)
    header_count = 0
    assistant_start = None
    for i, tok in enumerate(full_token_strs):
        if '<|end_header_id|>' in tok or tok == '<|end_header_id|>':
            header_count += 1
            if header_count == 3:  # Third header is assistant
                # Skip the newlines after header
                assistant_start = i + 1
                # Skip any newline tokens
                while assistant_start < len(tokens) and full_token_strs[assistant_start] in ['Ċ', 'ĊĊ', '\n', '\n\n']:
                    assistant_start += 1
                break

    if assistant_start is None:
        # Fallback: estimate based on prompt length
        prompt_formatted = CHAT_TEMPLATE.format(prompt=prompt)
        prompt_tokens = tokenizer.encode(prompt_formatted, add_special_tokens=False)
        assistant_start = len(prompt_tokens)

    return assistant_start, len(tokens) - 1


def generate_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
) -> str:
    """Generate a completion for a prompt."""
    formatted = CHAT_TEMPLATE.format(prompt=prompt)
    device = next(model.parameters()).device
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return full_text


def main():
    parser = argparse.ArgumentParser(
        description="Generate baseline attribution graphs with random target positions"
    )
    parser.add_argument(
        "--n-samples", type=int, default=500,
        help="Number of FineWeb samples to process (default: 500)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("graphs/fineweb_baseline_random"),
        help="Output directory for graphs"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--tau", type=float, default=0.005,
        help="Node threshold (default: 0.005)"
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of top logits (default: 5)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=50,
        help="Max tokens to generate for completion (default: 50)"
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--save-prompts", type=Path, default=None,
        help="Save prompts and positions to JSON file"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Sample FineWeb sentences
    sentences = sample_fineweb_sentences(args.n_samples, args.seed)

    # Load model
    print(f"Loading model {args.model}...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    device = next(model.parameters()).device

    # Initialize attributor
    config = RelPConfig(k=args.k, tau=args.tau, compute_edges=True)
    attributor = RelPAttributor(model, tokenizer, config)

    # Track metadata for all prompts
    all_metadata = []

    # PHASE 1: Generate all completions first
    print(f"\nPhase 1: Generating {len(sentences)} completions...", file=sys.stderr)
    completions = []
    for i, prompt in enumerate(tqdm(sentences, desc="Generating completions")):
        try:
            full_text = generate_completion(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens
            )

            start_pos, end_pos = find_assistant_response_range(
                tokenizer, full_text, prompt
            )

            if end_pos - start_pos < 3:
                print(f"  Skipping sample {i}: response too short", file=sys.stderr)
                continue

            # Randomly select a position within the assistant's response
            target_pos = random.randint(start_pos + 1, end_pos)

            completions.append({
                "index": i,
                "prompt": prompt,
                "full_text": full_text,
                "target_pos": target_pos,
                "start_pos": start_pos,
                "end_pos": end_pos,
            })
        except Exception as e:
            print(f"  Error generating completion {i}: {e}", file=sys.stderr)
            continue

    print(f"\nGenerated {len(completions)} completions", file=sys.stderr)

    # Clear CUDA cache before graph generation
    torch.cuda.empty_cache()

    # Re-initialize the attributor with fresh state
    del attributor
    torch.cuda.empty_cache()

    config = RelPConfig(k=args.k, tau=args.tau, compute_edges=True)
    attributor = RelPAttributor(model, tokenizer, config)

    # PHASE 2: Generate graphs for each completion
    print(f"\nPhase 2: Generating {len(completions)} graphs...", file=sys.stderr)

    for comp in tqdm(completions, desc="Generating graphs"):
        i = comp["index"]
        try:
            # Generate attribution graph from the target position
            graph = attributor.compute_attributions(
                comp["full_text"],
                k=args.k,
                tau=args.tau,
                compute_edges=True,
                target_position=comp["target_pos"]
            )

            # Add additional metadata
            graph["metadata"]["original_prompt"] = comp["prompt"]
            graph["metadata"]["assistant_response_start"] = comp["start_pos"]
            graph["metadata"]["assistant_response_end"] = comp["end_pos"]
            graph["metadata"]["is_baseline"] = True
            graph["metadata"]["sample_index"] = i

            # Save graph
            output_path = args.output_dir / f"baseline_{i:04d}.json"
            with open(output_path, "w") as f:
                json.dump(graph, f)

            # Track metadata
            all_metadata.append({
                "sample_index": i,
                "prompt": comp["prompt"],
                "target_position": comp["target_pos"],
                "assistant_start": comp["start_pos"],
                "assistant_end": comp["end_pos"],
                "output_file": str(output_path),
            })

        except Exception as e:
            print(f"  Error processing sample {i}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue

    # Save prompts metadata
    if args.save_prompts:
        with open(args.save_prompts, "w") as f:
            json.dump({
                "metadata": {
                    "source": "FineWeb-edu sample-10BT",
                    "total_prompts": len(all_metadata),
                    "seed": args.seed,
                    "tau": args.tau,
                    "k": args.k,
                    "max_new_tokens": args.max_new_tokens,
                },
                "samples": all_metadata
            }, f, indent=2)
        print(f"\nSaved prompt metadata to {args.save_prompts}", file=sys.stderr)

    print(f"\nGenerated {len(all_metadata)} graphs in {args.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
