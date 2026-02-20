#!/usr/bin/env python3
"""Generate a large corpus of medical completion prompts for edge aggregation analysis.

This script generates diverse medical prompts designed to activate medical knowledge
circuits in language models. The prompts are formatted as completion tasks.

Usage:
    python scripts/generate_medical_corpus.py --n 1000 -o data/medical_corpus.json
"""

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

SYSTEM_PROMPT = """You are a medical knowledge expert. Generate diverse medical completion prompts.

Each prompt should:
1. Be an incomplete sentence that elicits a specific medical fact
2. Cover diverse topics: anatomy, pharmacology, pathology, physiology, microbiology,
   neuroscience, cardiology, oncology, immunology, endocrinology, infectious disease
3. Be answerable with 1-3 words
4. Vary in difficulty and specificity

Format: Return a JSON object with a "prompts" array containing objects with:
- "prompt": The incomplete sentence
- "category": Medical category (e.g., "neuroscience", "pharmacology")

Examples:
- {"prompt": "The neurotransmitter associated with reward and pleasure is", "category": "neuroscience"}
- {"prompt": "Insulin is produced by the beta cells of the", "category": "endocrinology"}
- {"prompt": "The virus that causes AIDS is", "category": "infectious disease"}
- {"prompt": "Warfarin acts by inhibiting vitamin", "category": "pharmacology"}
- {"prompt": "The largest artery in the human body is the", "category": "anatomy"}
- {"prompt": "Parkinson's disease involves degeneration of neurons in the", "category": "neuroscience"}
- {"prompt": "The hormone that regulates blood calcium levels is", "category": "endocrinology"}
- {"prompt": "Penicillin works by inhibiting bacterial", "category": "pharmacology"}

Generate prompts with high diversity - avoid repetition of topics or phrasing patterns."""

USER_PROMPT_TEMPLATE = """Generate {n} diverse medical completion prompts.

Ensure broad coverage across these categories (aim for roughly equal distribution):
1. Neuroscience/Neurology - neurotransmitters, brain regions, neurological diseases
2. Pharmacology - drug mechanisms, drug classes, pharmacokinetics
3. Pathology/Pathophysiology - disease mechanisms, histology
4. Microbiology/Infectious Disease - pathogens, antibiotics, vaccines
5. Physiology - organ systems, regulatory mechanisms
6. Cardiology - heart anatomy, cardiovascular diseases
7. Endocrinology - hormones, glands, metabolic disorders
8. Oncology - cancer types, tumor markers, treatments
9. Immunology - immune cells, antibodies, autoimmune diseases
10. Anatomy - organs, structures, anatomical terms

Make each prompt unique. Vary sentence structures and medical specificity.

Return as JSON: {{"prompts": [...]}}"""


def generate_batch(client: OpenAI, n: int, model: str, existing_prompts: set) -> list[dict]:
    """Generate a batch of medical prompts."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(n=n)}
        ],
        temperature=0.9,  # Higher temperature for more diversity
        max_tokens=8000,
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)

    # Handle various response formats
    if isinstance(result, dict) and "prompts" in result:
        prompts = result["prompts"]
    elif isinstance(result, list):
        prompts = result
    else:
        for key, value in result.items():
            if isinstance(value, list):
                prompts = value
                break
        else:
            raise ValueError(f"Unexpected response format: {result.keys()}")

    # Filter out duplicates and invalid prompts
    valid_prompts = []
    for p in prompts:
        if not isinstance(p, dict):
            continue
        if "prompt" not in p:
            continue

        prompt_text = p["prompt"].lower().strip()

        # Skip if duplicate
        if prompt_text in existing_prompts:
            continue

        # Skip questions
        if prompt_text.endswith("?"):
            continue

        # Skip too short
        if len(prompt_text) < 20:
            continue

        existing_prompts.add(prompt_text)
        valid_prompts.append({
            "prompt": p["prompt"],
            "category": p.get("category", "general")
        })

    return valid_prompts


def main():
    parser = argparse.ArgumentParser(description="Generate medical completion prompts")
    parser.add_argument("--n", type=int, default=1000,
                        help="Total number of prompts to generate")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Prompts per API call")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="OpenAI model to use (gpt-4o-mini is fast/cheap)")
    parser.add_argument("-o", "--output", type=str, default="data/medical_corpus.json",
                        help="Output file path")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI()

    all_prompts = []
    seen_prompts = set()

    # Calculate batches needed (request slightly more to account for duplicates)
    target = args.n
    batches_needed = (target // args.batch_size) + 5  # Extra batches for duplicates

    print(f"Generating {target} medical prompts...")
    print(f"Using model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print()

    batch_num = 0
    while len(all_prompts) < target and batch_num < batches_needed * 2:
        batch_num += 1
        remaining = target - len(all_prompts)
        batch_request = min(args.batch_size, remaining + 10)  # Request slightly more

        print(f"Batch {batch_num}: Requesting {batch_request} prompts... ", end="", flush=True)

        try:
            new_prompts = generate_batch(client, batch_request, args.model, seen_prompts)
            all_prompts.extend(new_prompts)
            print(f"got {len(new_prompts)}, total: {len(all_prompts)}/{target}")

        except Exception as e:
            print(f"ERROR: {e}")
            time.sleep(2)  # Brief pause on error
            continue

        # Brief pause to avoid rate limits
        time.sleep(0.5)

        # Stop if we have enough
        if len(all_prompts) >= target:
            break

    # Trim to exact target
    all_prompts = all_prompts[:target]

    # Save results
    output = {
        "metadata": {
            "model": args.model,
            "total_prompts": len(all_prompts),
            "generation_date": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "prompts": all_prompts
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Generated {len(all_prompts)} medical prompts")
    print(f"Saved to: {output_path}")

    # Show category distribution
    categories = {}
    for p in all_prompts:
        cat = p.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print("\nCategory distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Show a few examples
    print("\nSample prompts:")
    import random
    samples = random.sample(all_prompts, min(10, len(all_prompts)))
    for p in samples:
        print(f"  [{p['category']}] {p['prompt']}")

    # Also create a YAML config file for use with aggregate_edge_stats.py
    yaml_path = output_path.with_suffix(".yaml")
    with open(yaml_path, "w") as f:
        f.write("# Medical prompts for edge aggregation\n")
        f.write("# Auto-generated by generate_medical_corpus.py\n\n")
        f.write("config:\n")
        f.write('  model_name: "meta-llama/Llama-3.1-8B-Instruct"\n\n')
        f.write("sequences:\n")
        for p in all_prompts:
            # Escape any quotes in prompt
            prompt_escaped = p["prompt"].replace('"', '\\"')
            f.write(f'  - prompt: "{prompt_escaped}"\n')
            f.write('    answer_prefix: ""\n')

    print(f"\nAlso created YAML config: {yaml_path}")


if __name__ == "__main__":
    main()
