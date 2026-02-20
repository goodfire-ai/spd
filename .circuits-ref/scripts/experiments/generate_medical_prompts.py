#!/usr/bin/env python3
"""
Stage 1: Generate medical completion prompts using GPT-5.

Each prompt is designed to elicit a specific factual answer.
We'll later filter for cases where Llama gets it wrong but has
the correct answer in top-5.
"""

import argparse
import json
from pathlib import Path

# Load API key from .env
from dotenv import load_dotenv
from openai import OpenAI

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

SYSTEM_PROMPT = """You are a medical knowledge expert helping to create test prompts for evaluating language models.

Generate medical completion prompts that:
1. Are factual statements with a SINGLE correct answer (1-2 words)
2. Use completion format that directly elicits the medical term
3. Cover diverse medical topics: anatomy, pharmacology, pathology, physiology, microbiology
4. Have answers that are specific medical terms (not common words)
5. Are answerable by a knowledgeable medical student

CRITICAL: The prompt must be designed so the FIRST WORD of the completion is the medical term itself.
- BAD: "The treatment for X is" → model says "The standard treatment..." (first word is "The")
- GOOD: "X is treated with" → model directly says "penicillin" or "insulin"
- GOOD: "The disease caused by Mycobacterium tuberculosis is called" → model says "tuberculosis"
- GOOD: "Dopamine is primarily produced in the" → model says "substantia" or "midbrain"

Format each prompt as JSON with:
- "prompt": The incomplete sentence that directly elicits the medical term
- "correct_answer": The single correct completion (1-2 words)
- "correct_answer_variants": List of acceptable alternative phrasings/spellings
- "category": Medical category
- "difficulty": easy, medium, or hard
- "wrong_but_plausible": List of 3-4 wrong answers that a confused student might give

Examples of GOOD prompts (answer is first word):
- "Parkinson's disease is caused by degeneration of neurons in the" → "substantia nigra"
- "The neurotransmitter most associated with reward is" → "dopamine"
- "Streptococcus pyogenes causes" → "pharyngitis" or "strep throat"
- "ACE inhibitors work by blocking" → "angiotensin"
- "Insulin is produced by the" → "pancreas"
- "The most common cause of community-acquired pneumonia is" → "Streptococcus pneumoniae"
- "Warfarin acts by inhibiting vitamin" → "K"
- "Cushing syndrome is caused by excess" → "cortisol"

Examples of BAD prompts (avoid):
- "The treatment for hypertension includes" (too many answers)
- "What causes diabetes?" (question format)
- "Diabetes is a disease that" (doesn't elicit specific term)
"""

USER_PROMPT_TEMPLATE = """Generate {n} diverse medical completion prompts.

Focus on these categories (roughly equal distribution):
- Neuroscience/Neurology (neurotransmitters, brain regions, diseases)
- Pharmacology (drug mechanisms, drug classes, side effects)
- Pathology (disease mechanisms, histology findings)
- Microbiology (pathogens, virulence factors, treatments)
- Physiology (hormones, organ systems, regulatory mechanisms)

Return as a JSON array of objects. Each object should have the fields described in your instructions."""


def generate_prompts(n: int = 20, model: str = "gpt-4o") -> list[dict]:
    """Generate medical prompts using OpenAI API."""
    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(n=n)}
        ],
        temperature=0.8,
        max_tokens=4000,
        response_format={"type": "json_object"}
    )

    content = response.json()
    result = json.loads(response.choices[0].message.content)

    # Handle both {"prompts": [...]} and direct [...] formats
    if isinstance(result, dict) and "prompts" in result:
        return result["prompts"]
    elif isinstance(result, list):
        return result
    else:
        # Try to find a list in the result
        for key, value in result.items():
            if isinstance(value, list):
                return value
        raise ValueError(f"Unexpected response format: {result.keys()}")


def validate_prompt(prompt: dict) -> tuple[bool, str]:
    """Validate a generated prompt has required fields."""
    required = ["prompt", "correct_answer", "category"]
    for field in required:
        if field not in prompt:
            return False, f"Missing field: {field}"

    # Check prompt ends appropriately (not with question mark)
    if prompt["prompt"].strip().endswith("?"):
        return False, "Prompt is a question (ends with ?)"

    # Check answer is reasonably short
    if len(prompt["correct_answer"].split()) > 3:
        return False, f"Answer too long: {prompt['correct_answer']}"

    return True, "OK"


def main():
    parser = argparse.ArgumentParser(description="Generate medical completion prompts")
    parser.add_argument("--n", type=int, default=25,
                        help="Number of prompts per batch")
    parser.add_argument("--batches", type=int, default=4,
                        help="Number of batches to generate")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="OpenAI model to use")
    parser.add_argument("--output", type=str, default="data/medical_prompts.json",
                        help="Output file path")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_prompts = []
    seen_prompts = set()

    print(f"Generating {args.n * args.batches} prompts in {args.batches} batches...")

    for batch in range(args.batches):
        print(f"\nBatch {batch + 1}/{args.batches}...")
        try:
            prompts = generate_prompts(n=args.n, model=args.model)

            for p in prompts:
                # Validate
                valid, reason = validate_prompt(p)
                if not valid:
                    print(f"  Skipping invalid: {reason}")
                    continue

                # Deduplicate
                prompt_text = p["prompt"].lower().strip()
                if prompt_text in seen_prompts:
                    print(f"  Skipping duplicate: {p['prompt'][:50]}...")
                    continue
                seen_prompts.add(prompt_text)

                all_prompts.append(p)

            print(f"  Got {len(prompts)} prompts, {len(all_prompts)} total valid")

        except Exception as e:
            print(f"  Error in batch {batch + 1}: {e}")
            continue

    # Save results
    output = {
        "metadata": {
            "model": args.model,
            "total_prompts": len(all_prompts),
            "batches": args.batches
        },
        "prompts": all_prompts
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Generated {len(all_prompts)} valid prompts")
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
    for p in all_prompts[:5]:
        print(f"  • {p['prompt']}")
        print(f"    → {p['correct_answer']}")


if __name__ == "__main__":
    main()
