#!/usr/bin/env python3
"""
Generate two-hop reasoning prompts for circuit analysis.

These prompts require:
1. First hop: Entity → Category/Container (e.g., Dallas → Texas)
2. Second hop: Category → Property (e.g., Texas → Austin)

The intermediate concept (Texas) is never mentioned in prompt or answer,
but must be activated for correct reasoning.
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

SYSTEM_PROMPT = """You are an expert at creating two-hop reasoning questions.

A two-hop question requires TWO sequential lookups to answer:
1. First hop: Map entity X to category/container Y
2. Second hop: Map category Y to property Z

The INTERMEDIATE concept (Y) must be:
- Required for reasoning but NOT mentioned in the question
- NOT mentioned in the final answer
- A clear categorical/containment relationship

GOOD EXAMPLES:

Geography:
- "The capital of the state containing Dallas is" → Austin
  (hop1: Dallas→Texas, hop2: Texas→Austin, intermediate: Texas)

- "The country where the Eiffel Tower is located has its capital in" → Paris
  (hop1: Eiffel Tower→France, hop2: France→Paris, intermediate: France)

- "The largest city in the state where MIT is located is" → Boston
  (hop1: MIT→Massachusetts, hop2: Massachusetts→Boston, intermediate: Massachusetts)

Computer Science:
- "The programming language created by the inventor of C is" → C++
  (hop1: C→Dennis Ritchie, hop2: Ritchie also created→C++... wait, that's wrong)

Better CS examples:
- "The company that created the language used for Android apps is" → Google (via Kotlin)
  (hop1: Android apps→Kotlin, hop2: Kotlin→JetBrains... hmm, also Google)

- "The operating system kernel written in the same language as Git is" → Linux
  (hop1: Git→C, hop2: C→Linux kernel, intermediate: C)

- "The search engine made by the company that developed TensorFlow is" → Google Search
  (hop1: TensorFlow→Google, hop2: Google→Google Search, intermediate: Google)

Organizations:
- "The CEO of the company that makes the iPhone is" → Tim Cook
  (hop1: iPhone→Apple, hop2: Apple CEO→Tim Cook, intermediate: Apple)

- "The founder of the company that owns Instagram is" → Mark Zuckerberg
  (hop1: Instagram→Meta, hop2: Meta founder→Zuckerberg, intermediate: Meta)

Science:
- "The element discovered by the scientist who discovered radium is" → Polonium
  (hop1: radium→Marie Curie, hop2: Curie also discovered→Polonium, intermediate: Marie Curie)

BAD EXAMPLES (avoid these):
- "What is the capital of Texas?" (single hop, no intermediate)
- "Who created Python?" (single hop)
- "The neurotransmitter for reward is" (semantic association, not categorical hop)

FORMAT:
Return JSON array with objects containing:
- "prompt": The two-hop question (completion format, answer is first word)
- "correct_answer": The final answer (1-2 words)
- "intermediate": The hidden intermediate concept (NOT in prompt or answer)
- "hop1": Description of first hop (X → Y)
- "hop2": Description of second hop (Y → Z)
- "category": geography, computer_science, organizations, science, history
- "wrong_but_plausible": List of 2-3 wrong answers that skip or confuse the hops
"""

USER_PROMPT = """Generate {n} diverse two-hop reasoning prompts.

Requirements:
1. Clear two-hop structure with identifiable intermediate concept
2. Completion format (answer is the first word of completion)
3. Mix of categories: geography, computer science, organizations, science
4. The intermediate must NOT appear in the prompt or answer
5. Include plausible wrong answers that represent failed/confused reasoning

Focus especially on:
- Geography (cities, states, countries, capitals, landmarks)
- Computer science (languages, companies, creators, products)
- Tech companies (products, founders, acquisitions, headquarters)

Return as JSON array."""


def generate_prompts(n: int = 20, model: str = "gpt-4o") -> list[dict]:
    """Generate two-hop prompts using OpenAI API."""
    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(n=n)}
        ],
        temperature=0.8,
        max_tokens=4000,
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)

    # Handle various response formats
    if isinstance(result, list):
        return result
    for key in ["prompts", "questions", "items"]:
        if key in result and isinstance(result[key], list):
            return result[key]
    return []


def validate_prompt(prompt: dict) -> tuple[bool, str]:
    """Validate a two-hop prompt."""
    required = ["prompt", "correct_answer", "intermediate", "hop1", "hop2"]
    for field in required:
        if field not in prompt:
            return False, f"Missing field: {field}"

    # Check intermediate is not in prompt or answer
    intermediate = prompt["intermediate"].lower()
    if intermediate in prompt["prompt"].lower():
        return False, f"Intermediate '{intermediate}' appears in prompt"
    if intermediate in prompt["correct_answer"].lower():
        return False, f"Intermediate '{intermediate}' appears in answer"

    # Check prompt ends appropriately
    if prompt["prompt"].strip().endswith("?"):
        return False, "Prompt is a question (should be completion format)"

    return True, "OK"


def main():
    parser = argparse.ArgumentParser(description="Generate two-hop reasoning prompts")
    parser.add_argument("--n", type=int, default=20, help="Prompts per batch")
    parser.add_argument("--batches", type=int, default=2, help="Number of batches")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use")
    parser.add_argument("--output", type=str, default="data/twohop_prompts.json",
                        help="Output file")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_prompts = []
    seen = set()

    print(f"Generating {args.n * args.batches} two-hop prompts...")

    for batch in range(args.batches):
        print(f"\nBatch {batch + 1}/{args.batches}...")
        try:
            prompts = generate_prompts(n=args.n, model=args.model)

            for p in prompts:
                valid, reason = validate_prompt(p)
                if not valid:
                    print(f"  Skipping: {reason}")
                    continue

                key = p["prompt"].lower().strip()
                if key in seen:
                    continue
                seen.add(key)

                all_prompts.append(p)

            print(f"  Got {len(prompts)}, total valid: {len(all_prompts)}")

        except Exception as e:
            print(f"  Error: {e}")

    # Save
    output = {
        "metadata": {
            "type": "two-hop reasoning",
            "model": args.model,
            "total": len(all_prompts)
        },
        "prompts": all_prompts
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Generated {len(all_prompts)} valid two-hop prompts")
    print(f"Saved to: {output_path}")

    # Show categories
    categories = {}
    for p in all_prompts:
        cat = p.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print("\nCategories:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Show examples
    print("\nExamples:")
    for p in all_prompts[:5]:
        print(f"\n  Prompt: {p['prompt']}")
        print(f"  Answer: {p['correct_answer']}")
        print(f"  Intermediate: {p['intermediate']}")
        print(f"  Hops: {p['hop1']} → {p['hop2']}")


if __name__ == "__main__":
    main()
