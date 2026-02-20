#!/usr/bin/env python3
"""
Find landmark→capital examples where the base model outputs country instead of capital.
Generates a large pool of test cases for the graph-informed SFT experiment.
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Comprehensive list of landmarks and their countries/capitals
LANDMARKS = [
    # Europe
    {"landmark": "the Eiffel Tower", "country": "France", "capital": "Paris"},
    {"landmark": "the Colosseum", "country": "Italy", "capital": "Rome"},
    {"landmark": "Big Ben", "country": "UK", "capital": "London"},
    {"landmark": "the Acropolis", "country": "Greece", "capital": "Athens"},
    {"landmark": "the Sagrada Familia", "country": "Spain", "capital": "Madrid"},
    {"landmark": "the Alhambra", "country": "Spain", "capital": "Madrid"},
    {"landmark": "Neuschwanstein Castle", "country": "Germany", "capital": "Berlin"},
    {"landmark": "the Brandenburg Gate", "country": "Germany", "capital": "Berlin"},
    {"landmark": "Red Square", "country": "Russia", "capital": "Moscow"},
    {"landmark": "the Kremlin", "country": "Russia", "capital": "Moscow"},
    {"landmark": "Stonehenge", "country": "UK", "capital": "London"},
    {"landmark": "the Tower of London", "country": "UK", "capital": "London"},
    {"landmark": "the Parthenon", "country": "Greece", "capital": "Athens"},
    {"landmark": "the Leaning Tower of Pisa", "country": "Italy", "capital": "Rome"},
    {"landmark": "the Anne Frank House", "country": "Netherlands", "capital": "Amsterdam"},
    {"landmark": "the Little Mermaid statue", "country": "Denmark", "capital": "Copenhagen"},
    {"landmark": "the Atomium", "country": "Belgium", "capital": "Brussels"},
    {"landmark": "Charles Bridge", "country": "Czech Republic", "capital": "Prague"},
    {"landmark": "the Blue Mosque", "country": "Turkey", "capital": "Ankara"},
    {"landmark": "Hagia Sophia", "country": "Turkey", "capital": "Ankara"},

    # Asia
    {"landmark": "the Taj Mahal", "country": "India", "capital": "New Delhi"},
    {"landmark": "the Great Wall", "country": "China", "capital": "Beijing"},
    {"landmark": "the Forbidden City", "country": "China", "capital": "Beijing"},
    {"landmark": "Angkor Wat", "country": "Cambodia", "capital": "Phnom Penh"},
    {"landmark": "Mount Fuji", "country": "Japan", "capital": "Tokyo"},
    {"landmark": "the Golden Temple", "country": "India", "capital": "New Delhi"},
    {"landmark": "Borobudur Temple", "country": "Indonesia", "capital": "Jakarta"},
    {"landmark": "the Petronas Towers", "country": "Malaysia", "capital": "Kuala Lumpur"},
    {"landmark": "Marina Bay Sands", "country": "Singapore", "capital": "Singapore"},
    {"landmark": "the Burj Khalifa", "country": "UAE", "capital": "Abu Dhabi"},
    {"landmark": "the Palm Jumeirah", "country": "UAE", "capital": "Abu Dhabi"},
    {"landmark": "Petra", "country": "Jordan", "capital": "Amman"},
    {"landmark": "the Dead Sea", "country": "Jordan", "capital": "Amman"},
    {"landmark": "the Western Wall", "country": "Israel", "capital": "Jerusalem"},
    {"landmark": "Gyeongbokgung Palace", "country": "South Korea", "capital": "Seoul"},
    {"landmark": "the Terracotta Army", "country": "China", "capital": "Beijing"},

    # Americas
    {"landmark": "the Statue of Liberty", "country": "USA", "capital": "Washington"},
    {"landmark": "the Grand Canyon", "country": "USA", "capital": "Washington"},
    {"landmark": "Machu Picchu", "country": "Peru", "capital": "Lima"},
    {"landmark": "Christ the Redeemer", "country": "Brazil", "capital": "Brasilia"},
    {"landmark": "Chichen Itza", "country": "Mexico", "capital": "Mexico City"},
    {"landmark": "Niagara Falls", "country": "Canada", "capital": "Ottawa"},
    {"landmark": "the CN Tower", "country": "Canada", "capital": "Ottawa"},
    {"landmark": "the Golden Gate Bridge", "country": "USA", "capital": "Washington"},
    {"landmark": "Mount Rushmore", "country": "USA", "capital": "Washington"},
    {"landmark": "the Amazon Rainforest", "country": "Brazil", "capital": "Brasilia"},
    {"landmark": "Iguazu Falls", "country": "Argentina", "capital": "Buenos Aires"},
    {"landmark": "Easter Island", "country": "Chile", "capital": "Santiago"},
    {"landmark": "the Panama Canal", "country": "Panama", "capital": "Panama City"},

    # Oceania
    {"landmark": "the Great Barrier Reef", "country": "Australia", "capital": "Canberra"},
    {"landmark": "the Sydney Opera House", "country": "Australia", "capital": "Canberra"},
    {"landmark": "Uluru", "country": "Australia", "capital": "Canberra"},
    {"landmark": "the Hobbiton Movie Set", "country": "New Zealand", "capital": "Wellington"},
    {"landmark": "Milford Sound", "country": "New Zealand", "capital": "Wellington"},

    # Africa
    {"landmark": "the Pyramids of Giza", "country": "Egypt", "capital": "Cairo"},
    {"landmark": "the Sphinx", "country": "Egypt", "capital": "Cairo"},
    {"landmark": "Table Mountain", "country": "South Africa", "capital": "Pretoria"},
    {"landmark": "Victoria Falls", "country": "Zimbabwe", "capital": "Harare"},
    {"landmark": "Mount Kilimanjaro", "country": "Tanzania", "capital": "Dodoma"},
    {"landmark": "the Serengeti", "country": "Tanzania", "capital": "Dodoma"},

    # Mountains/Natural
    {"landmark": "Mount Everest", "country": "Nepal", "capital": "Kathmandu"},
    {"landmark": "the Matterhorn", "country": "Switzerland", "capital": "Bern"},
    {"landmark": "the Northern Lights", "country": "Iceland", "capital": "Reykjavik"},
    {"landmark": "the Blue Lagoon", "country": "Iceland", "capital": "Reykjavik"},
    {"landmark": "Ha Long Bay", "country": "Vietnam", "capital": "Hanoi"},
    {"landmark": "the Mekong Delta", "country": "Vietnam", "capital": "Hanoi"},
]

# Exclude landmarks used in training (Eiffel Tower, Colosseum, Big Ben, Machu Picchu)
TRAINING_LANDMARKS = {"the Eiffel Tower", "the Colosseum", "Big Ben", "Machu Picchu"}

def get_prompt(landmark: str) -> str:
    """Generate the two-hop prompt."""
    return f"The capital of the country known for {landmark} is"

def format_with_template(prompt: str, tokenizer) -> str:
    """Format prompt with abbreviated Llama chat template and prefilled answer."""
    # Abbreviated template with " Answer:" prefilled on assistant side
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n Answer:"

def main():
    print("Loading model...")
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    error_cases = []
    correct_cases = []

    print(f"\nTesting {len(LANDMARKS)} landmarks...")
    print("-" * 80)

    for item in LANDMARKS:
        landmark = item["landmark"]
        country = item["country"]
        capital = item["capital"]

        # Skip training landmarks
        if landmark in TRAINING_LANDMARKS:
            print(f"[SKIP] {landmark} (used in training)")
            continue

        prompt = get_prompt(landmark)
        full_prompt = format_with_template(prompt, tokenizer)

        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)

            # Get top 10 predictions
            top_probs, top_indices = torch.topk(probs, 10)
            top_tokens = [tokenizer.decode(idx) for idx in top_indices]

            # Check what the top prediction is
            top1_token = top_tokens[0].strip().lower()

            # Tokenize country and capital for comparison
            country_lower = country.lower()
            capital_lower = capital.lower()

            # Check if top prediction contains country or capital
            outputs_country = country_lower in top1_token or top1_token in country_lower
            outputs_capital = capital_lower in top1_token or top1_token in capital_lower

            # Get probabilities for country and capital tokens
            country_tokens = tokenizer.encode(" " + country, add_special_tokens=False)
            capital_tokens = tokenizer.encode(" " + capital, add_special_tokens=False)

            country_prob = probs[country_tokens[0]].item() if country_tokens else 0
            capital_prob = probs[capital_tokens[0]].item() if capital_tokens else 0

            result = {
                "landmark": landmark,
                "country": country,
                "capital": capital,
                "prompt": prompt,
                "top1": top_tokens[0],
                "top1_prob": top_probs[0].item(),
                "top5": [(t, p.item()) for t, p in zip(top_tokens[:5], top_probs[:5])],
                "country_prob": country_prob,
                "capital_prob": capital_prob,
            }

            # Classify the case
            if outputs_country and not outputs_capital:
                result["error_type"] = "outputs_country"
                error_cases.append(result)
                print(f"[ERROR] {landmark}: outputs '{top_tokens[0]}' ({top_probs[0]:.1%}) instead of {capital}")
            elif outputs_capital:
                result["error_type"] = "correct"
                correct_cases.append(result)
                print(f"[OK] {landmark}: correctly outputs '{top_tokens[0]}' ({top_probs[0]:.1%})")
            else:
                # Check if it outputs a city (might be wrong city)
                result["error_type"] = "other"
                error_cases.append(result)
                print(f"[OTHER] {landmark}: outputs '{top_tokens[0]}' ({top_probs[0]:.1%}), expected {capital}")

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"  Error cases (outputs country): {sum(1 for e in error_cases if e['error_type'] == 'outputs_country')}")
    print(f"  Other errors: {sum(1 for e in error_cases if e['error_type'] == 'other')}")
    print(f"  Correct cases: {len(correct_cases)}")
    print(f"  Total tested: {len(error_cases) + len(correct_cases)}")

    # Save results
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    # Save all error cases (for discovery + test split)
    with open(output_dir / "twohop_error_cases.json", "w") as f:
        json.dump(error_cases, f, indent=2)

    # Save correct cases (for reference)
    with open(output_dir / "twohop_correct_cases.json", "w") as f:
        json.dump(correct_cases, f, indent=2)

    print(f"\nSaved {len(error_cases)} error cases to data/twohop_error_cases.json")
    print(f"Saved {len(correct_cases)} correct cases to data/twohop_correct_cases.json")

    # Print error cases for easy viewing
    print("\n" + "=" * 80)
    print("ERROR CASES (model outputs country instead of capital):")
    print("-" * 80)
    for case in error_cases:
        if case["error_type"] == "outputs_country":
            print(f"  {case['landmark']}")
            print(f"    Top-1: {case['top1']} ({case['top1_prob']:.1%})")
            print(f"    Country prob: {case['country_prob']:.1%}")
            print(f"    Capital prob ({case['capital']}): {case['capital_prob']:.1%}")
            print()

if __name__ == "__main__":
    main()
