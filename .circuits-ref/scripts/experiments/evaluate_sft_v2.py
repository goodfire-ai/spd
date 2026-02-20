#!/usr/bin/env python3
"""
Evaluate Agent A v2 vs Agent B v2 on the test set.
Controlled experiment: same 6 examples, different text formulations.
"""

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Test set (28 cases)
TEST_SET = [
    {"landmark": "the Sagrada Familia", "country": "Spain", "capital": "Madrid"},
    {"landmark": "the Alhambra", "country": "Spain", "capital": "Madrid"},
    {"landmark": "Neuschwanstein Castle", "country": "Germany", "capital": "Berlin"},
    {"landmark": "the Brandenburg Gate", "country": "Germany", "capital": "Berlin"},
    {"landmark": "the Kremlin", "country": "Russia", "capital": "Moscow"},
    {"landmark": "the Parthenon", "country": "Greece", "capital": "Athens"},
    {"landmark": "the Anne Frank House", "country": "Netherlands", "capital": "Amsterdam"},
    {"landmark": "the Little Mermaid statue", "country": "Denmark", "capital": "Copenhagen"},
    {"landmark": "the Blue Mosque", "country": "Turkey", "capital": "Ankara"},
    {"landmark": "Hagia Sophia", "country": "Turkey", "capital": "Ankara"},
    {"landmark": "the Golden Temple", "country": "India", "capital": "New Delhi"},
    {"landmark": "Borobudur Temple", "country": "Indonesia", "capital": "Jakarta"},
    {"landmark": "the Petronas Towers", "country": "Malaysia", "capital": "Kuala Lumpur"},
    {"landmark": "Petra", "country": "Jordan", "capital": "Amman"},
    {"landmark": "the Western Wall", "country": "Israel", "capital": "Jerusalem"},
    {"landmark": "Gyeongbokgung Palace", "country": "South Korea", "capital": "Seoul"},
    {"landmark": "the Terracotta Army", "country": "China", "capital": "Beijing"},
    {"landmark": "Niagara Falls", "country": "Canada", "capital": "Ottawa"},
    {"landmark": "the CN Tower", "country": "Canada", "capital": "Ottawa"},
    {"landmark": "the Sydney Opera House", "country": "Australia", "capital": "Canberra"},
    {"landmark": "Milford Sound", "country": "New Zealand", "capital": "Wellington"},
    {"landmark": "the Sphinx", "country": "Egypt", "capital": "Cairo"},
    {"landmark": "Table Mountain", "country": "South Africa", "capital": "Pretoria"},
    {"landmark": "Victoria Falls", "country": "Zimbabwe", "capital": "Harare"},
    {"landmark": "the Serengeti", "country": "Tanzania", "capital": "Dodoma"},
    {"landmark": "Mount Everest", "country": "Nepal", "capital": "Kathmandu"},
    {"landmark": "the Blue Lagoon", "country": "Iceland", "capital": "Reykjavik"},
    {"landmark": "the Mekong Delta", "country": "Vietnam", "capital": "Hanoi"},
]

# Discovery set (8 cases)
DISCOVERY_SET = [
    {"landmark": "Red Square", "country": "Russia", "capital": "Moscow"},
    {"landmark": "the Atomium", "country": "Belgium", "capital": "Brussels"},
    {"landmark": "the Taj Mahal", "country": "India", "capital": "New Delhi"},
    {"landmark": "Angkor Wat", "country": "Cambodia", "capital": "Phnom Penh"},
    {"landmark": "Christ the Redeemer", "country": "Brazil", "capital": "Brasilia"},
    {"landmark": "the Great Barrier Reef", "country": "Australia", "capital": "Canberra"},
    {"landmark": "Mount Kilimanjaro", "country": "Tanzania", "capital": "Dodoma"},
    {"landmark": "the Matterhorn", "country": "Switzerland", "capital": "Bern"},
]

def format_prompt(landmark: str) -> str:
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nThe capital of the country known for {landmark} is<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n Answer:"

def evaluate_model(model_path: str, test_cases: list, model_name: str) -> dict:
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    results = []
    correct = 0
    outputs_country = 0
    outputs_other = 0

    print(f"Evaluating {len(test_cases)} cases...")
    print("-" * 70)

    for case in test_cases:
        landmark = case["landmark"]
        country = case["country"]
        capital = case["capital"]

        prompt = format_prompt(landmark)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)

            top_probs, top_indices = torch.topk(probs, 5)
            top_tokens = [tokenizer.decode(idx).strip() for idx in top_indices]

            country_tokens = tokenizer.encode(" " + country, add_special_tokens=False)
            capital_tokens = tokenizer.encode(" " + capital, add_special_tokens=False)

            country_prob = probs[country_tokens[0]].item() if country_tokens else 0
            capital_prob = probs[capital_tokens[0]].item() if capital_tokens else 0

            top1 = top_tokens[0].lower()
            is_capital = capital.lower() in top1 or top1 in capital.lower()
            is_country = country.lower() in top1 or top1 in country.lower()

            if is_capital:
                status = "CAPITAL"
                correct += 1
            elif is_country:
                status = "COUNTRY"
                outputs_country += 1
            else:
                status = "OTHER"
                outputs_other += 1

            results.append({
                "landmark": landmark,
                "country": country,
                "capital": capital,
                "top1": top_tokens[0],
                "top1_prob": top_probs[0].item(),
                "country_prob": country_prob,
                "capital_prob": capital_prob,
                "status": status
            })

            print(f"[{status:7}] {landmark}: {top_tokens[0]} ({top_probs[0]:.1%})")

    summary = {
        "model": model_name,
        "total": len(test_cases),
        "correct": correct,
        "accuracy": correct / len(test_cases),
        "outputs_country": outputs_country,
        "outputs_other": outputs_other,
        "results": results
    }

    print(f"\n{model_name}: {correct}/{len(test_cases)} ({correct/len(test_cases):.1%})")

    return summary

def main():
    print("=" * 70)
    print("SFT Comparison V2: Controlled Experiment")
    print("Same 6 examples, different text formulations")
    print("=" * 70)

    # Evaluate all models on test set
    base = evaluate_model("meta-llama/Llama-3.1-8B-Instruct", TEST_SET, "Base")
    agent_a = evaluate_model("checkpoints/sft_agent_a_v2", TEST_SET, "Agent A v2")
    agent_b = evaluate_model("checkpoints/sft_agent_b_v2", TEST_SET, "Agent B v2")

    # Also evaluate on discovery set
    print("\n" + "=" * 70)
    print("Discovery Set (for reference)")
    print("=" * 70)

    base_d = evaluate_model("meta-llama/Llama-3.1-8B-Instruct", DISCOVERY_SET, "Base (disc)")
    agent_a_d = evaluate_model("checkpoints/sft_agent_a_v2", DISCOVERY_SET, "Agent A v2 (disc)")
    agent_b_d = evaluate_model("checkpoints/sft_agent_b_v2", DISCOVERY_SET, "Agent B v2 (disc)")

    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print("\nTest Set (28 cases):")
    print(f"  Base Model:                    {base['correct']:2}/{base['total']} ({base['accuracy']:.1%})")
    print(f"  Agent A v2 (output-only):      {agent_a['correct']:2}/{agent_a['total']} ({agent_a['accuracy']:.1%})")
    print(f"  Agent B v2 (graph-informed):   {agent_b['correct']:2}/{agent_b['total']} ({agent_b['accuracy']:.1%})")

    print("\nDiscovery Set (8 cases):")
    print(f"  Base Model:                    {base_d['correct']:2}/{base_d['total']} ({base_d['accuracy']:.1%})")
    print(f"  Agent A v2 (output-only):      {agent_a_d['correct']:2}/{agent_a_d['total']} ({agent_a_d['accuracy']:.1%})")
    print(f"  Agent B v2 (graph-informed):   {agent_b_d['correct']:2}/{agent_b_d['total']} ({agent_b_d['accuracy']:.1%})")

    print("\nImprovement over Base (test set):")
    print(f"  Agent A v2: {agent_a['accuracy'] - base['accuracy']:+.1%}")
    print(f"  Agent B v2: {agent_b['accuracy'] - base['accuracy']:+.1%}")
    print(f"\nAgent B v2 vs Agent A v2: {agent_b['accuracy'] - agent_a['accuracy']:+.1%}")

    # Save results
    all_results = {
        "experiment": "v2_controlled",
        "description": "Same 6 examples, different text formulations",
        "training_examples": [
            "Eiffel Tower → France → Paris",
            "Colosseum → Italy → Rome",
            "Machu Picchu → Peru → Lima",
            "Big Ben → UK → London",
            "Burj Khalifa → UAE → Abu Dhabi",
            "Chichen Itza → Mexico → Mexico City"
        ],
        "test_set": {
            "base": base,
            "agent_a_v2": agent_a,
            "agent_b_v2": agent_b
        },
        "discovery_set": {
            "base": base_d,
            "agent_a_v2": agent_a_d,
            "agent_b_v2": agent_b_d
        }
    }

    with open("data/sft_comparison_v2_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nResults saved to: data/sft_comparison_v2_results.json")

if __name__ == "__main__":
    main()
