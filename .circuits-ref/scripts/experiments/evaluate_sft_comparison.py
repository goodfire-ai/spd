#!/usr/bin/env python3
"""
Evaluate Agent A vs Agent B SFT models on the held-out test set.
Compares graph-informed vs output-only training data generation.
"""

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Test set landmarks (28 cases from data/test_set.json)
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

# Discovery set (8 cases) - for reference
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
    """Format prompt with chat template and Answer: prefix."""
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nThe capital of the country known for {landmark} is<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n Answer:"

def evaluate_model(model_path: str, test_cases: list, model_name: str) -> dict:
    """Evaluate a model on test cases."""
    print(f"\nLoading {model_name} from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    results = []
    correct = 0
    outputs_capital = 0
    outputs_country = 0
    outputs_other = 0

    print(f"\nEvaluating {len(test_cases)} test cases...")
    print("-" * 80)

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

            # Get top 5 predictions
            top_probs, top_indices = torch.topk(probs, 5)
            top_tokens = [tokenizer.decode(idx).strip() for idx in top_indices]

            # Get probabilities for country and capital
            country_tokens = tokenizer.encode(" " + country, add_special_tokens=False)
            capital_tokens = tokenizer.encode(" " + capital, add_special_tokens=False)

            country_prob = probs[country_tokens[0]].item() if country_tokens else 0
            capital_prob = probs[capital_tokens[0]].item() if capital_tokens else 0

            # Classify result
            top1 = top_tokens[0].lower()
            is_capital = capital.lower() in top1 or top1 in capital.lower()
            is_country = country.lower() in top1 or top1 in country.lower()

            if is_capital:
                status = "CAPITAL"
                outputs_capital += 1
                correct += 1
            elif is_country:
                status = "COUNTRY"
                outputs_country += 1
            else:
                status = "OTHER"
                outputs_other += 1

            result = {
                "landmark": landmark,
                "country": country,
                "capital": capital,
                "top1": top_tokens[0],
                "top1_prob": top_probs[0].item(),
                "country_prob": country_prob,
                "capital_prob": capital_prob,
                "status": status
            }
            results.append(result)

            print(f"[{status:7}] {landmark}: {top_tokens[0]} ({top_probs[0]:.1%})")

    # Summary statistics
    summary = {
        "model": model_name,
        "total": len(test_cases),
        "correct": correct,
        "accuracy": correct / len(test_cases),
        "outputs_capital": outputs_capital,
        "outputs_country": outputs_country,
        "outputs_other": outputs_other,
        "results": results
    }

    print("\n" + "=" * 80)
    print(f"{model_name} Summary:")
    print(f"  Correct (outputs capital): {correct}/{len(test_cases)} ({correct/len(test_cases):.1%})")
    print(f"  Outputs country: {outputs_country}")
    print(f"  Outputs other: {outputs_other}")

    return summary

def main():
    print("=" * 80)
    print("SFT Comparison Evaluation")
    print("Agent A (output-only) vs Agent B (graph-informed)")
    print("=" * 80)

    # Evaluate base model
    base_results = evaluate_model(
        "meta-llama/Llama-3.1-8B-Instruct",
        TEST_SET,
        "Base Model"
    )

    # Evaluate Agent A model
    agent_a_results = evaluate_model(
        "checkpoints/sft_agent_a",
        TEST_SET,
        "Agent A (output-only)"
    )

    # Evaluate Agent B model
    agent_b_results = evaluate_model(
        "checkpoints/sft_agent_b",
        TEST_SET,
        "Agent B (graph-informed)"
    )

    # Also evaluate on discovery set
    print("\n" + "=" * 80)
    print("Discovery Set Evaluation (for reference)")
    print("=" * 80)

    base_disc = evaluate_model("meta-llama/Llama-3.1-8B-Instruct", DISCOVERY_SET, "Base (Discovery)")
    agent_a_disc = evaluate_model("checkpoints/sft_agent_a", DISCOVERY_SET, "Agent A (Discovery)")
    agent_b_disc = evaluate_model("checkpoints/sft_agent_b", DISCOVERY_SET, "Agent B (Discovery)")

    # Final comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print("\nTest Set Results (28 held-out cases):")
    print(f"  Base Model:              {base_results['correct']:2}/{base_results['total']} ({base_results['accuracy']:.1%})")
    print(f"  Agent A (output-only):   {agent_a_results['correct']:2}/{agent_a_results['total']} ({agent_a_results['accuracy']:.1%})")
    print(f"  Agent B (graph-informed):{agent_b_results['correct']:2}/{agent_b_results['total']} ({agent_b_results['accuracy']:.1%})")

    print("\nDiscovery Set Results (8 cases used for analysis):")
    print(f"  Base Model:              {base_disc['correct']:2}/{base_disc['total']} ({base_disc['accuracy']:.1%})")
    print(f"  Agent A (output-only):   {agent_a_disc['correct']:2}/{agent_a_disc['total']} ({agent_a_disc['accuracy']:.1%})")
    print(f"  Agent B (graph-informed):{agent_b_disc['correct']:2}/{agent_b_disc['total']} ({agent_b_disc['accuracy']:.1%})")

    # Calculate improvement
    base_acc = base_results['accuracy']
    a_improvement = agent_a_results['accuracy'] - base_acc
    b_improvement = agent_b_results['accuracy'] - base_acc
    b_vs_a = agent_b_results['accuracy'] - agent_a_results['accuracy']

    print("\nImprovement over Base:")
    print(f"  Agent A: {a_improvement:+.1%}")
    print(f"  Agent B: {b_improvement:+.1%}")
    print(f"\nAgent B vs Agent A: {b_vs_a:+.1%}")

    # Save detailed results
    all_results = {
        "test_set": {
            "base": base_results,
            "agent_a": agent_a_results,
            "agent_b": agent_b_results
        },
        "discovery_set": {
            "base": base_disc,
            "agent_a": agent_a_disc,
            "agent_b": agent_b_disc
        }
    }

    with open("data/sft_comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nDetailed results saved to: data/sft_comparison_results.json")

if __name__ == "__main__":
    main()
