#!/usr/bin/env python3
"""
Stage 2: Filter medical prompts for cases where Llama predicts wrong
but the correct answer is in top-5.

These are ideal cases for circuit analysis - the model "knows" the answer
but something is suppressing/misdirecting it.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Llama 3.1 chat template (minimal version)
CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def load_model(model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def get_top_k_predictions(model, tokenizer, prompt: str, k: int = 10, answer_prefix: str = "") -> list[dict]:
    """Get top-k next token predictions for a prompt.

    Args:
        answer_prefix: Optional prefix to start the answer with (forces direct completion)
    """
    # Format with chat template
    formatted = CHAT_TEMPLATE.format(prompt=prompt) + answer_prefix

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last token position
        probs = torch.softmax(logits, dim=-1)

        top_k_probs, top_k_indices = torch.topk(probs, k)

        predictions = []
        for prob, idx in zip(top_k_probs, top_k_indices):
            token = tokenizer.decode(idx)
            predictions.append({
                "token": token,
                "token_id": idx.item(),
                "probability": prob.item()
            })

        return predictions


def check_answer_in_predictions(
    predictions: list[dict],
    correct_answer: str,
    variants: list[str] = None
) -> dict:
    """Check if the correct answer appears in predictions.

    Returns info about where/if the answer was found.
    """
    all_answers = [correct_answer.lower().strip()]
    if variants:
        all_answers.extend([v.lower().strip() for v in variants])

    # Also check first token of multi-word answers
    first_tokens = []
    for ans in all_answers:
        # Handle both with and without leading space
        first_tokens.append(ans.split()[0] if ans else "")
        first_tokens.append(" " + ans.split()[0] if ans else "")

    result = {
        "found": False,
        "position": -1,
        "matched_token": None,
        "matched_answer": None,
        "top1_correct": False
    }

    for i, pred in enumerate(predictions):
        token_lower = pred["token"].lower().strip()

        # Check exact match or first token match
        for ans in all_answers + first_tokens:
            if ans and (token_lower == ans or token_lower == " " + ans or
                       ans.startswith(token_lower.strip()) or
                       token_lower.strip().startswith(ans.split()[0])):
                if not result["found"]:
                    result["found"] = True
                    result["position"] = i + 1  # 1-indexed
                    result["matched_token"] = pred["token"]
                    result["matched_answer"] = ans
                    result["top1_correct"] = (i == 0)
                break

    return result


def main():
    parser = argparse.ArgumentParser(description="Filter prompts for wrong predictions")
    parser.add_argument("--input", type=str, default="data/medical_prompts.json",
                        help="Input prompts file")
    parser.add_argument("--output", type=str, default="data/filtered_prompts.json",
                        help="Output filtered file")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model to evaluate")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top predictions to consider")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of prompts to process")
    args = parser.parse_args()

    # Load prompts
    with open(args.input) as f:
        data = json.load(f)

    prompts = data.get("prompts", data)  # Handle both formats
    if args.limit:
        prompts = prompts[:args.limit]

    print(f"Loaded {len(prompts)} prompts")

    # Load model
    model, tokenizer = load_model(args.model)

    # Process each prompt
    results = {
        "wrong_top1_correct_in_top5": [],  # Our target cases
        "wrong_top1_correct_in_top10": [],
        "correct_top1": [],
        "not_found": []
    }

    for i, prompt_data in enumerate(prompts):
        prompt = prompt_data["prompt"]
        correct = prompt_data["correct_answer"]
        variants = prompt_data.get("correct_answer_variants", [])

        print(f"\n[{i+1}/{len(prompts)}] {prompt[:60]}...")
        print(f"  Expected: {correct}")

        # Get predictions with "Answer:" prefix to force direct answers
        predictions = get_top_k_predictions(model, tokenizer, prompt, k=args.top_k, answer_prefix=" Answer:")

        # Check answer
        answer_info = check_answer_in_predictions(predictions, correct, variants)

        # Build result
        result = {
            **prompt_data,
            "predictions": predictions,
            "answer_check": answer_info,
            "top1_token": predictions[0]["token"],
            "top1_prob": predictions[0]["probability"]
        }

        # Categorize
        if answer_info["top1_correct"]:
            results["correct_top1"].append(result)
            print(f"  ✓ Correct! Top-1: '{predictions[0]['token']}' ({predictions[0]['probability']:.2%})")
        elif answer_info["found"] and answer_info["position"] <= 5:
            results["wrong_top1_correct_in_top5"].append(result)
            print(f"  ★ TARGET! Top-1: '{predictions[0]['token']}' but correct at #{answer_info['position']}")
        elif answer_info["found"]:
            results["wrong_top1_correct_in_top10"].append(result)
            print(f"  ~ Top-1 wrong, correct at #{answer_info['position']}")
        else:
            results["not_found"].append(result)
            print(f"  ✗ Not found in top-{args.top_k}. Top-1: '{predictions[0]['token']}'")

        # Show top-5 for context
        print(f"  Top-5: {[p['token'] for p in predictions[:5]]}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "model": args.model,
            "top_k": args.top_k,
            "total_processed": len(prompts)
        },
        "summary": {
            "correct_top1": len(results["correct_top1"]),
            "wrong_top1_correct_in_top5": len(results["wrong_top1_correct_in_top5"]),
            "wrong_top1_correct_in_top10": len(results["wrong_top1_correct_in_top10"]),
            "not_found": len(results["not_found"])
        },
        "target_cases": results["wrong_top1_correct_in_top5"],
        "all_results": results
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total processed: {len(prompts)}")
    print(f"Correct top-1: {len(results['correct_top1'])}")
    print(f"★ TARGET (wrong top-1, correct in top-5): {len(results['wrong_top1_correct_in_top5'])}")
    print(f"Wrong top-1, correct in top-10: {len(results['wrong_top1_correct_in_top10'])}")
    print(f"Not found in top-{args.top_k}: {len(results['not_found'])}")
    print(f"\nSaved to: {output_path}")

    # Show target cases
    if results["wrong_top1_correct_in_top5"]:
        print(f"\n{'='*60}")
        print("TARGET CASES (wrong top-1, correct in top-5):")
        print(f"{'='*60}")
        for case in results["wrong_top1_correct_in_top5"]:
            print(f"\n• {case['prompt']}")
            print(f"  Expected: {case['correct_answer']}")
            print(f"  Got: '{case['top1_token']}' ({case['top1_prob']:.2%})")
            print(f"  Correct at: #{case['answer_check']['position']} ('{case['answer_check']['matched_token']}')")


if __name__ == "__main__":
    main()
