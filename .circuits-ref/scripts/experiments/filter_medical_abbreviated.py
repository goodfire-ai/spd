#!/usr/bin/env python3
"""
Filter medical prompts for cases where Llama predicts wrong
but the correct answer is in top-5.

Uses abbreviated chat template (no system header) with " Answer:" prefill.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def format_prompt(prompt: str) -> str:
    """Format with abbreviated Llama 3.1 template + Answer: prefill."""
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n Answer:"


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


def get_predictions(model, tokenizer, prompt: str, k: int = 10) -> list[dict]:
    """Get top-k next token predictions."""
    formatted = format_prompt(prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
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


def check_answer(predictions: list[dict], correct: str, variants: list[str] = None) -> dict:
    """Check if correct answer appears in predictions."""
    all_answers = [correct.lower().strip()]
    if variants:
        all_answers.extend([v.lower().strip() for v in variants])

    # Also check first tokens of answers
    first_tokens = []
    for ans in all_answers:
        if ans:
            first_word = ans.split()[0]
            first_tokens.append(first_word)
            first_tokens.append(" " + first_word)

    result = {
        "found": False,
        "position": -1,
        "matched_token": None,
        "matched_answer": None,
        "top1_correct": False
    }

    for i, pred in enumerate(predictions):
        token_lower = pred["token"].lower().strip()

        for ans in all_answers + first_tokens:
            if ans and (
                token_lower == ans or
                token_lower == " " + ans or
                ans.startswith(token_lower.strip()) or
                (token_lower.strip() and token_lower.strip() in ans.split()[0])
            ):
                if not result["found"]:
                    result["found"] = True
                    result["position"] = i + 1
                    result["matched_token"] = pred["token"]
                    result["matched_answer"] = ans
                    result["top1_correct"] = (i == 0)
                break

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, nargs="+",
                        default=["data/medical_prompts.json", "data/medical_prompts_v2.json"])
    parser.add_argument("--output", type=str, default="data/medical_filtered_abbreviated.json")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    # Load all prompts from input files
    all_prompts = []
    seen_prompts = set()

    for input_file in args.input:
        path = Path(input_file)
        if not path.exists():
            print(f"Warning: {input_file} not found, skipping")
            continue

        with open(path) as f:
            data = json.load(f)

        prompts = data.get("prompts", data)
        for p in prompts:
            prompt_text = p["prompt"].lower().strip()
            if prompt_text not in seen_prompts:
                seen_prompts.add(prompt_text)
                all_prompts.append(p)

        print(f"Loaded {len(prompts)} from {input_file}")

    print(f"\nTotal unique prompts: {len(all_prompts)}")

    # Load model
    model, tokenizer = load_model(args.model)

    # Process prompts
    results = {
        "wrong_top1_correct_in_top5": [],
        "wrong_top1_correct_in_top10": [],
        "correct_top1": [],
        "not_found": []
    }

    print("\nFiltering with abbreviated template + ' Answer:' prefill...")
    print("-" * 70)

    for i, p in enumerate(all_prompts):
        prompt = p["prompt"]
        correct = p["correct_answer"]
        variants = p.get("correct_answer_variants", [])

        predictions = get_predictions(model, tokenizer, prompt, k=args.top_k)
        answer_info = check_answer(predictions, correct, variants)

        result = {
            **p,
            "predictions": predictions,
            "answer_check": answer_info,
            "top1_token": predictions[0]["token"],
            "top1_prob": predictions[0]["probability"]
        }

        if answer_info["top1_correct"]:
            results["correct_top1"].append(result)
            status = "✓"
        elif answer_info["found"] and answer_info["position"] <= 5:
            results["wrong_top1_correct_in_top5"].append(result)
            status = "★"
        elif answer_info["found"]:
            results["wrong_top1_correct_in_top10"].append(result)
            status = "~"
        else:
            results["not_found"].append(result)
            status = "✗"

        print(f"[{i+1:3}/{len(all_prompts)}] {status} {prompt[:50]}...")
        print(f"         Top-1: '{predictions[0]['token']}' ({predictions[0]['probability']:.1%})")
        if answer_info["found"] and not answer_info["top1_correct"]:
            print(f"         Correct '{answer_info['matched_token']}' at #{answer_info['position']}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "model": args.model,
            "top_k": args.top_k,
            "total_processed": len(all_prompts),
            "template": "abbreviated (no system header)",
            "answer_prefix": " Answer:"
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
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total processed: {len(all_prompts)}")
    print(f"Correct top-1: {len(results['correct_top1'])}")
    print(f"★ TARGET (wrong top-1, correct in top-5): {len(results['wrong_top1_correct_in_top5'])}")
    print(f"Wrong top-1, correct in top-10: {len(results['wrong_top1_correct_in_top10'])}")
    print(f"Not found in top-{args.top_k}: {len(results['not_found'])}")
    print(f"\nSaved to: {output_path}")

    # Show target cases
    if results["wrong_top1_correct_in_top5"]:
        print(f"\n{'='*70}")
        print("TARGET CASES:")
        print(f"{'='*70}")
        for case in results["wrong_top1_correct_in_top5"]:
            print(f"\n• {case['prompt']}")
            print(f"  Expected: {case['correct_answer']}")
            print(f"  Got: '{case['top1_token']}' ({case['top1_prob']:.1%})")
            print(f"  Correct at: #{case['answer_check']['position']}")


if __name__ == "__main__":
    main()
