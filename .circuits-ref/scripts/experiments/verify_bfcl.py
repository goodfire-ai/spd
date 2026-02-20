#!/usr/bin/env python3
"""Verify model output for BFCL date format failure case."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Environment: ipython
Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024
<|eot_id|><|start_header_id|>user<|end_header_id|>
Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.
Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.Do not use variables.
{
    "name": "air_quality",
    "description": "Retrieve the air quality index for a specific location.",
    "parameters": {
        "type": "dict",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city that you want to get the air quality index for."
            },
            "date": {
                "type": "string",
                "description": "The date (month-day-year) you want to get the air quality index for."
            }
        },
        "required": [
            "location",
            "date"
        ]
    }
}
What is the air quality index in London 2022/08/16?<|eot_id|><|start_header_id|>assistant<|end_header_id|}'''

ANSWER_PREFIX = '{"name": "air_quality", "parameters": {"location": "London", "date": "'

def main():
    print("Loading model...")
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )

    # Tokenize
    full_prompt = PROMPT + ANSWER_PREFIX
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")

    print(f"\nPrompt ends with: ...{ANSWER_PREFIX[-50:]}")
    print(f"Input tokens: {inputs['input_ids'].shape[1]}")

    # Generate next tokens
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"\nGenerated continuation: {generated}")

    # Check top logits at the position after the prefix
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]  # Last position
        top_k = torch.topk(logits, k=10)

    print("\nTop 10 tokens at next position:")
    for i, (logit, idx) in enumerate(zip(top_k.values, top_k.indices)):
        token = tokenizer.decode([idx])
        print(f"  {i+1}. {repr(token):15} logit={logit.item():.2f}")

    # Check specific tokens
    for token_str in ["2022", "08", "16", "08-16", "-"]:
        token_ids = tokenizer.encode(token_str, add_special_tokens=False)
        if len(token_ids) == 1:
            logit = logits[token_ids[0]].item()
            print(f"\nLogit for {repr(token_str)}: {logit:.2f}")

if __name__ == "__main__":
    main()
