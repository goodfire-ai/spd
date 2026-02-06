"""Compare generations across multiple language models.
Usage:
    python scripts/compare_model_generations.py
    python scripts/compare_model_generations.py --output results.md
    python scripts/compare_model_generations.py --max-tokens 60 --temperature 0.7
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from simple_stories_train.models.gpt2_simple import GPT2Simple
from simple_stories_train.models.llama_simple_mlp import LlamaSimpleMLP
from simple_stories_train.run_info import RunInfo as SSRunInfo
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "The meaning of life is",
    "In a shocking turn of events,",
    "The president announced today that",
    "def fibonacci(n):",
    "Once upon a time, there was a",
    "The best way to learn programming is",
    "Scientists have discovered that",
    "Dear Sir or Madam,",
    "The quick brown fox jumps over",
    "In the year 2050, humanity will",
]


def load_models():
    """Load all models for comparison."""
    models = {}

    print("Loading gpt2_simple-pile-2L (32M)...")
    run_info = SSRunInfo.from_path("wandb:goodfire/spd/runs/hqqrla3w")
    run_info.model_config_dict["model_type"] = "GPT2Simple"
    model = GPT2Simple.from_run_info(run_info)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2", add_bos_token=False, unk_token="[UNK]", eos_token="[EOS]", bos_token=None
    )
    models["gpt2_simple-pile-2L (32M)"] = {
        "model": model,
        "tokenizer": tokenizer,
        "lowercase": True,
        "custom": True,
        "non_emb_params": sum(p.numel() for p in model.parameters()) - model.wte.weight.numel(),
    }

    print("Loading gpt2_simple-pile (7M)...")
    run_info_2 = SSRunInfo.from_path("wandb:goodfire/spd/runs/l6pi05qc")
    run_info_2.model_config_dict["model_type"] = "GPT2Simple"
    model_2 = GPT2Simple.from_run_info(run_info_2)
    model_2.eval()
    models["gpt2_simple-pile (7M)"] = {
        "model": model_2,
        "tokenizer": tokenizer,  # Same tokenizer
        "lowercase": True,
        "custom": True,
        "non_emb_params": sum(p.numel() for p in model_2.parameters()) - model_2.wte.weight.numel(),
    }

    print("Loading llama_simple_mlp-pile-4L-emb2048...")
    run_info_3 = SSRunInfo.from_path("wandb:goodfire/spd/runs/z7s4ltid")
    run_info_3.model_config_dict["model_type"] = "LlamaSimpleMLP"
    model_3 = LlamaSimpleMLP.from_run_info(run_info_3)
    model_3.eval()
    models["llama_simple_mlp-pile-4L-emb2048"] = {
        "model": model_3,
        "tokenizer": tokenizer,
        "lowercase": True,
        "custom": True,
        "non_emb_params": sum(p.numel() for p in model_3.parameters()) - model_3.wte.weight.numel(),
    }

    print("Loading llama_simple_mlp-pile-4L-emb768...")
    run_info_4 = SSRunInfo.from_path("wandb:goodfire/spd/runs/zsv8u5ni")
    model_4 = LlamaSimpleMLP.from_run_info(run_info_4)
    model_4.eval()
    models["llama_simple_mlp-pile-4L-emb768"] = {
        "model": model_4,
        "tokenizer": tokenizer,
        "lowercase": True,
        "custom": True,
        "non_emb_params": sum(p.numel() for p in model_4.parameters()) - model_4.wte.weight.numel(),
    }

    print("Loading llama_simple_mlp-pile-2L-emb2048...")
    run_info_5 = SSRunInfo.from_path("wandb:goodfire/spd/runs/x9wclpvx")
    model_5 = LlamaSimpleMLP.from_run_info(run_info_5)
    model_5.eval()
    models["llama_simple_mlp-pile-2L-emb2048"] = {
        "model": model_5,
        "tokenizer": tokenizer,
        "lowercase": True,
        "custom": True,
        "non_emb_params": sum(p.numel() for p in model_5.parameters()) - model_5.wte.weight.numel(),
    }

    print("Loading llama_simple_mlp-pile-2L-emb768...")
    run_info_6 = SSRunInfo.from_path("wandb:goodfire/spd/runs/ivdw6l06")
    model_6 = LlamaSimpleMLP.from_run_info(run_info_6)
    model_6.eval()
    models["llama_simple_mlp-pile-2L-emb768"] = {
        "model": model_6,
        "tokenizer": tokenizer,
        "lowercase": True,
        "custom": True,
        "non_emb_params": sum(p.numel() for p in model_6.parameters()) - model_6.wte.weight.numel(),
    }

    print("Loading llama_simple_mlp-pile-12L-emb768...")
    run_info_7 = SSRunInfo.from_path("wandb:goodfire/spd/runs/tprjv67x")
    model_7 = LlamaSimpleMLP.from_run_info(run_info_7)
    model_7.eval()
    models["llama_simple_mlp-pile-12L-emb768"] = {
        "model": model_7,
        "tokenizer": tokenizer,
        "lowercase": True,
        "custom": True,
        "non_emb_params": sum(p.numel() for p in model_7.parameters()) - model_7.wte.weight.numel(),
    }

    print("Loading pythia-70m...")
    pythia_70m = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
    pythia_70m_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    pythia_70m.eval()
    models["pythia-70m"] = {
        "model": pythia_70m,
        "tokenizer": pythia_70m_tokenizer,
        "lowercase": False,
        "custom": False,
        "non_emb_params": sum(p.numel() for p in pythia_70m.parameters())
        - pythia_70m.gpt_neox.embed_in.weight.numel(),
    }

    print("Loading pythia-160m...")
    pythia_160m = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m")
    pythia_160m_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    pythia_160m.eval()
    models["pythia-160m"] = {
        "model": pythia_160m,
        "tokenizer": pythia_160m_tokenizer,
        "lowercase": False,
        "custom": False,
        "non_emb_params": sum(p.numel() for p in pythia_160m.parameters())
        - pythia_160m.gpt_neox.embed_in.weight.numel(),
    }

    print("Loading gpt2-small (124M)...")
    gpt2_small = AutoModelForCausalLM.from_pretrained("gpt2")
    gpt2_small_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2_small.eval()
    models["gpt2-small (124M)"] = {
        "model": gpt2_small,
        "tokenizer": gpt2_small_tokenizer,
        "lowercase": False,
        "custom": False,
        "non_emb_params": sum(p.numel() for p in gpt2_small.parameters())
        - gpt2_small.transformer.wte.weight.numel(),
    }

    print("All models loaded!\n")
    return models


def generate(model_config, prompt, max_new_tokens=50, temperature=0.8, top_k=40):
    """Generate text from a model."""
    model = model_config["model"]
    tok = model_config["tokenizer"]
    lowercase = model_config["lowercase"]
    is_custom = model_config["custom"]

    device = next(model.parameters()).device

    input_text = prompt.lower() if lowercase else prompt
    input_ids = tok.encode(input_text, return_tensors="pt").to(device)

    generated = input_ids
    for _ in range(max_new_tokens):
        with torch.no_grad():
            if is_custom:
                output = model(generated)
                logits = output[0][:, -1, :]
            else:
                output = model(generated)
                logits = output.logits[:, -1, :]

        logits = logits / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)

        eos_id = tok.eos_token_id
        if eos_id is not None and next_token.item() == eos_id:
            break

    return tok.decode(generated[0], skip_special_tokens=True)


def compare_models(models, prompts, max_new_tokens=40, temperature=0.8, top_k=40):
    """Run comparison and return results as structured data."""
    results = []

    for prompt in prompts:
        prompt_results = {"prompt": prompt, "generations": {}}
        for name, config in models.items():
            try:
                output = generate(config, prompt, max_new_tokens, temperature, top_k)
                prompt_results["generations"][name] = output
            except Exception as e:
                prompt_results["generations"][name] = f"ERROR: {e}"
        results.append(prompt_results)

    return results


def format_markdown(results, models, args):
    """Format results as markdown."""
    lines = [
        "# Model Generation Comparison",
        "",
        "## Models",
        "",
        "| Model | Non-embedding Parameters |",
        "|-------|--------------------------|",
    ]

    for name, config in models.items():
        lines.append(f"| {name} | {config['non_emb_params']:,} |")

    lines.extend(
        [
            "",
            "## Settings",
            "",
            f"- **Max tokens**: {args.max_tokens}",
            f"- **Temperature**: {args.temperature}",
            f"- **Top-k**: {args.top_k}",
            "",
            "---",
            "",
            "## Generations",
            "",
        ]
    )

    for i, result in enumerate(results):
        lines.append(f"### Prompt {i + 1}")
        lines.append("")
        lines.append(f"> {result['prompt']}")
        lines.append("")

        for name, generation in result["generations"].items():
            lines.append(f"**{name}**")
            lines.append("```")
            lines.append(generation)
            lines.append("```")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare generations across language models")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("model_comparison.md"),
        help="Output file path (default: model_comparison.md)",
    )
    parser.add_argument("--max-tokens", type=int, default=40, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling (0 to disable)")
    args = parser.parse_args()

    models = load_models()

    print(f"Running comparison on {len(PROMPTS)} prompts...")
    results = compare_models(
        models,
        PROMPTS,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    markdown = format_markdown(results, models, args)
    args.output.write_text(markdown)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
