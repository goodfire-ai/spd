"""Compare generations from pile-trained LlamaSimpleMLP models, GPT-2 small, and Pythia models.

Usage:
    python scripts/compare_pile_generations.py
    python scripts/compare_pile_generations.py --output results.md
    python scripts/compare_pile_generations.py --max-tokens 60 --temperature 0.7
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from spd.pretrain.models.llama_simple_mlp import LlamaSimpleMLP
from spd.pretrain.run_info import PretrainRunInfo

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

CUSTOM_MODELS: dict[str, str] = {
    "pile_12L-768": "wandb:goodfire/spd/runs/t-a35bfd1a",
    "pile_4L-768": "wandb:goodfire/spd/runs/t-5541e2ae",
}


def load_custom_model(name: str, wandb_path: str) -> tuple[LlamaSimpleMLP, AutoTokenizer]:
    print(f"Loading {name}...")
    run_info = PretrainRunInfo.from_path(wandb_path)
    model = LlamaSimpleMLP.from_run_info(run_info)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2", add_bos_token=False, unk_token="[UNK]", eos_token="[EOS]", bos_token=None
    )
    return model, tokenizer


def load_hf_model(name: str, hf_id: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    print(f"Loading {name}...")
    model = AutoModelForCausalLM.from_pretrained(hf_id)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    return model, tokenizer


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    is_custom: bool,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids

    for _ in range(max_new_tokens):
        if is_custom:
            logits, _ = model(generated)
        else:
            logits = model(generated).logits
        logits = logits[:, -1, :] / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)

        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def non_emb_params(model: torch.nn.Module, emb_attr: str) -> int:
    emb = emb_attr.split(".")
    module = model
    for attr in emb:
        module = getattr(module, attr)
    return sum(p.numel() for p in model.parameters()) - module.weight.numel()


def format_markdown(
    results: list[dict],
    model_param_counts: dict[str, int],
    args: argparse.Namespace,
) -> str:
    lines = [
        "# Pile Model Generation Comparison",
        "",
        "## Models",
        "",
        "| Model | Non-embedding Parameters |",
        "|-------|--------------------------|",
    ]
    for name, count in model_param_counts.items():
        lines.append(f"| {name} | {count:,} |")

    lines.extend([
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
    ])

    for i, result in enumerate(results):
        lines.append(f"### Prompt {i + 1}")
        lines.append("")
        lines.append(f"> {result['prompt']}")
        lines.append("")
        for name, text in result["generations"].items():
            lines.append(f"**{name}**")
            lines.append("```")
            lines.append(text)
            lines.append("```")
            lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare pile model generations")
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("pile_comparison.md"), help="Output file path"
    )
    parser.add_argument("--max-tokens", type=int, default=40, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling (0 to disable)")
    args = parser.parse_args()

    # Load models
    models: dict[str, tuple[torch.nn.Module, AutoTokenizer, bool]] = {}
    for name, wandb_path in CUSTOM_MODELS.items():
        model, tokenizer = load_custom_model(name, wandb_path)
        models[name] = (model, tokenizer, True)

    HF_MODELS: dict[str, tuple[str, str]] = {
        "gpt2-small": ("gpt2", "transformer.wte"),
        "pythia-70m": ("EleutherAI/pythia-70m", "gpt_neox.embed_in"),
        "pythia-410m": ("EleutherAI/pythia-410m", "gpt_neox.embed_in"),
    }
    for name, (hf_id, _) in HF_MODELS.items():
        model, tokenizer = load_hf_model(name, hf_id)
        models[name] = (model, tokenizer, False)

    # Compute param counts
    model_param_counts = {}
    for name, wandb_path in CUSTOM_MODELS.items():
        model_param_counts[name] = non_emb_params(models[name][0], "wte")
    for name, (_, emb_attr) in HF_MODELS.items():
        model_param_counts[name] = non_emb_params(models[name][0], emb_attr)

    print(f"\nRunning comparison on {len(PROMPTS)} prompts...\n")
    results = []
    for prompt in PROMPTS:
        print(f"Prompt: {prompt}")
        prompt_result: dict = {"prompt": prompt, "generations": {}}
        for name, (model, tokenizer, is_custom) in models.items():
            text = generate(
                model,
                tokenizer,
                prompt,
                is_custom=is_custom,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            prompt_result["generations"][name] = text
            print(f"  {name}: {text[:80]}...")
        results.append(prompt_result)

    markdown = format_markdown(results, model_param_counts, args)
    args.output.write_text(markdown)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
