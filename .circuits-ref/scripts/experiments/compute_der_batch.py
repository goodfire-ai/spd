#!/usr/bin/env python3
"""
Compute direct effect ratios for neurons across multiple prompts.

Usage:
    # Single GPU
    python scripts/compute_der_batch.py --neurons data/neurons_missing_der.json \
        --prompts data/medical_corpus_579_substantive.json \
        --n-prompts 50 -o data/der_results.json

    # SLURM batch mode
    python scripts/compute_der_batch.py --neurons data/neurons_missing_der.json \
        --prompts data/medical_corpus_579_substantive.json \
        --batch-id 0 --n-batches 8 -o data/der_batch_0.json
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


@dataclass
class EffectResult:
    """Results for one neuron on one prompt."""
    neuron_id: str
    layer: int
    neuron: int
    prompt_idx: int
    direct_ratio: float


class DirectEffectAnalyzer:
    """Direct effect computation matching the original compute_direct_indirect.py."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.final_norm = model.model.norm
        self.lm_head = model.lm_head

    def clean_forward(self, prompt: str) -> dict:
        """Run clean forward, cache final hidden state and logits."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        final_hidden = None

        def capture_pre_norm(module, args):
            nonlocal final_hidden
            hidden = args[0] if isinstance(args, tuple) else args
            final_hidden = hidden[:, -1, :].detach().clone()

        hook = self.final_norm.register_forward_pre_hook(capture_pre_norm)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_clean = outputs.logits[:, -1, :].clone()

        hook.remove()

        return {
            'inputs': inputs,
            'final_hidden': final_hidden,
            'logits_clean': logits_clean,
        }

    def compute_direct_effect(self, cache: dict, layer: int, neuron: int,
                               activation: float) -> torch.Tensor:
        """Compute direct effect algebraically (no forward pass needed)."""
        down_proj = self.model.model.layers[layer].mlp.down_proj.weight
        V = down_proj[:, neuron] * activation

        h_plus_V = cache['final_hidden'] + V
        h_normed = self.final_norm(h_plus_V)
        logits_direct = self.lm_head(h_normed)

        return logits_direct - cache['logits_clean']

    def compute_total_effect(self, cache: dict, layer: int, neuron: int,
                              activation: float) -> torch.Tensor:
        """Compute total effect with one perturbed forward pass."""
        down_proj = self.model.model.layers[layer].mlp.down_proj.weight
        V = down_proj[:, neuron] * activation

        def inject_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden[:, -1, :] += V
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        hook = self.model.model.layers[layer].register_forward_hook(inject_hook)

        with torch.no_grad():
            outputs = self.model(**cache['inputs'])
            logits_perturbed = outputs.logits[:, -1, :].clone()

        hook.remove()

        return logits_perturbed - cache['logits_clean']

    def compute_direct_ratio(self, cache: dict, layer: int, neuron: int,
                              activation: float = 1.0) -> float:
        """Compute direct effect ratio for a neuron (requires forward pass)."""
        target_id = cache['logits_clean'].argmax().item()

        direct = self.compute_direct_effect(cache, layer, neuron, activation)
        total = self.compute_total_effect(cache, layer, neuron, activation)
        indirect = total - direct

        # Extract effects for target token
        d = direct[0, target_id].item()
        i = indirect[0, target_id].item()

        # Compute ratio
        if abs(d) + abs(i) > 1e-8:
            return abs(d) / (abs(d) + abs(i))
        else:
            return 0.5


def load_prompts(path: Path, n_prompts: int = 50) -> list[str]:
    """Load prompts from JSON file."""
    with open(path) as f:
        data = json.load(f)

    prompts = data.get('prompts', data)
    if isinstance(prompts, list) and isinstance(prompts[0], dict):
        prompts = [p.get('prompt', p.get('text', '')) for p in prompts]

    # Sample if we have more than needed
    if len(prompts) > n_prompts:
        random.seed(42)
        prompts = random.sample(prompts, n_prompts)

    return prompts


def compute_der_for_neuron(analyzer: DirectEffectAnalyzer,
                           layer: int, neuron: int,
                           prompts: list[str],
                           caches: list[dict]) -> dict:
    """Compute DER statistics for a single neuron across prompts."""
    ratios = []

    for cache in caches:
        try:
            ratio = analyzer.compute_direct_ratio(cache, layer, neuron)
            ratios.append(ratio)
        except Exception:
            continue

    if not ratios:
        return None

    ratios = np.array(ratios)
    mean_ratio = float(np.mean(ratios))

    # Classify effect type
    if mean_ratio > 0.7:
        effect_type = "logit"
    elif mean_ratio < 0.3:
        effect_type = "routing"
    else:
        effect_type = "mixed"

    return {
        'mean': mean_ratio,
        'std': float(np.std(ratios)),
        'min': float(np.min(ratios)),
        'max': float(np.max(ratios)),
        'effect_type': effect_type,
        'n_prompts': len(ratios)
    }


def main():
    parser = argparse.ArgumentParser(description="Compute direct effect ratios")
    parser.add_argument('--neurons', type=str, required=True,
                        help='Path to neurons JSON')
    parser.add_argument('--prompts', type=str, required=True,
                        help='Path to prompts JSON')
    parser.add_argument('--n-prompts', type=int, default=50,
                        help='Number of prompts to use')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output path')
    parser.add_argument('--batch-id', type=int,
                        help='Batch ID for SLURM array')
    parser.add_argument('--n-batches', type=int,
                        help='Total number of batches')
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    # Load neurons
    print(f"Loading neurons from {args.neurons}...")
    with open(args.neurons) as f:
        data = json.load(f)
    profiles = data.get('profiles', data)

    # Handle batch mode
    if args.batch_id is not None and args.n_batches is not None:
        batch_size = len(profiles) // args.n_batches + 1
        start = args.batch_id * batch_size
        end = min(start + batch_size, len(profiles))
        profiles = profiles[start:end]
        print(f"Batch {args.batch_id}: processing neurons {start}-{end} ({len(profiles)} neurons)")

    # Load prompts
    print(f"Loading prompts from {args.prompts}...")
    prompts = load_prompts(Path(args.prompts), args.n_prompts)
    print(f"Using {len(prompts)} prompts")

    # Initialize analyzer
    analyzer = DirectEffectAnalyzer(model, tokenizer)

    # Pre-compute caches for all prompts
    print("Pre-computing prompt caches...")
    caches = []
    for prompt in tqdm(prompts, desc="Caching prompts"):
        try:
            cache = analyzer.clean_forward(prompt)
            caches.append(cache)
        except Exception as e:
            print(f"Warning: Failed to cache prompt: {e}")

    # Compute DER for each neuron
    print(f"Computing DER for {len(profiles)} neurons...")
    results = {}

    for profile in tqdm(profiles, desc="Computing DER"):
        layer = profile['layer']
        neuron = profile['neuron']
        neuron_id = profile.get('neuron_id', f"L{layer}/N{neuron}")

        der = compute_der_for_neuron(analyzer, layer, neuron, prompts, caches)
        if der:
            results[neuron_id] = der

    # Save results
    output_data = {
        'results': results,
        'metadata': {
            'n_neurons': len(results),
            'n_prompts': len(prompts),
            'batch_id': args.batch_id,
            'n_batches': args.n_batches
        }
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved {len(results)} DER results to {args.output}")


if __name__ == '__main__':
    main()
