#!/usr/bin/env python3
"""
Compute direct vs indirect effects for neurons using efficient causal mediation.

Key insight from review: Direct effect can be computed algebraically without
a second forward pass, since freezing downstream guarantees h_final = h_clean + V.

Usage:
    # Validate on small sample
    python scripts/compute_direct_indirect.py --validate

    # Full run on all neurons
    python scripts/compute_direct_indirect.py --neurons data/medical_edge_stats_v2_enriched.json

    # SLURM batch mode
    python scripts/compute_direct_indirect.py --batch-id 0 --n-batches 100
"""

import argparse
import json
from dataclasses import dataclass

import torch
from tqdm import tqdm


@dataclass
class EffectResult:
    """Results for one neuron on one prompt."""
    neuron_id: str
    layer: int
    neuron: int
    prompt_id: int
    target_token: str
    target_id: int
    activation: float
    direct_effect: float
    indirect_effect: float
    total_effect: float
    direct_ratio: float
    sign_aligned: bool


class DirectIndirectAnalyzer:
    """Efficient direct/indirect effect computation."""

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
            'tokens': inputs['input_ids'][0].tolist()
        }

    def compute_direct_effect(self, cache: dict, layer: int, neuron: int,
                              activation: float) -> torch.Tensor:
        """
        Compute direct effect algebraically - NO forward pass needed.

        Since freezing downstream guarantees h_final = h_clean + V,
        we can compute: logits_direct = lm_head(final_norm(h_clean + V))
        """
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

    def analyze_neuron(self, cache: dict, layer: int, neuron: int,
                       activation: float, target_id: int,
                       prompt_id: int = 0) -> EffectResult:
        """Full analysis for one neuron on one prompt."""
        direct = self.compute_direct_effect(cache, layer, neuron, activation)
        total = self.compute_total_effect(cache, layer, neuron, activation)
        indirect = total - direct  # Exact vector decomposition

        # Extract effects for target token
        d = direct[0, target_id].item()
        t = total[0, target_id].item()
        i = indirect[0, target_id].item()

        # Compute ratio (handle edge cases)
        if abs(d) + abs(i) > 1e-8:
            direct_ratio = abs(d) / (abs(d) + abs(i))
        else:
            direct_ratio = 0.5

        return EffectResult(
            neuron_id=f"L{layer}/N{neuron}",
            layer=layer,
            neuron=neuron,
            prompt_id=prompt_id,
            target_token=self.tokenizer.decode([target_id]),
            target_id=target_id,
            activation=activation,
            direct_effect=d,
            indirect_effect=i,
            total_effect=t,
            direct_ratio=direct_ratio,
            sign_aligned=(d * i) > 0
        )


def validate_algebraic_shortcut(analyzer, cache, layer, neuron, activation):
    """
    Verify that algebraic direct effect matches frozen forward pass.
    This validates the core assumption of our efficiency improvement.
    """
    # Method 1: Algebraic (fast)
    direct_algebraic = analyzer.compute_direct_effect(cache, layer, neuron, activation)

    # Method 2: Frozen forward (slow, but ground truth)
    model = analyzer.model
    down_proj = model.model.layers[layer].mlp.down_proj.weight
    V = down_proj[:, neuron] * activation

    # Cache all layer outputs from clean forward
    clean_cache = {}

    def make_cache_hook(layer_idx):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            clean_cache[layer_idx] = hidden.detach().clone()
        return hook

    handles = [model.model.layers[i].register_forward_hook(make_cache_hook(i))
               for i in range(32)]

    with torch.no_grad():
        outputs = model(**cache['inputs'])
        logits_clean = outputs.logits[:, -1, :].clone()

    for h in handles:
        h.remove()

    # Frozen forward: inject V, freeze downstream
    def inject_hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden[:, -1, :] += V
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    def make_freeze_hook(layer_idx):
        def hook(module, input, output):
            frozen = clean_cache[layer_idx].clone()
            frozen[:, -1, :] += V
            if isinstance(output, tuple):
                return (frozen,) + output[1:]
            return frozen
        return hook

    handles = [model.model.layers[layer].register_forward_hook(inject_hook)]
    handles += [model.model.layers[i].register_forward_hook(make_freeze_hook(i))
                for i in range(layer + 1, 32)]

    with torch.no_grad():
        outputs = model(**cache['inputs'])
        logits_frozen = outputs.logits[:, -1, :].clone()

    for h in handles:
        h.remove()

    direct_frozen = logits_frozen - logits_clean

    # Compare
    max_diff = (direct_algebraic - direct_frozen).abs().max().item()
    mean_diff = (direct_algebraic - direct_frozen).abs().mean().item()

    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'match': max_diff < 0.01  # Should be very close
    }


def run_validation(model, tokenizer):
    """Validate the algebraic shortcut on a few examples."""
    print("=" * 60)
    print("VALIDATING ALGEBRAIC SHORTCUT")
    print("=" * 60)

    analyzer = DirectIndirectAnalyzer(model, tokenizer)

    prompt = "The neurotransmitter associated with reward and pleasure is"
    cache = analyzer.clean_forward(prompt)

    # Test a few neurons at different layers
    test_cases = [
        (5, 100, 1.0),
        (15, 1234, 2.0),
        (25, 5000, 0.5),
        (30, 8000, 1.5),
    ]

    all_pass = True
    for layer, neuron, activation in test_cases:
        result = validate_algebraic_shortcut(analyzer, cache, layer, neuron, activation)
        status = "✓ PASS" if result['match'] else "✗ FAIL"
        print(f"  L{layer}/N{neuron}: max_diff={result['max_diff']:.6f} {status}")
        if not result['match']:
            all_pass = False

    print()
    if all_pass:
        print("All validations passed! Algebraic shortcut is correct.")
    else:
        print("WARNING: Some validations failed. Check implementation.")

    return all_pass


def run_sample_analysis(model, tokenizer):
    """Run analysis on a small sample to demonstrate output."""
    print()
    print("=" * 60)
    print("SAMPLE ANALYSIS")
    print("=" * 60)

    analyzer = DirectIndirectAnalyzer(model, tokenizer)

    prompt = "The neurotransmitter associated with reward and pleasure is"
    cache = analyzer.clean_forward(prompt)

    # Target token: "dop" (most likely completion)
    target_id = cache['logits_clean'].argmax().item()
    target_token = tokenizer.decode([target_id])
    print(f"\nPrompt: {prompt}")
    print(f"Target token: '{target_token}' (id={target_id})")
    print()

    # Analyze neurons at different layers
    test_neurons = [
        (0, 491, 10.0),    # Early layer
        (15, 1816, 2.0),   # Mid layer
        (25, 6000, 1.0),   # Late-mid layer
        (30, 6962, 1.5),   # Late layer
    ]

    print(f"{'Neuron':<12} {'Direct':>10} {'Indirect':>10} {'Total':>10} {'Ratio':>8} {'Type':<15}")
    print("-" * 70)

    for layer, neuron, activation in test_neurons:
        result = analyzer.analyze_neuron(cache, layer, neuron, activation,
                                         target_id, prompt_id=0)

        if result.direct_ratio > 0.7:
            effect_type = "logit-dominant"
        elif result.direct_ratio < 0.3:
            effect_type = "routing-dominant"
        else:
            effect_type = "mixed"

        print(f"L{layer}/N{neuron:<6} {result.direct_effect:>+10.4f} "
              f"{result.indirect_effect:>+10.4f} {result.total_effect:>+10.4f} "
              f"{result.direct_ratio:>8.3f} {effect_type:<15}")


def main():
    parser = argparse.ArgumentParser(description="Compute direct/indirect effects")
    parser.add_argument('--validate', action='store_true',
                        help='Validate algebraic shortcut')
    parser.add_argument('--neurons', type=str,
                        help='Path to neuron profiles JSON')
    parser.add_argument('--prompts', type=str,
                        help='Path to prompts JSON')
    parser.add_argument('--output', type=str, default='effects_output.json',
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

    if args.validate:
        run_validation(model, tokenizer)
        run_sample_analysis(model, tokenizer)
        return

    # Full analysis mode
    if args.neurons:
        print(f"Loading neurons from {args.neurons}...")
        with open(args.neurons) as f:
            data = json.load(f)

        profiles = data['profiles'] if isinstance(data, dict) and 'profiles' in data else data

        # Handle batch mode
        if args.batch_id is not None and args.n_batches is not None:
            batch_size = len(profiles) // args.n_batches
            start = args.batch_id * batch_size
            end = start + batch_size if args.batch_id < args.n_batches - 1 else len(profiles)
            profiles = profiles[start:end]
            print(f"Batch {args.batch_id}: processing neurons {start}-{end}")

        # Run analysis
        analyzer = DirectIndirectAnalyzer(model, tokenizer)

        # Use a representative prompt
        prompt = "The neurotransmitter associated with reward and pleasure is"
        cache = analyzer.clean_forward(prompt)
        target_id = cache['logits_clean'].argmax().item()

        results = []
        for profile in tqdm(profiles, desc="Analyzing neurons"):
            layer = profile['layer']
            neuron = profile['neuron']
            nid = profile.get('neuron_id', f"L{layer}/N{neuron}")

            # Use unit activation (can be refined with actual activations)
            activation = 1.0

            result = analyzer.analyze_neuron(cache, layer, neuron, activation,
                                             target_id, prompt_id=0)
            results.append({
                'neuron_id': result.neuron_id,
                'layer': result.layer,
                'neuron': result.neuron,
                'direct_effect': result.direct_effect,
                'indirect_effect': result.indirect_effect,
                'total_effect': result.total_effect,
                'direct_ratio': result.direct_ratio,
                'sign_aligned': result.sign_aligned
            })

        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} results to {args.output}")


if __name__ == '__main__':
    main()
