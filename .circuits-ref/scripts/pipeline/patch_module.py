#!/usr/bin/env python3
"""
Activation patching and ablation experiments for module analysis.

Supports:
- Patching: Replace activations with values from a source (counterfactual) prompt
- Zero ablation: Set activations to zero
- Mean ablation: Set activations to their mean value across positions

Usage:
    # Patch from source prompt (counterfactual)
    python patch_module.py clusters.json --source "..." --target "..." --module 0

    # Zero ablation
    python patch_module.py clusters.json --target "..." --ablation zero --module 0

    # Mean ablation (mean over positions in target prompt)
    python patch_module.py clusters.json --target "..." --ablation mean --module 0

    # Mean ablation with pre-computed means from file
    python patch_module.py clusters.json --target "..." --ablation mean-file --mean-file means.json --module 0
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Minimal Llama 3.1 chat template without date/knowledge cutoff info
# Must match generate_graph.py exactly!
MINIMAL_CHAT_TEMPLATE = '''{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content'] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set system_message = '' %}
    {%- set loop_messages = messages %}
{%- endif %}
<|start_header_id|>system<|end_header_id|>

{{ system_message }}<|eot_id|>
{%- for message in loop_messages %}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>

{{ message['content'] }}
{%- if not loop.last or add_generation_prompt %}<|eot_id|>{% endif %}
{%- endfor %}
{%- if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>

{% endif %}'''



def load_module_neurons(clusters_file: Path, method: str):
    """Load neuron assignments for each module from clusters file."""
    with open(clusters_file) as f:
        data = json.load(f)

    # Find the method in the methods list
    methods = data.get("methods", [])
    method_data = None
    for m in methods:
        if m.get("method") == method:
            method_data = m
            break

    if method_data is None:
        available = [m.get("method") for m in methods]
        raise ValueError(f"Method '{method}' not found. Available: {available}")

    clusters = method_data.get("clusters", [])

    # Build module -> list of (layer, neuron, position) tuples
    module_neurons = defaultdict(list)
    for cluster in clusters:
        module_id = cluster["cluster_id"]
        for member in cluster["members"]:
            layer = int(member["layer"])
            neuron_idx = member["neuron"]
            position = member["position"]
            module_neurons[module_id].append((layer, neuron_idx, position))

    prompt = data.get("prompt", "")
    return dict(module_neurons), method_data, prompt


class ActivationPatcher:
    def __init__(self, model_path: str = "meta-llama/Llama-3.1-8B-Instruct"):
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Use default chat template (full Llama template with special tokens)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

        # Storage for cached activations
        self.cached_activations = {}
        self.hooks = []

    def _format_prompt(self, prompt: str) -> str:
        """Format prompt with chat template and Answer: prefix."""
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]
        # Use minimal template (matching generate_graph.py) to get consistent positions
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            chat_template=MINIMAL_CHAT_TEMPLATE
        )
        return formatted + " Answer:"

    def _clear_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.cached_activations = {}

    def _cache_activations(self, layer_idx: int, neuron_positions: list):
        """Create a hook that caches activations for specific neurons at specific positions."""
        neuron_pos_map = defaultdict(list)
        for neuron_idx, pos in neuron_positions:
            neuron_pos_map[pos].append(neuron_idx)

        def hook(module, args, kwargs):
            x = args[0]  # Shape: (batch, seq_len, intermediate_dim)

            for pos, neuron_indices in neuron_pos_map.items():
                if pos < x.shape[1]:
                    for neuron_idx in neuron_indices:
                        if neuron_idx < x.shape[2]:
                            key = (layer_idx, neuron_idx, pos)
                            self.cached_activations[key] = x[0, pos, neuron_idx].clone()

            return args, kwargs

        return hook

    def _patch_activations(self, layer_idx: int, neuron_positions: list, cached: dict):
        """Create a hook that patches in cached activations."""
        neuron_pos_map = defaultdict(list)
        for neuron_idx, pos in neuron_positions:
            neuron_pos_map[pos].append(neuron_idx)

        def hook(module, args, kwargs):
            x = args[0]
            modified = x.clone()

            for pos, neuron_indices in neuron_pos_map.items():
                if pos < modified.shape[1]:
                    for neuron_idx in neuron_indices:
                        key = (layer_idx, neuron_idx, pos)
                        if key in cached and neuron_idx < modified.shape[2]:
                            modified[0, pos, neuron_idx] = cached[key]

            return (modified,) + args[1:], kwargs

        return hook

    def _zero_ablate(self, layer_idx: int, neuron_positions: list):
        """Create a hook that zeros out specific neuron activations."""
        neuron_pos_map = defaultdict(list)
        for neuron_idx, pos in neuron_positions:
            neuron_pos_map[pos].append(neuron_idx)

        def hook(module, args, kwargs):
            x = args[0]
            modified = x.clone()

            for pos, neuron_indices in neuron_pos_map.items():
                if pos < modified.shape[1]:
                    for neuron_idx in neuron_indices:
                        if neuron_idx < modified.shape[2]:
                            modified[0, pos, neuron_idx] = 0.0

            return (modified,) + args[1:], kwargs

        return hook

    def _mean_ablate(self, layer_idx: int, neuron_positions: list, mean_values: dict = None):
        """
        Create a hook that replaces activations with mean values.

        If mean_values is provided, use those. Otherwise, compute mean over
        all positions in the current input for each neuron.
        """
        neuron_pos_map = defaultdict(list)
        for neuron_idx, pos in neuron_positions:
            neuron_pos_map[pos].append(neuron_idx)

        # Get unique neurons in this layer
        unique_neurons = set()
        for neuron_idx, pos in neuron_positions:
            unique_neurons.add(neuron_idx)

        def hook(module, args, kwargs):
            x = args[0]  # Shape: (batch, seq_len, intermediate_dim)
            modified = x.clone()

            # Compute means if not provided
            if mean_values is None:
                # Mean over all positions for each neuron
                neuron_means = {}
                for neuron_idx in unique_neurons:
                    if neuron_idx < x.shape[2]:
                        neuron_means[neuron_idx] = x[0, :, neuron_idx].mean().item()
            else:
                neuron_means = {n: mean_values.get((layer_idx, n), 0.0) for n in unique_neurons}

            for pos, neuron_indices in neuron_pos_map.items():
                if pos < modified.shape[1]:
                    for neuron_idx in neuron_indices:
                        if neuron_idx < modified.shape[2] and neuron_idx in neuron_means:
                            modified[0, pos, neuron_idx] = neuron_means[neuron_idx]

            return (modified,) + args[1:], kwargs

        return hook

    def get_logprobs(self, prompt: str, top_k: int = 10):
        """Get top-k log probabilities for next token."""
        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token position
            log_probs = torch.log_softmax(logits, dim=-1)

            top_log_probs, top_indices = torch.topk(log_probs, top_k)

            results = []
            for log_prob, idx in zip(top_log_probs, top_indices):
                token = self.tokenizer.decode([idx])
                results.append({
                    "token": token,
                    "token_id": idx.item(),
                    "log_prob": log_prob.item(),
                    "prob": torch.exp(log_prob).item(),
                })

            return results, inputs

    def cache_source_activations(self, prompt: str, module_neurons: dict, module_id: int):
        """Run source prompt and cache activations for a module."""
        self._clear_hooks()

        neurons = module_neurons[module_id]

        # Group by layer
        layer_neurons = defaultdict(list)
        for layer, neuron_idx, pos in neurons:
            layer_neurons[layer].append((neuron_idx, pos))

        # Set up caching hooks
        for layer_idx, neuron_positions in layer_neurons.items():
            mlp = self.model.model.layers[layer_idx].mlp
            hook = mlp.down_proj.register_forward_pre_hook(
                self._cache_activations(layer_idx, neuron_positions),
                with_kwargs=True
            )
            self.hooks.append(hook)

        # Run forward pass to cache
        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            self.model(**inputs)

        cached = dict(self.cached_activations)
        self._clear_hooks()

        return cached

    def patch_and_compare(self, source_prompt: str, target_prompt: str,
                          module_neurons: dict, module_id: int, top_k: int = 10):
        """
        Patch activations from source into target and compare logprobs.
        """
        print(f"\n{'='*60}")
        print(f"Patching Module {module_id}")
        print(f"{'='*60}")
        print(f"Source: {source_prompt}")
        print(f"Target: {target_prompt}")

        # Get baseline logprobs for target
        print("\nBaseline (target prompt, no patching):")
        baseline_logprobs, _ = self.get_logprobs(target_prompt, top_k)
        for i, entry in enumerate(baseline_logprobs[:5]):
            print(f"  {i+1}. {repr(entry['token']):15} p={entry['prob']:.4f} logp={entry['log_prob']:.3f}")

        # Get source logprobs for reference
        print("\nSource prompt (for reference):")
        source_logprobs, _ = self.get_logprobs(source_prompt, top_k)
        for i, entry in enumerate(source_logprobs[:5]):
            print(f"  {i+1}. {repr(entry['token']):15} p={entry['prob']:.4f} logp={entry['log_prob']:.3f}")

        # Cache activations from source
        print(f"\nCaching activations from source for module {module_id}...")
        neurons = module_neurons[module_id]
        print(f"  {len(neurons)} neuron-position pairs")
        cached = self.cache_source_activations(source_prompt, module_neurons, module_id)
        print(f"  Cached {len(cached)} activations")

        # Set up patching hooks
        layer_neurons = defaultdict(list)
        for layer, neuron_idx, pos in neurons:
            layer_neurons[layer].append((neuron_idx, pos))

        for layer_idx, neuron_positions in layer_neurons.items():
            mlp = self.model.model.layers[layer_idx].mlp
            hook = mlp.down_proj.register_forward_pre_hook(
                self._patch_activations(layer_idx, neuron_positions, cached),
                with_kwargs=True
            )
            self.hooks.append(hook)

        # Run with patching
        print("\nPatched (target prompt with source activations):")
        patched_logprobs, _ = self.get_logprobs(target_prompt, top_k)
        for i, entry in enumerate(patched_logprobs[:5]):
            print(f"  {i+1}. {repr(entry['token']):15} p={entry['prob']:.4f} logp={entry['log_prob']:.3f}")

        self._clear_hooks()

        # Compute changes
        print("\nChanges (patched - baseline):")
        baseline_dict = {e['token_id']: e for e in baseline_logprobs}
        patched_dict = {e['token_id']: e for e in patched_logprobs}

        # Find tokens that changed most
        all_tokens = set(baseline_dict.keys()) | set(patched_dict.keys())
        changes = []
        for tid in all_tokens:
            b_logp = baseline_dict.get(tid, {}).get('log_prob', -100)
            p_logp = patched_dict.get(tid, {}).get('log_prob', -100)
            token = baseline_dict.get(tid, patched_dict.get(tid, {})).get('token', '?')
            delta = p_logp - b_logp
            changes.append((token, tid, delta, b_logp, p_logp))

        changes.sort(key=lambda x: abs(x[2]), reverse=True)
        for token, tid, delta, b_logp, p_logp in changes[:10]:
            direction = "↑" if delta > 0 else "↓"
            print(f"  {repr(token):15} {direction} {delta:+.3f} (baseline={b_logp:.3f} → patched={p_logp:.3f})")

        return {
            "module_id": module_id,
            "intervention": "patch",
            "source_prompt": source_prompt,
            "target_prompt": target_prompt,
            "baseline_logprobs": baseline_logprobs,
            "source_logprobs": source_logprobs,
            "patched_logprobs": patched_logprobs,
            "num_cached": len(cached),
        }

    def ablate_and_compare(self, target_prompt: str, module_neurons: dict, module_id: int,
                           ablation_type: str = "zero", mean_values: dict = None, top_k: int = 10):
        """
        Ablate activations and compare logprobs.

        Args:
            target_prompt: The prompt to run
            module_neurons: Dict mapping module_id to list of (layer, neuron, pos)
            module_id: Which module to ablate
            ablation_type: "zero" or "mean"
            mean_values: Pre-computed mean values for mean ablation (optional)
            top_k: Number of top tokens to show
        """
        print(f"\n{'='*60}")
        print(f"Ablating Module {module_id} ({ablation_type})")
        print(f"{'='*60}")
        print(f"Target: {target_prompt}")

        # Get baseline logprobs
        print("\nBaseline (no ablation):")
        baseline_logprobs, _ = self.get_logprobs(target_prompt, top_k)
        for i, entry in enumerate(baseline_logprobs[:5]):
            print(f"  {i+1}. {repr(entry['token']):15} p={entry['prob']:.4f} logp={entry['log_prob']:.3f}")

        # Set up ablation hooks
        neurons = module_neurons[module_id]
        print(f"\nAblating {len(neurons)} neuron-position pairs...")

        layer_neurons = defaultdict(list)
        for layer, neuron_idx, pos in neurons:
            layer_neurons[layer].append((neuron_idx, pos))

        for layer_idx, neuron_positions in layer_neurons.items():
            mlp = self.model.model.layers[layer_idx].mlp

            if ablation_type == "zero":
                hook_fn = self._zero_ablate(layer_idx, neuron_positions)
            elif ablation_type == "mean":
                hook_fn = self._mean_ablate(layer_idx, neuron_positions, mean_values)
            else:
                raise ValueError(f"Unknown ablation type: {ablation_type}")

            hook = mlp.down_proj.register_forward_pre_hook(hook_fn, with_kwargs=True)
            self.hooks.append(hook)

        # Run with ablation
        print(f"\nAblated ({ablation_type}):")
        ablated_logprobs, _ = self.get_logprobs(target_prompt, top_k)
        for i, entry in enumerate(ablated_logprobs[:5]):
            print(f"  {i+1}. {repr(entry['token']):15} p={entry['prob']:.4f} logp={entry['log_prob']:.3f}")

        self._clear_hooks()

        # Compute changes
        print("\nChanges (ablated - baseline):")
        baseline_dict = {e['token_id']: e for e in baseline_logprobs}
        ablated_dict = {e['token_id']: e for e in ablated_logprobs}

        all_tokens = set(baseline_dict.keys()) | set(ablated_dict.keys())
        changes = []
        for tid in all_tokens:
            b_logp = baseline_dict.get(tid, {}).get('log_prob', -100)
            a_logp = ablated_dict.get(tid, {}).get('log_prob', -100)
            token = baseline_dict.get(tid, ablated_dict.get(tid, {})).get('token', '?')
            delta = a_logp - b_logp
            changes.append((token, tid, delta, b_logp, a_logp))

        changes.sort(key=lambda x: abs(x[2]), reverse=True)
        for token, tid, delta, b_logp, a_logp in changes[:10]:
            direction = "↑" if delta > 0 else "↓"
            print(f"  {repr(token):15} {direction} {delta:+.3f} (baseline={b_logp:.3f} → ablated={a_logp:.3f})")

        return {
            "module_id": module_id,
            "intervention": f"ablate-{ablation_type}",
            "target_prompt": target_prompt,
            "baseline_logprobs": baseline_logprobs,
            "ablated_logprobs": ablated_logprobs,
            "num_neurons": len(neurons),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Activation patching and ablation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Patch from source prompt (counterfactual)
  python patch_module.py clusters.json --source "A studies cancer" --target "A studies astrology" --module 0

  # Zero ablation (no source needed)
  python patch_module.py clusters.json --target "A studies astrology" --ablation zero --module 0

  # Mean ablation (mean over positions in target)
  python patch_module.py clusters.json --target "A studies astrology" --ablation mean --module 0

  # Test all modules
  python patch_module.py clusters.json --target "..." --ablation zero --all-modules
        """
    )
    parser.add_argument("clusters_file", type=Path, help="Path to clusters JSON file")
    parser.add_argument("--source", help="Source prompt (activations come from here). Required for patching.")
    parser.add_argument("--target", required=True, help="Target prompt (activations patched/ablated here)")
    parser.add_argument("--ablation", choices=["zero", "mean"], help="Ablation mode (if set, --source not needed)")
    parser.add_argument("--mean-file", type=Path, help="JSON file with pre-computed mean activations")
    parser.add_argument("--module", type=int, help="Module ID to patch/ablate")
    parser.add_argument("--all-modules", action="store_true", help="Test all modules")
    parser.add_argument("--method", default="infomap", help="Clustering method to use (default: infomap)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top tokens to show")
    args = parser.parse_args()

    if not args.all_modules and args.module is None:
        parser.error("--module is required unless --all-modules is specified")

    # Validate: need either --ablation or --source
    if args.ablation is None and args.source is None:
        parser.error("Either --source (for patching) or --ablation (for ablation) is required")

    # Load module data
    print(f"Loading clusters from: {args.clusters_file}")
    module_neurons, _, _ = load_module_neurons(args.clusters_file, args.method)
    print(f"Clustering method: {args.method}")
    print(f"Available modules: {sorted(module_neurons.keys())}")

    # Load mean values if provided
    mean_values = None
    if args.mean_file:
        print(f"Loading mean values from: {args.mean_file}")
        with open(args.mean_file) as f:
            mean_data = json.load(f)
            # Convert string keys back to tuples
            mean_values = {tuple(map(int, k.split(','))): v for k, v in mean_data.items()}

    # Initialize patcher
    patcher = ActivationPatcher()

    target_formatted = patcher._format_prompt(args.target)
    target_tokens = patcher.tokenizer(target_formatted, return_tensors="pt")
    print(f"\nTarget tokens: {target_tokens['input_ids'].shape[1]}")

    # If patching (not ablation), verify token alignment
    if args.source:
        source_formatted = patcher._format_prompt(args.source)
        source_tokens = patcher.tokenizer(source_formatted, return_tensors="pt")
        print(f"Source tokens: {source_tokens['input_ids'].shape[1]}")

        if source_tokens['input_ids'].shape[1] != target_tokens['input_ids'].shape[1]:
            print("\n⚠️  WARNING: Token counts differ! Position alignment may be off.")
            print("\nToken comparison:")
            src_ids = source_tokens['input_ids'][0].tolist()
            tgt_ids = target_tokens['input_ids'][0].tolist()
            max_len = max(len(src_ids), len(tgt_ids))
            for i in range(min(max_len, 30)):
                src_tok = patcher.tokenizer.decode([src_ids[i]]) if i < len(src_ids) else "<none>"
                tgt_tok = patcher.tokenizer.decode([tgt_ids[i]]) if i < len(tgt_ids) else "<none>"
                match = "✓" if i < len(src_ids) and i < len(tgt_ids) and src_ids[i] == tgt_ids[i] else "✗"
                print(f"  {i:2}: {match} {repr(src_tok):15} vs {repr(tgt_tok):15}")

    # Run experiments
    modules_to_test = sorted(module_neurons.keys()) if args.all_modules else [args.module]
    intervention_type = args.ablation if args.ablation else "patch"
    print(f"\nIntervention type: {intervention_type}")

    results = []
    for module_id in modules_to_test:
        if module_id not in module_neurons:
            print(f"\nModule {module_id} not found, skipping")
            continue

        if args.ablation:
            # Ablation mode
            result = patcher.ablate_and_compare(
                args.target,
                module_neurons, module_id,
                ablation_type=args.ablation,
                mean_values=mean_values,
                top_k=args.top_k
            )
        else:
            # Patching mode
            result = patcher.patch_and_compare(
                args.source, args.target,
                module_neurons, module_id,
                args.top_k
            )
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if not results:
        print("No results to summarize.")
        return

    baseline_top = results[0]['baseline_logprobs'][0]['token']
    print(f"Baseline top token: {repr(baseline_top)}")

    if args.source and 'source_logprobs' in results[0]:
        source_top = results[0]['source_logprobs'][0]['token']
        print(f"Source top token: {repr(source_top)}")
    else:
        source_top = None

    print()

    for r in results:
        # Get the modified logprobs (patched or ablated)
        modified_key = 'patched_logprobs' if 'patched_logprobs' in r else 'ablated_logprobs'
        modified_top = r[modified_key][0]['token']
        modified_prob = r[modified_key][0]['prob']

        # Check if top token changed
        changed = modified_top != baseline_top
        status = "TOP CHANGED" if changed else "same"

        print(f"  Module {r['module_id']:2}: {status:12} | top: {repr(modified_top):10} (p={modified_prob:.4f})")

        # If patching, show source token probability change
        if source_top and 'source_logprobs' in r:
            source_token_id = r['source_logprobs'][0]['token_id']
            baseline_source_prob = next((e['prob'] for e in r['baseline_logprobs'] if e['token_id'] == source_token_id), 0)
            modified_source_prob = next((e['prob'] for e in r[modified_key] if e['token_id'] == source_token_id), 0)
            print(f"             | source token {repr(source_top)}: {baseline_source_prob:.4f} → {modified_source_prob:.4f} ({modified_source_prob - baseline_source_prob:+.4f})")

    # Save results
    suffix = f"-{intervention_type}" if args.ablation else "-patching"
    output_file = args.clusters_file.parent / f"{args.clusters_file.stem}{suffix}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
