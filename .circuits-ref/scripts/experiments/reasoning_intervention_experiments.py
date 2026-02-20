#!/usr/bin/env python3
"""
Experiments to test if ablating spurious modules improves reasoning.

Two cases:
1. Roman Empire: Ablate British/Celtic/Norse associations (modules 13, 19, 20)
2. Hash Tables: Ablate cache-suppression mechanism (module 19)

Usage:
    python scripts/reasoning_intervention_experiments.py --experiment roman
    python scripts/reasoning_intervention_experiments.py --experiment hash
    python scripts/reasoning_intervention_experiments.py --experiment both
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Minimal Llama 3.1 chat template (must match generate_graph.py)
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


# Experiment configurations
EXPERIMENTS = {
    "roman": {
        "prompt": "The fall of the Roman Empire was accelerated by invasions from",
        "clusters_file": "outputs/knowledge_circuits/relp-the-fall-of-the-roman-empire-was-accelerated-by-in-clusters.json",
        "spurious_modules": [11, 12, 13],  # British/Celtic/Norse neurons (L19/N13885, L23/N2909)
        "expected_improvement": "Should boost Germanic tribes (Visigoths, Vandals, Huns) over British/Norse",
        "description": "Ablate British/Celtic/Anglo-Saxon/Norse associations that confuse 'invaders of Rome' with 'invaders of Britain'"
    },
    "hash": {
        "prompt": "Since hash tables have O(1) average lookup time, they are ideal for implementing",
        "clusters_file": "outputs/knowledge_circuits/relp-since-hash-tables-have-o1-average-lookup-time-they-clusters.json",
        "spurious_modules": [10, 13],  # Cache/Redis neurons (L23/N2603, L17/N7233)
        "expected_improvement": "Should boost cache/dictionary answers by removing suppression",
        "description": "Ablate the module that suppresses cache-related completions"
    }
}


def load_module_neurons(clusters_file: Path, method: str = "infomap"):
    """Load neuron assignments for each module from clusters file."""
    with open(clusters_file) as f:
        data = json.load(f)

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

    module_neurons = defaultdict(list)
    module_info = {}

    for cluster in clusters:
        module_id = cluster["cluster_id"]
        module_info[module_id] = {
            "name": cluster.get("name", f"Module {module_id}"),
            "size": len(cluster["members"]),
            "layer_range": cluster.get("layer_range", []),
        }
        for member in cluster["members"]:
            layer = int(member["layer"])
            neuron_idx = member["neuron"]
            position = member["position"]
            module_neurons[module_id].append((layer, neuron_idx, position))

    return dict(module_neurons), module_info


class InterventionExperimenter:
    def __init__(self, model_path: str = "meta-llama/Llama-3.1-8B-Instruct"):
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        self.hooks = []

    def _format_prompt(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            chat_template=MINIMAL_CHAT_TEMPLATE
        )
        return formatted + " Answer:"

    def _clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _create_ablation_hook(self, layer_idx: int, neuron_positions: list):
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

    def get_top_tokens(self, prompt: str, top_k: int = 20):
        """Get top-k tokens and their probabilities."""
        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)

            top_probs, top_indices = torch.topk(probs, top_k)

            results = []
            for prob, idx in zip(top_probs, top_indices):
                token = self.tokenizer.decode([idx])
                results.append({
                    "token": token,
                    "token_id": idx.item(),
                    "prob": prob.item(),
                })

            return results

    def run_ablation_experiment(self, prompt: str, module_neurons: dict,
                                 modules_to_ablate: list, module_info: dict):
        """Run ablation experiment and compare results."""

        print(f"\n{'='*70}")
        print(f"EXPERIMENT: Ablating modules {modules_to_ablate}")
        print(f"{'='*70}")
        print(f"Prompt: {prompt}")

        # Print module info
        print("\nModules to ablate:")
        total_neurons = 0
        for mid in modules_to_ablate:
            if mid in module_info:
                info = module_info[mid]
                neurons = module_neurons.get(mid, [])
                total_neurons += len(neurons)
                print(f"  Module {mid}: {info['name']} ({len(neurons)} neurons, layers {info['layer_range']})")

        # Baseline
        print("\n--- BASELINE (no ablation) ---")
        baseline = self.get_top_tokens(prompt, top_k=15)
        for i, entry in enumerate(baseline[:10]):
            print(f"  {i+1:2}. {repr(entry['token']):20} p={entry['prob']:.4f}")

        # Set up ablation hooks for all specified modules
        self._clear_hooks()

        for module_id in modules_to_ablate:
            if module_id not in module_neurons:
                print(f"  Warning: Module {module_id} not found")
                continue

            neurons = module_neurons[module_id]

            # Group by layer
            layer_neurons = defaultdict(list)
            for layer, neuron_idx, pos in neurons:
                layer_neurons[layer].append((neuron_idx, pos))

            for layer_idx, neuron_positions in layer_neurons.items():
                mlp = self.model.model.layers[layer_idx].mlp
                hook = mlp.down_proj.register_forward_pre_hook(
                    self._create_ablation_hook(layer_idx, neuron_positions),
                    with_kwargs=True
                )
                self.hooks.append(hook)

        # Ablated
        print(f"\n--- ABLATED ({total_neurons} neurons zeroed) ---")
        ablated = self.get_top_tokens(prompt, top_k=15)
        for i, entry in enumerate(ablated[:10]):
            print(f"  {i+1:2}. {repr(entry['token']):20} p={entry['prob']:.4f}")

        self._clear_hooks()

        # Compare
        print("\n--- CHANGES ---")
        baseline_dict = {e['token']: e['prob'] for e in baseline}
        ablated_dict = {e['token']: e['prob'] for e in ablated}

        all_tokens = set(baseline_dict.keys()) | set(ablated_dict.keys())
        changes = []
        for token in all_tokens:
            b_prob = baseline_dict.get(token, 0)
            a_prob = ablated_dict.get(token, 0)
            delta = a_prob - b_prob
            if abs(delta) > 0.001:  # Only show meaningful changes
                changes.append((token, delta, b_prob, a_prob))

        changes.sort(key=lambda x: x[1], reverse=True)  # Sort by increase

        print("  Increased:")
        for token, delta, b_prob, a_prob in changes[:5]:
            if delta > 0:
                print(f"    {repr(token):20} ↑ {delta:+.4f} ({b_prob:.4f} → {a_prob:.4f})")

        print("  Decreased:")
        for token, delta, b_prob, a_prob in changes[-5:]:
            if delta < 0:
                print(f"    {repr(token):20} ↓ {delta:+.4f} ({b_prob:.4f} → {a_prob:.4f})")

        # Check if top token changed
        baseline_top = baseline[0]['token']
        ablated_top = ablated[0]['token']

        print("\n--- VERDICT ---")
        if baseline_top != ablated_top:
            print(f"  ✓ TOP TOKEN CHANGED: {repr(baseline_top)} → {repr(ablated_top)}")
        else:
            print(f"  ○ Top token unchanged: {repr(baseline_top)}")
            # Check if top token prob changed significantly
            baseline_top_prob = baseline[0]['prob']
            ablated_top_prob = ablated[0]['prob']
            delta = ablated_top_prob - baseline_top_prob
            if abs(delta) > 0.01:
                direction = "increased" if delta > 0 else "decreased"
                print(f"    But probability {direction}: {baseline_top_prob:.4f} → {ablated_top_prob:.4f}")

        return {
            "baseline": baseline,
            "ablated": ablated,
            "modules_ablated": modules_to_ablate,
            "total_neurons_ablated": total_neurons,
            "top_changed": baseline_top != ablated_top,
            "baseline_top": baseline_top,
            "ablated_top": ablated_top,
        }


def run_experiment(exp_name: str, experimenter: InterventionExperimenter):
    """Run a single experiment."""
    config = EXPERIMENTS[exp_name]

    print(f"\n{'#'*70}")
    print(f"# EXPERIMENT: {exp_name.upper()}")
    print(f"{'#'*70}")
    print(f"\nDescription: {config['description']}")
    print(f"Expected: {config['expected_improvement']}")

    # Load cluster data
    clusters_path = Path(config['clusters_file'])
    if not clusters_path.exists():
        print(f"ERROR: Cluster file not found: {clusters_path}")
        return None

    module_neurons, module_info = load_module_neurons(clusters_path)

    # Run experiment
    result = experimenter.run_ablation_experiment(
        config['prompt'],
        module_neurons,
        config['spurious_modules'],
        module_info
    )

    result['experiment'] = exp_name
    result['config'] = config

    return result


def main():
    parser = argparse.ArgumentParser(description="Run reasoning intervention experiments")
    parser.add_argument("--experiment", choices=["roman", "hash", "both"], default="both",
                       help="Which experiment to run")
    parser.add_argument("--output", type=Path, help="Output JSON file for results")
    args = parser.parse_args()

    experimenter = InterventionExperimenter()

    results = []

    if args.experiment in ["roman", "both"]:
        result = run_experiment("roman", experimenter)
        if result:
            results.append(result)

    if args.experiment in ["hash", "both"]:
        result = run_experiment("hash", experimenter)
        if result:
            results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for r in results:
        exp = r['experiment']
        changed = "✓ TOP CHANGED" if r['top_changed'] else "○ same"
        print(f"\n{exp.upper()}:")
        print(f"  {changed}: {repr(r['baseline_top'])} → {repr(r['ablated_top'])}")
        print(f"  Ablated {r['total_neurons_ablated']} neurons across {len(r['modules_ablated'])} modules")

    # Save results
    if args.output:
        # Convert to JSON-serializable
        for r in results:
            r['config'] = {k: str(v) if isinstance(v, Path) else v
                          for k, v in r['config'].items()}

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
