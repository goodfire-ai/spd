#!/usr/bin/env python3
"""
Neuron-level ablation experiments targeting specific neurons identified as problematic.

Unlike module-level ablation, this targets individual neurons to avoid collateral damage.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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


# Specific neurons to ablate (from circuit analysis)
NEURON_EXPERIMENTS = {
    "roman": {
        "prompt": "The fall of the Roman Empire was accelerated by invasions from",
        "description": "Ablate British/Celtic/Anglo-Saxon/Norse neurons only",
        "target_neurons": [
            # L19/N13885: "British or Celtic culture, resistance against Roman rule"
            {"layer": 19, "neuron": 13885, "positions": [21, 28], "desc": "British/Celtic culture"},
            # L23/N2909: "Anglo-Saxon or Norse elements"
            {"layer": 23, "neuron": 2909, "positions": [21, 28], "desc": "Anglo-Saxon/Norse"},
        ],
        "expected": "Should reduce British/Norse associations while preserving Germanic tribe knowledge"
    },
    "hash": {
        "prompt": "Since hash tables have O(1) average lookup time, they are ideal for implementing",
        "description": "Test cache-related neurons (note: may be excitatory, not inhibitory)",
        "target_neurons": [
            # L23/N2603: "caching mechanisms, Redis"
            {"layer": 23, "neuron": 2603, "positions": [26, 33], "desc": "Caching mechanisms/Redis"},
            # L17/N7233: "caching and Redis technical terms"
            {"layer": 17, "neuron": 7233, "positions": [33], "desc": "Redis/caching jargon"},
        ],
        "expected": "Test if these neurons promote or suppress cache completions"
    },
    # Also test just the British neuron alone
    "roman_british_only": {
        "prompt": "The fall of the Roman Empire was accelerated by invasions from",
        "description": "Ablate ONLY the British/Celtic neuron (L19/N13885)",
        "target_neurons": [
            {"layer": 19, "neuron": 13885, "positions": [21, 28], "desc": "British/Celtic culture"},
        ],
        "expected": "More targeted - should only affect British associations"
    },
    # Test Norse neuron alone
    "roman_norse_only": {
        "prompt": "The fall of the Roman Empire was accelerated by invasions from",
        "description": "Ablate ONLY the Anglo-Saxon/Norse neuron (L23/N2909)",
        "target_neurons": [
            {"layer": 23, "neuron": 2909, "positions": [21, 28], "desc": "Anglo-Saxon/Norse"},
        ],
        "expected": "More targeted - should only affect Norse/Anglo-Saxon associations"
    },
    # LIFO/Stack experiment - ablate queue competitors
    "lifo_queue_ablate": {
        "prompt": "A data structure that follows Last-In-First-Out ordering is a",
        "description": "Ablate queue/FIFO neurons that compete with stack answer",
        "target_neurons": [
            # L24/N805: queue context (inf=2.12)
            {"layer": 24, "neuron": 805, "positions": [29], "desc": "Queue context"},
            # L21/N1200: "first come, first served" (inf=0.83)
            {"layer": 21, "neuron": 1200, "positions": [29], "desc": "FCFS/FIFO concept"},
        ],
        "expected": "Should boost 'stack' by reducing queue/FIFO competition"
    },
    # Also test the "Full stack" web dev noise
    "lifo_fullstack_ablate": {
        "prompt": "A data structure that follows Last-In-First-Out ordering is a",
        "description": "Ablate 'Full stack' web development noise",
        "target_neurons": [
            # L27/N109: "Full stack" web dev
            {"layer": 27, "neuron": 109, "positions": [29], "desc": "Full stack web dev"},
        ],
        "expected": "Remove web dev 'full stack' interference"
    },
}


class NeuronAblator:
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

    def _create_neuron_ablation_hook(self, layer_idx: int, neuron_idx: int, positions: list):
        """Create a hook that zeros a specific neuron at specific positions."""
        def hook(module, args, kwargs):
            x = args[0]
            modified = x.clone()

            for pos in positions:
                if pos < modified.shape[1] and neuron_idx < modified.shape[2]:
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

    def run_neuron_ablation(self, prompt: str, target_neurons: list):
        """Ablate specific neurons and measure effect."""

        print(f"\n{'='*70}")
        print("NEURON-LEVEL ABLATION")
        print(f"{'='*70}")
        print(f"Prompt: {prompt}")

        # Count total interventions
        total_interventions = sum(len(n['positions']) for n in target_neurons)
        print(f"\nTarget neurons ({total_interventions} total interventions):")
        for n in target_neurons:
            print(f"  L{n['layer']}/N{n['neuron']} at positions {n['positions']}: {n['desc']}")

        # Baseline
        print("\n--- BASELINE ---")
        baseline = self.get_top_tokens(prompt, top_k=15)
        for i, entry in enumerate(baseline[:10]):
            print(f"  {i+1:2}. {repr(entry['token']):20} p={entry['prob']:.4f}")

        # Set up ablation hooks for specific neurons
        self._clear_hooks()

        for neuron_spec in target_neurons:
            layer = neuron_spec['layer']
            neuron = neuron_spec['neuron']
            positions = neuron_spec['positions']

            mlp = self.model.model.layers[layer].mlp
            hook = mlp.down_proj.register_forward_pre_hook(
                self._create_neuron_ablation_hook(layer, neuron, positions),
                with_kwargs=True
            )
            self.hooks.append(hook)

        # Ablated
        print(f"\n--- ABLATED ({total_interventions} neuron-positions zeroed) ---")
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
            if abs(delta) > 0.0005:
                changes.append((token, delta, b_prob, a_prob))

        changes.sort(key=lambda x: x[1], reverse=True)

        print("  Increased:")
        for token, delta, b_prob, a_prob in changes[:5]:
            if delta > 0:
                print(f"    {repr(token):20} ↑ {delta:+.4f} ({b_prob:.4f} → {a_prob:.4f})")

        print("  Decreased:")
        for token, delta, b_prob, a_prob in changes[-5:]:
            if delta < 0:
                print(f"    {repr(token):20} ↓ {delta:+.4f} ({b_prob:.4f} → {a_prob:.4f})")

        # Verdict
        baseline_top = baseline[0]['token']
        ablated_top = ablated[0]['token']

        print("\n--- VERDICT ---")
        if baseline_top != ablated_top:
            print(f"  ✓ TOP TOKEN CHANGED: {repr(baseline_top)} → {repr(ablated_top)}")
        else:
            print(f"  ○ Top token unchanged: {repr(baseline_top)}")
            delta = ablated[0]['prob'] - baseline[0]['prob']
            if abs(delta) > 0.005:
                direction = "increased" if delta > 0 else "decreased"
                print(f"    Probability {direction}: {baseline[0]['prob']:.4f} → {ablated[0]['prob']:.4f} ({delta:+.4f})")

        return {
            "baseline": baseline,
            "ablated": ablated,
            "target_neurons": target_neurons,
            "total_interventions": total_interventions,
            "top_changed": baseline_top != ablated_top,
            "baseline_top": baseline_top,
            "ablated_top": ablated_top,
        }


def main():
    parser = argparse.ArgumentParser(description="Neuron-level ablation experiments")
    parser.add_argument("--experiment", choices=list(NEURON_EXPERIMENTS.keys()) + ["all"],
                       default="all", help="Which experiment to run")
    parser.add_argument("--output", type=Path, help="Output JSON file")
    args = parser.parse_args()

    ablator = NeuronAblator()

    experiments_to_run = list(NEURON_EXPERIMENTS.keys()) if args.experiment == "all" else [args.experiment]

    results = []
    for exp_name in experiments_to_run:
        config = NEURON_EXPERIMENTS[exp_name]

        print(f"\n{'#'*70}")
        print(f"# EXPERIMENT: {exp_name}")
        print(f"{'#'*70}")
        print(f"Description: {config['description']}")
        print(f"Expected: {config['expected']}")

        result = ablator.run_neuron_ablation(
            config['prompt'],
            config['target_neurons']
        )
        result['experiment'] = exp_name
        result['config'] = config
        results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for r in results:
        exp = r['experiment']
        changed = "✓ CHANGED" if r['top_changed'] else "○ same"
        interventions = r['total_interventions']
        print(f"\n{exp}:")
        print(f"  {changed}: {repr(r['baseline_top'])} → {repr(r['ablated_top'])}")
        print(f"  Ablated {interventions} neuron-position pairs")

    if args.output:
        # Make JSON serializable
        for r in results:
            r['config'] = {k: v for k, v in r['config'].items() if k != 'target_neurons'}

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
