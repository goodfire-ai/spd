#!/usr/bin/env python3
"""Steer a module by amplifying or suppressing its neurons.

Usage:
    python scripts/steer_module.py clusters.json --module 1 --mode amplify --strength 2.0
    python scripts/steer_module.py clusters.json --module 1 --mode suppress
    python scripts/steer_module.py clusters.json --module 1 --mode amplify --prompt "Custom prompt"

This script loads a model, identifies neurons in a module, and runs inference
with those neurons amplified or suppressed to test steering predictions.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ModuleSteerer:
    """Steers model generation by modifying neuron activations in specific modules."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        module_neurons: dict[int, list[tuple[int, int, int]]],  # module_id -> [(layer, neuron, position), ...]
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.module_neurons = module_neurons
        self.hooks = []
        self.steering_config = None

    def _get_neurons_by_layer(self, module_id: int) -> dict[int, list[tuple[int, int]]]:
        """Group neurons by layer for efficient hooking."""
        by_layer = defaultdict(list)
        for layer, neuron, position in self.module_neurons.get(module_id, []):
            by_layer[layer].append((neuron, position))
        return dict(by_layer)

    def _create_steering_hook(self, layer_idx: int, neuron_positions: list[tuple[int, int]], mode: str, strength: float):
        """Create a hook that modifies specific neuron activations."""
        def hook(module, input, output):
            # output shape: (batch, seq_len, hidden_dim) for MLP output
            # or (batch, seq_len, intermediate_dim) for gate/up projections

            # For Llama MLP, we hook after the down_proj which gives (batch, seq, hidden)
            # But neurons are in the intermediate space, so we need to hook gate_proj or up_proj

            # Actually, let's hook the activation after gate * up, before down_proj
            # This is tricky because Llama uses SwiGLU: down(gate(x) * up(x))

            # For now, let's modify the output directly as a proxy
            # This isn't perfect but gives directional signal

            modified = output.clone()

            for neuron_idx, position in neuron_positions:
                if position < modified.shape[1]:
                    if mode == "suppress":
                        # Zero out this neuron's contribution at this position
                        # Since we're at MLP output, we approximate by zeroing the position
                        # A better approach would hook intermediate activations
                        pass  # See note below
                    elif mode == "amplify":
                        # Amplify the activation
                        pass

            return modified

        return hook

    def _create_mlp_intermediate_hook(
        self,
        layer_idx: int,
        neuron_positions: list[tuple[int, int]],
        mode: str,
        strength: float,
        all_positions: bool = True
    ):
        """Create a hook for MLP intermediate activations.

        For Llama's SwiGLU MLP: output = down_proj(act(gate_proj(x)) * up_proj(x))
        We hook down_proj input to modify the intermediate activations.

        If all_positions=True, we modify the neuron at ALL sequence positions.
        This is needed during generation since new tokens are at new positions.
        """
        # Get unique neuron indices (ignore position if all_positions=True)
        if all_positions:
            neuron_indices = list(set(n for n, p in neuron_positions))
        else:
            neuron_indices = None

        call_count = [0]  # Mutable for closure

        def hook(module, args, kwargs):
            # down_proj input is (batch, seq, intermediate_dim)
            if len(args) == 0:
                return None

            x = args[0]
            modified = x.clone()

            call_count[0] += 1
            if call_count[0] == 1:
                print(f"    Hook L{layer_idx} called, input shape: {x.shape}, modifying {len(neuron_indices) if all_positions else len(neuron_positions)} neurons")

            if all_positions:
                # Modify these neurons at ALL positions
                for neuron_idx in neuron_indices:
                    if neuron_idx < modified.shape[2]:
                        if mode == "suppress":
                            modified[:, :, neuron_idx] = 0.0
                        elif mode == "amplify":
                            modified[:, :, neuron_idx] *= strength
            else:
                # Original behavior: specific positions only
                for neuron_idx, position in neuron_positions:
                    if position < modified.shape[1] and neuron_idx < modified.shape[2]:
                        if mode == "suppress":
                            modified[:, position, neuron_idx] = 0.0
                        elif mode == "amplify":
                            modified[:, position, neuron_idx] *= strength

            return (modified,) + args[1:], kwargs

        return hook

    def setup_steering(self, module_id: int, mode: str = "suppress", strength: float = 2.0):
        """Set up hooks to steer a specific module."""
        self.clear_hooks()

        neurons_by_layer = self._get_neurons_by_layer(module_id)

        if not neurons_by_layer:
            print(f"Warning: No neurons found for module {module_id}")
            return

        print(f"Setting up {mode} steering for module {module_id}")
        print(f"  Layers affected: {sorted(neurons_by_layer.keys())}")
        print(f"  Total neurons: {sum(len(v) for v in neurons_by_layer.values())}")

        for layer_idx, neuron_positions in neurons_by_layer.items():
            # Hook the down_proj input - this is where intermediate activations live
            # down_proj receives: act(gate_proj(x)) * up_proj(x)
            try:
                down_proj = self.model.model.layers[layer_idx].mlp.down_proj
                hook = self._create_mlp_intermediate_hook(layer_idx, neuron_positions, mode, strength)
                handle = down_proj.register_forward_pre_hook(hook, with_kwargs=True)
                self.hooks.append(handle)
            except (IndexError, AttributeError) as e:
                print(f"  Warning: Could not hook layer {layer_idx}: {e}")

        self.steering_config = {"module_id": module_id, "mode": mode, "strength": strength}

    def clear_hooks(self):
        """Remove all steering hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.steering_config = None

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate text with current steering configuration."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for reproducibility
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated

    def compare_outputs(self, prompt: str, module_id: int, mode: str = "suppress", strength: float = 2.0) -> dict[str, Any]:
        """Compare baseline vs steered outputs."""
        # Baseline (no steering)
        self.clear_hooks()
        baseline = self.generate(prompt)

        # Steered
        self.setup_steering(module_id, mode, strength)
        steered = self.generate(prompt)
        self.clear_hooks()

        return {
            "prompt": prompt,
            "module_id": module_id,
            "mode": mode,
            "strength": strength,
            "baseline": baseline,
            "steered": steered,
            "changed": baseline != steered,
        }


def load_module_neurons(clusters_path: Path, method: str = "spectral") -> tuple[dict[int, list], dict, str]:
    """Load module neuron assignments from clusters file."""
    with open(clusters_path) as f:
        data = json.load(f)

    # Find the requested method
    method_data = None
    for m in data.get("methods", []):
        if m["method"] == method:
            method_data = m
            break

    if method_data is None:
        available = [m["method"] for m in data.get("methods", [])]
        raise ValueError(f"Method '{method}' not found. Available: {available}")

    # Extract neurons per module
    module_neurons = {}
    for cluster in method_data.get("clusters", []):
        cluster_id = cluster["cluster_id"]
        neurons = []
        for member in cluster["members"]:
            layer = int(member["layer"]) if member["layer"].isdigit() else None
            if layer is not None:
                neurons.append((layer, member["neuron"], member["position"]))
        module_neurons[cluster_id] = neurons

    prompt = data.get("prompt", "")

    return module_neurons, method_data, prompt


# Minimal Llama 3.1 chat template (matches generate_graph.py)
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


def format_prompt_for_model(tokenizer, prompt: str, answer_prefix: str = " Answer:") -> str:
    """Apply chat template to prompt with optional answer prefix.

    Matches the format used in generate_graph.py for attribution.
    """
    messages = [{"role": "user", "content": prompt}]

    if answer_prefix:
        messages.append({"role": "assistant", "content": answer_prefix})
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            chat_template=MINIMAL_CHAT_TEMPLATE,
            continue_final_message=True,
        )
    else:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=MINIMAL_CHAT_TEMPLATE,
        )
    return formatted


def main():
    parser = argparse.ArgumentParser(description="Steer a module by modifying neuron activations")
    parser.add_argument("clusters_file", type=Path, help="Path to clusters JSON file")
    parser.add_argument("--module", type=int, help="Module ID to steer (required unless --all-modules)")
    parser.add_argument("--mode", choices=["amplify", "suppress"], default="suppress",
                        help="Steering mode (default: suppress)")
    parser.add_argument("--strength", type=float, default=2.0,
                        help="Amplification strength (default: 2.0, only used for amplify)")
    parser.add_argument("--method", default="spectral", help="Clustering method to use")
    parser.add_argument("--prompt", help="Custom prompt (default: use prompt from clusters file)")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--all-modules", action="store_true", help="Test all modules")
    args = parser.parse_args()

    if not args.all_modules and args.module is None:
        parser.error("--module is required unless --all-modules is specified")

    # Load module data
    print(f"Loading clusters from: {args.clusters_file}")
    module_neurons, method_data, default_prompt = load_module_neurons(args.clusters_file, args.method)

    prompt = args.prompt or default_prompt
    if not prompt:
        prompt = "What is the capital of the state containing Dallas?"

    print(f"Using prompt: {prompt}")
    print(f"Clustering method: {args.method}")
    print(f"Available modules: {sorted(module_neurons.keys())}")

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    # Format prompt
    formatted_prompt = format_prompt_for_model(tokenizer, prompt)

    # Create steerer
    steerer = ModuleSteerer(model, tokenizer, module_neurons)

    # Run experiments
    modules_to_test = sorted(module_neurons.keys()) if args.all_modules else [args.module]

    results = []
    for module_id in modules_to_test:
        print(f"\n{'='*60}")
        print(f"Testing Module {module_id} ({args.mode})")
        print(f"{'='*60}")

        result = steerer.compare_outputs(
            formatted_prompt,
            module_id,
            mode=args.mode,
            strength=args.strength,
        )
        results.append(result)

        print("\nBaseline output:")
        print(f"  {result['baseline'][-200:]}")  # Last 200 chars
        print("\nSteered output:")
        print(f"  {result['steered'][-200:]}")
        print(f"\nChanged: {result['changed']}")

    # Summary
    if args.all_modules:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for r in results:
            status = "CHANGED" if r["changed"] else "same"
            print(f"  Module {r['module_id']}: {status}")

    # Save results
    output_path = args.clusters_file.parent / f"{args.clusters_file.stem}-steering-{args.mode}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
