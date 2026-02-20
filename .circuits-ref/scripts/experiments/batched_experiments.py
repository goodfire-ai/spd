#!/usr/bin/env python3
"""
Batched causal intervention experiments.

Accepts JSON input specifying experiments to run and outputs results.
Optimized for running multiple ablation/patching experiments efficiently.

Usage:
    python batched_experiments.py input.json --output results.json

    # Or pipe JSON
    cat experiments.json | python batched_experiments.py - --output results.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from schemas.experiments import (
    BatchExperimentInput,
    BatchExperimentOutput,
    ExperimentResult,
    ExperimentSpec,
)

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
    for cluster in clusters:
        module_id = cluster["cluster_id"]
        for member in cluster["members"]:
            layer = int(member["layer"])
            neuron_idx = member["neuron"]
            position = member["position"]
            module_neurons[module_id].append((layer, neuron_idx, position))

    return dict(module_neurons)


class BatchedExperimentRunner:
    def __init__(self, model_path: str = "meta-llama/Llama-3.1-8B-Instruct"):
        print("Loading model...", file=sys.stderr)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.chat_template = MINIMAL_CHAT_TEMPLATE
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        self.hooks = []
        self.cached_activations = {}

    def _format_prompt(self, prompt: str, answer_prefix: str = None) -> str:
        """Format prompt with chat template and optional answer prefix."""
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if answer_prefix:
            formatted = formatted + answer_prefix
        return formatted

    def _clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.cached_activations = {}

    def get_logprobs(self, prompt: str, top_k: int = 10, answer_prefix: str = None,
                      track_tokens: list[str] = None):
        """Get top-k log probabilities for next token, plus any specifically tracked tokens.

        Args:
            prompt: The input prompt
            top_k: Number of top tokens to return
            answer_prefix: Optional prefix to add after assistant header
            track_tokens: Optional list of specific tokens to always include in results,
                         even if not in top-k (e.g., [" serotonin", " dopamine"])
        """
        formatted = self._format_prompt(prompt, answer_prefix)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)

            top_log_probs, top_indices = torch.topk(log_probs, top_k)

            results = []
            seen_tokens = set()

            # Add top-k tokens
            for log_prob, idx in zip(top_log_probs, top_indices):
                token = self.tokenizer.decode([idx])
                results.append({
                    "token": token,
                    "token_id": idx.item(),
                    "log_prob": log_prob.item(),
                    "prob": torch.exp(log_prob).item(),
                    "in_top_k": True,
                })
                seen_tokens.add(token)

            # Add specifically tracked tokens (even if not in top-k)
            if track_tokens:
                for token in track_tokens:
                    if token in seen_tokens:
                        continue
                    # Tokenize and get the first token ID
                    token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                    if token_ids:
                        idx = token_ids[0]
                        log_prob = log_probs[idx].item()
                        results.append({
                            "token": token,
                            "token_id": idx,
                            "log_prob": log_prob,
                            "prob": torch.exp(torch.tensor(log_prob)).item(),
                            "in_top_k": False,
                        })
                        seen_tokens.add(token)

            return results

    def _create_zero_ablation_hook(self, layer_idx: int, neuron_positions: list):
        """Create hook for zero ablation."""
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

    def _create_mean_ablation_hook(self, layer_idx: int, neuron_positions: list):
        """Create hook for mean ablation."""
        neuron_pos_map = defaultdict(list)
        for neuron_idx, pos in neuron_positions:
            neuron_pos_map[pos].append(neuron_idx)

        unique_neurons = set(n for n, p in neuron_positions)

        def hook(module, args, kwargs):
            x = args[0]
            modified = x.clone()

            # Compute mean over positions for each neuron
            neuron_means = {}
            for neuron_idx in unique_neurons:
                if neuron_idx < x.shape[2]:
                    neuron_means[neuron_idx] = x[0, :, neuron_idx].mean().item()

            for pos, neuron_indices in neuron_pos_map.items():
                if pos < modified.shape[1]:
                    for neuron_idx in neuron_indices:
                        if neuron_idx < modified.shape[2] and neuron_idx in neuron_means:
                            modified[0, pos, neuron_idx] = neuron_means[neuron_idx]

            return (modified,) + args[1:], kwargs

        return hook

    def _create_steer_hook(self, layer_idx: int, neuron_positions: list, scale: float):
        """Create hook for steering (scaling activations)."""
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
                            modified[0, pos, neuron_idx] *= scale

            return (modified,) + args[1:], kwargs

        return hook

    def _cache_activations_hook(self, layer_idx: int, neuron_positions: list):
        """Create hook for caching activations (for patching)."""
        neuron_pos_map = defaultdict(list)
        for neuron_idx, pos in neuron_positions:
            neuron_pos_map[pos].append(neuron_idx)

        def hook(module, args, kwargs):
            x = args[0]

            for pos, neuron_indices in neuron_pos_map.items():
                if pos < x.shape[1]:
                    for neuron_idx in neuron_indices:
                        if neuron_idx < x.shape[2]:
                            key = (layer_idx, neuron_idx, pos)
                            self.cached_activations[key] = x[0, pos, neuron_idx].item()

            return args, kwargs

        return hook

    def _create_patch_hook(self, layer_idx: int, neuron_positions: list, cached: dict):
        """Create hook for patching (replacing with cached values)."""
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

    def cache_source_activations(self, prompt: str, module_neurons: dict, module_id: int):
        """Run source prompt and cache activations for a module."""
        self._clear_hooks()

        neurons = module_neurons[module_id]
        layer_neurons = defaultdict(list)
        for layer, neuron_idx, pos in neurons:
            layer_neurons[layer].append((neuron_idx, pos))

        for layer_idx, neuron_positions in layer_neurons.items():
            mlp = self.model.model.layers[layer_idx].mlp
            hook = mlp.down_proj.register_forward_pre_hook(
                self._cache_activations_hook(layer_idx, neuron_positions),
                with_kwargs=True
            )
            self.hooks.append(hook)

        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            self.model(**inputs)

        cached = dict(self.cached_activations)
        self._clear_hooks()

        return cached

    def run_experiment(
        self,
        spec: ExperimentSpec,
        module_neurons: dict,
        top_k: int = 10
    ) -> ExperimentResult:
        """Run a single experiment and return result."""
        self._clear_hooks()

        module_id = spec.module_id
        if module_id not in module_neurons:
            raise ValueError(f"Module {module_id} not found in clusters")

        neurons = module_neurons[module_id]
        target_prompt = spec.target_prompt
        answer_prefix = getattr(spec, 'answer_prefix', None)
        track_tokens = getattr(spec, 'track_tokens', None)

        # Get baseline (with any specifically tracked tokens)
        baseline_logprobs = self.get_logprobs(target_prompt, top_k, answer_prefix, track_tokens)
        baseline_top = baseline_logprobs[0]

        # Find target token in baseline
        target_token = spec.target_token
        baseline_target_logprob = -100.0
        for entry in baseline_logprobs:
            if entry["token"] == target_token:
                baseline_target_logprob = entry["log_prob"]
                break

        # Group neurons by layer
        layer_neurons = defaultdict(list)
        for layer, neuron_idx, pos in neurons:
            layer_neurons[layer].append((neuron_idx, pos))

        # Set up intervention hooks
        cached = None
        if spec.experiment_type == "patch":
            if not spec.source_prompt:
                raise ValueError("Patch experiment requires source_prompt")
            cached = self.cache_source_activations(spec.source_prompt, module_neurons, module_id)

        for layer_idx, neuron_positions in layer_neurons.items():
            mlp = self.model.model.layers[layer_idx].mlp

            if spec.experiment_type == "zero_ablate":
                hook_fn = self._create_zero_ablation_hook(layer_idx, neuron_positions)
            elif spec.experiment_type == "mean_ablate":
                hook_fn = self._create_mean_ablation_hook(layer_idx, neuron_positions)
            elif spec.experiment_type == "steer":
                scale = spec.steer_scale or 2.0
                hook_fn = self._create_steer_hook(layer_idx, neuron_positions, scale)
            elif spec.experiment_type == "patch":
                hook_fn = self._create_patch_hook(layer_idx, neuron_positions, cached)
            else:
                raise ValueError(f"Unknown experiment type: {spec.experiment_type}")

            hook = mlp.down_proj.register_forward_pre_hook(hook_fn, with_kwargs=True)
            self.hooks.append(hook)

        # Run with intervention (track same tokens as baseline)
        result_logprobs = self.get_logprobs(target_prompt, top_k, answer_prefix, track_tokens)
        result_top = result_logprobs[0]

        self._clear_hooks()

        # Find target token in result
        result_target_logprob = -100.0
        for entry in result_logprobs:
            if entry["token"] == target_token:
                result_target_logprob = entry["log_prob"]
                break

        # Compute token group probabilities if specified
        group_probabilities = None
        observed_shift = None
        shift_correct = None

        if spec.token_groups:
            group_probabilities = {}
            baseline_probs_by_token = {e["token"]: e["prob"] for e in baseline_logprobs}
            result_probs_by_token = {e["token"]: e["prob"] for e in result_logprobs}

            for group in spec.token_groups:
                group_name = group["name"]
                tokens = group["tokens"]

                # Sum probabilities for all tokens in group
                baseline_group_prob = sum(baseline_probs_by_token.get(t, 0) for t in tokens)
                result_group_prob = sum(result_probs_by_token.get(t, 0) for t in tokens)

                group_probabilities[group_name] = {
                    "baseline_prob": baseline_group_prob,
                    "result_prob": result_group_prob,
                    "delta": result_group_prob - baseline_group_prob,
                    "tokens": tokens
                }

            # Determine observed shift pattern
            if spec.shift_from_group and spec.shift_to_group:
                from_delta = group_probabilities.get(spec.shift_from_group, {}).get("delta", 0)
                to_delta = group_probabilities.get(spec.shift_to_group, {}).get("delta", 0)

                if from_delta < -0.05 and to_delta > 0.05:
                    observed_shift = f"{spec.shift_from_group} -> {spec.shift_to_group}"
                    shift_correct = True
                elif from_delta > 0.05 and to_delta < -0.05:
                    observed_shift = f"{spec.shift_to_group} -> {spec.shift_from_group}"
                    shift_correct = False
                elif abs(from_delta) < 0.05 and abs(to_delta) < 0.05:
                    observed_shift = "no_change"
                    shift_correct = spec.expected_shift == "no_change"
                else:
                    observed_shift = "mixed"
                    shift_correct = False

        return ExperimentResult(
            experiment_id=spec.experiment_id,
            circuit_id=spec.circuit_id,
            module_id=module_id,
            experiment_type=spec.experiment_type,
            baseline_top_token=baseline_top["token"],
            baseline_top_prob=baseline_top["prob"],
            baseline_target_logprob=baseline_target_logprob,
            result_top_token=result_top["token"],
            result_top_prob=result_top["prob"],
            result_target_logprob=result_target_logprob,
            logprob_delta=result_target_logprob - baseline_target_logprob,
            top_token_changed=(baseline_top["token"] != result_top["token"]),
            baseline_top_k=[{"token": e["token"], "prob": e["prob"], "log_prob": e["log_prob"]}
                          for e in baseline_logprobs],
            result_top_k=[{"token": e["token"], "prob": e["prob"], "log_prob": e["log_prob"]}
                         for e in result_logprobs],
            num_neurons_affected=len(neurons),
            source_prompt=spec.source_prompt,
            steer_scale=spec.steer_scale,
            hypothesis=spec.hypothesis,
            group_probabilities=group_probabilities,
            observed_shift=observed_shift,
            shift_correct=shift_correct,
        )

    def run_batch(
        self,
        batch_input: BatchExperimentInput,
        clusters_file: Path,
        cluster_method: str = "infomap",
        top_k: int = 10,
        verbose: bool = True
    ) -> BatchExperimentOutput:
        """Run a batch of experiments."""
        module_neurons = load_module_neurons(clusters_file, cluster_method)

        results = []
        experiments = [ExperimentSpec.from_dict(e) for e in batch_input.experiments]

        # Sort by priority (higher first)
        experiments.sort(key=lambda e: -e.priority)

        for i, spec in enumerate(experiments):
            if verbose:
                print(f"[{i+1}/{len(experiments)}] Running {spec.experiment_id}...",
                      file=sys.stderr)

            try:
                result = self.run_experiment(spec, module_neurons, top_k)
                results.append(result.to_dict())

                if verbose:
                    delta = result.logprob_delta
                    direction = "↑" if delta > 0 else "↓"
                    print(f"  {spec.target_token}: {direction} {delta:+.3f} "
                          f"(top: {result.baseline_top_token} → {result.result_top_token})",
                          file=sys.stderr)

            except Exception as e:
                print(f"  ERROR: {e}", file=sys.stderr)
                # Create error result
                results.append({
                    "experiment_id": spec.experiment_id,
                    "circuit_id": spec.circuit_id,
                    "module_id": spec.module_id,
                    "experiment_type": spec.experiment_type,
                    "error": str(e),
                })

        return BatchExperimentOutput(
            circuit_id=batch_input.circuit_id,
            results=results
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run batched causal intervention experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example input JSON:
{
  "circuit_id": "grant_10k",
  "clusters_file": "outputs/clusters.json",
  "experiments": [
    {
      "experiment_id": "exp_001",
      "circuit_id": "grant_10k",
      "module_id": 8,
      "experiment_type": "zero_ablate",
      "target_prompt": "A professor requests $10,000...",
      "hypothesis": "Module 8 suppresses approval",
      "target_token": " yes",
      "expected_direction": "increases",
      "expected_magnitude": "large",
      "confidence": 0.8,
      "priority": 1
    }
  ]
}
        """
    )
    parser.add_argument("input", help="Input JSON file (or - for stdin)")
    parser.add_argument("--output", "-o", help="Output JSON file (default: stdout)")
    parser.add_argument("--clusters", help="Override clusters file from input JSON")
    parser.add_argument("--method", default="infomap", help="Clustering method")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k tokens to track")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    # Load input
    if args.input == "-":
        input_data = json.load(sys.stdin)
    else:
        with open(args.input) as f:
            input_data = json.load(f)

    batch_input = BatchExperimentInput.from_dict(input_data)
    clusters_file = Path(args.clusters or input_data.get("clusters_file"))

    if not clusters_file.exists():
        print(f"Error: Clusters file not found: {clusters_file}", file=sys.stderr)
        sys.exit(1)

    # Run experiments
    runner = BatchedExperimentRunner()
    output = runner.run_batch(
        batch_input,
        clusters_file,
        cluster_method=args.method,
        top_k=args.top_k,
        verbose=not args.quiet
    )

    # Write output
    output_json = json.dumps(output.to_dict(), indent=2)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_json)
        print(f"Results written to: {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
