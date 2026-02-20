#!/usr/bin/env python3
"""Analyze neuron output projections to understand what tokens each neuron promotes/suppresses.

Each MLP neuron has a fixed output direction in residual stream space (via down_proj).
By projecting this onto the unembedding matrix, we can see what tokens the neuron
"votes for" when it fires positively.

This is context-independent analysis - it tells us the neuron's intrinsic effect
regardless of what input triggered it.

Usage:
    # Analyze specific neurons
    python scripts/neuron_output_projection.py --neurons 15:7890 19:10945 20:3972

    # Analyze all neurons in a layer
    python scripts/neuron_output_projection.py --layer 15

    # Analyze neurons from a graph file
    python scripts/neuron_output_projection.py --from-graph outputs/my-graph.json

    # Output to JSON
    python scripts/neuron_output_projection.py --neurons 15:7890 -o neuron_profiles.json
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TokenContribution:
    """A token and its logit contribution from a neuron."""
    token_id: int
    token: str
    logit_contribution: float


@dataclass
class NeuronOutputProfile:
    """Output projection analysis for a single neuron."""
    layer: int
    neuron: int
    neuron_id: str  # "L{layer}/N{neuron}"

    # Top tokens this neuron promotes when firing positively
    top_promoted: list[TokenContribution]

    # Top tokens this neuron suppresses when firing positively
    top_suppressed: list[TokenContribution]

    # Statistics about the output direction
    output_norm: float  # L2 norm of output direction
    max_logit_contribution: float
    min_logit_contribution: float
    mean_logit_contribution: float
    std_logit_contribution: float


def analyze_neuron_output_projection(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer: int,
    neuron: int,
    top_k: int = 50,
) -> NeuronOutputProfile:
    """Analyze what tokens a neuron promotes/suppresses via its output projection.

    Args:
        model: The language model
        tokenizer: Tokenizer for decoding tokens
        layer: Layer index (0-31 for Llama 3.1 8B)
        neuron: Neuron index within the layer (0-14335)
        top_k: Number of top promoted/suppressed tokens to return

    Returns:
        NeuronOutputProfile with analysis results
    """
    # Get the down_proj weight matrix for this layer
    # Shape: [d_model, d_ffn] = [4096, 14336] for Llama 3.1 8B
    down_proj = model.model.layers[layer].mlp.down_proj.weight

    # Extract this neuron's output direction (column of down_proj)
    # Shape: [d_model] = [4096]
    output_direction = down_proj[:, neuron].float()

    # Get the unembedding matrix (lm_head)
    # Shape: [vocab_size, d_model] = [128256, 4096]
    lm_head = model.lm_head.weight.float()

    # Project output direction onto token space
    # Shape: [vocab_size]
    logit_contributions = lm_head @ output_direction

    # Compute statistics
    output_norm = output_direction.norm().item()
    max_contrib = logit_contributions.max().item()
    min_contrib = logit_contributions.min().item()
    mean_contrib = logit_contributions.mean().item()
    std_contrib = logit_contributions.std().item()

    # Get top promoted tokens (highest logit contribution)
    top_promoted_values, top_promoted_indices = logit_contributions.topk(top_k)
    top_promoted = []
    for i in range(top_k):
        token_id = top_promoted_indices[i].item()
        token_str = tokenizer.decode([token_id])
        top_promoted.append(TokenContribution(
            token_id=token_id,
            token=token_str,
            logit_contribution=top_promoted_values[i].item(),
        ))

    # Get top suppressed tokens (lowest logit contribution)
    top_suppressed_values, top_suppressed_indices = logit_contributions.topk(top_k, largest=False)
    top_suppressed = []
    for i in range(top_k):
        token_id = top_suppressed_indices[i].item()
        token_str = tokenizer.decode([token_id])
        top_suppressed.append(TokenContribution(
            token_id=token_id,
            token=token_str,
            logit_contribution=top_suppressed_values[i].item(),
        ))

    return NeuronOutputProfile(
        layer=layer,
        neuron=neuron,
        neuron_id=f"L{layer}/N{neuron}",
        top_promoted=top_promoted,
        top_suppressed=top_suppressed,
        output_norm=output_norm,
        max_logit_contribution=max_contrib,
        min_logit_contribution=min_contrib,
        mean_logit_contribution=mean_contrib,
        std_logit_contribution=std_contrib,
    )


def analyze_neurons_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    neurons: list[tuple[int, int]],
    top_k: int = 50,
    show_progress: bool = True,
) -> list[NeuronOutputProfile]:
    """Analyze multiple neurons efficiently.

    Args:
        model: The language model
        tokenizer: Tokenizer for decoding tokens
        neurons: List of (layer, neuron) tuples
        top_k: Number of top promoted/suppressed tokens per neuron
        show_progress: Whether to show progress bar

    Returns:
        List of NeuronOutputProfile for each neuron
    """
    results = []
    iterator = tqdm(neurons, desc="Analyzing neurons") if show_progress else neurons

    for layer, neuron in iterator:
        profile = analyze_neuron_output_projection(model, tokenizer, layer, neuron, top_k)
        results.append(profile)

    return results


def extract_neurons_from_graph(graph_path: Path) -> list[tuple[int, int]]:
    """Extract unique neurons from an attribution graph or cluster file.

    Args:
        graph_path: Path to the graph JSON file

    Returns:
        List of (layer, neuron) tuples
    """
    with open(graph_path) as f:
        graph = json.load(f)

    neurons = set()

    # Handle graph format (nodes list at top level)
    for node in graph.get("nodes", []):
        # Skip embeddings and logits (layer 32 is output projection)
        layer = node.get("layer")
        if layer in ("E", "32", None):
            continue

        try:
            layer_int = int(layer)
            # Skip invalid layer indices (0-31 for Llama 3.1 8B)
            if layer_int < 0 or layer_int > 31:
                continue
            neuron = node.get("feature")
            if neuron is not None:
                neurons.add((layer_int, neuron))
        except ValueError:
            continue

    # Handle cluster format (nested under methods/clusters/members)
    for method in graph.get("methods", []):
        for cluster in method.get("clusters", []):
            for member in cluster.get("members", []):
                layer = member.get("layer")
                neuron = member.get("neuron")
                if layer is None or neuron is None:
                    continue
                try:
                    layer_int = int(layer)
                    if 0 <= layer_int <= 31:
                        neurons.add((layer_int, neuron))
                except ValueError:
                    continue

    return sorted(neurons)


def format_profile_text(profile: NeuronOutputProfile, top_n: int = 10) -> str:
    """Format a neuron profile as human-readable text.

    Args:
        profile: The neuron profile to format
        top_n: Number of top tokens to show

    Returns:
        Formatted string
    """
    lines = [
        f"=== {profile.neuron_id} ===",
        f"Output norm: {profile.output_norm:.4f}",
        f"Logit range: [{profile.min_logit_contribution:.4f}, {profile.max_logit_contribution:.4f}]",
        f"Logit mean/std: {profile.mean_logit_contribution:.4f} / {profile.std_logit_contribution:.4f}",
        "",
        "TOP PROMOTED TOKENS (when neuron fires positively):",
    ]

    for i, tc in enumerate(profile.top_promoted[:top_n]):
        token_repr = repr(tc.token)
        lines.append(f"  {i+1:2d}. {token_repr:30s} logit: {tc.logit_contribution:+.4f}")

    lines.append("")
    lines.append("TOP SUPPRESSED TOKENS (when neuron fires positively):")

    for i, tc in enumerate(profile.top_suppressed[:top_n]):
        token_repr = repr(tc.token)
        lines.append(f"  {i+1:2d}. {token_repr:30s} logit: {tc.logit_contribution:+.4f}")

    return "\n".join(lines)


def profile_to_dict(profile: NeuronOutputProfile) -> dict:
    """Convert profile to JSON-serializable dict."""
    return {
        "layer": profile.layer,
        "neuron": profile.neuron,
        "neuron_id": profile.neuron_id,
        "top_promoted": [asdict(tc) for tc in profile.top_promoted],
        "top_suppressed": [asdict(tc) for tc in profile.top_suppressed],
        "output_norm": profile.output_norm,
        "max_logit_contribution": profile.max_logit_contribution,
        "min_logit_contribution": profile.min_logit_contribution,
        "mean_logit_contribution": profile.mean_logit_contribution,
        "std_logit_contribution": profile.std_logit_contribution,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze neuron output projections to understand token promotion/suppression"
    )

    # Neuron selection (mutually exclusive)
    neuron_group = parser.add_mutually_exclusive_group(required=True)
    neuron_group.add_argument(
        "--neurons", nargs="+", type=str,
        help="Specific neurons as layer:neuron (e.g., 15:7890 19:10945)"
    )
    neuron_group.add_argument(
        "--layer", type=int,
        help="Analyze all neurons in a specific layer (0-31)"
    )
    neuron_group.add_argument(
        "--from-graph", type=Path,
        help="Extract neurons from an attribution graph JSON file"
    )

    # Output options
    parser.add_argument(
        "-o", "--output", type=Path,
        help="Output JSON file (if not specified, prints to stdout)"
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="Number of top promoted/suppressed tokens to track (default: 50)"
    )
    parser.add_argument(
        "--display-top", type=int, default=10,
        help="Number of tokens to display in text output (default: 10)"
    )

    # Model options
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or path"
    )

    args = parser.parse_args()

    # Parse neuron specifications
    neurons = []
    if args.neurons:
        for spec in args.neurons:
            try:
                layer, neuron = spec.split(":")
                neurons.append((int(layer), int(neuron)))
            except ValueError:
                print(f"Error: Invalid neuron spec '{spec}'. Use format layer:neuron", file=sys.stderr)
                sys.exit(1)
    elif args.layer is not None:
        # All neurons in a layer (14336 for Llama 3.1 8B)
        neurons = [(args.layer, n) for n in range(14336)]
    elif args.from_graph:
        neurons = extract_neurons_from_graph(args.from_graph)
        print(f"Extracted {len(neurons)} neurons from {args.from_graph}", file=sys.stderr)

    if not neurons:
        print("No neurons to analyze", file=sys.stderr)
        sys.exit(1)

    # Load model
    print(f"Loading model {args.model}...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Analyze neurons
    print(f"Analyzing {len(neurons)} neurons...", file=sys.stderr)
    profiles = analyze_neurons_batch(
        model, tokenizer, neurons,
        top_k=args.top_k,
        show_progress=True,
    )

    # Output results
    if args.output:
        output_data = {
            "metadata": {
                "model": args.model,
                "top_k": args.top_k,
                "num_neurons": len(profiles),
            },
            "profiles": [profile_to_dict(p) for p in profiles],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved {len(profiles)} profiles to {args.output}", file=sys.stderr)
    else:
        # Print text format to stdout
        for profile in profiles:
            print(format_profile_text(profile, top_n=args.display_top))
            print()


if __name__ == "__main__":
    main()
