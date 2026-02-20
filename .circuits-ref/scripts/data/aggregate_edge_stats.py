#!/usr/bin/env python3
"""Aggregate edge statistics across many prompts to understand neuron connectivity patterns.

For each neuron that appears in attribution graphs, this tool aggregates:
- Upstream sources: What neurons/embeddings consistently feed into this neuron?
- Downstream targets: What neurons/logits does this neuron consistently feed into?
- Output token associations: What output tokens does this neuron contribute to?

By combining domain-specific prompts with random baseline sentences, we can identify
whether a neuron is domain-specific or general-purpose.

Usage:
    # Run on medical prompts + FineWeb baseline
    python scripts/aggregate_edge_stats.py \
        --domain-prompts configs/medical_prompts.yaml \
        --fineweb-samples 50 \
        -o edge_stats.json

    # Focus on specific neurons
    python scripts/aggregate_edge_stats.py \
        --domain-prompts configs/medical_prompts.yaml \
        --neurons 15:7890 19:10945 \
        -o edge_stats.json

    # From a directory of existing graphs
    python scripts/aggregate_edge_stats.py \
        --graphs-dir outputs/medical/ \
        -o edge_stats.json
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from circuits.relp import RelPAttributor


@dataclass
class EdgeStats:
    """Statistics for edges to/from a neuron."""
    count: int = 0
    total_weight: float = 0.0
    weights: list[float] = field(default_factory=list)

    @property
    def avg_weight(self) -> float:
        return self.total_weight / self.count if self.count > 0 else 0.0

    @property
    def frequency(self) -> float:
        """Frequency relative to total appearances (set externally)."""
        return 0.0  # Computed during aggregation


@dataclass
class NeuronEdgeProfile:
    """Aggregated edge statistics for a single neuron."""
    layer: int
    neuron: int
    neuron_id: str

    # How many prompts this neuron appeared in
    appearance_count: int = 0

    # Domain vs baseline appearances
    domain_appearance_count: int = 0
    baseline_appearance_count: int = 0

    # Upstream sources: neuron_id -> EdgeStats
    upstream_sources: dict[str, dict] = field(default_factory=dict)

    # Downstream targets: neuron_id -> EdgeStats
    downstream_targets: dict[str, dict] = field(default_factory=dict)

    # Output token associations
    output_token_counts: dict[str, int] = field(default_factory=dict)

    # Co-occurrence: neuron_id -> count (neurons appearing in same graph)
    cooccurrence_counts: dict[str, int] = field(default_factory=dict)


def parse_node_id(node_id: str) -> tuple[int | None, int | None, int | None]:
    """Parse a node_id into (layer, neuron, position).

    Returns (None, None, None) for embeddings or invalid IDs.
    """
    parts = node_id.split("_")
    if len(parts) != 3:
        return None, None, None

    layer_str, neuron_str, pos_str = parts

    # Handle embeddings
    if layer_str == "E":
        return None, None, None

    try:
        layer = int(layer_str)
        neuron = int(neuron_str)
        position = int(pos_str)
        return layer, neuron, position
    except ValueError:
        return None, None, None


def normalize_node_id(node_id: str) -> str:
    """Normalize a node_id by stripping position, keeping only layer and neuron.

    This allows aggregation across positions since RelP tracks attribution
    across positions (attention moves vectors between positions).

    Examples:
        "15_1816_26" -> "15_1816" (MLP neuron)
        "E_128000_0" -> "E_128000" (embedding)
        "L_12345_10" -> "L_12345" (logit)

    Returns the original node_id if format is unexpected.
    """
    parts = node_id.split("_")
    if len(parts) >= 2:
        # Keep layer/type and neuron/token, drop position
        return f"{parts[0]}_{parts[1]}"
    return node_id


def extract_edges_from_graph(graph: dict) -> tuple[
    dict[str, list[tuple[str, float]]],  # incoming edges per node
    dict[str, list[tuple[str, float]]],  # outgoing edges per node
    dict[str, str],  # node_id -> logit token (for logit nodes)
]:
    """Extract edge information from a graph.

    Returns:
        incoming: dict mapping node_id -> list of (source_id, weight)
        outgoing: dict mapping node_id -> list of (target_id, weight)
        logit_tokens: dict mapping logit node_id -> token string
    """
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    logit_tokens = {}

    # Build node info map
    for node in graph.get("nodes", []):
        node_id = node.get("node_id")
        if node.get("isLogit"):
            # Extract token from clerp (format: "Answer: token p=0.95")
            clerp = node.get("clerp", "")
            # Handle both formats: "Answer: token (p=X)" and "token (p=X)"
            if clerp.startswith("Answer:"):
                token = clerp.split("p=")[0].replace("Answer:", "").strip()
            elif "(p=" in clerp:
                # Format: " Dop (p=0.9492)" -> " Dop"
                token = clerp.split("(p=")[0].strip()
            else:
                token = clerp.strip()
            if token:
                logit_tokens[node_id] = token

    # Process edges
    for link in graph.get("links", []):
        source = link.get("source")
        target = link.get("target")
        weight = link.get("weight", 0.0)

        if source and target:
            incoming[target].append((source, weight))
            outgoing[source].append((target, weight))

    return dict(incoming), dict(outgoing), logit_tokens


def aggregate_graph_edges(
    graph: dict,
    profiles: dict[tuple[int, int], NeuronEdgeProfile],
    is_domain: bool,
    target_neurons: set[tuple[int, int]] | None = None,
):
    """Aggregate edge statistics from a single graph into profiles.

    Args:
        graph: The attribution graph
        profiles: Dict of (layer, neuron) -> NeuronEdgeProfile to update
        is_domain: Whether this is a domain prompt (vs baseline)
        target_neurons: If specified, only track these neurons
    """
    incoming, outgoing, logit_tokens = extract_edges_from_graph(graph)

    # First pass: collect all MLP neurons in this graph (for co-occurrence tracking)
    neurons_in_graph: set[tuple[int, int]] = set()
    for node in graph.get("nodes", []):
        node_id = node.get("node_id")
        layer, neuron, position = parse_node_id(node_id)
        if layer is not None and layer != 32:  # Valid MLP neuron
            if target_neurons is None or (layer, neuron) in target_neurons:
                neurons_in_graph.add((layer, neuron))

    # Second pass: process each neuron and update profiles
    for node in graph.get("nodes", []):
        node_id = node.get("node_id")
        layer, neuron, position = parse_node_id(node_id)

        # Skip non-MLP nodes
        if layer is None or layer == 32:  # layer 32 is logits
            continue

        # Filter to target neurons if specified
        if target_neurons and (layer, neuron) not in target_neurons:
            continue

        key = (layer, neuron)

        # Create profile if needed
        if key not in profiles:
            profiles[key] = NeuronEdgeProfile(
                layer=layer,
                neuron=neuron,
                neuron_id=f"L{layer}/N{neuron}",
            )

        profile = profiles[key]
        profile.appearance_count += 1
        if is_domain:
            profile.domain_appearance_count += 1
        else:
            profile.baseline_appearance_count += 1

        # Update co-occurrence counts with all other neurons in this graph
        for other_key in neurons_in_graph:
            if other_key != key:  # Don't count self
                other_layer, other_neuron = other_key
                other_id = f"L{other_layer}/N{other_neuron}"
                profile.cooccurrence_counts[other_id] = (
                    profile.cooccurrence_counts.get(other_id, 0) + 1
                )

        # Aggregate upstream sources (normalized to remove position)
        for source_id, weight in incoming.get(node_id, []):
            # Normalize to aggregate across positions
            norm_source = normalize_node_id(source_id)
            if norm_source not in profile.upstream_sources:
                profile.upstream_sources[norm_source] = {
                    "count": 0,
                    "total_weight": 0.0,
                    "weights": [],
                }
            stats = profile.upstream_sources[norm_source]
            stats["count"] += 1
            stats["total_weight"] += weight
            stats["weights"].append(weight)

        # Aggregate downstream targets (normalized to remove position)
        for target_id, weight in outgoing.get(node_id, []):
            # Normalize to aggregate across positions
            norm_target = normalize_node_id(target_id)
            if norm_target not in profile.downstream_targets:
                profile.downstream_targets[norm_target] = {
                    "count": 0,
                    "total_weight": 0.0,
                    "weights": [],
                }
            stats = profile.downstream_targets[norm_target]
            stats["count"] += 1
            stats["total_weight"] += weight
            stats["weights"].append(weight)

            # Track output token if target is a logit
            if target_id in logit_tokens:
                token = logit_tokens[target_id]
                profile.output_token_counts[token] = (
                    profile.output_token_counts.get(token, 0) + 1
                )


def generate_graph_for_prompt(
    attributor: RelPAttributor,
    tokenizer: AutoTokenizer,
    prompt: str,
    answer_prefix: str = "",
    tau: float = 0.005,
    k: int = 5,
) -> dict | None:
    """Generate an attribution graph for a single prompt.

    Args:
        attributor: The RelP attributor
        tokenizer: Tokenizer
        prompt: The prompt text
        answer_prefix: Optional answer prefix
        tau: Node threshold
        k: Number of top logits

    Returns:
        Graph dict or None on error
    """
    # Apply chat template
    formatted = apply_chat_template(prompt, answer_prefix)

    try:
        graph = attributor.compute_attributions(
            formatted,
            k=k,
            tau=tau,
            compute_edges=True,
        )
        return graph
    except Exception as e:
        print(f"Error generating graph: {e}", file=sys.stderr)
        return None


# Minimal Llama 3.1 chat template
CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{answer_prefix}"""


def apply_chat_template(prompt: str, answer_prefix: str = "") -> str:
    """Apply Llama 3.1 chat template."""
    return CHAT_TEMPLATE.format(prompt=prompt, answer_prefix=answer_prefix)


def load_prompts_from_config(config_path: Path) -> list[dict]:
    """Load prompts from a YAML config file.

    Returns list of {"prompt": str, "answer_prefix": str}
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    prompts = []
    for seq in config.get("sequences", []):
        prompts.append({
            "prompt": seq.get("prompt", ""),
            "answer_prefix": seq.get("answer_prefix", ""),
        })

    return prompts


def sample_fineweb_sentences(n_samples: int, seed: int = 42) -> list[str]:
    """Sample random sentences from FineWeb dataset.

    Uses HuggingFace datasets to stream a small sample.

    Args:
        n_samples: Number of sentences to sample
        seed: Random seed

    Returns:
        List of sentence strings
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Warning: datasets not installed, skipping FineWeb samples", file=sys.stderr)
        return []

    random.seed(seed)

    # Use FineWeb-Edu sample (smaller, educational content)
    print("Loading FineWeb samples...", file=sys.stderr)

    try:
        # Stream to avoid downloading full dataset
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )

        sentences = []
        # Sample more than needed, then randomly select
        buffer = []
        for i, example in enumerate(dataset):
            if i >= n_samples * 10:  # Buffer 10x samples
                break
            text = example.get("text", "")
            # Extract first sentence-like chunk (up to 200 chars)
            if text:
                # Simple sentence extraction
                chunk = text[:500].split(".")[0].strip()
                if 20 < len(chunk) < 200:
                    buffer.append(chunk)

        # Random sample from buffer
        if buffer:
            sentences = random.sample(buffer, min(n_samples, len(buffer)))

        print(f"Sampled {len(sentences)} FineWeb sentences", file=sys.stderr)
        return sentences

    except Exception as e:
        print(f"Warning: Could not load FineWeb: {e}", file=sys.stderr)
        return []


def compute_summary_stats(profiles: dict[tuple[int, int], NeuronEdgeProfile]) -> dict:
    """Compute summary statistics for the profile."""
    stats = {
        "total_neurons": len(profiles),
        "domain_specific_neurons": 0,  # Appear mostly in domain prompts
        "baseline_specific_neurons": 0,  # Appear mostly in baseline
        "general_neurons": 0,  # Appear in both
    }

    for profile in profiles.values():
        domain_ratio = (
            profile.domain_appearance_count / profile.appearance_count
            if profile.appearance_count > 0
            else 0
        )

        if domain_ratio > 0.7:
            stats["domain_specific_neurons"] += 1
        elif domain_ratio < 0.3:
            stats["baseline_specific_neurons"] += 1
        else:
            stats["general_neurons"] += 1

    return stats


def profile_to_dict(profile: NeuronEdgeProfile) -> dict:
    """Convert profile to JSON-serializable dict with computed statistics."""
    # Compute top upstream sources by frequency
    upstream_list = []
    for source_id, stats in profile.upstream_sources.items():
        freq = stats["count"] / profile.appearance_count if profile.appearance_count > 0 else 0
        avg_weight = stats["total_weight"] / stats["count"] if stats["count"] > 0 else 0
        upstream_list.append({
            "source": source_id,
            "count": stats["count"],
            "frequency": freq,
            "avg_weight": avg_weight,
        })
    upstream_list.sort(key=lambda x: x["frequency"], reverse=True)

    # Compute top downstream targets by frequency
    downstream_list = []
    for target_id, stats in profile.downstream_targets.items():
        freq = stats["count"] / profile.appearance_count if profile.appearance_count > 0 else 0
        avg_weight = stats["total_weight"] / stats["count"] if stats["count"] > 0 else 0
        downstream_list.append({
            "target": target_id,
            "count": stats["count"],
            "frequency": freq,
            "avg_weight": avg_weight,
        })
    downstream_list.sort(key=lambda x: x["frequency"], reverse=True)

    # Compute output token associations
    token_list = []
    for token, count in profile.output_token_counts.items():
        freq = count / profile.appearance_count if profile.appearance_count > 0 else 0
        token_list.append({
            "token": token,
            "count": count,
            "frequency": freq,
        })
    token_list.sort(key=lambda x: x["frequency"], reverse=True)

    # Domain specificity score
    domain_ratio = (
        profile.domain_appearance_count / profile.appearance_count
        if profile.appearance_count > 0
        else 0
    )

    # Compute top co-occurring neurons by frequency
    cooccurrence_list = []
    for other_id, count in profile.cooccurrence_counts.items():
        freq = count / profile.appearance_count if profile.appearance_count > 0 else 0
        cooccurrence_list.append({
            "neuron_id": other_id,
            "count": count,
            "frequency": freq,
        })
    cooccurrence_list.sort(key=lambda x: x["frequency"], reverse=True)

    return {
        "layer": profile.layer,
        "neuron": profile.neuron,
        "neuron_id": profile.neuron_id,
        "appearance_count": profile.appearance_count,
        "domain_appearance_count": profile.domain_appearance_count,
        "baseline_appearance_count": profile.baseline_appearance_count,
        "domain_specificity": domain_ratio,
        "top_upstream_sources": upstream_list[:20],  # Top 20
        "top_downstream_targets": downstream_list[:20],
        "output_token_associations": token_list[:20],
        "top_cooccurring_neurons": cooccurrence_list[:50],  # Top 50 co-occurring neurons
    }


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate edge statistics across prompts to understand neuron connectivity"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--domain-prompts", type=Path,
        help="YAML config file with domain-specific prompts"
    )
    input_group.add_argument(
        "--graphs-dir", type=Path,
        help="Directory containing pre-generated graph JSON files"
    )

    # Baseline options
    parser.add_argument(
        "--fineweb-samples", type=int, default=0,
        help="Number of random FineWeb sentences for baseline (default: 0)"
    )
    parser.add_argument(
        "--baseline-prompts", type=Path,
        help="Optional YAML config with baseline prompts (alternative to FineWeb)"
    )

    # Neuron filtering
    parser.add_argument(
        "--neurons", nargs="+", type=str,
        help="Only track specific neurons (format: layer:neuron, e.g., 15:7890)"
    )

    # Graph generation options
    parser.add_argument(
        "--tau", type=float, default=0.005,
        help="Node threshold for graph generation (default: 0.005)"
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of top logits to trace (default: 5)"
    )

    # Output
    parser.add_argument(
        "-o", "--output", type=Path, required=True,
        help="Output JSON file"
    )

    # Model options
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or path"
    )

    args = parser.parse_args()

    # Parse target neurons if specified
    target_neurons = None
    if args.neurons:
        target_neurons = set()
        for spec in args.neurons:
            try:
                layer, neuron = spec.split(":")
                target_neurons.add((int(layer), int(neuron)))
            except ValueError:
                print(f"Error: Invalid neuron spec '{spec}'", file=sys.stderr)
                sys.exit(1)

    # Initialize profiles dict
    profiles: dict[tuple[int, int], NeuronEdgeProfile] = {}

    if args.graphs_dir:
        # Load from existing graphs
        graph_files = list(args.graphs_dir.glob("*.json"))
        print(f"Found {len(graph_files)} graph files in {args.graphs_dir}", file=sys.stderr)

        for graph_path in tqdm(graph_files, desc="Processing graphs"):
            try:
                with open(graph_path) as f:
                    graph = json.load(f)
                # Assume all are domain graphs when loading from directory
                aggregate_graph_edges(graph, profiles, is_domain=True, target_neurons=target_neurons)
            except Exception as e:
                print(f"Error loading {graph_path}: {e}", file=sys.stderr)

    else:
        # Generate graphs on the fly
        print(f"Loading model {args.model}...", file=sys.stderr)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        attributor = RelPAttributor(model, tokenizer)

        # Load domain prompts
        domain_prompts = load_prompts_from_config(args.domain_prompts)
        print(f"Loaded {len(domain_prompts)} domain prompts", file=sys.stderr)

        # Load baseline prompts
        baseline_prompts = []
        if args.baseline_prompts:
            baseline_prompts = load_prompts_from_config(args.baseline_prompts)
        elif args.fineweb_samples > 0:
            fineweb_sentences = sample_fineweb_sentences(args.fineweb_samples)
            baseline_prompts = [{"prompt": s, "answer_prefix": ""} for s in fineweb_sentences]

        print(f"Loaded {len(baseline_prompts)} baseline prompts", file=sys.stderr)

        # Process domain prompts
        print("Processing domain prompts...", file=sys.stderr)
        for p in tqdm(domain_prompts, desc="Domain prompts"):
            graph = generate_graph_for_prompt(
                attributor, tokenizer,
                p["prompt"], p.get("answer_prefix", ""),
                tau=args.tau, k=args.k,
            )
            if graph:
                aggregate_graph_edges(graph, profiles, is_domain=True, target_neurons=target_neurons)

        # Process baseline prompts
        if baseline_prompts:
            print("Processing baseline prompts...", file=sys.stderr)
            for p in tqdm(baseline_prompts, desc="Baseline prompts"):
                graph = generate_graph_for_prompt(
                    attributor, tokenizer,
                    p["prompt"], p.get("answer_prefix", ""),
                    tau=args.tau, k=args.k,
                )
                if graph:
                    aggregate_graph_edges(graph, profiles, is_domain=False, target_neurons=target_neurons)

    # Compute summary statistics
    summary = compute_summary_stats(profiles)

    # Prepare output
    output_data = {
        "metadata": {
            "model": args.model,
            "tau": args.tau,
            "k": args.k,
            "num_domain_prompts": len(domain_prompts) if not args.graphs_dir else "unknown",
            "num_baseline_prompts": len(baseline_prompts) if not args.graphs_dir else 0,
        },
        "summary": summary,
        "profiles": [profile_to_dict(p) for p in profiles.values()],
    }

    # Sort profiles by appearance count
    output_data["profiles"].sort(key=lambda x: x["appearance_count"], reverse=True)

    # Save output
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved {len(profiles)} neuron profiles to {args.output}", file=sys.stderr)
    print("Summary:", file=sys.stderr)
    print(f"  Total neurons: {summary['total_neurons']}", file=sys.stderr)
    print(f"  Domain-specific: {summary['domain_specific_neurons']}", file=sys.stderr)
    print(f"  Baseline-specific: {summary['baseline_specific_neurons']}", file=sys.stderr)
    print(f"  General: {summary['general_neurons']}", file=sys.stderr)


if __name__ == "__main__":
    main()
