#!/usr/bin/env python3
"""
Build comprehensive neuron profiles combining:
1. NeuronDB max-activating labels
2. Edge statistics (upstream/downstream patterns)
3. Output projection analysis (token promotion/suppression)

This creates a complete picture of what a neuron does for agentic autointerp.

Usage:
    # Profile specific neurons
    python scripts/build_neuron_profile.py --neurons 15:7890 20:5432 --edge-stats data/medical_edge_stats_1000_labeled.json

    # Profile all neurons from edge stats
    python scripts/build_neuron_profile.py --edge-stats data/medical_edge_stats_1000_labeled.json --top-k 100 -o profiles.json

    # Profile neurons from a graph
    python scripts/build_neuron_profile.py --from-graph outputs/my-graph.json --edge-stats data/edge_stats.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_neuron_spec(spec: str) -> tuple[int, int]:
    """Parse 'layer:neuron' or 'Llayer/Nneuron' format."""
    if ":" in spec:
        layer, neuron = spec.split(":")
        return int(layer), int(neuron)
    elif "/" in spec:
        # L15/N7890 format
        layer = int(spec.split("/")[0][1:])
        neuron = int(spec.split("/")[1][1:])
        return layer, neuron
    else:
        raise ValueError(f"Invalid neuron spec: {spec}")


def load_edge_stats(path: Path) -> dict[tuple[int, int], dict]:
    """Load edge stats indexed by (layer, neuron)."""
    with open(path) as f:
        data = json.load(f)

    profiles = {}
    for p in data.get("profiles", []):
        key = (p["layer"], p["neuron"])
        profiles[key] = p
    return profiles


def compute_output_projection(
    model,
    tokenizer,
    layer: int,
    neuron: int,
    top_k: int = 20,
) -> dict:
    """Compute output projection for a single neuron."""
    down_proj = model.model.layers[layer].mlp.down_proj.weight
    output_direction = down_proj[:, neuron].float()

    lm_head = model.lm_head.weight.float()
    logit_contributions = lm_head @ output_direction

    # Get top promoted/suppressed
    top_vals, top_idx = logit_contributions.topk(k=top_k)
    bot_vals, bot_idx = logit_contributions.topk(k=top_k, largest=False)

    promoted = []
    for i in range(top_k):
        token_id = top_idx[i].item()
        promoted.append({
            "token": tokenizer.decode([token_id]),
            "token_id": token_id,
            "logit": top_vals[i].item(),
        })

    suppressed = []
    for i in range(top_k):
        token_id = bot_idx[i].item()
        suppressed.append({
            "token": tokenizer.decode([token_id]),
            "token_id": token_id,
            "logit": bot_vals[i].item(),
        })

    # Stats
    output_norm = output_direction.norm().item()
    logit_mean = logit_contributions.mean().item()
    logit_std = logit_contributions.std().item()

    return {
        "output_norm": output_norm,
        "logit_mean": logit_mean,
        "logit_std": logit_std,
        "logit_range": [logit_contributions.min().item(), logit_contributions.max().item()],
        "top_promoted": promoted,
        "top_suppressed": suppressed,
    }


def build_comprehensive_profile(
    model,
    tokenizer,
    layer: int,
    neuron: int,
    edge_stats: dict[tuple[int, int], dict] | None = None,
    output_proj_top_k: int = 20,
) -> dict:
    """
    Build comprehensive profile for a single neuron.

    Combines:
    - Output projection analysis
    - Edge statistics (if available)
    - NeuronDB label (from edge stats if present)
    """
    neuron_id = f"L{layer}/N{neuron}"

    profile = {
        "neuron_id": neuron_id,
        "layer": layer,
        "neuron": neuron,
    }

    # Output projection analysis
    print(f"  Computing output projection for {neuron_id}...")
    output_proj = compute_output_projection(
        model, tokenizer, layer, neuron, top_k=output_proj_top_k
    )
    profile["output_projection"] = output_proj

    # Edge statistics
    if edge_stats and (layer, neuron) in edge_stats:
        edge_profile = edge_stats[(layer, neuron)]
        profile["edge_stats"] = {
            "appearance_count": edge_profile.get("appearance_count", 0),
            "domain_specificity": edge_profile.get("domain_specificity", 0),
            "top_upstream_sources": edge_profile.get("top_upstream_sources", []),
            "top_downstream_targets": edge_profile.get("top_downstream_targets", []),
            "output_token_associations": edge_profile.get("output_token_associations", []),
        }
        # NeuronDB label if present
        if "max_act_label" in edge_profile:
            profile["max_act_label"] = edge_profile["max_act_label"]
    else:
        profile["edge_stats"] = None
        profile["max_act_label"] = None

    return profile


def format_profile_for_display(profile: dict) -> str:
    """Format profile for human-readable display."""
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"NEURON: {profile['neuron_id']}")
    lines.append(f"{'='*60}")

    # NeuronDB Label
    if profile.get("max_act_label"):
        label = profile["max_act_label"]
        if len(label) > 200:
            label = label[:200] + "..."
        lines.append("\nMAX-ACTIVATING LABEL (NeuronDB):")
        lines.append(f"  {label}")

    # Output Projection
    op = profile.get("output_projection", {})
    lines.append("\nOUTPUT PROJECTION:")
    lines.append(f"  Output norm: {op.get('output_norm', 0):.4f}")
    lines.append(f"  Logit range: [{op.get('logit_range', [0,0])[0]:.4f}, {op.get('logit_range', [0,0])[1]:.4f}]")

    lines.append("\n  Top Promoted Tokens:")
    for t in op.get("top_promoted", [])[:5]:
        lines.append(f"    {repr(t['token']):20s} logit: {t['logit']:+.4f}")

    lines.append("\n  Top Suppressed Tokens:")
    for t in op.get("top_suppressed", [])[:5]:
        lines.append(f"    {repr(t['token']):20s} logit: {t['logit']:+.4f}")

    # Edge Stats
    es = profile.get("edge_stats")
    if es:
        lines.append("\nEDGE STATISTICS:")
        lines.append(f"  Appearances: {es.get('appearance_count', 0)}")
        lines.append(f"  Domain specificity: {es.get('domain_specificity', 0):.2f}")

        if es.get("top_upstream_sources"):
            lines.append("\n  Top Upstream Sources:")
            for src in es["top_upstream_sources"][:3]:
                lines.append(f"    ← {src['source']}: freq={src['frequency']:.2f}, weight={src['avg_weight']:.3f}")

        if es.get("top_downstream_targets"):
            lines.append("\n  Top Downstream Targets:")
            for tgt in es["top_downstream_targets"][:3]:
                lines.append(f"    → {tgt['target']}: freq={tgt['frequency']:.2f}, weight={tgt['avg_weight']:.3f}")
    else:
        lines.append("\nEDGE STATISTICS: Not available")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Build comprehensive neuron profiles"
    )

    # Input options
    parser.add_argument(
        "--neurons",
        nargs="+",
        help="Specific neurons to profile (format: layer:neuron or Llayer/Nneuron)"
    )
    parser.add_argument(
        "--from-graph",
        type=Path,
        help="Profile all neurons from a graph JSON"
    )
    parser.add_argument(
        "--edge-stats",
        type=Path,
        help="Edge stats JSON for upstream/downstream patterns"
    )

    # Options
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Profile top K neurons from edge stats by appearance count"
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use for output projection"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output JSON file"
    )
    parser.add_argument(
        "--min-layer",
        type=int,
        default=15,
        help="Minimum layer for output projection analysis (early layers have weak effects)"
    )

    args = parser.parse_args()

    # Determine neurons to profile
    neuron_list = []

    if args.neurons:
        for spec in args.neurons:
            try:
                layer, neuron = parse_neuron_spec(spec)
                neuron_list.append((layer, neuron))
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)

    elif args.from_graph:
        # Extract neurons from graph
        with open(args.from_graph) as f:
            graph = json.load(f)
        for node in graph.get("nodes", []):
            layer = node.get("layer")
            if layer in ("E", "32", None):
                continue
            try:
                layer_int = int(layer)
                neuron = node.get("feature")
                if neuron is not None and layer_int >= args.min_layer:
                    neuron_list.append((layer_int, neuron))
            except (ValueError, TypeError):
                continue

    elif args.edge_stats:
        # Take top-k from edge stats
        edge_data = load_edge_stats(args.edge_stats)
        sorted_profiles = sorted(
            edge_data.items(),
            key=lambda x: x[1].get("appearance_count", 0),
            reverse=True
        )
        for (layer, neuron), _ in sorted_profiles[:args.top_k]:
            if layer >= args.min_layer:
                neuron_list.append((layer, neuron))

    else:
        print("Error: Specify --neurons, --from-graph, or --edge-stats", file=sys.stderr)
        sys.exit(1)

    if not neuron_list:
        print("No neurons to profile", file=sys.stderr)
        sys.exit(1)

    # Load edge stats if provided
    edge_stats = None
    if args.edge_stats:
        print(f"Loading edge stats from {args.edge_stats}...")
        edge_stats = load_edge_stats(args.edge_stats)

    # Load model
    print(f"Loading model {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Build profiles
    print(f"Building profiles for {len(neuron_list)} neurons...")
    profiles = []

    for layer, neuron in neuron_list:
        profile = build_comprehensive_profile(
            model, tokenizer, layer, neuron,
            edge_stats=edge_stats,
        )
        profiles.append(profile)

        # Print for display
        print(format_profile_for_display(profile))
        print()

    # Save output
    if args.output:
        print(f"Saving profiles to {args.output}...")
        with open(args.output, "w") as f:
            json.dump({"profiles": profiles}, f, indent=2)

    print(f"\nDone! Profiled {len(profiles)} neurons.")


if __name__ == "__main__":
    main()
