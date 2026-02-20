#!/usr/bin/env python3
"""
Precompute input projections for all neurons in edge stats.


For each neuron, computes the static input sensitivity: which vocabulary tokens
would most activate or suppress this neuron based on the MLP weights.

Uses the combined formula: SiLU(embedding @ gate_proj) * (embedding @ up_proj)
This accounts for Llama's gated MLP architecture.
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_layer_input_projections(
    model,
    tokenizer,
    layer: int,
    top_k: int = 10,
    device: str = "cuda",
) -> dict[int, dict]:
    """
    Compute input projections for all neurons in a layer.

    Returns dict mapping neuron_idx -> {
        "activates": [{"token": str, "token_id": int, "weight": float}, ...],
        "suppresses": [{"token": str, "token_id": int, "weight": float}, ...],
        "input_norm": float,
    }
    """
    # Get weights
    embeddings = model.model.embed_tokens.weight.float().to(device)  # [vocab, d_model]
    up_proj = model.model.layers[layer].mlp.up_proj.weight.float().to(device)  # [d_ffn, d_model]
    gate_proj = model.model.layers[layer].mlp.gate_proj.weight.float().to(device)  # [d_ffn, d_model]

    vocab_size, d_model = embeddings.shape
    d_ffn = up_proj.shape[0]

    # Compute projections: [vocab, d_ffn]
    # up_vals[v, n] = embedding[v] @ up_proj[n]
    up_vals = embeddings @ up_proj.T  # [vocab, d_ffn]
    gate_vals = embeddings @ gate_proj.T  # [vocab, d_ffn]

    # Combined: SiLU(gate) * up
    # SiLU(x) = x * sigmoid(x)
    combined = torch.nn.functional.silu(gate_vals) * up_vals  # [vocab, d_ffn]

    # Also compute input norm (norm of up_proj row for each neuron)
    input_norms = up_proj.norm(dim=1)  # [d_ffn]

    results = {}

    for neuron_idx in range(d_ffn):
        sensitivity = combined[:, neuron_idx]

        # Get top activating tokens
        top_vals, top_idx = sensitivity.topk(top_k)
        # Get top suppressing tokens
        bot_vals, bot_idx = sensitivity.topk(top_k, largest=False)

        activates = []
        for val, idx in zip(top_vals, top_idx):
            token = tokenizer.decode([idx.item()])
            activates.append({
                "token": token,
                "token_id": idx.item(),
                "weight": round(val.item(), 6),
            })

        suppresses = []
        for val, idx in zip(bot_vals, bot_idx):
            token = tokenizer.decode([idx.item()])
            suppresses.append({
                "token": token,
                "token_id": idx.item(),
                "weight": round(val.item(), 6),
            })

        results[neuron_idx] = {
            "activates": activates,
            "suppresses": suppresses,
            "input_norm": round(input_norms[neuron_idx].item(), 6),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Precompute input projections for neurons")
    parser.add_argument(
        "--edge-stats",
        type=Path,
        default=Path("data/medical_edge_stats_v6_enriched.json"),
        help="Path to edge stats JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: overwrite input)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top activating/suppressing tokens to store",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated list of layers to process (default: all layers in edge stats)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use",
    )
    args = parser.parse_args()

    output_path = args.output or args.edge_stats

    # Load edge stats
    print(f"Loading edge stats from {args.edge_stats}...", flush=True)
    with open(args.edge_stats) as f:
        edge_stats = json.load(f)

    profiles = edge_stats.get("profiles", [])
    print(f"Found {len(profiles)} neuron profiles", flush=True)

    # Get unique layers
    layers_in_stats = sorted(set(p["layer"] for p in profiles))
    if args.layers:
        layers_to_process = [int(x) for x in args.layers.split(",")]
    else:
        layers_to_process = layers_in_stats

    print(f"Layers to process: {layers_to_process}", flush=True)

    # Load model
    print(f"\nLoading model {args.model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Build neuron_id -> profile index mapping
    profile_idx = {p["neuron_id"]: i for i, p in enumerate(profiles)}

    # Process each layer
    for layer in layers_to_process:
        print(f"\nProcessing layer {layer}...", flush=True)

        # Compute projections for this layer
        layer_projections = compute_layer_input_projections(
            model, tokenizer, layer, top_k=args.top_k
        )

        # Update profiles
        updated = 0
        for p in profiles:
            if p["layer"] != layer:
                continue

            neuron_id = p["neuron_id"]
            neuron_idx = int(neuron_id.split("/")[1][1:])  # L31/N8359 -> 8359

            if neuron_idx in layer_projections:
                p["input_projection"] = layer_projections[neuron_idx]
                updated += 1

        print(f"  Updated {updated} profiles", flush=True)

        # Clear cache for this layer
        del layer_projections
        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    print(f"\nSaving to {output_path}...", flush=True)
    with open(output_path, "w") as f:
        json.dump(edge_stats, f, indent=2)

    print("Done!", flush=True)


if __name__ == "__main__":
    main()
