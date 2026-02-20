#!/usr/bin/env python3
"""Activation fingerprinting for cluster evaluation.

For each sampled cluster, captures neuron activations across 30 diverse prompts
and computes within-cluster pairwise correlation. High correlation = neurons
genuinely co-activate = real functional circuit.

Usage:
    sbatch --partition=h200-reserved --gres=gpu:1 --mem=200G --time=2:00:00 \
        --export=ALL,PYTHONUNBUFFERED=1 --chdir=... \
        --wrap=".venv/bin/python scripts/activation_fingerprint_eval.py"
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuron_scientist.tools import get_model_and_tokenizer, set_model_config


def main():
    print("Activation Fingerprinting for Cluster Evaluation")

    spec = json.load(open("data/thorough_eval/activation_fingerprint_spec.json"))
    neurons = spec["neurons"]  # [{layer, neuron}, ...]
    prompts = spec["prompts"]
    cluster_specs = spec["clusters"]  # {config_id: [{cluster_id, neurons}, ...]}

    print(f"  {len(neurons)} neurons, {len(prompts)} prompts")
    print(f"  Configs: {list(cluster_specs.keys())}")

    set_model_config("qwen3-32b")
    model, tokenizer = get_model_and_tokenizer()

    # Build neuron lookup: (layer, neuron) -> index
    neuron_idx = {(n["layer"], n["neuron"]): i for i, n in enumerate(neurons)}
    n_neurons = len(neurons)

    # Capture activations: [n_prompts, n_neurons]
    activation_matrix = np.zeros((len(prompts), n_neurons), dtype=np.float32)

    # Group neurons by layer for efficient hooking
    layers_needed = defaultdict(list)
    for n in neurons:
        layers_needed[n["layer"]].append(n["neuron"])

    for pi, prompt in enumerate(prompts):
        if pi % 5 == 0:
            print(f"  Prompt {pi+1}/{len(prompts)}: {prompt[:50]}...")

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        captured = {}
        hooks = []

        for layer_idx, neuron_list in layers_needed.items():
            def make_hook(li, nl):
                def hook_fn(module, input, output):
                    h = input[0].detach()  # [batch, seq, intermediate]
                    for nidx in nl:
                        # Max activation across positions
                        captured[(li, nidx)] = h[0, :, nidx].max().item()
                return hook_fn
            hook = model.model.layers[layer_idx].mlp.down_proj.register_forward_hook(
                make_hook(layer_idx, neuron_list))
            hooks.append(hook)

        with torch.no_grad():
            model(input_ids)

        for h in hooks:
            h.remove()

        for (l, n), val in captured.items():
            idx = neuron_idx.get((l, n))
            if idx is not None:
                activation_matrix[pi, idx] = val

    # Compute per-cluster correlation
    print("\nComputing per-cluster correlations...")
    results = {}

    for config_id, clusters in cluster_specs.items():
        config_results = []
        for cluster in clusters:
            cid = cluster["cluster_id"]
            neuron_pairs = cluster["neurons"]  # [(layer, neuron), ...]

            # Get column indices for this cluster's neurons
            col_indices = []
            for ln in neuron_pairs:
                idx = neuron_idx.get(tuple(ln))
                if idx is not None:
                    col_indices.append(idx)

            if len(col_indices) < 3:
                config_results.append({
                    "cluster_id": cid, "n_neurons": len(col_indices),
                    "mean_correlation": None, "reason": "too few neurons"
                })
                continue

            # Extract sub-matrix and compute pairwise Pearson
            sub = activation_matrix[:, col_indices]  # [n_prompts, n_cluster_neurons]
            corr = np.corrcoef(sub.T)  # [n, n]
            n = len(col_indices)
            pairwise = [corr[i, j] for i in range(n) for j in range(i+1, n)
                        if not np.isnan(corr[i, j])]

            config_results.append({
                "cluster_id": cid,
                "n_neurons": len(col_indices),
                "mean_correlation": round(float(np.mean(pairwise)), 4) if pairwise else None,
                "median_correlation": round(float(np.median(pairwise)), 4) if pairwise else None,
                "min_correlation": round(float(np.min(pairwise)), 4) if pairwise else None,
                "max_correlation": round(float(np.max(pairwise)), 4) if pairwise else None,
                "pct_positive": round(100 * sum(1 for p in pairwise if p > 0) / len(pairwise), 1) if pairwise else None,
            })

        results[config_id] = config_results

    # Summary
    print(f"\n{'='*60}")
    print("ACTIVATION FINGERPRINT RESULTS")
    print(f"{'='*60}")

    for config_id, clusters in results.items():
        valid = [c for c in clusters if c["mean_correlation"] is not None]
        if valid:
            corrs = [c["mean_correlation"] for c in valid]
            print(f"\n  Config {config_id}:")
            print(f"    Clusters evaluated: {len(valid)}")
            print(f"    Mean correlation: {np.mean(corrs):.4f}")
            print(f"    Median correlation: {np.median(corrs):.4f}")
            print(f"    >0.1: {sum(1 for c in corrs if c > 0.1)}/{len(valid)}")
            print(f"    >0.3: {sum(1 for c in corrs if c > 0.3)}/{len(valid)}")

    # Save
    output_path = "data/thorough_eval/activation_fingerprints.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
