#!/usr/bin/env python3
"""Thorough clustering evaluation suite.

Phase 1: Embedding coherence (automated, all clusters)
Phase 2: Sample clusters for LLM eval + cross-config comparison
Phase 3: Prepare activation fingerprinting data

Usage:
    .venv/bin/python scripts/thorough_eval.py
"""

import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np

random.seed(42)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIGS = {
    27: {"name": "relp_sq_q90_t20", "file": "data/clustering_sweep/config_027_assignments.json"},
    29: {"name": "inter_sq_q85_t10", "file": "data/clustering_sweep/config_029_assignments.json"},
    53: {"name": "weight_sq_q90_t10", "file": "data/clustering_sweep/config_053_assignments.json"},
}

EVAL_DIR = "data/thorough_eval"


def load_assignments(config_id):
    info = CONFIGS[config_id]
    return json.load(open(info["file"]))


def assignments_to_clusters(assigns):
    clusters = defaultdict(list)
    for node_str, cid in assigns.items():
        parts = node_str.split("/")
        layer = int(parts[0][1:])
        neuron = int(parts[1][1:])
        clusters[cid].append((layer, neuron))
    return dict(clusters)


def get_labels_batch(neurons_list):
    """Batch lookup labels from DuckDB."""
    import duckdb
    db = duckdb.connect("data/qwen32b_neurons.duckdb", read_only=True)

    labels = {}
    # Batch query
    for batch_start in range(0, len(neurons_list), 5000):
        batch = neurons_list[batch_start:batch_start + 5000]
        conditions = " OR ".join(
            f"(layer={l} AND neuron={n})" for l, n in batch
        )
        if not conditions:
            continue
        rows = db.execute(
            f"SELECT layer, neuron, label FROM neurons WHERE {conditions}"
        ).fetchall()
        for l, n, label in rows:
            labels[(l, n)] = label or "?"

    db.close()
    return labels


def phase1_embedding_coherence():
    """Compute sentence-embedding coherence for ALL eligible clusters."""
    print("\n" + "=" * 60)
    print("PHASE 1: Embedding Coherence (automated)")
    print("=" * 60)

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("ERROR: sentence-transformers or sklearn not installed")
        return {}

    model = SentenceTransformer("all-MiniLM-L6-v2")
    results = {}

    for config_id, info in CONFIGS.items():
        print(f"\n  Config {config_id} ({info['name']})...")
        assigns = load_assignments(config_id)
        clusters = assignments_to_clusters(assigns)

        # Filter to size 5-100
        eligible = {k: v for k, v in clusters.items() if 5 <= len(v) <= 100}
        print(f"    {len(eligible)} eligible clusters (size 5-100)")

        # Get all labels needed
        all_neurons = []
        for neurons in eligible.values():
            all_neurons.extend(neurons)
        labels = get_labels_batch(all_neurons)

        # Compute coherence per cluster
        coherences = []
        for cid, neurons in eligible.items():
            neuron_labels = [labels.get((l, n), "?") for l, n in neurons]
            neuron_labels = [l for l in neuron_labels if l != "?"]
            if len(neuron_labels) < 3:
                continue

            embeddings = model.encode(neuron_labels)
            sim_matrix = cosine_similarity(embeddings)
            n = len(neuron_labels)
            pairwise = [sim_matrix[i, j] for i in range(n) for j in range(i + 1, n)]
            coherences.append({
                "cluster_id": cid,
                "size": len(neurons),
                "n_labels": len(neuron_labels),
                "coherence": float(np.mean(pairwise)),
            })

        coherence_vals = [c["coherence"] for c in coherences]
        results[config_id] = {
            "n_clusters": len(coherences),
            "mean": round(float(np.mean(coherence_vals)), 4),
            "median": round(float(np.median(coherence_vals)), 4),
            "p25": round(float(np.percentile(coherence_vals, 25)), 4),
            "p75": round(float(np.percentile(coherence_vals, 75)), 4),
            "p90": round(float(np.percentile(coherence_vals, 90)), 4),
            "pct_above_0.3": round(100 * sum(1 for v in coherence_vals if v > 0.3) / len(coherence_vals), 1),
            "pct_above_0.4": round(100 * sum(1 for v in coherence_vals if v > 0.4) / len(coherence_vals), 1),
            "all_coherences": coherences,
        }

        print(f"    Mean coherence: {results[config_id]['mean']:.4f}")
        print(f"    Median: {results[config_id]['median']:.4f}")
        print(f"    >0.3: {results[config_id]['pct_above_0.3']}%")
        print(f"    >0.4: {results[config_id]['pct_above_0.4']}%")

    return results


def phase2_sample_clusters():
    """Sample 50 clusters per config for LLM eval with 3 raters."""
    print("\n" + "=" * 60)
    print("PHASE 2: Sampling clusters for LLM eval")
    print("=" * 60)

    all_samples = {}

    for config_id, info in CONFIGS.items():
        assigns = load_assignments(config_id)
        clusters = assignments_to_clusters(assigns)
        eligible = {k: v for k, v in clusters.items() if 5 <= len(v) <= 100}

        # Stratified sample: 17 small, 17 medium, 16 large
        small = [k for k, v in eligible.items() if len(v) < 15]
        medium = [k for k, v in eligible.items() if 15 <= len(v) < 40]
        large = [k for k, v in eligible.items() if len(v) >= 40]

        sample_ids = (
            random.sample(small, min(17, len(small))) +
            random.sample(medium, min(17, len(medium))) +
            random.sample(large, min(16, len(large)))
        )[:50]

        # Get labels
        all_neurons = []
        for cid in sample_ids:
            all_neurons.extend(eligible[cid])
        labels = get_labels_batch(all_neurons)

        samples = []
        for cid in sample_ids:
            neurons = eligible[cid]
            neuron_data = []
            for l, n in sorted(neurons):
                neuron_data.append({
                    "layer": l, "neuron": n,
                    "label": labels.get((l, n), "?"),
                })
            samples.append({
                "config_id": config_id,
                "config_name": info["name"],
                "cluster_id": cid,
                "size": len(neurons),
                "neurons": neuron_data,
            })

        all_samples[config_id] = samples
        print(f"  Config {config_id}: sampled {len(samples)} clusters")

    # Split into batches for agents (10 clusters each, mixed across configs)
    # 150 clusters total, 10 per batch = 15 batches, × 3 raters = 45 agents
    all_clusters = []
    for config_id, samples in all_samples.items():
        all_clusters.extend(samples)

    random.shuffle(all_clusters)

    n_per_batch = 10
    n_batches = (len(all_clusters) + n_per_batch - 1) // n_per_batch

    for i in range(n_batches):
        batch = all_clusters[i * n_per_batch:(i + 1) * n_per_batch]
        for rater in range(3):
            path = f"{EVAL_DIR}/llm_eval_batch_{i:02d}_rater_{rater}.json"
            with open(path, "w") as f:
                json.dump(batch, f, indent=2)

    print(f"  Created {n_batches} batches × 3 raters = {n_batches * 3} eval files")
    print(f"  Total clusters: {len(all_clusters)}")

    return all_samples, n_batches


def phase3_cross_config_comparison(all_samples):
    """Find neurons clustered differently across configs and prepare comparison data."""
    print("\n" + "=" * 60)
    print("PHASE 3: Cross-config comparison")
    print("=" * 60)

    # Build per-config assignment maps
    config_assigns = {}
    for config_id in CONFIGS:
        config_assigns[config_id] = load_assignments(config_id)

    # Find neurons that appear in at least 2 configs
    neuron_configs = defaultdict(dict)
    for config_id, assigns in config_assigns.items():
        for node_str, cid in assigns.items():
            neuron_configs[node_str][config_id] = cid

    multi_config = {n: c for n, c in neuron_configs.items() if len(c) >= 2}
    print(f"  Neurons in ≥2 configs: {len(multi_config):,}")

    # Find groups where configs DISAGREE
    # Pick 30 clusters from config 27 (RelP), find where their neurons land in configs 53 (weight)
    assigns_27 = config_assigns[27]
    assigns_53 = config_assigns[53]
    clusters_27 = assignments_to_clusters(assigns_27)
    clusters_53 = assignments_to_clusters(assigns_53)

    eligible_27 = {k: v for k, v in clusters_27.items() if 5 <= len(v) <= 30}
    sample_27 = random.sample(list(eligible_27.keys()), min(30, len(eligible_27)))

    comparisons = []
    all_neurons_needed = []
    for cid27 in sample_27:
        neurons_27 = eligible_27[cid27]
        all_neurons_needed.extend(neurons_27)

        # Where do these neurons land in config 53?
        weight_clusters = Counter()
        for l, n in neurons_27:
            node_str = f"L{l}/N{n}"
            if node_str in assigns_53:
                weight_clusters[assigns_53[node_str]] += 1

        comparisons.append({
            "relp_cluster_id": cid27,
            "relp_size": len(neurons_27),
            "neurons": [{"layer": l, "neuron": n} for l, n in sorted(neurons_27)],
            "weight_distribution": dict(weight_clusters.most_common(5)),
            "n_weight_clusters": len(weight_clusters),
            "weight_dominant_pct": round(
                100 * weight_clusters.most_common(1)[0][1] / len(neurons_27), 1
            ) if weight_clusters else 0,
        })

    # Add labels
    labels = get_labels_batch(all_neurons_needed)
    for comp in comparisons:
        for n in comp["neurons"]:
            n["label"] = labels.get((n["layer"], n["neuron"]), "?")

    # Save comparison data
    path = f"{EVAL_DIR}/cross_config_comparisons.json"
    with open(path, "w") as f:
        json.dump(comparisons, f, indent=2)

    # Also prepare evaluation batches (10 per agent)
    n_batches = (len(comparisons) + 9) // 10
    for i in range(n_batches):
        batch = comparisons[i * 10:(i + 1) * 10]
        path = f"{EVAL_DIR}/cross_config_batch_{i:02d}.json"
        with open(path, "w") as f:
            json.dump(batch, f, indent=2)

    print(f"  Created {len(comparisons)} comparisons in {n_batches} batches")

    return comparisons


def phase4_activation_fingerprint_prep(all_samples):
    """Prepare data for GPU-based activation fingerprinting."""
    print("\n" + "=" * 60)
    print("PHASE 4: Activation fingerprinting prep")
    print("=" * 60)

    # Select 20 clusters per config for activation fingerprinting
    fp_clusters = {}
    for config_id, samples in all_samples.items():
        # Pick 20 from the 50 sampled
        fp_sample = random.sample(samples, min(20, len(samples)))
        fp_clusters[config_id] = fp_sample

    # Collect all unique neurons needed
    all_neurons = set()
    for config_id, clusters in fp_clusters.items():
        for cluster in clusters:
            for n in cluster["neurons"]:
                all_neurons.add((n["layer"], n["neuron"]))

    print(f"  Unique neurons for fingerprinting: {len(all_neurons):,}")
    print(f"  Clusters per config: {[len(v) for v in fp_clusters.values()]}")

    # Save the fingerprinting job spec
    fp_spec = {
        "neurons": [{"layer": l, "neuron": n} for l, n in sorted(all_neurons)],
        "clusters": {
            str(config_id): [
                {"cluster_id": c["cluster_id"], "config_id": config_id,
                 "neurons": [(n["layer"], n["neuron"]) for n in c["neurons"]]}
                for c in clusters
            ]
            for config_id, clusters in fp_clusters.items()
        },
        "prompts": [
            # Diverse prompts for activation capture
            "The capital of France is",
            "In quantum mechanics, the wave function",
            "She couldn't stop crying after hearing the",
            "The recipe calls for two cups of",
            "According to the latest research on climate",
            "def fibonacci(n):",
            "The Supreme Court ruled that",
            "In the year 1066, William the Conqueror",
            "The patient presented with acute",
            "To solve this equation, first",
            "The endangered species was last seen in",
            "He apologized for his mistake but",
            "The chemical reaction between sodium and",
            "In machine learning, gradient descent",
            "The novel explores themes of identity and",
            "The price of gold increased by",
            "During the Renaissance, artists like",
            "The function returns the maximum value of",
            "She felt a deep sense of gratitude for",
            "The experiment demonstrated that under pressure",
            "In Buddhist philosophy, suffering arises from",
            "The algorithm has a time complexity of",
            "The treaty was signed between the two",
            "Photosynthesis converts carbon dioxide and water into",
            "The quarterback threw a perfect spiral to",
            "Marx argued that the means of production",
            "The bridge collapsed due to structural",
            "In organic chemistry, benzene is an example of",
            "The painting depicts a serene landscape with",
            "The server responded with a 404 error because",
        ],
    }

    path = f"{EVAL_DIR}/activation_fingerprint_spec.json"
    with open(path, "w") as f:
        json.dump(fp_spec, f, indent=2)

    print(f"  Saved fingerprint spec: {len(fp_spec['neurons'])} neurons, "
          f"{len(fp_spec['prompts'])} prompts")

    return fp_spec


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)

    t0 = time.time()

    # Phase 1: Embedding coherence
    embed_results = phase1_embedding_coherence()
    with open(f"{EVAL_DIR}/embedding_coherence.json", "w") as f:
        json.dump(embed_results, f, indent=2)

    # Phase 2: Sample for LLM eval
    all_samples, n_llm_batches = phase2_sample_clusters()

    # Phase 3: Cross-config comparison
    comparisons = phase3_cross_config_comparison(all_samples)

    # Phase 4: Activation fingerprinting prep
    fp_spec = phase4_activation_fingerprint_prep(all_samples)

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"THOROUGH EVAL SETUP COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*60}")

    print("\nPhase 1 (Embedding Coherence):")
    for cid, r in embed_results.items():
        print(f"  Config {cid}: mean={r['mean']:.4f}, >0.3={r['pct_above_0.3']}%, "
              f">0.4={r['pct_above_0.4']}% ({r['n_clusters']} clusters)")

    print(f"\nPhase 2 (LLM Eval): {n_llm_batches} batches × 3 raters = "
          f"{n_llm_batches * 3} agent calls")
    print(f"  Files: {EVAL_DIR}/llm_eval_batch_*_rater_*.json")

    print(f"\nPhase 3 (Cross-Config): {len(comparisons)} comparisons")
    print(f"  Files: {EVAL_DIR}/cross_config_batch_*.json")

    print(f"\nPhase 4 (Activation FP): {len(fp_spec['neurons'])} neurons × "
          f"{len(fp_spec['prompts'])} prompts")
    print(f"  File: {EVAL_DIR}/activation_fingerprint_spec.json")

    print("\nNext steps:")
    print("  1. Launch LLM eval agents (see llm_eval_batch files)")
    print("  2. Launch cross-config eval agents (see cross_config_batch files)")
    print("  3. Submit activation fingerprinting SLURM job")


if __name__ == "__main__":
    main()
