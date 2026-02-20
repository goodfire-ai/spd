#!/usr/bin/env python3
"""Test scaling of attribution graph generation with node count."""

import sys
import time

sys.path.insert(0, '.')

from circuits.pipeline import PipelineConfig, run_pipeline

PROMPT = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|}

Environment: ipython
Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024
<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.
Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.Do not use variables.
{
    "name": "air_quality",
    "description": "Retrieve the air quality index for a specific location.",
    "parameters": {
        "type": "dict",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city that you want to get the air quality index for."
            },
            "date": {
                "type": "string",
                "description": "The date (month-day-year) you want to get the air quality index for."
            }
        },
        "required": [
            "location",
            "date"
        ]
    }
}
What is the air quality index in London 2022/08/16?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{"name": "air_quality", "parameters": {"location": "London", "date": "'''

# Test different tau values to get different node counts
TAU_VALUES = [0.05, 0.03, 0.02, 0.015, 0.01, 0.008]

def main():
    results = []

    for tau in TAU_VALUES:
        print(f"\n{'='*60}")
        print(f"Testing tau={tau}")
        print(f"{'='*60}")

        config = PipelineConfig(
            use_chat_template=False,
            contrastive_tokens=["202", "08"],
            tau=tau,
            label_neurons=False,  # Skip DB for speed
            run_llm_analysis=False,  # Skip LLM for speed
            output_dir=f"outputs/scaling_test/tau_{tau}",
            save_intermediate=False,
        )

        start = time.time()
        result = run_pipeline(PROMPT, config=config, verbose=True)
        elapsed = time.time() - start

        # Extract metrics from result
        graph = result.get('graph', {})
        nodes = len([n for n in graph.get('nodes', []) if n.get('node_type') == 'feature'])
        edges = len(graph.get('edges', []))

        results.append({
            'tau': tau,
            'nodes': nodes,
            'edges': edges,
            'time': elapsed
        })

        print(f"  Nodes: {nodes}, Edges: {edges}, Time: {elapsed:.1f}s")

    print("\n" + "="*60)
    print("SCALING RESULTS")
    print("="*60)
    print(f"{'tau':>8} {'nodes':>8} {'edges':>10} {'time':>8} {'time/node':>10} {'time/edge':>10}")
    print("-"*60)
    for r in results:
        time_per_node = r['time'] / r['nodes'] if r['nodes'] > 0 else 0
        time_per_edge = r['time'] / r['edges'] if r['edges'] > 0 else 0
        print(f"{r['tau']:>8.3f} {r['nodes']:>8} {r['edges']:>10} {r['time']:>7.1f}s {time_per_node:>9.3f}s {time_per_edge:>9.5f}s")

    # Estimate scaling
    print("\n" + "="*60)
    print("SCALING ANALYSIS")
    print("="*60)

    if len(results) >= 2:
        # Check if O(n), O(n²), or O(edges)
        import math

        r1, r2 = results[0], results[-1]
        n1, n2 = r1['nodes'], r2['nodes']
        e1, e2 = r1['edges'], r2['edges']
        t1, t2 = r1['time'], r2['time']

        if n1 > 0 and n2 > 0 and n1 != n2:
            # Linear scaling: t = k*n => k = t/n
            k_linear = (t2 - t1) / (n2 - n1) if n2 != n1 else 0

            # Quadratic scaling: t = k*n² => log(t2/t1) / log(n2/n1) ≈ 2
            ratio_n = n2 / n1
            ratio_t = t2 / t1
            exponent = math.log(ratio_t) / math.log(ratio_n) if ratio_n > 1 else 0

            # Edge scaling: t = k*edges
            k_edge = (t2 - t1) / (e2 - e1) if e2 != e1 else 0

            print(f"Node ratio: {n2}/{n1} = {ratio_n:.2f}x")
            print(f"Time ratio: {t2:.1f}/{t1:.1f} = {ratio_t:.2f}x")
            print(f"Edge ratio: {e2}/{e1} = {e2/e1:.2f}x")
            print(f"\nEstimated exponent (t ∝ n^α): α = {exponent:.2f}")
            print("  - If α ≈ 1: O(n) linear")
            print("  - If α ≈ 2: O(n²) quadratic")
            print(f"\nPer-node marginal cost: {k_linear:.3f}s/node")
            print(f"Per-edge marginal cost: {k_edge:.5f}s/edge")

if __name__ == "__main__":
    main()
