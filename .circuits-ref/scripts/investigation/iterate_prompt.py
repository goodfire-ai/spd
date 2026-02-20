#!/usr/bin/env python3
"""Run one L28/N774 investigation and extract the characterization.

Usage:
    .venv/bin/python scripts/iterate_prompt.py [variant_number]
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

VARIANT = int(sys.argv[1]) if len(sys.argv) > 1 else 0


async def main():
    from neuron_scientist.agent import investigate_neuron

    t0 = time.time()
    print(f"=== Variant {VARIANT} ===", flush=True)

    result = await investigate_neuron(
        neuron_id="L28/N774",
        initial_label="Fires on statistical ratio phrases promoting ten",
        edge_stats_path=Path("data/fineweb_50k_edge_stats_enriched.json"),
        output_dir=Path("neuron_reports/json"),
        max_experiments=50,
        model="sonnet",
        gpu_server_url="http://localhost:8478",
    )

    elapsed = time.time() - t0

    # Extract characterization
    inv_path = Path("neuron_reports/json/L28_N774_investigation.json")
    d = json.loads(inv_path.read_text())
    char = d.get("characterization", {})

    # Get top activating prompts with tokens
    ev = d.get("evidence", {})
    act = sorted(ev.get("activating_prompts", []),
                 key=lambda x: x.get("activation", 0), reverse=True)

    print(f"\nCompleted in {elapsed:.0f}s, {d.get('total_experiments', '?')} experiments")
    print(f"\n--- INPUT FUNCTION ---\n{char.get('input_function', '')[:400]}")
    print(f"\n--- OUTPUT FUNCTION ---\n{char.get('output_function', '')[:400]}")
    print(f"\n--- FINAL HYPOTHESIS ---\n{char.get('final_hypothesis', '')[:400]}")

    print("\n--- TOP 5 ACTIVATING PROMPTS ---")
    for p in act[:5]:
        print(f"  act={p.get('activation',0):.2f} token=\"{p.get('token','?')}\" | {p.get('prompt','')[:80]}")

    # Save variant result
    variant_path = Path(f"neuron_reports/json/prompt_variant_{VARIANT}.json")
    variant_path.write_text(json.dumps({
        "variant": VARIANT,
        "elapsed_s": elapsed,
        "experiments": d.get("total_experiments"),
        "input_function": char.get("input_function", ""),
        "output_function": char.get("output_function", ""),
        "final_hypothesis": char.get("final_hypothesis", ""),
        "top_5_activating": [
            {"act": p.get("activation", 0), "token": p.get("token", "?"),
             "prompt": p.get("prompt", "")[:100]}
            for p in act[:5]
        ],
    }, indent=2))
    print(f"\nSaved to {variant_path}")


asyncio.run(main())
