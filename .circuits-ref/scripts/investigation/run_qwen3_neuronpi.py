#!/usr/bin/env python3
"""Run NeuronPI investigation on Qwen3-32B model.

This script configures the tools to use Qwen3-32B instead of Llama and runs
NeuronPI investigations.

Usage:
    # Investigate a single neuron
    python scripts/run_qwen3_neuronpi.py --neuron L33/N4047

    # Investigate with initial hypothesis
    python scripts/run_qwen3_neuronpi.py --neuron L33/N4047 \
        --hypothesis "Wine/grape detector that promotes wine-related tokens"

    # Investigate multiple neurons
    python scripts/run_qwen3_neuronpi.py --neurons L33/N4047 L7/N11846

    # Use a different Qwen model
    python scripts/run_qwen3_neuronpi.py --neuron L7/N11846 --model qwen3-8b
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neuron_scientist.pi_agent import run_neuron_pi
from neuron_scientist.tools import MODEL_CONFIGS, get_model_config, set_model_config


async def investigate_neuron(
    neuron_id: str,
    initial_hypothesis: str = "",
    output_dir: Path = None,
    model: str = "sonnet",
    scientist_model: str = "opus",
    max_review_iterations: int = 2,
    skip_review: bool = False,
):
    """Run NeuronPI investigation on a single neuron.

    Args:
        neuron_id: Neuron ID in format "L{layer}/N{neuron}"
        initial_hypothesis: Optional starting hypothesis
        output_dir: Output directory for results
        model: Claude model for PI orchestration
        scientist_model: Claude model for scientist agent
        max_review_iterations: Max GPT review iterations
        skip_review: Skip GPT review step
    """
    if output_dir is None:
        output_dir = Path("neuron_reports/json")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = get_model_config()
    print(f"\n{'='*60}")
    print(f"NeuronPI Investigation: {neuron_id}")
    print(f"Model: {config.name}")
    print(f"Hypothesis: {initial_hypothesis or '(none - will be generated)'}")
    print(f"{'='*60}\n")

    try:
        result = await run_neuron_pi(
            neuron_id=neuron_id,
            initial_label="",
            initial_hypothesis=initial_hypothesis,
            edge_stats_path=None,
            output_dir=output_dir,
            model=model,
            scientist_model=scientist_model,
            max_review_iterations=max_review_iterations,
            skip_review=skip_review,
            skip_dashboard=False,
        )

        print(f"\n{'='*60}")
        print(f"Investigation Complete: {neuron_id}")
        print(f"Final verdict: {result.final_verdict[:100]}...")
        print(f"Iterations: {result.iterations}")
        if result.investigation:
            print(f"Confidence: {result.investigation.confidence}")
        print(f"{'='*60}\n")

        return result

    except Exception as e:
        print(f"\nERROR investigating {neuron_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    parser = argparse.ArgumentParser(
        description="Run NeuronPI investigation on Qwen3 models"
    )

    # Model selection
    parser.add_argument(
        "--model-config",
        choices=list(MODEL_CONFIGS.keys()),
        default="qwen3-32b",
        help="Model configuration to use (default: qwen3-32b)"
    )

    # Neuron selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--neuron",
        type=str,
        help="Single neuron ID to investigate (e.g., L33/N4047)"
    )
    group.add_argument(
        "--neurons",
        nargs="+",
        type=str,
        help="Multiple neuron IDs to investigate"
    )
    group.add_argument(
        "--neurons-file",
        type=str,
        help="JSON file containing neuron list"
    )

    # Hypothesis
    parser.add_argument(
        "--hypothesis",
        type=str,
        default="",
        help="Initial hypothesis for the investigation"
    )
    parser.add_argument(
        "--hypotheses-file",
        type=str,
        help="JSON file mapping neuron IDs to hypotheses"
    )

    # Claude model selection
    parser.add_argument(
        "--claude-model",
        choices=["opus", "sonnet", "haiku"],
        default="sonnet",
        help="Claude model for PI orchestration (default: sonnet)"
    )
    parser.add_argument(
        "--scientist-model",
        choices=["opus", "sonnet", "haiku"],
        default="opus",
        help="Claude model for scientist agent (default: opus)"
    )

    # Investigation options
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Max GPT review iterations (default: 2)"
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="Skip GPT review step"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("neuron_reports/json"),
        help="Output directory for investigation results"
    )

    args = parser.parse_args()

    # Set model configuration
    print(f"Setting model config: {args.model_config}")
    set_model_config(args.model_config)

    # Get list of neurons to investigate
    neurons = []
    hypotheses = {}

    if args.neuron:
        neurons = [args.neuron]
        if args.hypothesis:
            hypotheses[args.neuron] = args.hypothesis
    elif args.neurons:
        neurons = args.neurons
    elif args.neurons_file:
        with open(args.neurons_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        neurons.append(item)
                    elif isinstance(item, dict):
                        nid = f"L{item['layer']}/N{item['neuron']}"
                        neurons.append(nid)
                        if 'hypothesis' in item:
                            hypotheses[nid] = item['hypothesis']
                    elif isinstance(item, list):
                        neurons.append(f"L{item[0]}/N{item[1]}")

    # Load hypotheses from file if provided
    if args.hypotheses_file:
        with open(args.hypotheses_file) as f:
            hypotheses.update(json.load(f))

    if not neurons:
        print("No neurons to investigate!")
        sys.exit(1)

    print(f"\nWill investigate {len(neurons)} neurons:")
    for nid in neurons:
        h = hypotheses.get(nid, "(no hypothesis)")[:50]
        print(f"  - {nid}: {h}...")

    # Run investigations
    results = []
    for neuron_id in neurons:
        hypothesis = hypotheses.get(neuron_id, args.hypothesis)
        result = await investigate_neuron(
            neuron_id=neuron_id,
            initial_hypothesis=hypothesis,
            output_dir=args.output_dir,
            model=args.claude_model,
            scientist_model=args.scientist_model,
            max_review_iterations=args.max_iterations,
            skip_review=args.skip_review,
        )
        results.append((neuron_id, result))

    # Summary
    print("\n" + "="*60)
    print("INVESTIGATION SUMMARY")
    print("="*60)
    success = sum(1 for _, r in results if r is not None)
    print(f"Completed: {success}/{len(neurons)}")
    for nid, result in results:
        status = "SUCCESS" if result else "FAILED"
        verdict = result.final_verdict[:50] + "..." if result else "N/A"
        print(f"  {nid}: {status} - {verdict}")


if __name__ == "__main__":
    asyncio.run(main())
