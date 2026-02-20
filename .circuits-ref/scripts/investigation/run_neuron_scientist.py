#!/usr/bin/env python3
"""Run the Neuron Scientist agent on a single neuron.

Uses the Claude Agent SDK to autonomously investigate a neuron through
hypothesis-driven experimentation.

Usage:
    # Basic investigation
    python scripts/run_neuron_scientist.py --neuron L15/N7890

    # With initial hypothesis
    python scripts/run_neuron_scientist.py \
        --neuron L15/N7890 \
        --hypothesis "This neuron responds to medical terminology"

    # With edge stats for connectivity analysis
    python scripts/run_neuron_scientist.py \
        --neuron L15/N7890 \
        --edge-stats data/fabric_edge_stats.json

    # Quick mode (fewer experiments)
    python scripts/run_neuron_scientist.py \
        --neuron L15/N7890 \
        --max-experiments 20
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neuron_scientist import investigate_neuron


async def main_async(args):
    """Async main function."""

    # Create output path
    if args.output:
        output_dir = args.output
    else:
        output_dir = Path("neuron_reports/json")

    # Run investigation
    print("=" * 60)
    print("NEURON SCIENTIST AGENT")
    print("=" * 60)
    print(f"Target: {args.neuron}")
    print(f"Model: {args.model}")
    print(f"Prompt version: V{args.prompt_version}")
    print(f"Max experiments: {args.max_experiments}")
    if args.hypothesis:
        print(f"Initial hypothesis: {args.hypothesis}")
    if args.edge_stats:
        print(f"Edge stats: {args.edge_stats}")
    if args.labels:
        print(f"Labels: {args.labels}")
    print("=" * 60)
    print()

    investigation = await investigate_neuron(
        neuron_id=args.neuron,
        initial_label=args.label or "",
        initial_hypothesis=args.hypothesis or "",
        edge_stats_path=args.edge_stats,
        labels_path=args.labels,
        output_dir=output_dir,
        max_experiments=args.max_experiments,
        model=args.model,
        prompt_version=args.prompt_version,
    )

    # Print summary
    print()
    print("=" * 60)
    print("INVESTIGATION COMPLETE")
    print("=" * 60)
    print(f"Neuron: {investigation.neuron_id}")
    print(f"Experiments run: {investigation.total_experiments}")
    print(f"Confidence: {investigation.confidence:.0%}")
    print()
    print(f"Activating prompts found: {len(investigation.activating_prompts)}")
    print(f"Non-activating prompts: {len(investigation.non_activating_prompts)}")
    print(f"Ablation experiments: {len(investigation.ablation_effects)}")
    print()

    if investigation.hypotheses_tested:
        print("Hypotheses tested:")
        for h in investigation.hypotheses_tested[:5]:
            status = h.get('status', 'pending')
            hyp = h.get('hypothesis', 'N/A')
            print(f"  [{status}] {hyp}")

    if investigation.key_findings:
        print("\nKey findings:")
        for f in investigation.key_findings[:5]:
            print(f"  - {f}")

    safe_id = args.neuron.replace("/", "_")
    print(f"\nFull report: {output_dir}/{safe_id}_investigation.json")


def main():
    parser = argparse.ArgumentParser(
        description="Run Neuron Scientist agent investigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--neuron", type=str, required=True,
        help="Neuron ID to investigate (e.g., L15/N7890)"
    )
    parser.add_argument(
        "--hypothesis", type=str, default=None,
        help="Initial hypothesis to test"
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="Initial label from batch labeling"
    )
    parser.add_argument(
        "--edge-stats", type=Path, default=Path("data/medical_edge_stats_v6_enriched.json"),
        help="Path to edge statistics JSON (default: enriched v6 with DER and output projections)"
    )
    parser.add_argument(
        "--labels", type=Path, default=Path("data/neuron_labels_combined.json"),
        help="Path to neuron labels JSON (default: combined FineWeb + v6 labels)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory for investigation report"
    )
    parser.add_argument(
        "--max-experiments", type=int, default=100,
        help="Maximum experiments to run (default: 100)"
    )
    parser.add_argument(
        "--model", type=str, default="opus", choices=["opus", "sonnet", "haiku"],
        help="Model to use (default: opus)"
    )
    parser.add_argument(
        "--prompt-version", type=int, default=5, choices=[1, 2, 3, 4, 5],
        help="System prompt version: 1=original, 2=V2, 3=V3, 4=V4 two-phase, 5=V5 simplified (default: 5)"
    )

    args = parser.parse_args()

    # Run async
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
