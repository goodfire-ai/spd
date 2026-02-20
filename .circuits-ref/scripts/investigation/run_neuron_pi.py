#!/usr/bin/env python3
"""Run NeuronPI - the Principal Investigator orchestrator.

This script runs the full neuron investigation pipeline:
1. Initial investigation with NeuronScientist
2. GPT peer review (via Codex MCP)
3. Revision cycles based on feedback
4. Dashboard generation

Usage:
    # Full pipeline
    python scripts/run_neuron_pi.py --neuron L15/N7890

    # With options
    python scripts/run_neuron_pi.py \\
        --neuron L15/N7890 \\
        --hypothesis "Medical terminology detector" \\
        --edge-stats data/medical_edge_stats.json \\
        --max-iterations 3 \\
        --model sonnet \\
        --scientist-model opus

    # Skip GPT review (investigation + dashboard only)
    python scripts/run_neuron_pi.py --neuron L15/N7890 --skip-review

    # Investigation only (no dashboard)
    python scripts/run_neuron_pi.py --neuron L15/N7890 --skip-dashboard

    # Both skipped (just run investigation)
    python scripts/run_neuron_pi.py --neuron L15/N7890 --skip-review --skip-dashboard
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

# Load environment
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


async def main():
    parser = argparse.ArgumentParser(
        description="Run NeuronPI - Principal Investigator orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        "--neuron", "-n",
        required=True,
        help="Neuron ID to investigate (e.g., L15/N7890)",
    )

    # Optional context
    parser.add_argument(
        "--label", "-l",
        default="",
        help="Initial label for the neuron",
    )
    parser.add_argument(
        "--hypothesis", "-H",
        default="",
        help="Starting hypothesis to test",
    )
    parser.add_argument(
        "--edge-stats", "-e",
        type=Path,
        default=None,
        help="Path to edge statistics JSON",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/neuron_labels_combined.json"),
        help="Path to neuron labels JSON (default: combined FineWeb + v6)",
    )

    # Output
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("neuron_reports/json"),
        help="Output directory for JSON files (default: neuron_reports/json)",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        default="sonnet",
        choices=["opus", "sonnet", "haiku"],
        help="Model for PI orchestration (default: sonnet)",
    )
    parser.add_argument(
        "--scientist-model",
        default="opus",
        choices=["opus", "sonnet", "haiku"],
        help="Model for NeuronScientist investigations (default: opus)",
    )

    # Pipeline control
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Max review iterations (default: 3)",
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="Skip GPT review (investigation + dashboard only)",
    )
    parser.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="Skip dashboard generation",
    )
    parser.add_argument(
        "--review-only",
        action="store_true",
        help="Review-only mode: skip initial investigation, load existing one",
    )
    parser.add_argument(
        "--investigation", "-i",
        type=Path,
        default=None,
        help="Path to existing investigation JSON (required for --review-only)",
    )

    args = parser.parse_args()

    # Validate review-only mode
    if args.review_only:
        if not args.investigation:
            # Try to find investigation based on neuron ID
            safe_id = args.neuron.replace("/", "_")
            default_path = args.output_dir / f"{safe_id}_investigation.json"
            if default_path.exists():
                args.investigation = default_path
            else:
                parser.error("--review-only requires --investigation or existing investigation file")
        if not args.investigation.exists():
            parser.error(f"Investigation file not found: {args.investigation}")

    # Import here to avoid loading models at parse time
    from neuron_scientist.pi_agent import run_neuron_pi

    print(f"\n{'='*60}")
    print("NeuronPI - Principal Investigator")
    print(f"{'='*60}")
    print(f"Neuron: {args.neuron}")
    if args.review_only:
        print("MODE: Review-only")
        print(f"Investigation: {args.investigation}")
    else:
        print(f"Label: {args.label or '(none)'}")
        print(f"Hypothesis: {args.hypothesis or '(none)'}")
    print(f"Edge stats: {args.edge_stats or '(none)'}")
    print(f"Labels: {args.labels}")
    print(f"Output dir: {args.output_dir}")
    print(f"PI model: {args.model}")
    print(f"Scientist model: {args.scientist_model}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Skip review: {args.skip_review}")
    print(f"Skip dashboard: {args.skip_dashboard}")
    print(f"{'='*60}\n")

    # Run pipeline
    result = await run_neuron_pi(
        neuron_id=args.neuron,
        initial_label=args.label,
        initial_hypothesis=args.hypothesis,
        edge_stats_path=args.edge_stats,
        labels_path=args.labels,
        output_dir=args.output_dir,
        model=args.model,
        scientist_model=args.scientist_model,
        max_review_iterations=args.max_iterations,
        skip_review=args.skip_review,
        skip_dashboard=args.skip_dashboard,
        existing_investigation_path=args.investigation if args.review_only else None,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("Pipeline Complete")
    print(f"{'='*60}")
    print(f"Neuron: {result.neuron_id}")
    print(f"Final verdict: {result.final_verdict}")
    print(f"Iterations: {result.iterations}")
    print(f"Reviews: {len(result.review_history)}")

    if result.investigation:
        print("\nInvestigation:")
        print(f"  Confidence: {result.investigation.confidence}")
        print(f"  Experiments: {result.investigation.total_experiments}")
        print(f"  Input function: {result.investigation.input_function[:100]}...")
        print(f"  Output function: {result.investigation.output_function[:100]}...")

    if result.review_history:
        print("\nReview history:")
        for i, review in enumerate(result.review_history):
            print(f"  [{i+1}] {review.verdict} - {review.confidence_assessment}")
            if review.gaps:
                print(f"      Gaps: {', '.join(review.gaps[:3])}")

    print("\nOutput files:")
    print(f"  Investigation: {result.investigation_path}")
    print(f"  Dashboard JSON: {result.dashboard_json_path}")
    if result.dashboard_path:
        print(f"  Dashboard HTML: {result.dashboard_path}")

    if result.error:
        print(f"\nError: {result.error}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
