#!/usr/bin/env python3
"""Run NeuronSkeptic on an existing investigation.

This script runs adversarial testing on a completed investigation to
stress-test the hypothesis before submitting to review.

Usage:
    # Run skeptic on an existing investigation
    python scripts/run_skeptic.py \
        --investigation neuron_reports/json/L15_N7890_investigation.json

    # With specific model
    python scripts/run_skeptic.py \
        --investigation neuron_reports/json/L15_N7890_investigation.json \
        --model opus

    # Output to specific directory
    python scripts/run_skeptic.py \
        --investigation neuron_reports/json/L15_N7890_investigation.json \
        --output-dir outputs/skeptic_reports/
"""

import argparse
import asyncio
import json
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
        description="Run NeuronSkeptic adversarial testing on an investigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--investigation", "-i",
        required=True,
        type=Path,
        help="Path to investigation JSON file",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("neuron_reports/json"),
        help="Output directory for skeptic report (default: neuron_reports/json)",
    )
    parser.add_argument(
        "--model", "-m",
        default="sonnet",
        choices=["opus", "sonnet", "haiku"],
        help="Claude model for skeptic agent (default: sonnet)",
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
        help="Path to neuron labels JSON",
    )
    parser.add_argument(
        "--target-model",
        default="qwen3-32b",
        choices=["llama-3.1-8b", "qwen3-32b", "qwen3-8b"],
        help="Target model being investigated (default: qwen3-32b)",
    )

    args = parser.parse_args()

    # Set model config for the target model
    from neuron_scientist.tools import set_model_config
    set_model_config(args.target_model)

    # Validate investigation file
    if not args.investigation.exists():
        print(f"Error: Investigation file not found: {args.investigation}")
        return 1

    # Load investigation
    print(f"Loading investigation: {args.investigation}")
    with open(args.investigation) as f:
        data = json.load(f)

    from neuron_scientist.schemas import NeuronInvestigation
    investigation = NeuronInvestigation.from_dict(data)

    print(f"\n{'='*60}")
    print("NeuronSkeptic - Adversarial Testing")
    print(f"{'='*60}")
    print(f"Neuron: {investigation.neuron_id}")
    print(f"Hypothesis: {investigation.final_hypothesis[:80]}...")
    print(f"Scientist confidence: {investigation.confidence}")
    print(f"Model: {args.model}")
    print(f"{'='*60}\n")

    # Run skeptic
    from neuron_scientist import run_skeptic

    report = await run_skeptic(
        neuron_id=investigation.neuron_id,
        investigation=investigation,
        edge_stats_path=args.edge_stats,
        labels_path=args.labels,
        model=args.model,
    )

    # Save report
    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = investigation.neuron_id.replace("/", "_")
    report_path = args.output_dir / f"{safe_id}_skeptic_report.json"

    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("Skeptic Report Complete")
    print(f"{'='*60}")
    print(f"Verdict: {report.verdict}")
    print(f"Confidence adjustment: {report.confidence_adjustment:+.2f}")
    print(f"Selectivity score: {report.selectivity_score:.2f}")
    print(f"False positive rate: {report.false_positive_rate:.2f}")
    print(f"False negative rate: {report.false_negative_rate:.2f}")
    print(f"\nAlternative hypotheses tested: {len(report.alternative_hypotheses)}")
    for alt in report.alternative_hypotheses:
        print(f"  - {alt.alternative}: {alt.verdict}")
    print(f"\nBoundary tests: {len(report.boundary_tests)}")
    passed = sum(1 for t in report.boundary_tests if t.passed)
    print(f"  - Passed: {passed}/{len(report.boundary_tests)}")
    print(f"\nConfounds found: {len(report.confounds)}")
    for c in report.confounds:
        print(f"  - [{c.severity}] {c.factor}: {c.description}")
    print("\nKey challenges:")
    for challenge in report.key_challenges[:5]:
        print(f"  - {challenge}")
    if report.revised_hypothesis:
        print(f"\nRevised hypothesis: {report.revised_hypothesis}")
    print(f"\nReport saved to: {report_path}")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
