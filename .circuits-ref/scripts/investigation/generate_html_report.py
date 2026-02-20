#!/usr/bin/env python3
"""Generate beautiful HTML reports from neuron investigation data.

Usage:
    # V2 mode (default) - uses investigation.json as primary input
    python scripts/generate_html_report.py outputs/investigations/L4_N10555_investigation.json --v2

    # Batch mode (processes all *_investigation.json files)
    python scripts/generate_html_report.py --batch outputs/investigations/ --v2

    # Custom output directory
    python scripts/generate_html_report.py investigation.json -o frontend/reports/ --v2

    # Use Opus for highest quality
    python scripts/generate_html_report.py investigation.json --model opus --v2

    # V1 mode (legacy) - requires dashboard.json
    python scripts/generate_html_report.py outputs/investigations/L4_N10555_dashboard.json --v1
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


async def process_single_v1(
    dashboard_path: Path,
    output_dir: Path,
    model: str,
) -> Path:
    """Process a single dashboard file using V1 agent."""
    from neuron_scientist.dashboard_agent import DashboardHTMLAgent

    agent = DashboardHTMLAgent(
        dashboard_path=dashboard_path,
        output_dir=output_dir,
        model=model,
    )
    return await agent.generate()


async def process_single_v2(
    investigation_path: Path,
    output_dir: Path,
    model: str,
    negative_investigation_path: Path | None = None,
) -> Path:
    """Process using V2 agent with investigation data."""
    from neuron_scientist.dashboard_agent_v2 import DashboardAgentV2

    agent = DashboardAgentV2(
        investigation_path=investigation_path,
        output_dir=output_dir,
        model=model,
        negative_investigation_path=negative_investigation_path,
    )
    return await agent.generate()


async def process_batch(
    input_dir: Path,
    output_dir: Path,
    model: str,
    max_concurrent: int = 3,
    use_v2: bool = True,
) -> list[Path]:
    """Process all investigation/dashboard files in a directory."""
    if use_v2:
        # Look for investigation files
        input_files = list(input_dir.glob("*_investigation.json"))
        file_type = "investigation"
    else:
        # V1 mode - look for dashboard files
        input_files = list(input_dir.glob("*_dashboard.json"))
        file_type = "dashboard"

    print(f"Found {len(input_files)} {file_type} files in {input_dir}")
    print(f"Using {'V2' if use_v2 else 'V1'} agent")

    results = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(path: Path) -> Path:
        async with semaphore:
            try:
                if use_v2:
                    return await process_single_v2(path, output_dir, model)
                else:
                    return await process_single_v1(path, output_dir, model)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                import traceback
                traceback.print_exc()
                return None

    tasks = [process_with_semaphore(p) for p in input_files]
    results = await asyncio.gather(*tasks)

    return [r for r in results if r is not None]


def main():
    parser = argparse.ArgumentParser(
        description="Generate beautiful HTML reports from neuron investigation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Path to investigation JSON file (or directory with --batch)",
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("frontend/reports"),
        help="Output directory for HTML files (default: frontend/reports/)",
    )

    parser.add_argument(
        "--model",
        choices=["sonnet", "opus", "haiku"],
        default="opus",
        help="Claude model to use (default: opus)",
    )

    parser.add_argument(
        "--v2",
        action="store_true",
        help="Use V2 agent with freeform evidence section (default)",
    )

    parser.add_argument(
        "--v1",
        action="store_true",
        help="Use V1 agent (legacy, requires dashboard.json)",
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all *_investigation.json files in the input directory",
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum concurrent generations in batch mode (default: 3)",
    )

    parser.add_argument(
        "--negative-investigation",
        type=Path,
        default=None,
        help="Path to negative-polarity investigation JSON for merged bipolar dashboard",
    )

    args = parser.parse_args()

    if args.input is None:
        parser.print_help()
        return 1

    # Default to V2 unless explicitly using --v1
    use_v2 = not args.v1

    if args.batch:
        if not args.input.is_dir():
            print(f"Error: {args.input} is not a directory. Use --batch with a directory path.")
            return 1

        results = asyncio.run(process_batch(
            args.input,
            args.output,
            args.model,
            args.max_concurrent,
            use_v2=use_v2,
        ))
        print(f"\nGenerated {len(results)} HTML reports in {args.output}")

    else:
        if not args.input.exists():
            print(f"Error: {args.input} does not exist")
            return 1

        if use_v2:
            print("Using V2 agent with freeform evidence section")
            neg_path = getattr(args, 'negative_investigation', None)
            if neg_path:
                print(f"Merging with negative investigation: {neg_path}")
            result = asyncio.run(process_single_v2(
                args.input,
                args.output,
                args.model,
                negative_investigation_path=neg_path,
            ))
        else:
            print("Using V1 agent (legacy)")
            result = asyncio.run(process_single_v1(
                args.input,
                args.output,
                args.model,
            ))

        print(f"\nGenerated: {result}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
