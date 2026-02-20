#!/usr/bin/env python3
"""
Full-Scale Neuron Labeling Script.

Orchestrates the complete labeling of all 173,923 neurons in the fineweb edge stats
using a two-pass approach:

1. OUTPUT pass (L31→L0): What does each neuron DO when it fires?
2. INPUT pass (L0→L31): What TRIGGERS each neuron to fire?

Layer dependencies require sequential processing:
- Output pass: L31 first (no downstream deps), then L30 (uses L31 labels), etc.
- Input pass: L0 first (no upstream deps), then L1 (uses L0 labels), etc.

Within each layer, neurons are processed in parallel batches.

Usage:
    # Full run (both passes)
    python scripts/run_full_labeling.py

    # Output pass only
    python scripts/run_full_labeling.py --output-only

    # Input pass only (requires output pass complete)
    python scripts/run_full_labeling.py --input-only

    # Resume from checkpoint
    python scripts/run_full_labeling.py --resume

    # Dry run (show what would be done)
    python scripts/run_full_labeling.py --dry-run

Estimated time: ~29 hours total (~14.5 hours per pass)
Estimated cost: ~$5,700 (at GPT-5 Jan 2026 pricing)
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def get_neuron_counts(edge_stats_path: Path, min_appearances: int = 100) -> dict:
    """Count neurons per layer that meet minimum appearance threshold."""
    console.print(f"[dim]Loading edge stats from {edge_stats_path}...[/dim]")

    with open(edge_stats_path) as f:
        edge_stats = json.load(f)

    profiles = edge_stats.get("profiles", [])

    # Baseline neurons to exclude
    baseline_neurons = {"L0/N491", "L0/N8268", "L0/N10585", "L1/N2427"}

    layer_counts = {}
    total = 0

    for p in profiles:
        if p.get("appearance_count", 0) < min_appearances:
            continue
        if p["neuron_id"] in baseline_neurons:
            continue

        layer = p["layer"]
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
        total += 1

    return {
        "total": total,
        "by_layer": layer_counts,
        "layers": sorted(layer_counts.keys()),
    }


def estimate_cost_and_time(neuron_count: int, batch_size: int = 200) -> dict:
    """Estimate cost and time for labeling.

    Based on GPT-5 Jan 2026 pricing:
    - Input: $1.25 / 1M tokens
    - Output: $10.00 / 1M tokens

    Token estimates per call:
    - Input: ~3,500 tokens
    - Output: ~1,200 tokens

    Rate limits (Tier 2):
    - TPM: 1M tokens/minute
    - Effective: ~200 calls/min (TPM constrained)
    """
    calls_per_pass = neuron_count
    total_calls = calls_per_pass * 2  # output + input passes

    # Token estimates
    input_tokens_per_call = 3500
    output_tokens_per_call = 1200
    total_input_tokens = total_calls * input_tokens_per_call
    total_output_tokens = total_calls * output_tokens_per_call

    # Cost (GPT-5 Jan 2026)
    input_cost = (total_input_tokens / 1_000_000) * 1.25
    output_cost = (total_output_tokens / 1_000_000) * 10.00
    total_cost = input_cost + output_cost

    # Time estimate
    calls_per_minute = 200  # TPM constrained
    total_minutes = total_calls / calls_per_minute
    total_hours = total_minutes / 60

    return {
        "total_calls": total_calls,
        "calls_per_pass": calls_per_pass,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "total_minutes": total_minutes,
        "total_hours": total_hours,
        "hours_per_pass": total_hours / 2,
    }


def display_plan(neuron_counts: dict, estimates: dict):
    """Display the labeling plan."""
    table = Table(
        title="Full-Scale Neuron Labeling Plan",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Total neurons", f"{neuron_counts['total']:,}")
    table.add_row("Layers", f"{len(neuron_counts['layers'])}")
    table.add_row("", "")
    table.add_row("[bold]Cost Estimate[/bold]", "")
    table.add_row("API calls", f"{estimates['total_calls']:,}")
    table.add_row("Input tokens", f"{estimates['input_tokens']/1e6:.1f}M")
    table.add_row("Output tokens", f"{estimates['output_tokens']/1e6:.1f}M")
    table.add_row("Input cost", f"${estimates['input_cost']:,.0f}")
    table.add_row("Output cost", f"${estimates['output_cost']:,.0f}")
    table.add_row("[bold]Total cost[/bold]", f"[bold]${estimates['total_cost']:,.0f}[/bold]")
    table.add_row("", "")
    table.add_row("[bold]Time Estimate[/bold]", "")
    table.add_row("Hours per pass", f"{estimates['hours_per_pass']:.1f}")
    table.add_row("[bold]Total hours[/bold]", f"[bold]{estimates['total_hours']:.1f}[/bold]")

    console.print(table)

    # Layer breakdown
    layer_table = Table(
        title="Neurons by Layer",
        box=box.SIMPLE,
        show_header=True,
    )
    layer_table.add_column("Layer", justify="right")
    layer_table.add_column("Neurons", justify="right")
    layer_table.add_column("Est. Time", justify="right")

    for layer in sorted(neuron_counts['by_layer'].keys(), reverse=True):
        count = neuron_counts['by_layer'][layer]
        minutes = count / 200  # 200 calls/min
        layer_table.add_row(f"L{layer}", f"{count:,}", f"{minutes:.1f} min")

    console.print(layer_table)


def run_pass(
    pass_type: str,
    edge_stats_path: Path,
    db_path: Path,
    state_path: Path,
    model: str,
    batch_size: int,
    min_appearances: int,
    baseline_path: Path | None,
    start_layer: int | None = None,
    end_layer: int | None = None,
    resume: bool = False,
) -> bool:
    """Run a single labeling pass.

    Returns True if successful, False on error.
    """
    # Import here to avoid loading everything for dry-run
    from interactive_labeling import InteractiveLabeler

    # Determine layer range based on pass type
    if pass_type == "output":
        default_start = 31
        default_end = 0
    else:  # input
        default_start = 0
        default_end = 31

    actual_start = start_layer if start_layer is not None else default_start
    actual_end = end_layer if end_layer is not None else default_end

    console.print(Panel(
        f"[bold]{pass_type.upper()} Pass[/bold]\n\n"
        f"Edge stats: {edge_stats_path}\n"
        f"Database: {db_path}\n"
        f"State: {state_path}\n"
        f"Model: {model}\n"
        f"Batch size: {batch_size}\n"
        f"Layers: {actual_start} → {actual_end}\n"
        f"Resume: {resume}",
        title=f"Starting {pass_type.upper()} Pass",
        border_style="green" if pass_type == "input" else "blue"
    ))

    try:
        labeler = InteractiveLabeler(
            edge_stats_path=edge_stats_path,
            db_path=db_path,
            state_path=state_path,
            model=model,
            min_appearances=min_appearances,
            batch_size=batch_size,
            browse_mode=False,
            label_pass=pass_type,
            baseline_path=baseline_path,
        )

        if resume:
            if labeler.session.load_state():
                console.print(f"[green]Resumed at L{labeler.session.current_layer}[/green]")
            else:
                console.print("[yellow]No previous state, starting fresh[/yellow]")
                labeler.session.current_layer = actual_start
        else:
            labeler.session.current_layer = actual_start

        asyncio.run(labeler.run_auto_all_layers(actual_start, actual_end))
        return True

    except Exception as e:
        console.print(f"[red]Error during {pass_type} pass: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Full-scale neuron labeling (173K neurons, ~$5.7K, ~29 hours)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full run (both passes)
    python scripts/run_full_labeling.py

    # Output pass only (L31→L0)
    python scripts/run_full_labeling.py --output-only

    # Input pass only (L0→L31) - requires output pass complete
    python scripts/run_full_labeling.py --input-only

    # Resume from checkpoint
    python scripts/run_full_labeling.py --resume

    # Dry run (show plan without executing)
    python scripts/run_full_labeling.py --dry-run
        """
    )

    parser.add_argument(
        "--edge-stats",
        type=Path,
        default=Path("data/fineweb_50k_edge_stats_enriched.json"),
        help="Path to edge statistics JSON (default: fineweb 50K)"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/neuron_function_db_full.json"),
        help="Path to output label database"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Path to baseline edge stats for domain specificity (optional)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        help="OpenAI model to use (default: gpt-5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=800,
        help="Parallel API calls per batch (default: 800, optimized for high-tier accounts)"
    )
    parser.add_argument(
        "--min-appearances",
        type=int,
        default=10,
        help="Minimum appearances to include a neuron (default: 10)"
    )
    parser.add_argument(
        "--output-only",
        action="store_true",
        help="Run only the output pass (L31→L0)"
    )
    parser.add_argument(
        "--input-only",
        action="store_true",
        help="Run only the input pass (L0→L31)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous checkpoint"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without executing"
    )
    parser.add_argument(
        "--start-layer",
        type=int,
        default=None,
        help="Override start layer (for partial runs)"
    )
    parser.add_argument(
        "--end-layer",
        type=int,
        default=None,
        help="Override end layer (for partial runs)"
    )

    args = parser.parse_args()

    # Validate
    if not args.edge_stats.exists():
        console.print(f"[red]Error: Edge stats not found: {args.edge_stats}[/red]")
        sys.exit(1)

    if args.output_only and args.input_only:
        console.print("[red]Error: Cannot specify both --output-only and --input-only[/red]")
        sys.exit(1)

    # Get neuron counts and estimates
    neuron_counts = get_neuron_counts(args.edge_stats, args.min_appearances)

    if args.output_only or args.input_only:
        # Single pass
        estimates = estimate_cost_and_time(neuron_counts['total'], args.batch_size)
        estimates['total_calls'] //= 2
        estimates['total_cost'] /= 2
        estimates['total_hours'] /= 2
    else:
        estimates = estimate_cost_and_time(neuron_counts['total'], args.batch_size)

    # Display plan
    display_plan(neuron_counts, estimates)

    if args.dry_run:
        console.print("\n[yellow]Dry run - no changes made[/yellow]")
        return

    # Confirm before expensive operation
    console.print(f"\n[bold yellow]This will cost approximately ${estimates['total_cost']:,.0f} and take ~{estimates['total_hours']:.0f} hours.[/bold yellow]")

    try:
        response = input("Continue? [y/N] ")
        if response.lower() != 'y':
            console.print("[yellow]Aborted[/yellow]")
            return
    except (EOFError, KeyboardInterrupt):
        # Non-interactive mode (e.g., SLURM) - proceed
        console.print("[dim]Non-interactive mode, proceeding...[/dim]")

    # Run passes
    start_time = time.time()

    # State files for each pass
    output_state = Path("data/.labeling_state_output_full.json")
    input_state = Path("data/.labeling_state_input_full.json")

    # Baseline path (if exists)
    baseline_path = args.baseline if args.baseline and args.baseline.exists() else None

    success = True

    if not args.input_only:
        console.print("\n" + "=" * 60)
        console.print("[bold]PHASE 1: OUTPUT PASS (L31 → L0)[/bold]")
        console.print("=" * 60 + "\n")

        success = run_pass(
            pass_type="output",
            edge_stats_path=args.edge_stats,
            db_path=args.db,
            state_path=output_state,
            model=args.model,
            batch_size=args.batch_size,
            min_appearances=args.min_appearances,
            baseline_path=baseline_path,
            start_layer=args.start_layer if args.start_layer is not None else 31,
            end_layer=args.end_layer if args.end_layer is not None else 0,
            resume=args.resume,
        )

        if not success:
            console.print("[red]Output pass failed, aborting[/red]")
            sys.exit(1)

    if not args.output_only:
        console.print("\n" + "=" * 60)
        console.print("[bold]PHASE 2: INPUT PASS (L0 → L31)[/bold]")
        console.print("=" * 60 + "\n")

        success = run_pass(
            pass_type="input",
            edge_stats_path=args.edge_stats,
            db_path=args.db,
            state_path=input_state,
            model=args.model,
            batch_size=args.batch_size,
            min_appearances=args.min_appearances,
            baseline_path=baseline_path,
            start_layer=args.start_layer if args.start_layer is not None else 0,
            end_layer=args.end_layer if args.end_layer is not None else 31,
            resume=args.resume,
        )

        if not success:
            console.print("[red]Input pass failed[/red]")
            sys.exit(1)

    # Summary
    elapsed = time.time() - start_time
    console.print("\n" + "=" * 60)
    console.print(Panel(
        f"[bold green]Labeling Complete![/bold green]\n\n"
        f"Total time: {elapsed/3600:.1f} hours\n"
        f"Database: {args.db}\n"
        f"Neurons labeled: {neuron_counts['total']:,}",
        title="Summary",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
