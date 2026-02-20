#!/usr/bin/env python3
"""Launch large-scale overnight graph generation.

This is a convenience wrapper around parallel_baseline_generation.py for
running large-scale (50K+) graph generation jobs overnight.

Example usage:
    # Generate 50,000 graphs using 80 GPUs with mixed dataset
    python scripts/launch_overnight_generation.py \
        --n-samples 50000 \
        --n-gpus 80 \
        --dataset mixed \
        --output-dir graphs/fabric_v1

    # Dry run to preview the job
    python scripts/launch_overnight_generation.py \
        --n-samples 50000 \
        --n-gpus 80 \
        --dry-run

    # Conservative run with 20K samples
    python scripts/launch_overnight_generation.py \
        --n-samples 20000 \
        --n-gpus 40 \
        --dataset fineweb

Capacity estimates (H200 cluster):
    - 10 nodes × 8 GPUs = 80 parallel workers
    - ~6 graphs/minute per GPU
    - 8 hours overnight = 480 minutes
    - 480 min × 80 GPUs × 6 graphs/min = ~230,000 graphs possible
    - Conservative target: 20,000-50,000 graphs
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def estimate_runtime(n_samples: int, n_gpus: int) -> str:
    """Estimate runtime based on observed performance."""
    # Observed: ~10 seconds per graph
    graphs_per_gpu = n_samples / n_gpus
    seconds = graphs_per_gpu * 10
    hours = seconds / 3600
    return f"{hours:.1f} hours"


def main():
    parser = argparse.ArgumentParser(
        description="Launch large-scale overnight graph generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--n-samples", type=int, required=True,
        help="Total number of graphs to generate (e.g., 50000)"
    )
    parser.add_argument(
        "--n-gpus", type=int, required=True,
        help="Number of GPUs to use (e.g., 80 for 10 nodes)"
    )
    parser.add_argument(
        "--dataset", type=str, default="mixed",
        choices=["fineweb", "chat", "mixed"],
        help="Dataset type (default: mixed)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: graphs/fabric_<timestamp>)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Create job files but don't submit"
    )
    parser.add_argument(
        "--partition", default="h200-reserved",
        help="SLURM partition (default: h200-reserved)"
    )
    parser.add_argument(
        "--time-limit", default="4:00:00",
        help="Time limit per task (default: 4:00:00)"
    )
    parser.add_argument(
        "--mem", default="64G",
        help="Memory per task (default: 64G)"
    )
    parser.add_argument(
        "--tau", type=float, default=0.005,
        help="Node threshold (default: 0.005)"
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Top logits to trace (default: 5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Set default output directory with timestamp
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        args.output_dir = Path(f"graphs/fabric_{args.dataset}_{timestamp}")

    # Print configuration summary
    print("=" * 60)
    print("OVERNIGHT GRAPH GENERATION")
    print("=" * 60)
    print(f"Dataset:        {args.dataset}")
    print(f"Total samples:  {args.n_samples:,}")
    print(f"GPUs:           {args.n_gpus}")
    print(f"Samples/GPU:    ~{args.n_samples // args.n_gpus:,}")
    print(f"Output:         {args.output_dir}")
    print(f"Partition:      {args.partition}")
    print(f"Time limit:     {args.time_limit}")
    print(f"Memory:         {args.mem}")
    print(f"Tau:            {args.tau}")
    print(f"K:              {args.k}")
    print(f"Seed:           {args.seed}")
    print("-" * 60)
    print(f"Estimated time: {estimate_runtime(args.n_samples, args.n_gpus)}")
    print("=" * 60)

    # Build command
    cmd = [
        sys.executable,
        "scripts/parallel_baseline_generation.py",
        "--n-samples", str(args.n_samples),
        "--n-gpus", str(args.n_gpus),
        "--dataset", args.dataset,
        "--output-dir", str(args.output_dir),
        "--partition", args.partition,
        "--time", args.time_limit,
        "--mem", args.mem,
        "--tau", str(args.tau),
        "--k", str(args.k),
        "--seed", str(args.seed),
    ]

    if args.dry_run:
        cmd.append("--dry-run")

    print(f"\nRunning: {' '.join(cmd)}\n")

    # Execute
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nError: Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    if not args.dry_run:
        print("\n" + "=" * 60)
        print("JOB SUBMITTED SUCCESSFULLY")
        print("=" * 60)
        print("\nMonitoring commands:")
        print("  squeue -u $USER                    # View job status")
        print("  watch -n 30 'squeue -u $USER'      # Auto-refresh status")
        print("  sacct -j <JOB_ID> --format=JobID,State,Elapsed,MaxRSS")
        print(f"\nOutput will be saved to: {args.output_dir}")
        print("\nFiles generated:")
        print(f"  {args.output_dir}/graph_*.json    # Individual graphs")
        print(f"  {args.output_dir}/metadata.json   # All prompts + positions")
        print(f"  {args.output_dir}/neuron_index.json  # Neuron -> graph mapping")


if __name__ == "__main__":
    main()
