#!/usr/bin/env python3
"""Command-line interface for submitting SPD experiments to SLURM."""

import argparse
import sys
from pathlib import Path

from spd.slurm_utils import submit_experiment_job, get_slurm_partition


def main():
    parser = argparse.ArgumentParser(
        description="Submit SPD experiments to SLURM with partition support"
    )
    parser.add_argument(
        "experiment",
        choices=["tms", "resid_mlp", "lm"],
        help="Experiment type to run"
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the experiment config file"
    )
    parser.add_argument(
        "--partition",
        help="SLURM partition to use (overrides SLURM_PARTITION env var)"
    )
    parser.add_argument(
        "--time",
        default="24:00:00",
        help="Time limit (default: 24:00:00)"
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=4,
        help="Number of CPUs per task (default: 4)"
    )
    parser.add_argument(
        "--memory",
        default="16G",
        help="Memory requirement (default: 16G)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="Number of GPUs to request"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for SLURM output files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be submitted without actually submitting"
    )

    args = parser.parse_args()

    # Check if config file exists
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Show current partition configuration
    current_partition = args.partition or get_slurm_partition()
    if current_partition:
        print(f"Using SLURM partition: {current_partition}")
    else:
        print("Using default SLURM partition")

    # Collect SLURM options
    slurm_options = {
        "time": args.time,
        "cpus_per_task": args.cpus,
        "memory": args.memory,
    }

    if args.gpu is not None:
        slurm_options["gpu"] = args.gpu

    if args.dry_run:
        slurm_options["dry_run"] = True

    # Submit the job
    job_id = submit_experiment_job(
        experiment_name=args.experiment,
        config_path=args.config,
        partition=args.partition,
        output_dir=args.output_dir,
        **slurm_options
    )

    if job_id:
        print(f"Successfully submitted job {job_id}")
    elif not args.dry_run:
        print("Failed to submit job", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()