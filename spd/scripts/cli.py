import argparse
from typing import Final

from spd.log import LogFormat
from spd.registry import EXPERIMENT_REGISTRY

_SPD_RUN_EXAMPLES: Final[str] = """
Examples:
    # Run subset of experiments locally
    spd-run --experiments tms_5-2,resid_mlp1 --local

    # Run parameter sweep locally
    spd-run --experiments tms_5-2 --sweep --local

    # Run subset of experiments (no sweep)
    spd-run --experiments tms_5-2,resid_mlp1

    # Run parameter sweep on a subset of experiments with default sweep_params.yaml
    spd-run --experiments tms_5-2,resid_mlp2 --sweep

    # Run parameter sweep on an experiment with custom sweep params at spd/scripts/my_sweep.yaml
    spd-run --experiments tms_5-2 --sweep my_sweep.yaml

    # Run all experiments (no sweep)
    spd-run

    # Use custom W&B project
    spd-run --experiments tms_5-2 --project my-spd-project

    # Run all experiments on CPU
    spd-run --experiments tms_5-2 --cpu

    # Run with data parallelism over 4 GPUs (only supported for lm experiments)
    spd-run --experiments ss_llama --dp 4
"""


def cli():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        prog="spd-run",
        description="SPD runner for experiments with optional parameter sweeps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_SPD_RUN_EXAMPLES,
    )

    # main arguments
    parser.add_argument(
        "-e",
        "--experiments",
        type=str,
        default=None,
        help=(
            "Comma-separated list of experiment names. If not specified, runs all experiments. "
            f"Available: {list(EXPERIMENT_REGISTRY.keys())}"
        ),
    )
    parser.add_argument(
        "--local",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run locally instead of submitting to SLURM",
    )

    # Sweep arguments
    parser.add_argument(
        "--sweep",
        nargs="?",
        const=True,
        default=False,
        help="Enable parameter sweep. If `--sweep` passed with argument, uses default sweep_params.yaml. "
        "Otherwise, specify a single path to custom sweep parameters file.",
    )

    parser.add_argument(
        "-n",
        "--n-agents",
        type=int,
        default=None,
        help="Maximum number of concurrent SLURM tasks. Required for sweeps unless running locally. "
        "For non-sweep runs, defaults to the number of experiments.",
    )

    # Report and project settings
    parser.add_argument(
        "--create-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create W&B report for aggregated view",
    )

    parser.add_argument(
        "--project",
        type=str,
        default="spd",
        help="W&B project name (default: spd). Will be created if it doesn't exist.",
    )

    parser.add_argument(
        "--report-title",
        type=str,
        default=None,
        help="Title for the W&B report. Generated automatically if not provided.",
    )

    # Execution settings
    parser.add_argument(
        "--cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use CPU instead of GPU",
    )

    parser.add_argument(
        "--dp",
        "--data-parallelism",
        type=int,
        default=1,
        help="Number of GPUs for data parallelism (1-8). Only supported for lm experiments. "
        "Cannot be used with local mode (default: 1)",
    )

    parser.add_argument(
        "--job-suffix",
        type=str,
        default=None,
        help="Optional suffix for SLURM job names",
    )

    # Git and logging settings
    parser.add_argument(
        "--create-snapshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create a git snapshot branch for the run",
    )

    parser.add_argument(
        "--use-wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use W&B for logging and tracking",
    )

    parser.add_argument(
        "--log-format",
        type=str,
        choices=LogFormat.__args__,
        default="default",
        help="Logging format for script output. 'terse' removes timestamps/level (default: 'default')",
    )

    parser.add_argument(
        "-p",
        "--partition",
        type=str,
        default="h100-reserved",
        help="SLURM partition to use (default: 'h100-reserved')",
    )

    args: argparse.Namespace = parser.parse_args()

    # Call main with parsed arguments
    from spd.scripts.run import main
    main(
        experiments=args.experiments,
        sweep=args.sweep,
        n_agents=args.n_agents,
        create_report=args.create_report,
        job_suffix=args.job_suffix,
        cpu=args.cpu,
        partition=args.partition,
        dp=args.dp,
        project=args.project,
        local=args.local,
        log_format=args.log_format,
        create_snapshot=args.create_snapshot,
        use_wandb=args.use_wandb,
        report_title=args.report_title,
    )


if __name__ == "__main__":
    cli()
