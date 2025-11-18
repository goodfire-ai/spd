#!/usr/bin/env python3
"""Test script to check node reliability by running multiple short training runs.

This script helps diagnose whether a specific node (h200-dev-145-040) is causing failures
by running the same experiment multiple times on both the suspect node and a control node.

Usage:
    python test_node_reliability.py --n-runs 10 --steps 300
    python test_node_reliability.py --n-runs 5 --steps 200 --project test-nodes
"""

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

from spd.log import logger
from spd.scripts.run import generate_commands, generate_run_id
from spd.settings import REPO_ROOT
from spd.utils.git_utils import create_git_snapshot
from spd.utils.slurm_utils import create_slurm_array_script, submit_slurm_array


def run_node_test(
    n_runs: int,
    steps: int,
    suspect_node: str,
    control_node: str,
    project: str,
    dp: int,
) -> tuple[str, str]:
    """Run multiple training runs on suspect and control nodes.

    Args:
        n_runs: Number of runs to execute on each node
        steps: Number of training steps per run
        suspect_node: Node suspected of being buggy
        control_node: Control node for comparison
        project: W&B project name
        dp: Number of GPUs for data parallelism

    Returns:
        Tuple of (suspect_job_id, control_job_id)
    """
    # Create git snapshot
    snapshot_branch, commit_hash = create_git_snapshot(branch_name_prefix="node-test")
    logger.info(f"Created git snapshot: {snapshot_branch} ({commit_hash[:8]})")

    # Generate commands for each run
    # We'll create N copies of the same command with different run IDs
    logger.section(f"Setting up {n_runs} runs on each node ({n_runs * 2} total)")

    # Create commands for suspect node
    suspect_commands = []
    for i in range(n_runs):
        run_id = f"suspect-{suspect_node}-{generate_run_id()}-{i}"
        cmds = generate_commands(
            experiments_list=["ss_gpt2_simple"],
            run_id=run_id,
            sweep_params_file=None,
            project=project,
            dp=dp,
        )
        suspect_commands.extend(cmds)

    # Create commands for control node
    control_commands = []
    for i in range(n_runs):
        run_id = f"control-{control_node}-{generate_run_id()}-{i}"
        cmds = generate_commands(
            experiments_list=["ss_gpt2_simple"],
            run_id=run_id,
            sweep_params_file=None,
            project=project,
            dp=dp,
        )
        control_commands.extend(cmds)

    logger.info(f"Generated {len(suspect_commands)} commands for suspect node")
    logger.info(f"Generated {len(control_commands)} commands for control node")

    # Submit jobs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Submit suspect node jobs
        logger.section(f"Submitting jobs to SUSPECT node: {suspect_node}")
        suspect_script = temp_path / f"suspect_{suspect_node}.sh"
        create_slurm_array_script(
            script_path=suspect_script,
            job_name=f"test-suspect-{suspect_node}",
            commands=suspect_commands,
            snapshot_branch=snapshot_branch,
            n_gpus_per_job=dp,
            partition="h200-reserved",
            time_limit="02:00:00",
            max_concurrent_tasks=n_runs,  # Run all in parallel
            nodelist=suspect_node,
        )
        suspect_job_id = submit_slurm_array(suspect_script)
        logger.info(f"Suspect node job ID: {suspect_job_id}")

        # Small delay to avoid overwhelming SLURM
        time.sleep(2)

        # Submit control node jobs
        logger.section(f"Submitting jobs to CONTROL node: {control_node}")
        control_script = temp_path / f"control_{control_node}.sh"
        create_slurm_array_script(
            script_path=control_script,
            job_name=f"test-control-{control_node}",
            commands=control_commands,
            snapshot_branch=snapshot_branch,
            n_gpus_per_job=dp,
            partition="h200-reserved",
            time_limit="02:00:00",
            max_concurrent_tasks=n_runs,  # Run all in parallel
            nodelist=control_node,
        )
        control_job_id = submit_slurm_array(control_script)
        logger.info(f"Control node job ID: {control_job_id}")

    return suspect_job_id, control_job_id


def print_monitoring_instructions(
    suspect_node: str,
    control_node: str,
    suspect_job_id: str,
    control_job_id: str,
    n_runs: int,
) -> None:
    """Print instructions for monitoring the test runs."""
    logger.section("MONITORING INSTRUCTIONS")

    logger.info("Watch job status:")
    logger.info(f"  squeue -j {suspect_job_id},{control_job_id}")
    logger.info("")

    logger.info("View logs:")
    logger.info(f"  Suspect ({suspect_node}): ~/slurm_logs/slurm-{suspect_job_id}_*.out")
    logger.info(f"  Control ({control_node}): ~/slurm_logs/slurm-{control_job_id}_*.out")
    logger.info("")

    logger.info("Check for failures in logs:")
    logger.info(f"  grep -i 'error\\|fail\\|abort' ~/slurm_logs/slurm-{suspect_job_id}_*.out")
    logger.info(f"  grep -i 'error\\|fail\\|abort' ~/slurm_logs/slurm-{control_job_id}_*.out")
    logger.info("")

    logger.info("Count completed/failed runs after jobs finish:")
    logger.info(
        f"  sacct -j {suspect_job_id} --format=JobID,State,ExitCode,NodeList | tail -n {n_runs + 1}"
    )
    logger.info(
        f"  sacct -j {control_job_id} --format=JobID,State,ExitCode,NodeList | tail -n {n_runs + 1}"
    )
    logger.info("")

    logger.info("Compare success rates:")
    logger.info(f"  Suspect: sacct -j {suspect_job_id} --format=State | grep -c COMPLETED")
    logger.info(f"  Control: sacct -j {control_job_id} --format=State | grep -c COMPLETED")


def main() -> None:
    """Run node reliability test."""
    parser = argparse.ArgumentParser(
        description="Test node reliability by running multiple short training runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of runs to execute on each node (default: 10)",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Number of training steps per run (default: 300). "
        "NOTE: You must manually update ss_gpt2_simple_config.yaml to set this value!",
    )

    parser.add_argument(
        "--suspect-node",
        type=str,
        default="h200-dev-145-040",
        help="Node suspected of being buggy (default: h200-dev-145-040)",
    )

    parser.add_argument(
        "--control-node",
        type=str,
        default="h200-reserved-145-005",
        help="Control node for comparison (default: h200-reserved-145-005)",
    )

    parser.add_argument(
        "--project",
        type=str,
        default="spd-node-test",
        help="W&B project name (default: spd-node-test)",
    )

    parser.add_argument(
        "--dp",
        type=int,
        default=8,
        help="Number of GPUs for data parallelism (default: 8)",
    )

    args = parser.parse_args()

    # Validate inputs
    if args.n_runs <= 0:
        raise ValueError("--n-runs must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    logger.section("NODE RELIABILITY TEST")
    logger.values(
        {
            "Suspect node": args.suspect_node,
            "Control node": args.control_node,
            "Runs per node": args.n_runs,
            "Steps per run": args.steps,
            "Total runs": args.n_runs * 2,
            "W&B project": args.project,
            "Data parallelism": args.dp,
        }
    )

    logger.warning(
        f"\n⚠️  IMPORTANT: Make sure you've set 'steps: {args.steps}' in "
        "spd/experiments/lm/ss_gpt2_simple_config.yaml!\n"
    )

    # Check if config file has the right number of steps
    config_path = REPO_ROOT / "spd/experiments/lm/ss_gpt2_simple_config.yaml"
    with open(config_path) as f:
        config_content = f.read()
        if f"steps: {args.steps}" not in config_content:
            logger.error(
                f"Config file doesn't contain 'steps: {args.steps}'. "
                "Please update it before running this test!"
            )
            response = input("Continue anyway? (y/N): ")
            if response.lower() != "y":
                logger.info("Aborting.")
                return

    # Run the test
    suspect_job_id, control_job_id = run_node_test(
        n_runs=args.n_runs,
        steps=args.steps,
        suspect_node=args.suspect_node,
        control_node=args.control_node,
        project=args.project,
        dp=args.dp,
    )

    logger.section("JOBS SUBMITTED SUCCESSFULLY")
    logger.values(
        {
            "Suspect job ID": suspect_job_id,
            "Control job ID": control_job_id,
        }
    )

    # Print monitoring instructions
    print_monitoring_instructions(
        suspect_node=args.suspect_node,
        control_node=args.control_node,
        suspect_job_id=suspect_job_id,
        control_job_id=control_job_id,
        n_runs=args.n_runs,
    )


if __name__ == "__main__":
    main()
