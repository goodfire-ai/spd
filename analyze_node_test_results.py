#!/usr/bin/env python3
"""Analyze results from node reliability test.

This script analyzes the SLURM job results to determine if the suspect node
has a higher failure rate than the control node.

Usage:
    python analyze_node_test_results.py <suspect_job_id> <control_job_id>
"""

import argparse
import subprocess
from collections import Counter

from spd.log import logger


def get_job_states(job_id: str) -> list[str]:
    """Get the states of all array tasks for a job.

    Args:
        job_id: SLURM job ID

    Returns:
        List of job states (e.g., ["COMPLETED", "FAILED", "COMPLETED", ...])
    """
    result = subprocess.run(
        ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
        capture_output=True,
        text=True,
        check=True,
    )

    # Filter out the main job entry (we only want array tasks)
    states = []
    for line in result.stdout.strip().split("\n"):
        state = line.strip()
        if state and state not in ["PENDING", "RUNNING"]:
            states.append(state)

    # sacct returns multiple entries per task (batch step, etc.), so we need to deduplicate
    # We'll take every other entry starting from the second one (the .batch entries)
    if len(states) > 1:
        states = states[1::2]  # Skip the first and take every other

    return states


def analyze_results(suspect_job_id: str, control_job_id: str) -> None:
    """Analyze and compare results from suspect and control nodes.

    Args:
        suspect_job_id: Job ID for suspect node runs
        control_job_id: Job ID for control node runs
    """
    logger.section("NODE TEST RESULTS ANALYSIS")

    # Get job states
    logger.info("Fetching job states from SLURM...")
    suspect_states = get_job_states(suspect_job_id)
    control_states = get_job_states(control_job_id)

    # Count states
    suspect_counts = Counter(suspect_states)
    control_counts = Counter(control_states)

    # Calculate success rates
    suspect_total = len(suspect_states)
    control_total = len(control_states)

    suspect_completed = suspect_counts.get("COMPLETED", 0)
    control_completed = control_counts.get("COMPLETED", 0)

    suspect_failed = suspect_total - suspect_completed
    control_failed = control_total - control_completed

    suspect_success_rate = (
        suspect_completed / suspect_total * 100 if suspect_total > 0 else 0
    )
    control_success_rate = (
        control_completed / control_total * 100 if control_total > 0 else 0
    )

    # Display results
    logger.section(f"SUSPECT NODE (Job {suspect_job_id})")
    logger.values(
        {
            "Total runs": suspect_total,
            "Completed": suspect_completed,
            "Failed": suspect_failed,
            "Success rate": f"{suspect_success_rate:.1f}%",
        }
    )

    if suspect_failed > 0:
        logger.info("\nFailure breakdown:")
        for state, count in suspect_counts.items():
            if state != "COMPLETED":
                logger.info(f"  {state}: {count}")

    logger.section(f"CONTROL NODE (Job {control_job_id})")
    logger.values(
        {
            "Total runs": control_total,
            "Completed": control_completed,
            "Failed": control_failed,
            "Success rate": f"{control_success_rate:.1f}%",
        }
    )

    if control_failed > 0:
        logger.info("\nFailure breakdown:")
        for state, count in control_counts.items():
            if state != "COMPLETED":
                logger.info(f"  {state}: {count}")

    # Analysis
    logger.section("ANALYSIS")

    if suspect_success_rate < control_success_rate:
        diff = control_success_rate - suspect_success_rate
        logger.warning(
            f"⚠️  Suspect node has {diff:.1f}% lower success rate than control node!"
        )
        logger.warning(
            f"Suspect: {suspect_completed}/{suspect_total} success "
            f"vs Control: {control_completed}/{control_total} success"
        )

        if suspect_failed >= 2 and control_failed == 0:
            logger.error(
                "\n🔴 STRONG EVIDENCE: Suspect node has multiple failures while control has none!"
            )
        elif suspect_failed > control_failed * 2:
            logger.error(
                "\n🟠 MODERATE EVIDENCE: Suspect node has significantly more failures!"
            )
        else:
            logger.warning(
                "\n🟡 WEAK EVIDENCE: Suspect node has more failures but difference is small."
            )

    elif control_success_rate < suspect_success_rate:
        diff = suspect_success_rate - control_success_rate
        logger.info(
            f"Control node has {diff:.1f}% lower success rate - suspect node seems fine!"
        )
    else:
        logger.info("Both nodes have identical success rates - no evidence of node issue.")

    # Print log file locations for further investigation
    if suspect_failed > 0 or control_failed > 0:
        logger.section("FURTHER INVESTIGATION")
        logger.info("Check error logs:")
        logger.info(f"  Suspect: grep -l 'error\\|Error\\|ERROR' ~/slurm_logs/slurm-{suspect_job_id}_*.out")
        logger.info(f"  Control: grep -l 'error\\|Error\\|ERROR' ~/slurm_logs/slurm-{control_job_id}_*.out")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze results from node reliability test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "suspect_job_id",
        type=str,
        help="SLURM job ID for suspect node runs",
    )

    parser.add_argument(
        "control_job_id",
        type=str,
        help="SLURM job ID for control node runs",
    )

    args = parser.parse_args()

    analyze_results(args.suspect_job_id, args.control_job_id)


if __name__ == "__main__":
    main()
