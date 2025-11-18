#!/usr/bin/env python3
"""Check the status of SLURM nodes and suggest good candidates for testing.

Usage:
    python check_nodes.py
    python check_nodes.py --node h200-dev-145-040
"""

import argparse
import subprocess

from spd.log import logger


def check_node_status(node_name: str | None = None) -> None:
    """Check the status of a specific node or all h200 nodes.

    Args:
        node_name: Name of specific node to check. If None, shows all h200 nodes.
    """
    if node_name:
        # Check specific node
        result = subprocess.run(
            ["scontrol", "show", "node", node_name],
            capture_output=True,
            text=True,
            check=True,
        )

        logger.section(f"STATUS FOR NODE: {node_name}")

        # Parse output for key information
        for line in result.stdout.split("\n"):
            if any(
                key in line
                for key in ["State=", "Reason=", "Partitions=", "AllocTRES=", "AvailTRES="]
            ):
                logger.info(line.strip())

    else:
        # List all h200 nodes
        result = subprocess.run(
            ["sinfo", "-N", "-o", "%N %P %T %E", "--sort=N"],
            capture_output=True,
            text=True,
            check=True,
        )

        logger.section("ALL H200 NODES")

        # Filter for h200 nodes
        lines = result.stdout.strip().split("\n")
        header = lines[0]
        logger.info(header)
        logger.info("-" * len(header))

        seen_nodes = set()
        for line in lines[1:]:
            if "h200" in line:
                node_name = line.split()[0]
                # Only show each node once (they appear multiple times for different partitions)
                if node_name not in seen_nodes:
                    logger.info(line)
                    seen_nodes.add(node_name)

        # Suggest good nodes for testing
        logger.section("RECOMMENDATIONS")

        logger.info("Looking for idle/mixed nodes in h200-reserved partition...")

        idle_nodes = []
        mixed_nodes = []

        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 3 and "h200-reserved" in parts[1]:
                node = parts[0]
                state = parts[2]

                if state == "idle" and node not in idle_nodes:
                    idle_nodes.append(node)
                elif state == "mixed" and node not in mixed_nodes:
                    mixed_nodes.append(node)

        if idle_nodes:
            logger.info("\nBest candidates (idle nodes):")
            for node in idle_nodes[:3]:  # Show top 3
                logger.info(f"  ✓ {node}")

        if mixed_nodes:
            logger.info("\nAlternative candidates (mixed nodes):")
            for node in mixed_nodes[:3]:  # Show top 3
                logger.info(f"  ~ {node}")

        if not idle_nodes and not mixed_nodes:
            logger.warning("No idle or mixed nodes found in h200-reserved partition!")

        # Check suspect node specifically
        logger.section("CHECKING SUSPECT NODE: h200-dev-145-040")
        suspect_result = subprocess.run(
            ["scontrol", "show", "node", "h200-dev-145-040"],
            capture_output=True,
            text=True,
        )

        if suspect_result.returncode == 0:
            for line in suspect_result.stdout.split("\n"):
                if "State=" in line or "Reason=" in line:
                    logger.info(line.strip())

            if "DRAIN" in suspect_result.stdout:
                logger.warning(
                    "\n⚠️  Suspect node is DRAINED - you'll need admin help to undrain it"
                )
                logger.info(
                    "Or choose a different suspect node from the 'idle' or 'mixed' nodes above"
                )
        else:
            logger.error(f"Could not find node h200-dev-145-040: {suspect_result.stderr}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check SLURM node status and suggest testing candidates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--node",
        type=str,
        default=None,
        help="Check status of a specific node. If not specified, shows all h200 nodes.",
    )

    args = parser.parse_args()

    check_node_status(args.node)


if __name__ == "__main__":
    main()
