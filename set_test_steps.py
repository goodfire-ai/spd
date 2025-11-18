#!/usr/bin/env python3
"""Helper script to set the number of training steps in ss_gpt2_simple_config.yaml.

This is useful when setting up node reliability tests that need to run quickly.

Usage:
    python set_test_steps.py 300
    python set_test_steps.py 200 --backup
"""

import argparse
import shutil
from pathlib import Path

from spd.log import logger
from spd.settings import REPO_ROOT


def set_config_steps(steps: int, backup: bool = False) -> None:
    """Update the steps value in ss_gpt2_simple_config.yaml.

    Args:
        steps: New number of training steps
        backup: If True, create a backup of the original config
    """
    config_path = REPO_ROOT / "spd/experiments/lm/ss_gpt2_simple_config.yaml"

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return

    # Create backup if requested
    if backup:
        backup_path = config_path.with_suffix(".yaml.backup")
        shutil.copy2(config_path, backup_path)
        logger.info(f"Created backup: {backup_path}")

    # Read the config
    with open(config_path) as f:
        lines = f.readlines()

    # Find and update the steps line
    updated = False
    original_steps = None
    for i, line in enumerate(lines):
        if line.startswith("steps:"):
            original_steps = line.split(":")[1].strip()
            lines[i] = f"steps: {steps}\n"
            updated = True
            break

    if not updated:
        logger.error("Could not find 'steps:' line in config file")
        return

    # Write back
    with open(config_path, "w") as f:
        f.writelines(lines)

    logger.section("CONFIG UPDATED")
    logger.values(
        {
            "File": str(config_path),
            "Original steps": original_steps,
            "New steps": steps,
        }
    )

    logger.info("\n✓ Config file updated successfully!")
    logger.info(f"  You can now run: python test_node_reliability.py --steps {steps}")


def restore_config_backup() -> None:
    """Restore config from backup if it exists."""
    config_path = REPO_ROOT / "spd/experiments/lm/ss_gpt2_simple_config.yaml"
    backup_path = config_path.with_suffix(".yaml.backup")

    if not backup_path.exists():
        logger.error(f"No backup file found: {backup_path}")
        return

    shutil.copy2(backup_path, config_path)
    logger.info(f"Restored config from backup: {backup_path}")
    logger.info(f"Backup still exists at: {backup_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Set the number of training steps in ss_gpt2_simple_config.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "steps",
        type=int,
        nargs="?",
        help="Number of training steps to set",
    )

    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a backup of the config file before modifying",
    )

    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore config from backup file (ignores steps argument)",
    )

    args = parser.parse_args()

    if args.restore:
        restore_config_backup()
    elif args.steps is None:
        parser.error("steps argument is required unless --restore is used")
    else:
        if args.steps <= 0:
            parser.error("steps must be positive")
        set_config_steps(args.steps, backup=args.backup)


if __name__ == "__main__":
    main()
