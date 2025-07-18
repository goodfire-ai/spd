"""Git utilities for creating code snapshots."""

import datetime
import subprocess
import tempfile
from pathlib import Path

from spd.log import logger
from spd.settings import REPO_ROOT


def repo_current_branch() -> str:
    """Return the active Git branch by invoking the `git` CLI.

    Uses `git rev-parse --abbrev-ref HEAD`, which prints either the branch
    name (e.g. `main`) or `HEAD` if the repo is in a detached-HEAD state.

    Returns:
        The name of the current branch, or `HEAD` if in detached state.

    Raises:
        subprocess.CalledProcessError: If the `git` command fails.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def create_git_snapshot(branch_name_prefix: str) -> str:
    """Create a git snapshot branch with current changes.

    Creates a timestamped branch containing all current changes (staged and unstaged). Uses a
    temporary worktree to avoid affecting the current working directory. Will push the snapshot
    branch to origin if possible, but will continue without error if push permissions are lacking.

    Returns:
        Branch name of the created snapshot

    Raises:
        subprocess.CalledProcessError: If git commands fail (except for push)
    """
    # Generate timestamped branch name
    timestamp_utc = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    snapshot_branch = f"{branch_name_prefix}-{timestamp_utc}"

    # Create temporary worktree path
    with tempfile.TemporaryDirectory() as temp_dir:
        worktree_path = Path(temp_dir) / f"spd-snapshot-{timestamp_utc}"

        try:
            # Create worktree with new branch
            subprocess.run(
                ["git", "worktree", "add", "-b", snapshot_branch, str(worktree_path)],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
            )

            # Copy current working tree to worktree (including untracked files)
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--delete",
                    "--exclude=.git",
                    "--filter=:- .gitignore",
                    f"{REPO_ROOT}/",
                    f"{worktree_path}/",
                ],
                check=True,
                capture_output=True,
            )

            # Stage all changes in the worktree
            subprocess.run(["git", "add", "-A"], cwd=worktree_path, check=True, capture_output=True)

            # Check if there are changes to commit
            result = subprocess.run(
                ["git", "diff", "--cached", "--quiet"], cwd=worktree_path, capture_output=True
            )

            # Commit changes if any exist
            if result.returncode != 0:  # Non-zero means there are changes
                subprocess.run(
                    ["git", "commit", "-m", f"Sweep snapshot {timestamp_utc}", "--no-verify"],
                    cwd=worktree_path,
                    check=True,
                    capture_output=True,
                )

            try:
                subprocess.run(
                    ["git", "push", "-u", "origin", snapshot_branch],
                    cwd=worktree_path,
                    check=True,
                    capture_output=True,
                )
                logger.info(f"Successfully pushed snapshot branch '{snapshot_branch}' to origin")
            except subprocess.CalledProcessError as e:
                logger.warning(
                    f"Could not push snapshot branch '{snapshot_branch}' to origin. "
                    f"The branch was created locally but won't be accessible to other users. "
                    f"Error: {e.stderr.decode().strip() if e.stderr else 'Unknown error'}"
                )

        finally:
            # Clean up worktree (branch remains in main repo)
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree_path)],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
            )

    return snapshot_branch
