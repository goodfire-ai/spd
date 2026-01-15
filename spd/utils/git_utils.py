"""Git utilities for creating code snapshots."""

import subprocess
import tempfile
import time
import shutil
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


def repo_is_clean() -> bool:
    """Return True if the current git repository has no uncommitted or untracked changes."""
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    return status == ""


def repo_current_commit_hash() -> str:
    """Return the current commit hash of the active HEAD."""
    commit_hash: str = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return commit_hash


def create_git_snapshot(run_id: str) -> tuple[str, str]:
    """Create a git snapshot branch with current changes.

    Creates a timestamped branch containing all current changes (staged and unstaged). Uses a
    temporary worktree to avoid affecting the current working directory. Will push the snapshot
    branch to origin if possible, but will continue without error if push permissions are lacking.

    Returns:
        (branch_name, commit_hash) where commit_hash is the HEAD of the snapshot branch
        (this will be the new snapshot commit if changes existed, otherwise the base commit).

    Raises:
        subprocess.CalledProcessError: If git commands fail (except for push)
    """
    # prefix branch name
    snapshot_branch: str = f"snapshot/{run_id}"

    # Create temporary worktree path
    with tempfile.TemporaryDirectory() as temp_dir:
        worktree_path = Path(temp_dir) / f"spd-snapshot-{run_id}"

        try:
            # Create worktree with new branch
            subprocess.run(
                ["git", "worktree", "add", "-b", snapshot_branch, str(worktree_path)],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
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
                text=True,
            )

            # Stage all changes in the worktree
            subprocess.run(
                ["git", "add", "-A"], cwd=worktree_path, check=True, capture_output=True, text=True
            )

            # Check if there are changes to commit
            diff_result = subprocess.run(
                ["git", "diff", "--cached", "--quiet"], cwd=worktree_path, capture_output=True, text=True
            )

            # Commit changes if any exist
            if diff_result.returncode != 0:  # Non-zero means there are changes
                subprocess.run(
                    ["git", "commit", "-m", f"run id {run_id}", "--no-verify"],
                    cwd=worktree_path,
                    check=True,
                    capture_output=True,
                    text=True,
                )

            # Get the commit hash of HEAD (either new commit or base commit if nothing changed)
            rev_parse = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=worktree_path,
                check=True,
                capture_output=True,
                text=True,
            )
            commit_hash = rev_parse.stdout.strip()

            # Try push (non-fatal if fails)
            try:
                subprocess.run(
                    ["git", "push", "-u", "origin", snapshot_branch],
                    cwd=worktree_path,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.info(f"Successfully pushed snapshot branch '{snapshot_branch}' to origin")
            except subprocess.CalledProcessError as e:
                err = (e.stderr or "").strip() if isinstance(e.stderr, str) else ""
                logger.warning(
                    f"Could not push snapshot branch '{snapshot_branch}' to origin. "
                    f"The branch was created locally but won't be accessible to other users. "
                    f"Error: {err or 'Unknown error'}"
                )

        finally:
            # Clean up worktree (branch remains in main repo).
            # This *should* always succeed, but in practice it can fail transiently (e.g. git locks)
            # or after partial cleanup. Cleanup failures should not block launching jobs.
            remove_stderr: str | None = None
            removed_ok = False

            for attempt in range(5):
                remove_res = subprocess.run(
                    ["git", "worktree", "remove", "--force", str(worktree_path)],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                )
                if remove_res.returncode == 0:
                    removed_ok = True
                    break

                remove_stderr = (remove_res.stderr or "").strip()
                # If git thinks it isn't (or is no longer) a worktree, treat as already cleaned up.
                if "is not a working tree" in remove_stderr or "No such file or directory" in remove_stderr:
                    removed_ok = True
                    break

                # Backoff a bit for transient failures (e.g. repository lock contention).
                time.sleep(0.25 * (attempt + 1))

            # Best-effort: prune stale metadata. This helps if the temp dir disappears but git still
            # thinks the worktree exists.
            subprocess.run(
                ["git", "worktree", "prune", "--expire", "now"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

            # Best-effort: remove the directory if it still exists (TemporaryDirectory will also
            # clean it up, but this makes behavior more predictable when git removal fails).
            if worktree_path.exists():
                shutil.rmtree(worktree_path, ignore_errors=True)

            if not removed_ok:
                logger.warning(
                    "Failed to remove temporary git worktree. This is non-fatal but may leave stale "
                    f"metadata until pruned. worktree_path={worktree_path} stderr={remove_stderr!r}"
                )

    return snapshot_branch, commit_hash
