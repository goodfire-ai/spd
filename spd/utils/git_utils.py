"""Git utilities for creating code snapshots."""

import subprocess
import tempfile
from pathlib import Path

from spd.log import logger
from spd.settings import REPO_ROOT, SLURM_WORKSPACES_DIR


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


def repo_is_clean(catch_except_as_false: bool = False) -> bool:
    """Return True if the current git repository has no uncommitted or untracked changes.

    # TODO: this may error in CI environments: https://github.com/goodfire-ai/spd/actions/runs/18560369066/job/52907611203
    `fatal: detected dubious ownership in repository at '/__w/spd/spd'`

    for now, if `catch_except_as_false` is True, we catch any exceptions and return False.

    """
    try:
        status: str = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return status == ""
    except Exception as e:
        if catch_except_as_false:
            return False
        else:
            raise e


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
            diff_result = subprocess.run(
                ["git", "diff", "--cached", "--quiet"], cwd=worktree_path, capture_output=True
            )

            # Commit changes if any exist
            if diff_result.returncode != 0:  # Non-zero means there are changes
                subprocess.run(
                    ["git", "commit", "-m", f"run id {run_id}", "--no-verify"],
                    cwd=worktree_path,
                    check=True,
                    capture_output=True,
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

    return snapshot_branch, commit_hash


def create_snapshot_workspace(run_id: str) -> Path:
    """Create a shared workspace by copying REPO_ROOT (including .venv) at submission time.

    This captures the exact state of REPO_ROOT (code + venv) when spd-run is executed.
    Much faster than git clone + uv sync (~15s vs ~70s) because it copies the existing
    venv instead of rebuilding it.

    Args:
        run_id: Unique identifier for the run

    Returns:
        Path to the workspace directory

    Raises:
        subprocess.CalledProcessError: If rsync or uv commands fail
    """
    workspace_path = SLURM_WORKSPACES_DIR / f"spd-{run_id}"
    SLURM_WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating snapshot workspace at {workspace_path}...")

    # rsync REPO_ROOT to workspace, respecting .gitignore but including .venv
    # Filter rules are processed in order: include .venv first, then apply .gitignore
    logger.info("Copying codebase and venv with rsync...")
    subprocess.run(
        [
            "rsync",
            "-a",
            "--include=.venv",
            "--include=.venv/**",
            "--filter=:- .gitignore",
            "--exclude=.git",
            f"{REPO_ROOT}/",
            f"{workspace_path}/",
        ],
        check=True,
        capture_output=True,
    )

    # Fix editable install to point to workspace instead of REPO_ROOT
    # This is fast (~1-2s) because --no-deps skips dependency resolution
    logger.info("Fixing editable install paths...")
    subprocess.run(
        ["uv", "pip", "install", "-e", ".", "--no-deps", "-q"],
        cwd=workspace_path,
        check=True,
        capture_output=True,
    )

    logger.info(f"Snapshot workspace ready: {workspace_path}")
    return workspace_path
