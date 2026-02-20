#!/usr/bin/env python3
"""
Clean up agent-generated sessions from Claude Code's session index.

This script identifies and removes sessions that were created by the Claude Agent SDK
based on their prompt patterns. It modifies the sessions-index.json and optionally
deletes the corresponding .jsonl session files.

Usage:
    # Dry run - see what would be deleted
    python cleanup_agent_sessions.py --dry-run

    # Actually delete agent sessions
    python cleanup_agent_sessions.py

    # Delete sessions matching custom patterns
    python cleanup_agent_sessions.py --patterns "Generate a beautiful" "# Investigation:"

    # Target a specific project
    python cleanup_agent_sessions.py --project /path/to/project
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Default patterns that identify agent-generated sessions
DEFAULT_AGENT_PATTERNS = [
    r"^Generate a beautiful HTML dashboard",
    r"^# Investigation: Neuron",
    r"^## Investigation: Neuron",
    # Add more patterns as needed
]

# Additional patterns for cleanup
EMPTY_SESSION_PATTERNS = [
    r"^No prompt$",
]


def get_claude_projects_dir() -> Path:
    """Get the Claude projects directory."""
    return Path.home() / ".claude" / "projects"


def get_project_session_dir(project_path: str) -> Path:
    """Convert a project path to its Claude session directory."""
    # Claude converts paths by replacing / with -
    # The resulting name starts with a dash (e.g., "-mnt-polished-lake-...")
    safe_name = project_path.replace("/", "-")
    return get_claude_projects_dir() / safe_name


def find_project_dirs() -> list[Path]:
    """Find all project directories with sessions."""
    projects_dir = get_claude_projects_dir()
    if not projects_dir.exists():
        return []
    return [d for d in projects_dir.iterdir() if d.is_dir() and (d / "sessions-index.json").exists()]


def load_sessions_index(project_dir: Path) -> dict:
    """Load the sessions-index.json file."""
    index_path = project_dir / "sessions-index.json"
    with open(index_path) as f:
        return json.load(f)


def save_sessions_index(project_dir: Path, data: dict):
    """Save the sessions-index.json file."""
    index_path = project_dir / "sessions-index.json"
    # Backup first
    backup_path = index_path.with_suffix(".json.bak")
    if index_path.exists():
        import shutil
        shutil.copy(index_path, backup_path)

    with open(index_path, "w") as f:
        json.dump(data, f, indent=2)


def matches_agent_pattern(prompt: str, patterns: list[str]) -> bool:
    """Check if a prompt matches any agent pattern."""
    for pattern in patterns:
        if re.search(pattern, prompt):
            return True
    return False


def cleanup_sessions(
    project_dir: Path,
    patterns: list[str],
    dry_run: bool = True,
    delete_files: bool = True,
    include_empty: bool = False,
    show_kept: bool = False,
) -> tuple[int, int]:
    """
    Clean up agent sessions from a project.

    Returns:
        Tuple of (sessions_removed, bytes_freed)
    """
    index_data = load_sessions_index(project_dir)
    entries = index_data.get("entries", [])

    # Combine patterns
    all_patterns = list(patterns)
    if include_empty:
        all_patterns.extend(EMPTY_SESSION_PATTERNS)

    keep_entries = []
    remove_entries = []

    for entry in entries:
        prompt = entry.get("firstPrompt", "")
        if matches_agent_pattern(prompt, all_patterns):
            remove_entries.append(entry)
        else:
            keep_entries.append(entry)

    bytes_freed = 0
    files_to_delete = []

    for entry in remove_entries:
        session_file = Path(entry.get("fullPath", ""))
        if session_file.exists():
            bytes_freed += session_file.stat().st_size
            files_to_delete.append(session_file)

    if dry_run:
        print(f"\n[DRY RUN] Would remove {len(remove_entries)} sessions ({bytes_freed / 1024 / 1024:.1f} MB)")
        print(f"[DRY RUN] Would keep {len(keep_entries)} sessions")

        if remove_entries:
            print("\nSample sessions to remove:")
            for entry in remove_entries[:5]:
                prompt_preview = entry.get("firstPrompt", "")[:60] + "..."
                print(f"  - {entry['sessionId'][:8]}: {prompt_preview}")
            if len(remove_entries) > 5:
                print(f"  ... and {len(remove_entries) - 5} more")

        if show_kept and keep_entries:
            print("\nSessions to keep:")
            for entry in keep_entries:
                prompt_preview = entry.get("firstPrompt", "")[:60]
                if len(entry.get("firstPrompt", "")) > 60:
                    prompt_preview += "..."
                print(f"  - {entry['sessionId'][:8]}: {prompt_preview}")
    else:
        # Actually perform the cleanup
        if delete_files:
            for f in files_to_delete:
                try:
                    f.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete {f}: {e}")

        # Update the index
        index_data["entries"] = keep_entries
        save_sessions_index(project_dir, index_data)

        print(f"Removed {len(remove_entries)} sessions ({bytes_freed / 1024 / 1024:.1f} MB)")
        print(f"Kept {len(keep_entries)} sessions")
        print(f"Backup saved to {project_dir / 'sessions-index.json.bak'}")

    return len(remove_entries), bytes_freed


def main():
    parser = argparse.ArgumentParser(
        description="Clean up agent-generated Claude Code sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--project", "-p",
        type=str,
        help="Target a specific project path (default: current directory)"
    )
    parser.add_argument(
        "--all-projects", "-a",
        action="store_true",
        help="Clean up all projects"
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=DEFAULT_AGENT_PATTERNS,
        help="Regex patterns to match agent prompts"
    )
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Keep .jsonl files (only update index)"
    )
    parser.add_argument(
        "--list-patterns",
        action="store_true",
        help="List default patterns and exit"
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Also remove 'No prompt' sessions"
    )
    parser.add_argument(
        "--show-kept",
        action="store_true",
        help="Show sessions that would be kept"
    )

    args = parser.parse_args()

    if args.list_patterns:
        print("Default agent patterns:")
        for p in DEFAULT_AGENT_PATTERNS:
            print(f"  {p}")
        return 0

    if args.all_projects:
        project_dirs = find_project_dirs()
        if not project_dirs:
            print("No Claude projects found")
            return 1
    else:
        # Use specified project or current directory
        project_path = args.project or os.getcwd()
        project_dir = get_project_session_dir(project_path)

        if not project_dir.exists():
            print(f"No session directory found for: {project_path}")
            print(f"Expected: {project_dir}")
            return 1

        project_dirs = [project_dir]

    total_removed = 0
    total_bytes = 0

    for project_dir in project_dirs:
        print(f"\n{'=' * 60}")
        print(f"Project: {project_dir.name}")
        print('=' * 60)

        removed, bytes_freed = cleanup_sessions(
            project_dir,
            args.patterns,
            dry_run=args.dry_run,
            delete_files=not args.keep_files,
            include_empty=args.include_empty,
            show_kept=args.show_kept,
        )
        total_removed += removed
        total_bytes += bytes_freed

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {total_removed} sessions, {total_bytes / 1024 / 1024:.1f} MB")
    if args.dry_run:
        print("\nRun without --dry-run to actually delete these sessions.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
