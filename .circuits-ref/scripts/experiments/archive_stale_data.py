#!/usr/bin/env python3
"""Archive stale investigation data and reports.

Archives files older than a retention period to compressed archives,
keeping only recent data in the working directories.

Usage:
    # Dry run (default) - show what would be archived
    python scripts/archive_stale_data.py

    # Actually archive files older than 7 days
    python scripts/archive_stale_data.py --execute --days 7

    # Archive only investigations
    python scripts/archive_stale_data.py --execute --only investigations

    # Delete instead of archive
    python scripts/archive_stale_data.py --execute --delete
"""

import argparse
import tarfile
from datetime import datetime
from pathlib import Path

# Directories to manage
MANAGED_DIRS = {
    "investigations": Path("outputs/investigations"),
    "reports": Path("frontend/reports"),
}

# Archive destination
ARCHIVE_DIR = Path("archives")


def get_file_age_days(path: Path) -> float:
    """Get file age in days."""
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime).total_seconds() / 86400


def find_stale_files(
    directory: Path,
    days: int,
    extensions: set[str] | None = None,
) -> list[tuple[Path, float]]:
    """Find files older than the specified number of days.

    Returns list of (path, age_in_days) tuples.
    """
    stale = []
    if not directory.exists():
        return stale

    for path in directory.iterdir():
        if path.is_file():
            if extensions and path.suffix not in extensions:
                continue
            age = get_file_age_days(path)
            if age > days:
                stale.append((path, age))

    return sorted(stale, key=lambda x: -x[1])  # Sort by age, oldest first


def format_size(size_bytes: int) -> str:
    """Format size in human-readable form."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def archive_files(
    files: list[Path],
    archive_name: str,
    delete_after: bool = True,
) -> Path:
    """Archive files to a compressed tarball.

    Args:
        files: List of files to archive
        archive_name: Name for the archive (without extension)
        delete_after: Whether to delete files after archiving

    Returns:
        Path to the created archive
    """
    ARCHIVE_DIR.mkdir(exist_ok=True)
    archive_path = ARCHIVE_DIR / f"{archive_name}.tar.gz"

    with tarfile.open(archive_path, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=f.name)

    if delete_after:
        for f in files:
            f.unlink()

    return archive_path


def main():
    parser = argparse.ArgumentParser(
        description="Archive stale investigation data and reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=7,
        help="Archive files older than this many days (default: 7)",
    )
    parser.add_argument(
        "--execute", "-x",
        action="store_true",
        help="Actually perform archiving (default: dry run)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete files instead of archiving them",
    )
    parser.add_argument(
        "--only",
        choices=list(MANAGED_DIRS.keys()),
        help="Only process this directory",
    )
    parser.add_argument(
        "--list-archives",
        action="store_true",
        help="List existing archives and exit",
    )

    args = parser.parse_args()

    # List archives mode
    if args.list_archives:
        if not ARCHIVE_DIR.exists():
            print("No archives directory found.")
            return

        archives = sorted(ARCHIVE_DIR.glob("*.tar.gz"))
        if not archives:
            print("No archives found.")
            return

        print(f"Archives in {ARCHIVE_DIR}/:")
        total_size = 0
        for archive in archives:
            size = archive.stat().st_size
            total_size += size
            age = get_file_age_days(archive)
            print(f"  {archive.name}: {format_size(size)} ({age:.0f} days old)")
        print(f"\nTotal archive size: {format_size(total_size)}")
        return

    # Determine which directories to process
    dirs_to_process = {args.only: MANAGED_DIRS[args.only]} if args.only else MANAGED_DIRS

    # Collect all stale files
    all_stale = {}
    total_size = 0
    total_files = 0

    for name, directory in dirs_to_process.items():
        if name == "investigations":
            extensions = {".json", ".txt"}
        elif name == "reports":
            extensions = {".html", ".json"}
        else:
            extensions = None

        stale = find_stale_files(directory, args.days, extensions)
        if stale:
            all_stale[name] = stale
            size = sum(f.stat().st_size for f, _ in stale)
            total_size += size
            total_files += len(stale)

            print(f"\n{name.upper()} ({directory}):")
            print(f"  Found {len(stale)} files older than {args.days} days ({format_size(size)})")

            # Show oldest 5 files
            for path, age in stale[:5]:
                print(f"    {path.name}: {age:.1f} days old")
            if len(stale) > 5:
                print(f"    ... and {len(stale) - 5} more")

    if not all_stale:
        print(f"\nNo files older than {args.days} days found.")
        return

    print(f"\n{'='*60}")
    print(f"SUMMARY: {total_files} files, {format_size(total_size)} total")
    print(f"{'='*60}")

    if not args.execute:
        print("\n[DRY RUN] No changes made. Use --execute to archive files.")
        return

    # Execute archiving or deletion
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for name, stale in all_stale.items():
        files = [f for f, _ in stale]

        if args.delete:
            print(f"\nDeleting {len(files)} files from {name}...")
            for f in files:
                f.unlink()
            print(f"  Deleted {len(files)} files")
        else:
            archive_name = f"{name}_{timestamp}"
            print(f"\nArchiving {len(files)} files from {name}...")
            archive_path = archive_files(files, archive_name)
            print(f"  Created: {archive_path} ({format_size(archive_path.stat().st_size)})")

    print("\nDone!")


if __name__ == "__main__":
    main()
