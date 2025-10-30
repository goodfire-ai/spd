#!/usr/bin/env python3
"""Download external JavaScript/CSS library dependencies.

Process:
1. Read deps.json to get list of library URLs
2. Create lib_dir if it doesn't exist
3. For each dependency:
   - Download to {fname}.temp
   - If {fname} exists: compare to temp
     - If same: delete temp, skip
     - If different: raise exception with diff details
   - If doesn't exist: rename temp to {fname}
"""

import argparse
import json
import urllib.request
from pathlib import Path


def get_filename_from_url(url: str) -> str:
    """Extract filename from URL."""
    filename: str = url.split("/")[-1]
    return filename


def download_file(url: str, dest_path: Path) -> None:
    """Download a file from URL to destination path."""
    print(f"Downloading {url}...")
    with urllib.request.urlopen(url) as response:
        content: bytes = response.read()
    dest_path.write_bytes(content)


def files_are_identical(path1: Path, path2: Path) -> bool:
    """Check if two files have identical content."""
    identical: bool = path1.read_bytes() == path2.read_bytes()
    return identical


def process_dependencies(deps_file: Path, lib_dir: Path) -> None:
    """Download and process all dependencies from deps.json."""
    # Create lib directory if it doesn't exist
    lib_dir.mkdir(parents=True, exist_ok=True)
    print(f"Library directory: {lib_dir}")

    # Read dependencies
    if not deps_file.exists():
        raise FileNotFoundError(f"Dependencies file not found: {deps_file}")

    deps_data: dict[str, list[dict[str, str]]] = json.loads(deps_file.read_text())

    dependencies: list[dict[str, str]] = deps_data.get("dependencies", [])
    print(f"Found {len(dependencies)} dependencies to process\n")

    downloaded: list[str] = []
    skipped: list[str] = []
    errors: list[tuple[str, str]] = []

    for dep in dependencies:
        url: str = dep["url"]
        # Use custom name if provided, otherwise extract from URL
        filename: str = dep.get("name", get_filename_from_url(url))

        target_path: Path = lib_dir / filename
        temp_path: Path = lib_dir / f"{filename}.temp"

        try:
            # Download to temp file
            download_file(url, temp_path)

            if target_path.exists():
                # Compare with existing file
                if files_are_identical(target_path, temp_path):
                    print(f"  ✓ {filename} is up to date")
                    temp_path.unlink()  # Delete temp file
                    skipped.append(filename)
                else:
                    # Files differ - raise exception
                    temp_path.unlink()  # Clean up temp file
                    error_msg: str = (
                        f"\n{'='*60}\n"
                        f"ERROR: Local file differs from remote!\n"
                        f"File: {filename}\n"
                        f"Local: {target_path}\n"
                        f"Remote: {url}\n"
                        f"\nThe local file has been modified or the remote has been updated.\n"
                        f"Please review the differences and either:\n"
                        f"  - Delete the local file to download the new version\n"
                        f"  - Keep the local modifications (remote will not be downloaded)\n"
                        f"{'='*60}\n"
                    )
                    raise Exception(error_msg)
            else:
                # No existing file - rename temp to final name
                temp_path.rename(target_path)
                print(f"  ✓ {filename} downloaded")
                downloaded.append(filename)

        except Exception as e:
            errors.append((filename, str(e)))
            if temp_path.exists():
                temp_path.unlink()
            raise

    # Print summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Downloaded: {len(downloaded)}")
    print(f"  Skipped (up to date): {len(skipped)}")
    print(f"  Errors: {len(errors)}")

    if downloaded:
        print("\nDownloaded files:")
        for f in downloaded:
            print(f"  - {f}")

    if errors:
        print("\nErrors:")
        for filename, error in errors:
            print(f"  - {filename}: {error}")


def main() -> None:
    """Main entry point with CLI argument parsing."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Download external JavaScript/CSS library dependencies"
    )
    parser.add_argument(
        "--deps-file",
        type=Path,
        default=None,
        help="Path to dependencies JSON file (default: deps.json relative to script)",
    )
    parser.add_argument(
        "--lib-dir",
        type=Path,
        default=None,
        help="Path to library directory (default: lib/ relative to script)",
    )

    args: argparse.Namespace = parser.parse_args()

    # Determine paths
    script_dir: Path = Path(__file__).parent
    lib_dir: Path = args.lib_dir if args.lib_dir else script_dir / "lib"
    deps_file: Path = args.deps_file if args.deps_file else script_dir / "deps.json"

    process_dependencies(deps_file, lib_dir)


if __name__ == "__main__":
    main()
