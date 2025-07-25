#!/usr/bin/env python3
"""
claude-refactor.py - Direct integration for Claude Code refactoring workflows
Automatically packs relevant files and passes them to Gemini for planning
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


def run_command(cmd: List[str]) -> tuple[bool, str, str]:
    """Run a command and return success, stdout, stderr."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr


def pack_files_for_refactor(files: List[str], output_path: str) -> bool:
    """Pack specific files for refactoring analysis."""
    cmd = [
        "repomix",
        "--format",
        "xml",
        "--output",
        output_path,
        "--compression-level",
        "6",
        "--stats",
        "--token-count",
    ]

    # Add each file explicitly
    for file in files:
        if os.path.exists(file):
            cmd.extend(["--include", file])

    # Add the current directory as target
    cmd.append(".")

    success, stdout, stderr = run_command(cmd)
    if success:
        print(f"✓ Packed {len(files)} files for analysis")
        # Try to extract token count
        if "Total tokens:" in stdout:
            tokens = stdout.split("Total tokens:")[1].split("\n")[0].strip()
            print(f"  Token count: {tokens}")
    else:
        print(f"✗ Failed to pack files: {stderr}")

    return success


def analyze_refactor_with_gemini(
    context_file: str, refactor_description: str, files: List[str]
) -> str:
    """Send packed context to Gemini for refactoring analysis."""

    file_list = "\n".join([f"- {f}" for f in files])

    prompt = f"""Analyze this codebase for the following refactoring task: {refactor_description}

Files involved in this refactoring:
{file_list}

Please provide:
1. **Current State Analysis**
   - How these files are currently structured
   - Dependencies and coupling between them
   - Technical debt or issues identified

2. **Refactoring Strategy**
   - Step-by-step refactoring approach
   - Order of operations to maintain functionality
   - Intermediate states that keep tests passing

3. **Specific Changes Required**
   - For each file, what needs to change
   - New files that need to be created
   - Files that can be deleted

4. **Testing Plan**
   - How to verify each step
   - New tests needed
   - Existing tests that might break

5. **Risk Assessment**
   - What could go wrong
   - How to mitigate risks
   - Rollback strategy if needed

Provide concrete code examples for the key changes."""

    cmd = ["gemini", "-p", f"@{context_file}", prompt]

    success, stdout, stderr = run_command(cmd)
    if success:
        return stdout
    else:
        return f"Error running Gemini: {stderr}"


def main():
    """Main function for Claude Code integration."""
    if len(sys.argv) < 3:
        print('Usage: claude-refactor.py "refactor description" file1 file2 ...')
        print(
            'Example: claude-refactor.py "extract authentication logic" src/api.py src/auth.py'
        )
        sys.exit(1)

    refactor_description = sys.argv[1]
    files = sys.argv[2:]

    # Validate files exist
    valid_files = []
    for file in files:
        if os.path.exists(file):
            valid_files.append(file)
        else:
            print(f"Warning: File not found: {file}")

    if not valid_files:
        print("Error: No valid files provided")
        sys.exit(1)

    print(f"🔄 Refactoring: {refactor_description}")
    print(f"📁 Files: {len(valid_files)}")

    # Create temp directory
    temp_dir = Path(".temp")
    temp_dir.mkdir(exist_ok=True)

    # Generate context file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    context_file = temp_dir / f"refactor_{timestamp}.xml"

    # Pack files
    if not pack_files_for_refactor(valid_files, str(context_file)):
        sys.exit(1)

    # Analyze with Gemini
    print("\n🤖 Analyzing refactoring approach with Gemini...")
    result = analyze_refactor_with_gemini(
        str(context_file), refactor_description, valid_files
    )

    print("\n" + "=" * 80)
    print(result)
    print("=" * 80)

    # Save result
    output_file = temp_dir / f"refactor_plan_{timestamp}.md"
    with open(output_file, "w") as f:
        f.write(f"# Refactoring Plan: {refactor_description}\n\n")
        f.write(f"**Files:** {', '.join(valid_files)}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        f.write(result)

    print(f"\n✅ Refactoring plan saved to: {output_file}")

    # Clean up context file
    context_file.unlink()


if __name__ == "__main__":
    main()
