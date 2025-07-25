#!/usr/bin/env python3
"""
planwithgemini.py - Intelligent planning with repomix and Gemini
Combines repomix's file packing capabilities with Gemini's analysis for comprehensive planning.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple


# Color codes for terminal output
class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    NC = "\033[0m"  # No Color


def print_colored(message: str, color: str = Colors.NC):
    """Print colored message to terminal."""
    print(f"{color}{message}{Colors.NC}")


def check_dependencies() -> Tuple[bool, List[str]]:
    """Check if required tools are installed."""
    missing = []

    # Check repomix
    try:
        subprocess.run(["repomix", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("repomix (install with: npm install -g repomix)")

    # Check gemini
    try:
        subprocess.run(["gemini", "--help"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("gemini (install from gemini CLI repository)")

    return len(missing) == 0, missing


def extract_keywords(task: str) -> List[str]:
    """Extract relevant keywords from task description for smart file selection."""
    # Common words to exclude
    stop_words = {
        "the",
        "and",
        "or",
        "for",
        "with",
        "from",
        "to",
        "in",
        "on",
        "at",
        "of",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "implement",
        "create",
        "add",
        "update",
        "modify",
        "change",
        "refactor",
    }

    # Extract words
    words = re.findall(r"\b[a-z]+\b", task.lower())

    # Filter out stop words and short words
    keywords = [w for w in words if w not in stop_words and len(w) > 2]

    # Also extract CamelCase and snake_case terms
    special_terms = re.findall(r"[A-Z][a-z]+|[a-z]+_[a-z]+", task)
    keywords.extend([t.lower() for t in special_terms])

    return list(set(keywords))


def build_repomix_command(
    mode: str, task: str, output_file: str, config_file: str, additional_args: Dict
) -> List[str]:
    """Build repomix command based on mode and parameters."""
    base_cmd = [
        "repomix",
        "--format",
        "xml",
        "--output",
        output_file,
        "--stats",
        "--token-count",
    ]

    if config_file and os.path.exists(config_file):
        base_cmd.extend(["--config", config_file])

    if mode == "auto":
        # Smart file selection based on keywords
        keywords = extract_keywords(task)
        print_colored(f"Keywords identified: {', '.join(keywords)}", Colors.BLUE)

        for keyword in keywords:
            base_cmd.extend(
                ["--include", f"**/*{keyword}*", "--include", f"**/{keyword}/**"]
            )
        base_cmd.extend(["--compression-level", "6"])

    elif mode == "full":
        # Full codebase analysis
        base_cmd.extend(
            [
                "--compression-level",
                "9",
                "--exclude",
                "**/node_modules/**",
                "--exclude",
                "**/dist/**",
                "--exclude",
                "**/.git/**",
                "--exclude",
                "**/coverage/**",
            ]
        )

    elif mode == "quick":
        # Quick compressed analysis
        base_cmd.extend(
            ["--compression-level", "9", "--remove-comments", "--remove-empty-lines"]
        )

    elif mode == "security":
        # Security-focused analysis
        base_cmd.extend(
            [
                "--include",
                "**/auth/**",
                "--include",
                "**/security/**",
                "--include",
                "**/*config*",
                "--include",
                "**/*.env*",
                "--security-check",
            ]
        )

    # Add any additional includes/excludes
    if "includes" in additional_args:
        for pattern in additional_args["includes"]:
            base_cmd.extend(["--include", pattern])

    if "excludes" in additional_args:
        for pattern in additional_args["excludes"]:
            base_cmd.extend(["--exclude", pattern])

    # Add target directory
    base_cmd.append(additional_args.get("target_dir", "."))

    return base_cmd


def create_gemini_prompt(mode: str, task: str) -> str:
    """Create appropriate Gemini prompt based on mode and task."""
    prompts = {
        "auto": f"""Analyze this codebase context and create a detailed implementation plan for: {task}

Focus on:
1. Understanding current architecture and patterns
2. Identifying integration points for the new feature
3. Potential risks and dependencies
4. Step-by-step implementation approach with specific code changes
5. Testing and validation strategy
6. Performance considerations

Provide a structured plan with:
- Executive summary
- Implementation phases with time estimates
- Specific files to modify with rationale
- Risk assessment and mitigation
- Success criteria
- Code examples where helpful""",
        "full": f"""Perform a comprehensive analysis of this entire codebase for: {task}

Provide:
1. Current Architecture Assessment
   - Design patterns and principles in use
   - Technical debt and improvement areas
   - Component coupling and cohesion analysis

2. Implementation Strategy
   - Detailed refactoring/implementation plan
   - Migration approach if applicable
   - Backward compatibility considerations

3. System-wide Impact Analysis
   - Direct and indirect effects
   - Breaking changes identification
   - Performance implications

4. Risk Management
   - Critical risks and mitigation strategies
   - Rollback procedures
   - Incremental deployment approach

5. Testing and Validation
   - Comprehensive testing strategy
   - Performance benchmarks
   - Monitoring requirements

Structure as a detailed, actionable plan with phases, milestones, and specific tasks.""",
        "quick": f"""Quickly analyze this codebase and provide a concise implementation plan for: {task}

Focus on:
1. Key changes required
2. Main risks and how to address them
3. Implementation steps (high-level)
4. Testing approach
5. Estimated timeline

Keep the plan practical and actionable.""",
        "security": f"""Perform a security-focused analysis for: {task}

Examine:
1. Authentication and Authorization
   - Current implementation review
   - Vulnerabilities and weaknesses
   - Best practice compliance

2. Data Security
   - Sensitive data handling
   - Encryption usage
   - Data flow analysis

3. API Security
   - Endpoint protection
   - Input validation
   - Rate limiting and abuse prevention

4. Security Improvements
   - Prioritized recommendations
   - Implementation approach
   - Testing requirements

Provide actionable security improvements with implementation details.""",
    }

    return prompts.get(mode, f"Analyze this codebase and provide insights for: {task}")


def run_repomix(cmd: List[str], verbose: bool = False) -> Tuple[bool, str]:
    """Run repomix command and return success status and output."""
    if verbose:
        print_colored(f"Running: {' '.join(cmd)}", Colors.YELLOW)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if verbose and result.stdout:
            print(result.stdout)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print_colored(f"Error running repomix: {e.stderr}", Colors.RED)
        return False, e.stderr


def run_gemini(
    context_file: str, prompt: str, output_file: Optional[str] = None
) -> Tuple[bool, str]:
    """Run Gemini analysis on the context file."""
    print_colored("Running Gemini analysis...", Colors.BLUE)

    cmd = ["gemini", "-p", f"@{context_file}", prompt]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        if output_file:
            with open(output_file, "w") as f:
                f.write(result.stdout)
            print_colored(f"Plan saved to: {output_file}", Colors.GREEN)
        else:
            print(result.stdout)

        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print_colored(f"Error running Gemini: {e.stderr}", Colors.RED)
        return False, e.stderr


def get_file_stats(file_path: str) -> Dict[str, str]:
    """Get basic statistics about a file."""
    if not os.path.exists(file_path):
        return {}

    stats = os.stat(file_path)
    size = stats.st_size

    # Convert to human-readable size
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            size_str = f"{size:.1f} {unit}"
            break
        size /= 1024.0
    else:
        size_str = f"{size:.1f} TB"

    return {
        "size": size_str,
        "modified": datetime.fromtimestamp(stats.st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent planning with repomix and Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s auto "implement caching layer"
  %(prog)s full "refactor authentication system" -o auth-refactor-plan.md
  %(prog)s security "audit API endpoints"
  %(prog)s custom "analyze performance" -- --include "**/perf/**"
        """,
    )

    parser.add_argument(
        "mode",
        choices=["auto", "full", "quick", "security", "custom"],
        help="Analysis mode",
    )
    parser.add_argument("task", help="Task description for planning")
    parser.add_argument("-c", "--config", help="Custom config file")
    parser.add_argument("-o", "--output", help="Save plan to file")
    parser.add_argument(
        "-i",
        "--include",
        action="append",
        default=[],
        help="Additional include patterns",
    )
    parser.add_argument(
        "-e",
        "--exclude",
        action="append",
        default=[],
        help="Additional exclude patterns",
    )
    parser.add_argument("-t", "--target", default=".", help="Target directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--keep-context", action="store_true", help="Keep context file after analysis"
    )

    args, unknown = parser.parse_known_args()

    # Check dependencies
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        print_colored("Missing dependencies:", Colors.RED)
        for dep in missing:
            print_colored(f"  - {dep}", Colors.RED)
        sys.exit(1)

    # Set up paths
    temp_dir = Path(".temp")
    temp_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    context_file = temp_dir / f"context_{args.mode}_{timestamp}.xml"

    config_file = args.config or "repomix-gemini.config.json"

    print_colored(f"Planning with Gemini - Mode: {args.mode}", Colors.GREEN)
    print_colored(f"Task: {args.task}", Colors.GREEN)
    print()

    # Build repomix command
    additional_args = {
        "includes": args.include,
        "excludes": args.exclude,
        "target_dir": args.target,
    }

    if args.mode == "custom" and unknown:
        # For custom mode, use the additional arguments as repomix args
        cmd = ["repomix"] + unknown + ["--output", str(context_file)]
    else:
        cmd = build_repomix_command(
            args.mode, args.task, str(context_file), config_file, additional_args
        )

    # Run repomix
    print_colored(f"Analyzing codebase with repomix ({args.mode} mode)...", Colors.BLUE)
    success, output = run_repomix(cmd, args.verbose)

    if not success:
        sys.exit(1)

    # Check if context file was created
    if not context_file.exists():
        print_colored("Error: Failed to generate context with repomix", Colors.RED)
        sys.exit(1)

    # Show context stats
    stats = get_file_stats(str(context_file))
    if stats:
        print_colored(f"Context file generated: {stats['size']}", Colors.PURPLE)

    # Parse token count from repomix output if available
    token_match = re.search(r"Total tokens:\s*(\d+)", output)
    if token_match:
        tokens = int(token_match.group(1))
        print_colored(f"Total tokens: {tokens:,}", Colors.PURPLE)

    # Create Gemini prompt
    prompt = create_gemini_prompt(args.mode, args.task)

    # Run Gemini analysis
    success, output = run_gemini(str(context_file), prompt, args.output)

    if not success:
        sys.exit(1)

    # Cleanup
    if not args.keep_context and context_file.exists():
        context_file.unlink()
    elif args.keep_context:
        print_colored(f"Context file kept at: {context_file}", Colors.YELLOW)

    print_colored("Planning complete!", Colors.GREEN)


if __name__ == "__main__":
    main()
