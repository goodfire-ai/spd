#!/usr/bin/env python3
"""
Automatic context-aware planning tool that intelligently selects relevant files
and optionally uses external tools like repomix and Gemini for enhanced analysis.
"""

import os
import sys
import json
import subprocess
import tempfile
import argparse
import glob
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import shutil


class AutoContextPlanner:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).absolute()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="plan_context_"))

    def __del__(self):
        # Cleanup temp directory
        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def extract_keywords_from_description(
        self, description: str
    ) -> Dict[str, List[str]]:
        """Extract relevant keywords and patterns from natural language description."""
        description_lower = description.lower()

        keywords = {"primary": [], "technical": [], "actions": [], "components": []}

        # Common technical terms mapping
        tech_mappings = {
            "cache": ["cache", "redis", "memcache", "caching"],
            "api": ["api", "route", "endpoint", "controller", "rest"],
            "auth": ["auth", "login", "session", "token", "jwt", "oauth"],
            "database": ["db", "database", "model", "schema", "migration"],
            "queue": ["queue", "job", "worker", "celery", "rq", "bull"],
            "websocket": ["websocket", "ws", "socket", "realtime", "io"],
            "notification": ["notification", "alert", "message", "notify"],
            "payment": ["payment", "stripe", "billing", "subscription"],
            "email": ["email", "mail", "smtp", "sendgrid"],
            "test": ["test", "spec", "unittest", "pytest", "jest"],
            "security": ["security", "encryption", "https", "ssl", "csrf"],
            "performance": ["performance", "optimize", "fast", "slow", "speed"],
            "refactor": ["refactor", "reorganize", "restructure", "cleanup"],
        }

        # Extract technical keywords
        for term, related in tech_mappings.items():
            if any(word in description_lower for word in related):
                keywords["technical"].extend(related)

        # Extract action words
        action_words = [
            "implement",
            "add",
            "create",
            "build",
            "refactor",
            "optimize",
            "integrate",
            "update",
            "fix",
            "improve",
            "enhance",
        ]
        keywords["actions"] = [
            word for word in action_words if word in description_lower
        ]

        # Extract component names (simple word extraction)
        words = re.findall(r"\b\w+\b", description_lower)
        # Filter common words and keep potentially relevant component names
        common_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "i",
            "want",
            "need",
            "should",
            "would",
            "could",
            "our",
            "my",
            "your",
            "their",
            "it",
            "this",
            "that",
        }
        keywords["primary"] = [w for w in words if w not in common_words and len(w) > 2]

        return keywords

    def find_relevant_files(
        self, keywords: Dict[str, List[str]]
    ) -> List[Tuple[str, int]]:
        """Find files relevant to the keywords with relevance scoring."""
        relevant_files = {}

        # Common source directories
        source_dirs = [
            "src",
            "app",
            "lib",
            "api",
            "server",
            "client",
            "packages",
            "backend",
            "frontend",
        ]
        search_dirs = [d for d in source_dirs if (self.project_root / d).exists()]
        if not search_dirs:
            search_dirs = ["."]

        # Also search for all Python/JS/TS files if no specific patterns match
        all_source_files = []
        for search_dir in search_dirs:
            search_path = self.project_root / search_dir
            if search_path.exists():
                for ext in ["py", "js", "ts", "jsx", "tsx"]:
                    files = list(search_path.rglob(f"*.{ext}"))
                    all_source_files.extend(files)

        # File extensions to consider
        extensions = ["py", "js", "ts", "jsx", "tsx", "java", "go", "rs", "rb", "php"]

        # Search for files
        for keyword in set(keywords["primary"] + keywords["technical"]):
            if len(keyword) < 3:  # Skip very short keywords
                continue

            for search_dir in search_dirs:
                search_path = self.project_root / search_dir
                # Search for files containing keyword in name
                for ext in extensions:
                    pattern = f"**/*{keyword}*.{ext}"
                    try:
                        for file_path in search_path.rglob(f"*{keyword}*.{ext}"):
                            if self._should_exclude_file(file_path):
                                continue
                            rel_path = file_path.relative_to(self.project_root)
                            relevant_files[str(rel_path)] = (
                                relevant_files.get(str(rel_path), 0) + 2
                            )
                    except Exception:
                        pass

                # Search for files containing keyword in content
                try:
                    grep_cmd = [
                        "grep",
                        "-r",
                        "-l",
                        "-i",
                        keyword,
                        str(self.project_root / search_dir),
                    ]
                    result = subprocess.run(
                        grep_cmd, capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        for line in result.stdout.strip().split("\n"):
                            if line:
                                file_path = Path(line)
                                if self._should_exclude_file(file_path):
                                    continue
                                rel_path = file_path.relative_to(self.project_root)
                                relevant_files[str(rel_path)] = (
                                    relevant_files.get(str(rel_path), 0) + 1
                                )
                except (subprocess.TimeoutExpired, Exception):
                    pass

        # If we found few files, add all source files with lower relevance
        if len(relevant_files) < 10:
            for file_path in all_source_files:
                if self._should_exclude_file(file_path):
                    continue
                rel_path = file_path.relative_to(self.project_root)
                str_path = str(rel_path)
                if str_path not in relevant_files:
                    # Give base score of 0.5 for being a source file
                    relevant_files[str_path] = 0.5

        # Sort by relevance score
        sorted_files = sorted(relevant_files.items(), key=lambda x: x[1], reverse=True)
        return sorted_files[:50]  # Limit to top 50 files

    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded."""
        exclude_patterns = [
            "node_modules",
            ".git",
            "dist",
            "build",
            "__pycache__",
            ".pytest_cache",
            "coverage",
            ".next",
            "venv",
            ".env",
        ]

        # Check file name patterns
        file_name = file_path.name
        if file_name.endswith((".min.js", ".min.css", ".map", ".lock", ".sum")):
            return True

        # Check path components
        path_parts = file_path.parts
        for part in path_parts:
            if part in exclude_patterns:
                return True

        # Exclude test files initially (can be added later if needed)
        if re.search(r"(test_|_test|\.test\.|\.spec\.)", file_name):
            return True

        return False

    def check_external_tools(self) -> Dict[str, bool]:
        """Check availability of external tools."""
        tools = {
            "repomix": shutil.which("repomix") is not None,
            "gemini": shutil.which("gemini") is not None,
            "ripgrep": shutil.which("rg") is not None,
        }
        return tools

    def create_context_with_repomix(
        self, keywords: List[str], output_file: str
    ) -> bool:
        """Use repomix to create context file if available."""
        if not shutil.which("repomix"):
            return False

        try:
            include_patterns = []
            for keyword in keywords:
                include_patterns.extend(
                    [
                        f"**/*{keyword}*",
                        f"**/*{keyword}*.py",
                        f"**/*{keyword}*.js",
                        f"**/*{keyword}*.ts",
                    ]
                )

            cmd = [
                "repomix",
                "--format",
                "xml",
                "--output",
                output_file,
                "--stats",
                "--compression-level",
                "6",
            ]

            # Add include patterns
            for pattern in include_patterns[:10]:  # Limit patterns
                cmd.extend(["--include", pattern])

            # Add exclude patterns
            exclude_patterns = [
                "**/node_modules/**",
                "**/.git/**",
                "**/dist/**",
                "**/build/**",
                "**/*.test.*",
                "**/*.spec.*",
            ]
            for pattern in exclude_patterns:
                cmd.extend(["--exclude", pattern])

            cmd.append(str(self.project_root))

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except Exception:
            return False

    def analyze_with_gemini(
        self, context_file: str, task_description: str
    ) -> Optional[str]:
        """Use Gemini to analyze context if available."""
        if not shutil.which("gemini"):
            return None

        try:
            prompt = f"""Analyze this codebase context and create a detailed implementation plan for: {task_description}
            
Focus on:
1. Current architecture and patterns
2. Integration points for new feature
3. Potential risks and dependencies
4. Step-by-step implementation approach
5. Testing and validation strategy
6. Time estimates for each phase

Provide a comprehensive plan with:
- Executive summary
- Phase-based implementation steps
- Specific files to modify
- Code examples where helpful
- Risk assessment
- Success criteria"""

            cmd = ["gemini", "-p", f"@{context_file}", prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                return result.stdout
            return None
        except Exception:
            return None

    def create_fallback_plan(
        self,
        description: str,
        keywords: Dict[str, List[str]],
        relevant_files: List[Tuple[str, int]],
    ) -> str:
        """Create a plan without external tools."""
        plan = f"""# Implementation Plan: {description}

## Analysis Summary
Based on your description, I've identified the following key components:

### Keywords Extracted
- Primary: {", ".join(keywords["primary"][:10])}
- Technical: {", ".join(keywords["technical"][:10])}
- Actions: {", ".join(keywords["actions"])}

### Relevant Files Identified
"""

        # Group files by relevance
        high_relevance = [(f, s) for f, s in relevant_files if s >= 3]
        medium_relevance = [(f, s) for f, s in relevant_files if 1 <= s < 3]
        low_relevance = [(f, s) for f, s in relevant_files if s < 1]

        plan += f"\n**High Relevance ({len(high_relevance)} files)**:\n"
        for file, score in high_relevance[:10]:
            plan += f"- `{file}` (relevance score: {score})\n"

        if medium_relevance:
            plan += f"\n**Medium Relevance ({len(medium_relevance)} files)**:\n"
            for file, score in medium_relevance[:10]:
                plan += f"- `{file}` (relevance score: {score})\n"

        if low_relevance and not high_relevance and not medium_relevance:
            plan += f"\n**Available Source Files ({len(low_relevance)} files)**:\n"
            for file, score in low_relevance[:15]:
                plan += f"- `{file}`\n"

        plan += f"""

## Recommended Implementation Approach

### Phase 1: Analysis and Planning
1. Review the identified files to understand current architecture
2. Identify integration points for your feature
3. Document any dependencies or constraints

### Phase 2: Implementation
1. Start with core functionality
2. Integrate with existing components
3. Add error handling and validation

### Phase 3: Testing
1. Write unit tests for new components
2. Add integration tests
3. Perform manual testing

### Phase 4: Documentation and Deployment
1. Update documentation
2. Prepare deployment scripts
3. Create rollback plan

## Next Steps
1. Review the relevant files identified above
2. Use Claude Code to read and analyze the high-relevance files
3. Create a more detailed plan based on the actual code structure
4. Begin implementation starting with the core components

Note: For more comprehensive analysis, consider installing:
- `repomix` for better context extraction: `npm install -g repomix`
- `gemini` CLI for enhanced AI analysis
"""
        return plan

    def create_plan(self, description: str) -> str:
        """Main method to create an implementation plan."""
        print(f"Creating implementation plan for: {description}\n")

        # Extract keywords
        print("Extracting keywords and patterns...")
        keywords = self.extract_keywords_from_description(description)

        # Find relevant files
        print("Searching for relevant files...")
        relevant_files = self.find_relevant_files(keywords)
        print(f"Found {len(relevant_files)} potentially relevant files\n")

        # Check available tools
        tools = self.check_external_tools()
        print("External tools available:")
        for tool, available in tools.items():
            print(f"  - {tool}: {'✓' if available else '✗'}")
        print()

        # Try to use external tools if available
        if tools["repomix"] and tools["gemini"]:
            print("Using repomix and Gemini for enhanced analysis...")
            context_file = str(self.temp_dir / "context.xml")

            # Create context with repomix
            if self.create_context_with_repomix(
                keywords["primary"] + keywords["technical"], context_file
            ):
                # Analyze with Gemini
                gemini_result = self.analyze_with_gemini(context_file, description)
                if gemini_result:
                    print("Successfully created plan with Gemini analysis!")
                    return gemini_result

        # Fallback to internal analysis
        print("Creating plan with internal analysis...")
        return self.create_fallback_plan(description, keywords, relevant_files)


def main():
    parser = argparse.ArgumentParser(
        description="Create implementation plans with automatic context selection"
    )
    parser.add_argument(
        "description",
        nargs="+",
        help="Natural language description of what you want to implement",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Root directory of the project (default: current directory)",
    )
    parser.add_argument(
        "--output", help="Output file for the plan (default: print to stdout)"
    )

    args = parser.parse_args()
    description = " ".join(args.description)

    # Create planner and generate plan
    planner = AutoContextPlanner(args.project_root)
    plan = planner.create_plan(description)

    # Output plan
    if args.output:
        with open(args.output, "w") as f:
            f.write(plan)
        print(f"\nPlan saved to: {args.output}")
    else:
        print("\n" + "=" * 80 + "\n")
        print(plan)


if __name__ == "__main__":
    main()
