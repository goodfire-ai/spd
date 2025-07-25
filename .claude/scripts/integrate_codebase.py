#!/usr/bin/env python3
"""
Codebase Integration Script
Helps integrate external codebases into the research workflow.
"""

import argparse
import subprocess
import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import shutil
import sys
from datetime import datetime


class CodebaseIntegrator:
    def __init__(self):
        self.integration_dir = Path("external_codebases")
        self.ai_docs_dir = Path("ai_docs/codebases")
        # Ensure directories exist
        self.integration_dir.mkdir(exist_ok=True)
        self.ai_docs_dir.mkdir(parents=True, exist_ok=True)

    def clone_or_download(self, source: str, target_name: Optional[str] = None) -> Path:
        """Clone or download a codebase from various sources"""
        if source.startswith(("http://", "https://")) and source.endswith(".git"):
            # Git repository
            repo_name = target_name or source.split("/")[-1].replace(".git", "")
            target_path = self.integration_dir / repo_name

            if target_path.exists():
                print(f"✅ Repository already exists at: {target_path}")
                # Pull latest changes
                try:
                    result = subprocess.run(
                        ["git", "pull"], cwd=target_path, capture_output=True, text=True
                    )
                    if result.returncode != 0:
                        print(
                            f"⚠️  Warning: Could not pull latest changes: {result.stderr}"
                        )
                except Exception as e:
                    print(f"⚠️  Warning: Could not pull latest changes: {e}")
            else:
                print(f"📥 Cloning repository: {source}")
                try:
                    result = subprocess.run(
                        ["git", "clone", source, str(target_path)],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode != 0:
                        raise Exception(f"Git clone failed: {result.stderr}")
                    print(f"✅ Successfully cloned to: {target_path}")
                except subprocess.CalledProcessError as e:
                    raise Exception(f"Failed to clone repository: {e}")
                except FileNotFoundError:
                    raise Exception(
                        "Git is not installed. Please install git and try again."
                    )

            return target_path
        elif source.startswith(("http://", "https://")):
            # Direct download
            print(f"📥 Downloading: {source}")
            # Implementation for downloading archives
            pass
        else:
            # Local path
            source_path = Path(source)
            if source_path.exists():
                target_path = self.integration_dir / (target_name or source_path.name)
                if source_path.is_dir():
                    shutil.copytree(source_path, target_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(source_path, target_path)
                return target_path
            else:
                raise ValueError(f"Source not found: {source}")

    def analyze_codebase(self, codebase_path: Path) -> Dict[str, Any]:
        """Analyze the structure and contents of a codebase"""
        print(f"🔍 Analyzing codebase: {codebase_path}")

        analysis = {
            "path": str(codebase_path),
            "name": codebase_path.name,
            "structure": {},
            "languages": {},
            "key_files": [],
            "dependencies": {},
        }

        # Count files by extension
        for root, dirs, files in os.walk(codebase_path):
            # Skip hidden and build directories
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d not in ["node_modules", "__pycache__", "build", "dist"]
            ]

            for file in files:
                if file.startswith("."):
                    continue

                ext = Path(file).suffix.lower()
                if ext:
                    analysis["languages"][ext] = analysis["languages"].get(ext, 0) + 1

                # Identify key files
                if file in [
                    "README.md",
                    "setup.py",
                    "package.json",
                    "requirements.txt",
                    "Cargo.toml",
                ]:
                    analysis["key_files"].append(os.path.join(root, file))

        # Identify project type and dependencies
        try:
            if (codebase_path / "package.json").exists():
                with open(codebase_path / "package.json", "r") as f:
                    pkg = json.load(f)
                    analysis["project_type"] = "nodejs"
                    analysis["dependencies"]["npm"] = list(
                        pkg.get("dependencies", {}).keys()
                    )
        except Exception as e:
            print(f"⚠️  Could not parse package.json: {e}")

        try:
            if (codebase_path / "requirements.txt").exists():
                with open(codebase_path / "requirements.txt", "r") as f:
                    analysis["project_type"] = "python"
                    analysis["dependencies"]["pip"] = [
                        line.strip()
                        for line in f
                        if line.strip() and not line.startswith("#")
                    ]
        except Exception as e:
            print(f"⚠️  Could not parse requirements.txt: {e}")

        # Check for other project types
        if (codebase_path / "Cargo.toml").exists():
            analysis["project_type"] = "rust"
        elif (codebase_path / "go.mod").exists():
            analysis["project_type"] = "go"
        elif (codebase_path / "pom.xml").exists():
            analysis["project_type"] = "java"

        return analysis

    def create_integration_config(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create integration configuration based on analysis"""
        config = {
            "codebase": {
                "name": analysis["name"],
                "path": analysis["path"],
                "type": analysis.get("project_type", "unknown"),
            },
            "integration": {"commands": [], "hooks": [], "workflows": []},
        }

        # Add language-specific commands
        if ".py" in analysis["languages"]:
            config["integration"]["commands"].extend(
                [
                    {
                        "name": "analyze_python_functions",
                        "command": "python scripts/commands/analyze_function.py",
                        "args": "--codebase {path}",
                    }
                ]
            )

        # Add project-specific workflows
        if analysis.get("project_type") == "python":
            config["integration"]["workflows"].append(
                {
                    "name": "python_safety_analysis",
                    "steps": [
                        {
                            "tool": "multi_mind",
                            "args": {"focus": "security vulnerabilities"},
                        },
                        {"tool": "analyze_function", "args": {"focus": "complexity"}},
                    ],
                }
            )

        return config

    def generate_claude_md(
        self, analysis: Dict[str, Any], config: Dict[str, Any]
    ) -> str:
        """Generate CLAUDE.md content for the integrated codebase"""
        claude_md = f"""# {analysis["name"]} Integration

## Codebase Overview
- **Path**: `{analysis["path"]}`
- **Type**: {analysis.get("project_type", "Unknown")}
- **Primary Languages**: {", ".join(sorted(analysis["languages"].keys())[:5])}

## Key Files
{chr(10).join(f"- {file}" for file in analysis["key_files"][:10])}

## Integration Commands

### Analysis Commands
```bash
# Analyze functions
python scripts/commands/analyze_function.py <function_name> --codebase {analysis["path"]}

# Multi-mind analysis
python scripts/commands/multi_mind.py "Analyze security of {analysis["name"]}" --output ./analysis

# Generate specifications
python scripts/commands/spec_driven.py "<feature>" --output ./specs
```

### Orchestrated Workflows
```bash
# Comprehensive analysis
python scripts/commands/orchestrate_research.py \\
    --workflow comprehensive_analysis \\
    --codebase {analysis["path"]} \\
    --task "Analyze {analysis["name"]} architecture and security"

# Custom workflow
python scripts/commands/orchestrate_research.py \\
    --custom-steps custom_workflow.json \\
    --codebase {analysis["path"]}
```

## Quick Tasks

1. **Security Analysis**
   ```bash
   /multi-mind "Analyze security vulnerabilities in {analysis["name"]}"
   ```

2. **Architecture Review**
   ```bash
   gemini -p "@{analysis["path"]}" "Explain the architecture and design patterns"
   ```

3. **Code Quality Assessment**
   ```bash
   /analyze-function main --codebase {analysis["path"]}
   ```

## Notes
- This codebase was integrated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Use the orchestration tools to coordinate complex analyses
- Always verify findings with multiple tools
"""
        return claude_md

    def create_ai_docs(self, analysis: Dict[str, Any], codebase_path: Path):
        """Create comprehensive AI-friendly documentation"""
        docs_path = self.ai_docs_dir / analysis["name"]
        docs_path.mkdir(parents=True, exist_ok=True)

        # Create overview
        overview = f"""# {analysis["name"]} Overview

## Purpose
This codebase provides [TODO: Add purpose based on README or manual analysis]

## Technology Stack
- **Primary Language**: {max(analysis["languages"].items(), key=lambda x: x[1])[0] if analysis["languages"] else "Unknown"}
- **Project Type**: {analysis.get("project_type", "Unknown")}
- **Key Dependencies**: {", ".join(list(analysis["dependencies"].get("pip", [])[:5]) or list(analysis["dependencies"].get("npm", [])[:5]) or ["None identified"])}

## Architecture
[TODO: Add architecture overview after analysis]

## Key Features
[TODO: List main features]
"""
        with open(docs_path / "overview.md", "w") as f:
            f.write(overview)

        # Create structure map
        structure = f"""# {analysis["name"]} Structure Map

## Directory Layout
```
{self._generate_tree(codebase_path, max_depth=3)}
```

## Key Files
{chr(10).join(f"- `{file}` - [TODO: Add description]" for file in analysis["key_files"][:10])}

## Language Distribution
{chr(10).join(f"- {ext}: {count} files" for ext, count in sorted(analysis["languages"].items(), key=lambda x: x[1], reverse=True)[:10])}
"""
        with open(docs_path / "structure_map.md", "w") as f:
            f.write(structure)

        # Create context summary
        context = f"""# {analysis["name"]} Context Summary

## Quick Reference
This is an AI-optimized summary for quick context loading.

### What is {analysis["name"]}?
[TODO: One paragraph summary]

### Key Components
[TODO: List main components with brief descriptions]

### Common Tasks
[TODO: List common operations]

### Integration Points
[TODO: How to integrate with other systems]
"""
        with open(docs_path / "context_summary.md", "w") as f:
            f.write(context)

        print(f"📝 Created AI documentation in: {docs_path}")
        return docs_path

    def _generate_tree(self, path: Path, prefix="", max_depth=3, current_depth=0):
        """Generate a tree structure of the directory"""
        if current_depth >= max_depth:
            return prefix + "..."

        tree = ""
        items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
        for i, item in enumerate(items):
            if item.name.startswith(".") or item.name in [
                "__pycache__",
                "node_modules",
                ".git",
            ]:
                continue

            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            tree += prefix + current_prefix + item.name

            if item.is_dir() and current_depth < max_depth - 1:
                tree += "/\n"
                extension = "    " if is_last else "│   "
                tree += self._generate_tree(
                    item, prefix + extension, max_depth, current_depth + 1
                )
            else:
                tree += "\n"

        return tree

    def save_integration(
        self, analysis: Dict[str, Any], config: Dict[str, Any], docs_path: Path
    ):
        """Save integration configuration and documentation"""
        integration_name = analysis["name"]
        integration_path = Path(".claude/integrations") / integration_name
        integration_path.mkdir(parents=True, exist_ok=True)

        # Save analysis
        with open(integration_path / "analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)

        # Save config
        with open(integration_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save CLAUDE.md
        claude_md = self.generate_claude_md(analysis, config)
        with open(integration_path / "CLAUDE.md", "w") as f:
            f.write(claude_md)

        # Add reference to AI docs
        config["ai_docs_path"] = str(docs_path)
        with open(integration_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"✅ Integration saved to: {integration_path}")
        print(f"📄 CLAUDE.md generated for easy reference")

    def integrate(self, source: str, name: Optional[str] = None) -> Dict[str, Any]:
        """Main integration workflow"""
        # Download/clone codebase
        codebase_path = self.clone_or_download(source, name)

        # Analyze structure
        analysis = self.analyze_codebase(codebase_path)

        # Create integration config
        config = self.create_integration_config(analysis)

        # Create AI-friendly documentation
        docs_path = self.create_ai_docs(analysis, codebase_path)

        # Save integration
        self.save_integration(analysis, config, docs_path)

        return {
            "path": codebase_path,
            "analysis": analysis,
            "config": config,
            "docs_path": docs_path,
        }


def main():
    parser = argparse.ArgumentParser(description="Integrate External Codebase")
    parser.add_argument("source", help="GitHub URL, local path, or archive URL")
    parser.add_argument("--name", type=str, help="Custom name for the integration")
    parser.add_argument(
        "--analyze-only", action="store_true", help="Only analyze, don't save"
    )

    args = parser.parse_args()

    integrator = CodebaseIntegrator()

    try:
        result = integrator.integrate(args.source, args.name)

        print("\n" + "=" * 80)
        print("INTEGRATION COMPLETE")
        print("=" * 80)
        print(f"Codebase: {result['analysis']['name']}")
        print(f"Location: {result['path']}")
        print(
            f"Languages: {', '.join(list(result['analysis']['languages'].keys())[:5])}"
        )
        print(f"\n📁 Created Documentation:")
        print(f"   - Overview: {result['docs_path']}/overview.md")
        print(f"   - Structure: {result['docs_path']}/structure_map.md")
        print(f"   - Context: {result['docs_path']}/context_summary.md")
        print(f"\n🚀 Next steps:")
        print(f"1. Review and complete TODOs in: {result['docs_path']}/")
        print(
            f"2. Check integration config: .claude/integrations/{result['analysis']['name']}/"
        )
        print(f"3. Use in Claude: @{result['docs_path']}/context_summary.md")
        print(f"4. Run analysis: /integrate-external-codebase follow-up")

    except Exception as e:
        print(f"❌ Integration failed: {e}")
        print(f"\n💡 Troubleshooting tips:")
        print(f"   - Check if git is installed: git --version")
        print(f"   - Verify the URL is correct")
        print(f"   - For private repos, set up authentication first")
        print(f"   - Try with --name flag for custom naming")
        return 1


if __name__ == "__main__":
    main()
