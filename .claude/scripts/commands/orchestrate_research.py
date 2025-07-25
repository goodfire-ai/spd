#!/usr/bin/env python3
"""
Research Orchestration Script
Coordinates multiple AI models and tools for complex research workflows.
"""

import json
import yaml
import argparse
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import concurrent.futures
import tempfile
import time


class ResearchOrchestrator:
    def __init__(self, config_path: Optional[str] = None):
        self.config = self.load_config(config_path)
        self.results = {}
        self.workflow_log = []

    def load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load orchestration configuration"""
        default_config = {
            "tools": {
                "claude": {"command": "claude", "max_tokens": 100000},
                "gemini": {"command": "gemini", "model": "gemini-pro"},
                "multi_mind": {"command": "python scripts/commands/multi_mind.py"},
                "spec_driven": {"command": "python scripts/commands/spec_driven.py"},
                "analyze_function": {
                    "command": "python scripts/commands/analyze_function.py"
                },
            },
            "workflows": {
                "comprehensive_analysis": [
                    {"tool": "multi_mind", "args": {"agents": 6}},
                    {"tool": "gemini", "args": {"focus": "architecture"}},
                    {"tool": "claude", "args": {"focus": "implementation"}},
                ],
                "spec_to_implementation": [
                    {"tool": "spec_driven", "args": {}},
                    {"tool": "claude", "args": {"task": "implement"}},
                    {"tool": "analyze_function", "args": {"verify": True}},
                ],
            },
        }

        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                    loaded_config = yaml.safe_load(f)
                else:
                    loaded_config = json.load(f)
                default_config.update(loaded_config)

        return default_config

    def log_step(self, step: str, status: str, details: Any = None):
        """Log workflow step"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "status": status,
            "details": details,
        }
        self.workflow_log.append(entry)
        print(f"[{entry['timestamp']}] {step}: {status}")

    def run_claude(self, prompt: str, context: Optional[str] = None) -> str:
        """Run Claude with a prompt"""
        self.log_step("Claude Analysis", "started")

        full_prompt = prompt
        if context:
            full_prompt = f"{context}\n\n{prompt}"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(full_prompt)
            prompt_file = f.name

        try:
            result = subprocess.run(
                [self.config["tools"]["claude"]["command"], "-m", prompt_file],
                capture_output=True,
                text=True,
            )
            self.log_step("Claude Analysis", "completed")
            return result.stdout
        finally:
            os.unlink(prompt_file)

    def run_gemini(self, codebase_path: str, query: str) -> str:
        """Run Gemini analysis on codebase"""
        self.log_step("Gemini Analysis", "started")

        command = [
            self.config["tools"]["gemini"]["command"],
            "-p",
            f"@{codebase_path}",
            query,
        ]

        result = subprocess.run(command, capture_output=True, text=True)
        self.log_step("Gemini Analysis", "completed")
        return result.stdout

    def run_multi_mind(self, task: str, output_dir: str) -> Dict[str, Any]:
        """Run multi-mind analysis"""
        self.log_step("Multi-Mind Analysis", "started")

        output_file = Path(output_dir) / "multi_mind_results.json"
        command = [
            "python",
            "scripts/commands/multi_mind.py",
            task,
            "--output",
            str(output_file),
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        if output_file.exists():
            with open(output_file, "r") as f:
                results = json.load(f)
            self.log_step("Multi-Mind Analysis", "completed")
            return results
        else:
            self.log_step("Multi-Mind Analysis", "failed")
            return {"error": result.stderr}

    def run_spec_driven(self, feature: str, output_dir: str) -> Dict[str, Any]:
        """Run spec-driven development"""
        self.log_step("Spec Generation", "started")

        spec_dir = Path(output_dir) / "specifications"
        command = [
            "python",
            "scripts/commands/spec_driven.py",
            feature,
            "--output",
            str(spec_dir),
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        if (spec_dir / "specification.md").exists():
            self.log_step("Spec Generation", "completed")
            return {
                "spec_path": str(spec_dir / "specification.md"),
                "test_path": str(spec_dir / "test_suite.py"),
            }
        else:
            self.log_step("Spec Generation", "failed")
            return {"error": result.stderr}

    def execute_workflow(
        self, workflow_name: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a predefined workflow"""
        if workflow_name not in self.config["workflows"]:
            return {"error": f"Unknown workflow: {workflow_name}"}

        workflow = self.config["workflows"][workflow_name]
        workflow_results = []

        print(f"\n🚀 Executing workflow: {workflow_name}")
        print("=" * 80)

        for i, step in enumerate(workflow):
            tool = step["tool"]
            args = step.get("args", {})

            # Add context from previous steps
            if i > 0 and workflow_results:
                args["previous_results"] = workflow_results[-1]

            # Execute tool
            if tool == "multi_mind":
                result = self.run_multi_mind(
                    context.get("task", "Analysis task"), context.get("output_dir", ".")
                )
            elif tool == "gemini":
                result = self.run_gemini(
                    context.get("codebase_path", "."),
                    context.get("query", "Analyze the architecture"),
                )
            elif tool == "claude":
                result = self.run_claude(
                    context.get("prompt", "Provide analysis"),
                    context.get("context", ""),
                )
            elif tool == "spec_driven":
                result = self.run_spec_driven(
                    context.get("feature", "New feature"),
                    context.get("output_dir", "."),
                )
            else:
                result = {"error": f"Unknown tool: {tool}"}

            workflow_results.append(
                {
                    "tool": tool,
                    "result": result,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Add delay between steps to avoid rate limiting
            time.sleep(2)

        return {
            "workflow": workflow_name,
            "results": workflow_results,
            "log": self.workflow_log,
        }

    def create_custom_workflow(self, steps: List[Dict[str, Any]]) -> str:
        """Create a custom workflow from steps"""
        workflow_name = f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.config["workflows"][workflow_name] = steps

        # Save updated config
        config_path = Path(".claude/orchestration_config.json")
        config_path.parent.mkdir(exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

        return workflow_name

    def generate_report(
        self, results: Dict[str, Any], output_file: Optional[str] = None
    ):
        """Generate comprehensive report from workflow results"""
        report = f"""# Research Orchestration Report

Generated: {datetime.now().isoformat()}

## Workflow: {results["workflow"]}

## Execution Log
"""
        for log_entry in results["log"]:
            report += f"- [{log_entry['timestamp']}] {log_entry['step']}: {log_entry['status']}\n"

        report += "\n## Results\n\n"

        for i, step_result in enumerate(results["results"]):
            report += f"### Step {i + 1}: {step_result['tool']}\n"
            report += f"**Timestamp:** {step_result['timestamp']}\n\n"

            if isinstance(step_result["result"], dict):
                report += "```json\n"
                report += json.dumps(step_result["result"], indent=2)
                report += "\n```\n\n"
            else:
                report += str(step_result["result"]) + "\n\n"

        if output_file:
            with open(output_file, "w") as f:
                f.write(report)
            print(f"\n💾 Report saved to: {output_file}")

        return report


def main():
    parser = argparse.ArgumentParser(description="Research Orchestration Tool")
    parser.add_argument("--workflow", type=str, help="Predefined workflow name")
    parser.add_argument("--task", type=str, help="Task description")
    parser.add_argument("--codebase", type=str, default=".", help="Path to codebase")
    parser.add_argument(
        "--feature", type=str, help="Feature for spec-driven development"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./research_output", help="Output directory"
    )
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--report", type=str, help="Report output file")

    # Custom workflow options
    parser.add_argument(
        "--custom-steps", type=str, help="JSON file with custom workflow steps"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize orchestrator
    orchestrator = ResearchOrchestrator(config_path=args.config)

    # Prepare context
    context = {
        "task": args.task or "Analyze codebase",
        "codebase_path": args.codebase,
        "feature": args.feature or "New feature",
        "output_dir": str(output_dir),
        "query": args.task or "Provide comprehensive analysis",
    }

    # Execute workflow
    if args.custom_steps:
        with open(args.custom_steps, "r") as f:
            custom_steps = json.load(f)
        workflow_name = orchestrator.create_custom_workflow(custom_steps)
        results = orchestrator.execute_workflow(workflow_name, context)
    elif args.workflow:
        results = orchestrator.execute_workflow(args.workflow, context)
    else:
        # Default comprehensive analysis
        results = orchestrator.execute_workflow("comprehensive_analysis", context)

    # Generate report
    report = orchestrator.generate_report(results, args.report)

    if not args.report:
        print("\n" + "=" * 80)
        print("ORCHESTRATION SUMMARY")
        print("=" * 80)
        print(f"Workflow completed with {len(results['results'])} steps")
        print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
