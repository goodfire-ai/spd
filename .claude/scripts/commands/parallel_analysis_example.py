#!/usr/bin/env python3
"""
Multi-Mind Analysis Script
Orchestrates multiple specialized agents working in parallel for complex analysis tasks.
"""

import json
import subprocess
import argparse
from typing import List, Dict, Any
from pathlib import Path
import concurrent.futures
import tempfile
import os


class MultiMindAnalyzer:
    def __init__(self, claude_command: str = "claude"):
        self.claude_command = claude_command
        self.results = []

    def create_agent_prompt(self, role: str, task: str, context: str) -> str:
        """Create a specialized prompt for each agent"""
        return f"""<role>
You are a {role} with deep expertise in your domain. Focus exclusively on your area of specialization.
</role>

<task>
{task}
</task>

<context>
{context}
</context>

<instructions>
1. Analyze the task from your specialized perspective
2. Provide concrete, actionable insights
3. Highlight critical issues or opportunities
4. Be specific and technical in your domain
5. Format your response with clear sections
</instructions>

Provide your analysis:"""

    def run_agent(self, agent_config: Dict[str, str]) -> Dict[str, Any]:
        """Run a single agent analysis"""
        role = agent_config["role"]
        focus = agent_config["focus"]

        print(f"🧠 Starting {role} analysis...")

        # Create temporary file for the prompt
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            prompt = self.create_agent_prompt(
                role, focus, agent_config.get("context", "")
            )
            f.write(prompt)
            prompt_file = f.name

        try:
            # Run Claude with the prompt
            result = subprocess.run(
                [self.claude_command, "-m", prompt_file], capture_output=True, text=True
            )

            return {
                "role": role,
                "focus": focus,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None,
            }
        finally:
            # Clean up temporary file
            os.unlink(prompt_file)

    def synthesize_results(self, results: List[Dict[str, Any]]) -> str:
        """Synthesize findings from all agents"""
        synthesis_prompt = f"""<role>
You are a Research Synthesizer specializing in integrating diverse expert perspectives.
</role>

<task>
Synthesize the following expert analyses into a coherent, actionable report.
</task>

<expert_analyses>
{json.dumps(results, indent=2)}
</expert_analyses>

<instructions>
1. Identify consensus points across experts
2. Highlight important disagreements or tensions
3. Extract key insights and patterns
4. Provide integrated recommendations
5. Create an executive summary
</instructions>

Create the synthesis report:"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(synthesis_prompt)
            prompt_file = f.name

        try:
            result = subprocess.run(
                [self.claude_command, "-m", prompt_file], capture_output=True, text=True
            )
            return result.stdout
        finally:
            os.unlink(prompt_file)

    def analyze(self, task: str, agents: List[Dict[str, str]] = None) -> str:
        """Run multi-agent analysis"""
        if agents is None:
            # Default agent configuration
            agents = [
                {
                    "role": "Security Analyst",
                    "focus": f"Security implications of: {task}",
                },
                {
                    "role": "Performance Engineer",
                    "focus": f"Performance aspects of: {task}",
                },
                {
                    "role": "Software Architect",
                    "focus": f"Architectural design of: {task}",
                },
                {
                    "role": "Research Scientist",
                    "focus": f"Theoretical foundations of: {task}",
                },
                {"role": "Risk Assessor", "focus": f"Risk analysis of: {task}"},
                {
                    "role": "Implementation Strategist",
                    "focus": f"Implementation strategy for: {task}",
                },
            ]

        # Add context to each agent
        for agent in agents:
            agent["context"] = task

        # Run agents in parallel
        print(f"🚀 Launching {len(agents)} specialized agents in parallel...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(agents)) as executor:
            future_to_agent = {
                executor.submit(self.run_agent, agent): agent for agent in agents
            }

            for future in concurrent.futures.as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    print(f"✅ {result['role']} analysis complete")
                except Exception as e:
                    print(f"❌ {agent['role']} analysis failed: {e}")

        # Synthesize results
        print("\n🔄 Synthesizing expert analyses...")
        synthesis = self.synthesize_results(self.results)

        return synthesis

    def save_results(self, output_path: str):
        """Save detailed results to file"""
        output = {
            "individual_analyses": self.results,
            "synthesis": self.synthesize_results(self.results)
            if self.results
            else None,
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"💾 Detailed results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Mind Analysis Tool")
    parser.add_argument("task", help="The task or question to analyze")
    parser.add_argument(
        "--agents", type=str, help="JSON file with custom agent configurations"
    )
    parser.add_argument("--output", type=str, help="Output file for detailed results")
    parser.add_argument(
        "--claude-command", type=str, default="claude", help="Claude command to use"
    )

    args = parser.parse_args()

    analyzer = MultiMindAnalyzer(claude_command=args.claude_command)

    # Load custom agents if provided
    agents = None
    if args.agents:
        with open(args.agents, "r") as f:
            agents = json.load(f)

    # Run analysis
    synthesis = analyzer.analyze(args.task, agents)

    print("\n" + "=" * 80)
    print("SYNTHESIS REPORT")
    print("=" * 80)
    print(synthesis)

    # Save detailed results if requested
    if args.output:
        analyzer.save_results(args.output)


if __name__ == "__main__":
    main()
