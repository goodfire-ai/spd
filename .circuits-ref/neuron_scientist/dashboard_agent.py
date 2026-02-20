"""Dashboard HTML Generator Agent using Claude Agent SDK.

Transforms neuron_scientist dashboard JSON into beautiful Distill.pub-style HTML pages.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)

from .html_template import generate_html

# System prompt for the dashboard generator agent
SYSTEM_PROMPT = """You are a science communicator creating beautiful, accessible explanations of neural network internals.

Your task is to transform raw neuron investigation data into compelling, Distill.pub-style HTML pages.

## Your Process

1. First, call `get_dashboard_data` to see the full investigation data
2. Analyze the data and craft:
   - A creative 2-4 word **title** (e.g., "Monoamine Neurotransmitter Gate")
   - A **lead paragraph** (one sentence starting with "This neuron...")
   - A **body paragraph** (2-3 sentences elaborating on the neuron's behavior)
   - A **key finding** (the most surprising discovery, 2-3 sentences)
   - **Selectivity groups** as JSON for the circuit diagram
3. Call `write_html` with all your generated content

## Writing Guidelines

**Title**: 2-4 words, conceptual and memorable, not technical jargon

**Lead paragraph**:
- ONE compelling sentence
- Start with "This neuron..."
- Use <strong> tags for key concepts

**Body paragraph**:
- 2-3 sentences elaborating on the neuron's behavior
- Highlight interesting patterns: what it responds to, what it ignores, surprising exceptions
- Only mention selectivity if it's genuinely notable or surprising
- Use <em> for emphasis
- Vary your sentence structure - don't start every body with the same phrase

**Key finding**:
- The SINGLE most surprising or important discovery
- Often involves a refuted hypothesis
- Use <strong> for key terms

**Selectivity groups** (JSON format):
```json
{
  "fires": [
    {"label": "Fires on [category]", "examples": [{"text": "example with <mark>key</mark> word", "activation": 2.78}]}
  ],
  "ignores": [
    {"label": "Ignores [category]", "examples": [{"text": "example text", "activation": 0.08}]}
  ]
}
```

Focus on telling a STORY about what makes this neuron interesting.

## MANDATORY Quality Requirements

### No Placeholder Text (Issue 1)
**NEVER use placeholder or generic text in your output.**
- Forbidden patterns: "Test", "Test lead.", "Test body.", "Test finding.", "TODO", "TBD", "Example"
- If the investigation data is insufficient to generate meaningful content, explain what's missing rather than using placeholders
- Every field must contain substantive, specific content about THIS neuron
- Before calling write_html, verify none of your content matches placeholder patterns

### Complete Content - No Truncation (Issue 2)
**Never truncate content mid-sentence or mid-word.**
- Titles must be 2-4 COMPLETE words forming a coherent concept
- If you reach a length limit, restructure to end at a natural stopping point
- Bad: "Extremely sparse layer-0 neuron with weak selectiv" (cut off)
- Good: "Sparse Activation Gate" (complete)
- All sentences must end properly with punctuation, not mid-word

### Required Selectivity Groups (Issue 3)
**The selectivity_groups JSON is REQUIRED and must be populated if activation data exists.**
- If the investigation has positive_examples or negative_examples, you MUST transform them into selectivity groups
- Minimum: At least one "fires" category AND one "ignores" category
- Never leave selectivity_json as empty {} or with empty arrays if activation data exists
- Extract patterns from the data: what categories does it fire on? what does it ignore?

### Token Display - Strip Artifacts (Issue 6)
**Clean up tokenizer artifacts before displaying to users.**
- Replace 'Ġ' (BPE leading space marker) with readable format
- Display as: "inhibition (with leading space)" or just "inhibition"
- Never show raw "Ġinhibition" to users
- Apply this to ALL token displays: examples, key findings, selectivity groups

### Downstream Entry Validation (Issue 7)
**Validate downstream_neurons entries have proper neuron ID format.**
- Valid downstream entries have neuron_id like "L15/N7890"
- If entries lack neuron_id format (e.g., "dopamine →0.5"), these are OUTPUT EFFECTS not downstream neurons
- Place token/logit effects in a separate "Output Effects" context or omit from downstream section
- Only show actual neuron-to-neuron connections in Downstream Neurons section

### Explanatory Text for Empty Sections (Issue 10)
**If any section would be empty, display a brief explanation instead of leaving blank.**
- Empty Upstream for Layer 0: "This Layer 0 neuron connects directly to token embeddings."
- Empty Downstream for Layer 31: "This final-layer neuron projects directly to output logits."
- Empty Steering section: "Steering experiments not performed for this investigation."
- Empty Selectivity (rare): "Insufficient activation data to determine selectivity patterns."
- Never show section headers with no content beneath them
"""


class DashboardHTMLAgent:
    """Agent that generates beautiful HTML dashboards from investigation JSON."""

    def __init__(
        self,
        dashboard_path: Path,
        output_dir: Path = Path("frontend/reports"),
        model: str = "sonnet",
    ):
        """Initialize the dashboard generator.

        Args:
            dashboard_path: Path to dashboard JSON file
            output_dir: Directory to write output HTML
            model: Claude model to use ("opus", "sonnet", or "haiku")
        """
        self.dashboard_path = Path(dashboard_path)
        self.output_dir = Path(output_dir)
        self.model = model
        self.dashboard_data = None

    def _load_dashboard(self) -> dict[str, Any]:
        """Load dashboard JSON file."""
        with open(self.dashboard_path) as f:
            return json.load(f)

    def _create_mcp_tools(self):
        """Create MCP tools for HTML generation."""
        dashboard = self.dashboard_data
        output_dir = self.output_dir

        @tool(
            "get_dashboard_data",
            "Get the full dashboard data for the neuron investigation",
            {}
        )
        async def tool_get_dashboard_data(args: dict[str, Any]) -> dict[str, Any]:
            """Return the dashboard data for the agent to analyze."""
            # Return a curated view of the data
            summary_card = dashboard.get("summary_card", {})
            activation_patterns = dashboard.get("activation_patterns", {})
            findings = dashboard.get("findings", {})
            hypotheses = dashboard.get("hypothesis_timeline", {}).get("hypotheses", [])
            connectivity = dashboard.get("connectivity", {})

            positive = activation_patterns.get("positive_examples", [])[:10]
            negative = activation_patterns.get("negative_examples", [])[:10]

            data = {
                "neuron_id": dashboard.get("neuron_id", ""),
                "summary": summary_card.get("summary", ""),
                "input_function": summary_card.get("input_function", ""),
                "output_function": summary_card.get("output_function", ""),
                "confidence": summary_card.get("confidence", 0),
                "total_experiments": summary_card.get("total_experiments", 0),
                "positive_examples": [
                    {"prompt": ex.get("prompt", ""), "activation": ex.get("activation", 0)}
                    for ex in positive
                ],
                "negative_examples": [
                    {"prompt": ex.get("prompt", ""), "activation": ex.get("activation", 0)}
                    for ex in negative
                ],
                "key_findings": findings.get("key_findings", [])[:5],
                "open_questions": findings.get("open_questions", [])[:5],
                "hypotheses": [
                    {
                        "hypothesis": h.get("hypothesis", "")[:150],
                        "status": h.get("status", ""),
                        "prior": h.get("prior_probability", 50),
                        "posterior": h.get("posterior_probability", 50),
                    }
                    for h in hypotheses[:5]
                ],
                "upstream_neurons": [
                    {"id": n.get("neuron_id", ""), "label": n.get("label", "Unknown")}
                    for n in connectivity.get("upstream", [])[:5]
                ],
                "downstream_neurons": [
                    {"id": n.get("neuron_id", ""), "label": n.get("label", "Unknown")}
                    for n in connectivity.get("downstream", [])[:5]
                ],
            }

            return {
                "content": [{"type": "text", "text": json.dumps(data, indent=2)}]
            }

        @tool(
            "write_html",
            "Assemble all generated content into the final HTML dashboard and write to file",
            {
                "title": str,
                "narrative_lead": str,
                "narrative_body": str,
                "key_finding": str,
                "selectivity_json": str,
            }
        )
        async def tool_write_html(args: dict[str, Any]) -> dict[str, Any]:
            """Assemble and write the HTML dashboard."""
            neuron_id = dashboard.get("neuron_id", "L0/N0")
            summary_card = dashboard.get("summary_card", {})
            connectivity = dashboard.get("connectivity", {})
            activation_patterns = dashboard.get("activation_patterns", {})
            hypothesis_timeline = dashboard.get("hypothesis_timeline", {})
            findings = dashboard.get("findings", {})
            detailed_experiments = dashboard.get("detailed_experiments", {})
            relp_analysis = dashboard.get("relp_analysis", {})
            output_projections = dashboard.get("output_projections", {})

            # Parse selectivity JSON
            try:
                selectivity = json.loads(args.get("selectivity_json", "{}"))
                selectivity_fires = selectivity.get("fires", [])
                selectivity_ignores = selectivity.get("ignores", [])
            except json.JSONDecodeError:
                selectivity_fires = []
                selectivity_ignores = []

            # Build activation examples
            positive_examples = activation_patterns.get("positive_examples", [])
            negative_examples = activation_patterns.get("negative_examples", [])

            activation_examples = []
            for ex in positive_examples[:5]:
                activation_examples.append({
                    "prompt": ex.get("prompt", ""),
                    "activation": ex.get("activation", 0),
                    "is_positive": True,
                    "highlighted_text": ex.get("prompt", ""),
                })
            for ex in negative_examples[:3]:
                activation_examples.append({
                    "prompt": ex.get("prompt", ""),
                    "activation": ex.get("activation", 0),
                    "is_positive": False,
                    "highlighted_text": ex.get("prompt", ""),
                })

            # Extract trigger words for mock test
            trigger_words = []
            for proj in output_projections.get("promote", [])[:10]:
                token = proj.get("token", "")
                if len(token) > 2:
                    trigger_words.append(token)

            # Get steering results
            steering_results = detailed_experiments.get("steering", [])

            html = generate_html(
                neuron_id=neuron_id,
                title=args.get("title", "Neuron Investigation"),
                confidence=summary_card.get("confidence", 0.5),
                total_experiments=summary_card.get("total_experiments", 0),
                narrative_lead=args.get("narrative_lead", ""),
                narrative_body=args.get("narrative_body", ""),
                upstream_neurons=connectivity.get("upstream", []),
                downstream_neurons=connectivity.get("downstream", []),
                selectivity_fires=selectivity_fires,
                selectivity_ignores=selectivity_ignores,
                steering_results=steering_results,
                activation_examples=activation_examples,
                key_finding=args.get("key_finding", ""),
                hypotheses=hypothesis_timeline.get("hypotheses", []),
                open_questions=findings.get("open_questions", []),
                detailed_experiments=detailed_experiments,
                trigger_words=trigger_words,
                relp_analysis=relp_analysis,
            )

            # Write to file
            safe_id = neuron_id.replace("/", "_")
            output_path = output_dir / f"{safe_id}.html"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html)

            return {
                "content": [{"type": "text", "text": json.dumps({
                    "success": True,
                    "output_path": str(output_path),
                    "neuron_id": neuron_id,
                    "title": args.get("title", ""),
                })}]
            }

        return [
            tool_get_dashboard_data,
            tool_write_html,
        ]

    def _build_initial_prompt(self) -> str:
        """Build the initial prompt for the agent."""
        neuron_id = self.dashboard_data.get("neuron_id", "")

        return f"""Generate a beautiful HTML dashboard for neuron {neuron_id}.

First, call `get_dashboard_data` to see the investigation results.

Then craft compelling content and call `write_html` with:
- title: A creative 2-4 word title
- narrative_lead: One sentence starting with "This neuron..."
- narrative_body: 2-3 sentences elaborating on behavior (vary your opening!)
- key_finding: The most surprising discovery (2-3 sentences)
- selectivity_json: JSON with "fires" and "ignores" groups

Make it tell a story about what makes this neuron interesting! Don't use repetitive phrasing.
"""

    async def generate(self) -> Path:
        """Generate HTML dashboard using Claude Agent SDK.

        Returns:
            Path to generated HTML file
        """
        print(f"Loading dashboard from {self.dashboard_path}")
        self.dashboard_data = self._load_dashboard()
        neuron_id = self.dashboard_data.get("neuron_id", "unknown")

        print(f"Generating HTML dashboard for {neuron_id}")
        start_time = time.time()

        # Create MCP tools and server
        tools = self._create_mcp_tools()
        mcp_server = create_sdk_mcp_server(
            name="dashboard_tools",
            version="1.0.0",
            tools=tools,
        )

        # Build initial prompt
        initial_prompt = self._build_initial_prompt()

        # Configure options
        # Store transcripts in separate directory to avoid cluttering main project
        project_root = Path(__file__).parent.parent
        transcripts_dir = project_root / "neuron_reports" / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)

        options = ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            max_turns=10,
            model=self.model,
            mcp_servers={"dashboard_tools": mcp_server},
            cwd=transcripts_dir,
            add_dirs=[project_root],  # Allow access to main project files
            allowed_tools=[
                "mcp__dashboard_tools__get_dashboard_data",
                "mcp__dashboard_tools__write_html",
            ],
        )

        output_path = None

        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(initial_prompt)

                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                preview = block.text[:100].replace("\n", " ")
                                print(f"Agent: {preview}...")
                            elif isinstance(block, ToolUseBlock):
                                print(f"Tool: {block.name}")

                    elif isinstance(message, ToolResultBlock):
                        try:
                            result = json.loads(message.content)
                            if result.get("success") and result.get("output_path"):
                                output_path = Path(result["output_path"])
                                print(f"Generated: {output_path}")
                        except (json.JSONDecodeError, TypeError):
                            pass

                    elif isinstance(message, ResultMessage):
                        if message.subtype == "error":
                            print(f"Error: {message}")

        except Exception as e:
            print(f"Agent error: {e}")
            import traceback
            traceback.print_exc()

        duration = time.time() - start_time
        print(f"Dashboard generation complete in {duration:.1f}s")

        if output_path and output_path.exists():
            return output_path
        else:
            # Fall back to expected path
            safe_id = neuron_id.replace("/", "_")
            return self.output_dir / f"{safe_id}.html"


async def generate_dashboard(
    dashboard_path: Path,
    output_dir: Path = Path("frontend/reports"),
    model: str = "sonnet",
) -> Path:
    """Generate HTML dashboard from JSON.

    Args:
        dashboard_path: Path to dashboard JSON
        output_dir: Output directory
        model: Model to use

    Returns:
        Path to generated HTML
    """
    agent = DashboardHTMLAgent(
        dashboard_path=dashboard_path,
        output_dir=output_dir,
        model=model,
    )
    return await agent.generate()


def generate_dashboard_sync(
    dashboard_path: Path,
    output_dir: Path = Path("frontend/reports"),
    model: str = "sonnet",
) -> Path:
    """Synchronous wrapper for generate_dashboard."""
    return asyncio.run(generate_dashboard(dashboard_path, output_dir, model))
