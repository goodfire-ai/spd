"""Neuron Scientist Agent using Claude Agent SDK.

An autonomous agent that investigates individual neurons through
hypothesis-driven experimentation.

Example usage:
    from neuron_scientist import NeuronScientist, investigate_neuron

    # Quick investigation
    investigation = await investigate_neuron(
        neuron_id="L15/N7890",
        initial_hypothesis="This neuron responds to medical terminology",
    )

    # Full control
    scientist = NeuronScientist(
        neuron_id="L15/N7890",
        initial_label="medical-terms",
        edge_stats_path=Path("data/medical_edge_stats_v6_enriched.json"),
    )
    investigation = await scientist.investigate(max_experiments=100)

    # NeuronPI orchestrator (full pipeline with GPT review + skeptic)
    from neuron_scientist import NeuronPI, run_neuron_pi
    result = await run_neuron_pi(
        neuron_id="L15/N7890",
        initial_hypothesis="Medical terminology detector",
        max_review_iterations=3,
    )

    # Run just the skeptic on an existing investigation
    from neuron_scientist import NeuronSkeptic, run_skeptic
    skeptic_report = await run_skeptic(
        neuron_id="L15/N7890",
        investigation=investigation,
    )

    # Generate HTML report (V2 - recommended, uses consolidated investigation.json)
    from neuron_scientist import DashboardAgentV2, generate_dashboard_v2
    html_path = await generate_dashboard_v2(
        Path("outputs/L15_N7890_investigation.json"),
    )

    # Generate HTML report (V1 - legacy, requires separate dashboard.json)
    from neuron_scientist import DashboardHTMLAgent, generate_dashboard
    html_path = await generate_dashboard(Path("outputs/L15_N7890_dashboard.json"))
"""

from .agent import NeuronScientist, investigate_neuron
from .dashboard_agent import DashboardHTMLAgent, generate_dashboard, generate_dashboard_sync
from .dashboard_agent_v2 import DashboardAgentV2, generate_dashboard_v2, generate_dashboard_v2_sync
from .pi_agent import NeuronPI, run_neuron_pi
from .schemas import DashboardData, NeuronInvestigation, PIResult, ReviewResult, SkepticReport
from .skeptic_agent import NeuronSkeptic, run_skeptic

__all__ = [
    # Core agent
    "NeuronScientist",
    "NeuronInvestigation",
    "DashboardData",
    "investigate_neuron",
    # NeuronSkeptic (adversarial testing)
    "NeuronSkeptic",
    "SkepticReport",
    "run_skeptic",
    # NeuronPI orchestrator
    "NeuronPI",
    "PIResult",
    "ReviewResult",
    "run_neuron_pi",
    # Dashboard V1 (legacy)
    "DashboardHTMLAgent",
    "generate_dashboard",
    "generate_dashboard_sync",
    # Dashboard V2 (recommended)
    "DashboardAgentV2",
    "generate_dashboard_v2",
    "generate_dashboard_v2_sync",
]
