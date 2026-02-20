"""Data schemas for neuron investigation results - dashboard ready.

Evidence Truncation Limits
--------------------------
The following limits control how much evidence is serialized to JSON.
These exist to:
1. Keep JSON files manageable (~100KB instead of multi-MB)
2. Fit within LLM context windows for downstream review
3. Focus on the most relevant evidence

Adjust these if you need more evidence preserved:
"""

from dataclasses import dataclass, field
from typing import Any, Optional

# =============================================================================
# Evidence Truncation Constants (for serialization)
# =============================================================================
# These limits apply when converting NeuronInvestigation to JSON via to_dict()

# Activation examples: Most important evidence, keep more
MAX_ACTIVATING_PROMPTS = 20  # Prompts where neuron fires strongly
MAX_NON_ACTIVATING_PROMPTS = 10  # Negative examples (fewer needed)
MAX_NEGATIVELY_ACTIVATING_PROMPTS = 20  # Prompts where neuron fires most negatively

# Experimental results
MAX_ABLATION_EFFECTS = 10  # Ablation experiment results

# Note: Full data is preserved in memory; truncation only affects JSON output


@dataclass
class ActivationExample:
    """A single activation example for the dashboard."""
    prompt: str
    activation: float
    position: int
    token: str
    is_positive: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "activation": self.activation,
            "position": self.position,
            "token": self.token,
            "is_positive": self.is_positive,
        }


@dataclass
class AblationEffect:
    """Ablation effect for dashboard visualization."""
    token: str
    shift: float
    direction: str  # "promotes" or "suppresses"
    consistency: str  # "high", "medium", "low"

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "shift": self.shift,
            "direction": self.direction,
            "consistency": self.consistency,
        }


@dataclass
class HypothesisRecord:
    """Record of a hypothesis and its evidence for dashboard timeline."""
    hypothesis: str
    formed_at_experiment: int
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    confidence: float = 0.5
    status: str = "testing"  # "testing", "supported", "refuted", "refined"

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "formed_at_experiment": self.formed_at_experiment,
            "evidence_for": self.evidence_for,
            "evidence_against": self.evidence_against,
            "confidence": self.confidence,
            "status": self.status,
        }


@dataclass
class ConnectivityNode:
    """Node in connectivity graph for dashboard."""
    neuron_id: str
    label: str
    weight: float
    direction: str  # "upstream" or "downstream"
    is_logit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "neuron_id": self.neuron_id,
            "label": self.label,
            "weight": self.weight,
            "direction": self.direction,
            "is_logit": self.is_logit,
        }


@dataclass
class DashboardData:
    """Complete dashboard-ready data for a neuron investigation."""

    # Identity
    neuron_id: str
    layer: int
    neuron_idx: int

    # Summary card
    summary: str = ""
    input_function: str = ""
    output_function: str = ""
    function_type: str = ""  # semantic, routing, formatting, etc.
    confidence: float = 0.0
    total_experiments: int = 0

    # Initial info
    initial_label: str = ""
    transluce_positive: str = ""
    transluce_negative: str = ""

    # Statistics for summary card
    stats: dict[str, Any] = field(default_factory=dict)

    # Activation examples (for heatmap/list)
    positive_examples: list[ActivationExample] = field(default_factory=list)
    negative_examples: list[ActivationExample] = field(default_factory=list)
    minimal_triggers: list[str] = field(default_factory=list)

    # Ablation effects (for bar chart)
    ablation_effects: list[AblationEffect] = field(default_factory=list)
    consistent_promotes: list[str] = field(default_factory=list)
    consistent_suppresses: list[str] = field(default_factory=list)

    # Hypothesis timeline
    hypotheses: list[HypothesisRecord] = field(default_factory=list)
    final_hypothesis: str = ""

    # Connectivity graph
    upstream_nodes: list[ConnectivityNode] = field(default_factory=list)
    downstream_nodes: list[ConnectivityNode] = field(default_factory=list)

    # Key findings (bullet points for dashboard)
    key_findings: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)

    # Full agent reasoning (collapsible section)
    agent_reasoning: str = ""

    # Metadata
    timestamp: str = ""
    investigation_duration_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON for dashboard rendering."""
        return {
            "neuron_id": self.neuron_id,
            "layer": self.layer,
            "neuron_idx": self.neuron_idx,
            "summary_card": {
                "summary": self.summary,
                "input_function": self.input_function,
                "output_function": self.output_function,
                "function_type": self.function_type,
                "confidence": self.confidence,
                "total_experiments": self.total_experiments,
                "initial_label": self.initial_label,
                "transluce_positive": self.transluce_positive,
                "transluce_negative": self.transluce_negative,
            },
            "stats": self.stats,
            "activation_patterns": {
                "positive_examples": [e.to_dict() for e in self.positive_examples],
                "negative_examples": [e.to_dict() for e in self.negative_examples],
                "minimal_triggers": self.minimal_triggers,
            },
            "ablation_effects": {
                "effects": [e.to_dict() for e in self.ablation_effects],
                "consistent_promotes": self.consistent_promotes,
                "consistent_suppresses": self.consistent_suppresses,
            },
            "hypothesis_timeline": {
                "hypotheses": [h.to_dict() for h in self.hypotheses],
                "final_hypothesis": self.final_hypothesis,
            },
            "connectivity": {
                "upstream": [n.to_dict() for n in self.upstream_nodes],
                "downstream": [n.to_dict() for n in self.downstream_nodes],
            },
            "findings": {
                "key_findings": self.key_findings,
                "open_questions": self.open_questions,
            },
            "agent_reasoning": self.agent_reasoning,
            "metadata": {
                "timestamp": self.timestamp,
                "investigation_duration_sec": self.investigation_duration_sec,
            },
        }


# Keep backward compatibility
@dataclass
class NeuronInvestigation:
    """Complete investigation results (backward compatible)."""
    neuron_id: str
    layer: int
    neuron_idx: int

    # Investigation metadata
    timestamp: str = ""
    total_experiments: int = 0
    confidence: float = 0.0
    pre_skeptic_confidence: float | None = None  # Confidence before skeptic adjustment
    skeptic_confidence_adjustment: float = 0.0  # Delta applied by skeptic

    # Initial info
    initial_label: str = ""
    initial_hypothesis: str = ""

    # Agent findings
    final_hypothesis: str = ""
    input_function: str = ""
    output_function: str = ""
    function_type: str = ""

    # Polarity mode: "positive" or "negative" (which firing direction this investigation covers)
    polarity_mode: str = "positive"

    # Evidence
    activating_prompts: list[dict[str, Any]] = field(default_factory=list)
    non_activating_prompts: list[dict[str, Any]] = field(default_factory=list)
    negatively_activating_prompts: list[dict[str, Any]] = field(default_factory=list)  # Most negatively firing prompts
    ablation_effects: list[dict[str, Any]] = field(default_factory=list)
    connectivity: dict[str, Any] = field(default_factory=dict)
    wiring_analysis: dict[str, Any] = field(default_factory=dict)  # Weight-based upstream wiring with polarity
    output_wiring_analysis: dict[str, Any] = field(default_factory=dict)  # Weight-based downstream wiring with polarity

    # SwiGLU operating regime (detected during category selectivity)
    operating_regime: str | None = None        # "standard", "inverted", "mixed"
    regime_confidence: float = 0.0                # fraction of activations in dominant regime
    regime_data: dict[str, Any] | None = None  # full gate/up decomposition stats
    firing_sign_stats: dict[str, Any] | None = None  # {positive_pct, negative_pct, mean_gate_pre, mean_up_pre}
    relp_results: list[dict[str, Any]] = field(default_factory=list)  # RelP attribution findings
    steering_results: list[dict[str, Any]] = field(default_factory=list)  # Steering/dose-response results
    dose_response_results: list[dict[str, Any]] = field(default_factory=list)  # Dose-response curves

    # Output projections (what tokens this neuron promotes/suppresses)
    output_projections: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: {"promote": [], "suppress": []})

    # Prior knowledge used as seed (for Investigation Flow visualization)
    prior_claims: dict[str, Any] = field(default_factory=dict)  # {input_label, input_description, output_label, output_description}

    # Skeptic adversarial testing results (embedded for dashboard access)
    skeptic_report: dict[str, Any] | None = None  # SkepticReport.to_dict() when available

    # Revision history - tracks gaps identified and addressed across review iterations
    # Format: [{"iteration": 1, "gaps_identified": [...], "gaps_addressed": [...], "gaps_remaining": [...]}, ...]
    revision_history: list[dict[str, Any]] = field(default_factory=list)

    # Transcript summaries - narrative summaries from each agent in the pipeline
    # Format: [
    #   {"agent": "scientist", "iteration": 1, "summary": "...", "key_discoveries": [...], "surprises": [...], "decisions": [...]},
    #   {"agent": "skeptic", "iteration": 1, "summary": "...", "challenges": [...], "verdict": "...", "key_tests": [...]},
    #   {"agent": "gpt_reviewer", "iteration": 1, "summary": "...", "verdict": "...", "gaps_identified": [...], "recommendations": [...]},
    # ]
    # These provide the narrative "story" of the investigation for the dashboard agent
    transcript_summaries: list[dict[str, Any]] = field(default_factory=list)

    # Protocol validation - tracks which validation steps were completed
    # Added by save_structured_report, preserved across re-saves
    protocol_validation: dict[str, Any] = field(default_factory=dict)

    # =============================================================================
    # V4 Phase Tracking Fields
    # =============================================================================

    # Phase completion status
    input_phase_complete: bool = False
    output_phase_complete: bool = False

    # Input phase characterization (populated by complete_input_phase)
    input_characterization: dict[str, Any] = field(default_factory=lambda: {
        "summary": "",
        "triggers": [],
        "upstream_dependencies": [],
        "selectivity_data": {},
        "confidence": 0.0,
    })

    # Output phase characterization (populated by complete_output_phase)
    output_characterization: dict[str, Any] = field(default_factory=lambda: {
        "summary": "",
        "promotes": [],
        "suppresses": [],
        "multi_token_ablation": [],
        "multi_token_steering": [],
        "downstream_dependencies": [],
        "relp_evidence": [],
        "confidence": 0.0,
    })

    # V4 experimental results
    upstream_dependency_results: list[dict[str, Any]] = field(default_factory=list)
    upstream_steering_results: list[dict[str, Any]] = field(default_factory=list)  # V5: batch_steer_upstream_and_test
    downstream_dependency_results: list[dict[str, Any]] = field(default_factory=list)
    multi_token_ablation_results: list[dict[str, Any]] = field(default_factory=list)
    multi_token_steering_results: list[dict[str, Any]] = field(default_factory=list)

    # V5 Anomaly investigation
    anomaly_investigation: dict[str, Any] | None = None

    # Hypotheses tested
    hypotheses_tested: list[dict[str, Any]] = field(default_factory=list)

    # Visualization data (for dashboard figures)
    categorized_prompts: dict[str, list[dict[str, Any]]] = field(default_factory=dict)  # {category: [{prompt, activation, position, token}, ...]}
    homograph_tests: list[dict[str, Any]] = field(default_factory=list)  # [{word, contexts: [{label, example, activation, category}, ...]}]
    patching_experiments: list[dict[str, Any]] = field(default_factory=list)  # [{source_prompt, target_prompt, source_activation, target_activation, patched_logits, baseline_logits, effect_summary}, ...]

    # Category selectivity test results (comprehensive stats for visualization)
    # Single accumulated dict — each run merges into this via merge_selectivity_runs().
    # Structure:
    # {
    #   "global_mean": float, "global_std": float, "total_prompts": int,
    #   "categories": {category_name: {"type": str, "prompts": [{prompt, activation, z_score}, ...], "mean": float, "std": float, "z_mean": float}},
    #   "top_activating": [{prompt, activation, z_score, category}, ...],
    #   "selectivity_summary": str,
    # }
    category_selectivity_data: dict[str, Any] = field(default_factory=dict)

    # Summary
    key_findings: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    agent_reasoning: str = ""

    # Dashboard data
    dashboard: DashboardData | None = None

    def to_dict(self) -> dict[str, Any]:
        # --- Helper: strip bulky per-prompt raw data from dependency results ---
        def _slim_upstream(entries):
            """Strip per_prompt_breakdown (200K+ chars) — consumers use individual_ablation."""
            return [
                {k: v for k, v in entry.items() if k != "per_prompt_breakdown"}
                for entry in entries
            ]

        def _slim_downstream(entries):
            """Strip per_prompt_results when dependency_summary exists (500K+ chars).
            Preserves per_prompt_results as fallback only when summary is absent
            (see figure_tools.py generate_downstream_dependency_table).
            """
            return [
                {k: v for k, v in entry.items()
                 if k != "per_prompt_results" or not entry.get("dependency_summary")}
                for entry in entries
            ]

        def _cap_categorized_prompts(cat_prompts, max_per_category=5):
            """Keep top N prompts per category by absolute activation."""
            return {
                cat: sorted(prompts, key=lambda p: abs(p.get("activation", 0)), reverse=True)[:max_per_category]
                for cat, prompts in cat_prompts.items()
            }

        result = {
            "neuron_id": self.neuron_id,
            "layer": self.layer,
            "neuron_idx": self.neuron_idx,
            "polarity_mode": self.polarity_mode,
            "timestamp": self.timestamp,
            "total_experiments": self.total_experiments,
            "confidence": self.confidence,
            "pre_skeptic_confidence": self.pre_skeptic_confidence,
            "skeptic_confidence_adjustment": self.skeptic_confidence_adjustment,
            "characterization": {
                "final_hypothesis": self.final_hypothesis,
                "input_function": self.input_function,
                "output_function": self.output_function,
                "function_type": self.function_type,
            },
            "evidence": {
                "activating_prompts": self.activating_prompts[:MAX_ACTIVATING_PROMPTS],
                "non_activating_prompts": self.non_activating_prompts[:MAX_NON_ACTIVATING_PROMPTS],
                "negatively_activating_prompts": self.negatively_activating_prompts[:MAX_NEGATIVELY_ACTIVATING_PROMPTS],
                "ablation_effects": self.ablation_effects[:MAX_ABLATION_EFFECTS],
                "connectivity": self.connectivity,
                # relp/steering/dose live here only (no top-level duplication)
                "relp_results": sorted(
                    self.relp_results,
                    key=lambda x: (not x.get("neuron_found", False), x.get("source") != "corpus")
                ),
                "steering_results": self.steering_results,
                "dose_response_results": self.dose_response_results,
            },
            "output_projections": self.output_projections,
            "skeptic_report": self.skeptic_report,
            "transcript_summaries": self.transcript_summaries,
            "hypotheses_tested": self.hypotheses_tested,
            "categorized_prompts": _cap_categorized_prompts(self.categorized_prompts),
            "category_selectivity_data": self.category_selectivity_data,
            "key_findings": self.key_findings,
            "open_questions": self.open_questions,
            "protocol_validation": self.protocol_validation,
            "input_characterization": self.input_characterization,
            "output_characterization": self.output_characterization,
            # Dependency results — stripped of bulky per-prompt raw data
            "upstream_dependency_results": _slim_upstream(self.upstream_dependency_results),
            "upstream_steering_results": self.upstream_steering_results,
            "downstream_dependency_results": _slim_downstream(self.downstream_dependency_results),
            "multi_token_ablation_results": self.multi_token_ablation_results,
            "multi_token_steering_results": self.multi_token_steering_results,
            "anomaly_investigation": self.anomaly_investigation,
            "wiring_analysis": self.wiring_analysis,
            "output_wiring_analysis": self.output_wiring_analysis,
            # SwiGLU operating regime
            "operating_regime": self.operating_regime,
            "regime_confidence": self.regime_confidence,
            "regime_data": self.regime_data,
            "firing_sign_stats": self.firing_sign_stats,
        }

        # --- Conditionally include fields that are often empty ---
        if self.prior_claims:
            result["prior_claims"] = self.prior_claims
        if self.revision_history:
            result["revision_history"] = self.revision_history
        if self.homograph_tests:
            result["homograph_tests"] = self.homograph_tests
        if self.patching_experiments:
            result["patching_experiments"] = self.patching_experiments
        if self.agent_reasoning:
            result["agent_reasoning"] = self.agent_reasoning
        if self.initial_label:
            result["initial_label"] = self.initial_label
        if self.initial_hypothesis:
            result["initial_hypothesis"] = self.initial_hypothesis

        # Include dashboard data if available
        if self.dashboard:
            result["dashboard"] = self.dashboard.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NeuronInvestigation":
        """Reconstruct NeuronInvestigation from saved JSON.

        Args:
            data: Dict from to_dict() or loaded from JSON file

        Returns:
            Populated NeuronInvestigation instance
        """
        # Handle nested characterization format
        char = data.get("characterization", {})

        investigation = cls(
            neuron_id=data.get("neuron_id", ""),
            layer=data.get("layer", 0),
            neuron_idx=data.get("neuron_idx", 0),
        )

        # Polarity mode
        investigation.polarity_mode = data.get("polarity_mode", "positive")

        # Metadata
        investigation.timestamp = data.get("timestamp", "")
        investigation.total_experiments = data.get("total_experiments", 0)
        investigation.confidence = data.get("confidence", 0.0)
        investigation.pre_skeptic_confidence = data.get("pre_skeptic_confidence")
        investigation.skeptic_confidence_adjustment = data.get("skeptic_confidence_adjustment", 0.0)

        # Initial info
        investigation.initial_label = data.get("initial_label", "")
        investigation.initial_hypothesis = data.get("initial_hypothesis", "")

        # Agent findings from characterization
        investigation.final_hypothesis = char.get("final_hypothesis", "")
        investigation.input_function = char.get("input_function", "")
        investigation.output_function = char.get("output_function", "")
        investigation.function_type = char.get("function_type", "")

        # Evidence - handle both nested and flat formats
        evidence = data.get("evidence", {})
        investigation.activating_prompts = evidence.get("activating_prompts", data.get("activating_prompts", []))
        investigation.non_activating_prompts = evidence.get("non_activating_prompts", data.get("non_activating_prompts", []))
        investigation.negatively_activating_prompts = evidence.get("negatively_activating_prompts", data.get("negatively_activating_prompts", []))
        investigation.ablation_effects = evidence.get("ablation_effects", data.get("ablation_effects", []))
        investigation.connectivity = evidence.get("connectivity", data.get("connectivity", {}))
        investigation.relp_results = data.get("relp_results", evidence.get("relp_results", []))
        investigation.steering_results = data.get("steering_results", evidence.get("steering_results", []))
        investigation.dose_response_results = data.get("dose_response_results", evidence.get("dose_response_results", []))

        # Hypotheses
        investigation.hypotheses_tested = data.get("hypotheses_tested", [])

        # Visualization data
        investigation.categorized_prompts = data.get("categorized_prompts", {})
        investigation.homograph_tests = data.get("homograph_tests", [])
        investigation.patching_experiments = data.get("patching_experiments", [])
        # Category selectivity: stored as single accumulated dict.
        # Backward compat: old format was list of run dicts — merge them.
        raw_sel = data.get("category_selectivity_data", {})
        if isinstance(raw_sel, list):
            # Legacy list-of-runs format: merge into single dict
            from .tools import merge_selectivity_runs
            investigation.category_selectivity_data = merge_selectivity_runs(raw_sel) if raw_sel else {}
        elif isinstance(raw_sel, dict):
            investigation.category_selectivity_data = raw_sel
        else:
            investigation.category_selectivity_data = {}

        # Summary
        investigation.key_findings = data.get("key_findings", [])
        investigation.open_questions = data.get("open_questions", [])
        investigation.agent_reasoning = data.get("agent_reasoning", "")

        # Output projections and prior claims
        investigation.output_projections = data.get("output_projections", {"promote": [], "suppress": []})
        investigation.prior_claims = data.get("prior_claims", {})

        # Skeptic report (adversarial testing results)
        investigation.skeptic_report = data.get("skeptic_report")

        # Revision history (gaps tracked across review iterations)
        investigation.revision_history = data.get("revision_history", [])

        # Transcript summaries (narrative summaries from each agent)
        investigation.transcript_summaries = data.get("transcript_summaries", [])

        # Protocol validation (validation steps completed)
        investigation.protocol_validation = data.get("protocol_validation", {})

        # V4 phase tracking
        investigation.input_phase_complete = data.get("input_phase_complete", False)
        investigation.output_phase_complete = data.get("output_phase_complete", False)
        investigation.input_characterization = data.get("input_characterization", {
            "summary": "",
            "triggers": [],
            "upstream_dependencies": [],
            "selectivity_data": {},
            "confidence": 0.0,
        })
        investigation.output_characterization = data.get("output_characterization", {
            "summary": "",
            "promotes": [],
            "suppresses": [],
            "multi_token_ablation": [],
            "multi_token_steering": [],
            "downstream_dependencies": [],
            "relp_evidence": [],
            "confidence": 0.0,
        })
        investigation.upstream_dependency_results = data.get("upstream_dependency_results", [])
        investigation.upstream_steering_results = data.get("upstream_steering_results", [])
        investigation.downstream_dependency_results = data.get("downstream_dependency_results", [])
        investigation.multi_token_ablation_results = data.get("multi_token_ablation_results", [])
        investigation.multi_token_steering_results = data.get("multi_token_steering_results", [])
        investigation.anomaly_investigation = data.get("anomaly_investigation")

        # Wiring analysis (weight-based connectivity with SwiGLU polarity)
        investigation.wiring_analysis = data.get("wiring_analysis", {})
        investigation.output_wiring_analysis = data.get("output_wiring_analysis", {})

        # SwiGLU operating regime (backward compatible - defaults for older investigations)
        investigation.operating_regime = data.get("operating_regime")
        investigation.regime_confidence = data.get("regime_confidence", 0.0)
        investigation.regime_data = data.get("regime_data")
        investigation.firing_sign_stats = data.get("firing_sign_stats")

        return investigation

    def to_dashboard(self) -> DashboardData:
        """Convert to dashboard-ready format."""
        dashboard = DashboardData(
            neuron_id=self.neuron_id,
            layer=self.layer,
            neuron_idx=self.neuron_idx,
            summary=f"Investigation of {self.neuron_id}",
            input_function=self.input_function,
            output_function=self.output_function,
            function_type=self.function_type,
            confidence=self.confidence,
            total_experiments=self.total_experiments,
            initial_label=self.initial_label,
            timestamp=self.timestamp,
        )

        # Convert activation examples (entries may be dicts or plain strings)
        for ex in self.activating_prompts:
            if isinstance(ex, str):
                ex = {"prompt": ex, "activation": 0, "position": -1, "token": ""}
            dashboard.positive_examples.append(ActivationExample(
                prompt=ex.get("prompt", "")[:200],
                activation=ex.get("activation", 0),
                position=ex.get("position", -1),
                token=ex.get("token", ""),
                is_positive=True,
            ))

        for ex in self.non_activating_prompts:
            if isinstance(ex, str):
                ex = {"prompt": ex, "activation": 0, "position": -1, "token": ""}
            dashboard.negative_examples.append(ActivationExample(
                prompt=ex.get("prompt", "")[:200],
                activation=ex.get("activation", 0),
                position=ex.get("position", -1),
                token=ex.get("token", ""),
                is_positive=False,
            ))

        # Convert ablation effects
        for effect in self.ablation_effects:
            if "promotes" in effect:
                for token, shift in effect.get("promotes", []):
                    dashboard.ablation_effects.append(AblationEffect(
                        token=token,
                        shift=shift,
                        direction="promotes",
                        consistency="high",
                    ))
                    dashboard.consistent_promotes.append(token)
            if "suppresses" in effect:
                for token, shift in effect.get("suppresses", []):
                    dashboard.ablation_effects.append(AblationEffect(
                        token=token,
                        shift=shift,
                        direction="suppresses",
                        consistency="high",
                    ))
                    dashboard.consistent_suppresses.append(token)

        # Convert hypotheses
        for i, h in enumerate(self.hypotheses_tested):
            dashboard.hypotheses.append(HypothesisRecord(
                hypothesis=h.get("hypothesis", ""),
                formed_at_experiment=i * 10,
                confidence=h.get("confidence", 0.5),
                status=h.get("status", "tested"),
            ))

        dashboard.final_hypothesis = self.final_hypothesis
        dashboard.key_findings = self.key_findings
        dashboard.open_questions = self.open_questions
        dashboard.agent_reasoning = self.agent_reasoning

        # Stats
        dashboard.stats = {
            "activating_count": len(self.activating_prompts),
            "non_activating_count": len(self.non_activating_prompts),
            "ablation_count": len(self.ablation_effects),
            "hypotheses_count": len(self.hypotheses_tested),
        }

        return dashboard


# =============================================================================
# NeuronPI Review and Orchestration Schemas
# =============================================================================

@dataclass
class ReviewResult:
    """Result from GPT review of an investigation."""
    verdict: str  # "APPROVE" or "REQUEST_CHANGES"
    hypothesis_assessment: str = ""  # V4: per-hypothesis assessment summary
    gaps: list[str] = field(default_factory=list)
    feedback: str = ""
    raw_response: str = ""
    iteration: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "hypothesis_assessment": self.hypothesis_assessment,
            "gaps": self.gaps,
            "feedback": self.feedback,
            "raw_response": self.raw_response,
            "iteration": self.iteration,
        }


@dataclass
class PIResult:
    """Result from NeuronPI orchestration pipeline."""
    neuron_id: str
    investigation: NeuronInvestigation | None = None
    skeptic_report: Optional["SkepticReport"] = None  # NEW: adversarial findings
    review_history: list[ReviewResult] = field(default_factory=list)
    iterations: int = 0
    final_verdict: str = ""  # "APPROVE", "MAX_ITERATIONS", "ERROR"
    dashboard_path: str | None = None
    investigation_path: str | None = None
    dashboard_json_path: str | None = None
    error: str | None = None
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "neuron_id": self.neuron_id,
            "iterations": self.iterations,
            "final_verdict": self.final_verdict,
            "skeptic_report": self.skeptic_report.to_dict() if self.skeptic_report else None,
            "review_history": [r.to_dict() for r in self.review_history],
            "dashboard_path": self.dashboard_path,
            "investigation_path": self.investigation_path,
            "dashboard_json_path": self.dashboard_json_path,
            "error": self.error,
            "timestamp": self.timestamp,
        }


# =============================================================================
# NeuronSkeptic Schemas
# =============================================================================

@dataclass
class AlternativeHypothesis:
    """An alternative hypothesis tested by the skeptic."""
    original_hypothesis: str
    alternative: str
    test_description: str
    results: list[dict[str, Any]] = field(default_factory=list)
    verdict: str = ""  # "distinguished", "indistinguishable", "inconclusive"
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_hypothesis": self.original_hypothesis,
            "alternative": self.alternative,
            "test_description": self.test_description,
            "results": self.results,
            "verdict": self.verdict,
            "evidence": self.evidence,
        }


@dataclass
class BoundaryTest:
    """A boundary/edge case test."""
    description: str
    prompt: str
    expected_behavior: str  # "should_activate" or "should_not_activate"
    actual_activation: float
    passed: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "prompt": self.prompt,
            "expected_behavior": self.expected_behavior,
            "actual_activation": self.actual_activation,
            "passed": self.passed,
            "notes": self.notes,
        }


@dataclass
class Confound:
    """A potential confounding factor discovered."""
    factor: str  # e.g., "position", "length", "co-occurrence"
    description: str
    evidence: str
    severity: str  # "critical", "moderate", "minor"
    tested_prompts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor": self.factor,
            "description": self.description,
            "evidence": self.evidence,
            "severity": self.severity,
            "tested_prompts": self.tested_prompts,
        }


@dataclass
class SkepticReport:
    """Full report from NeuronSkeptic adversarial testing."""
    neuron_id: str
    original_hypothesis: str

    # Adversarial findings
    alternative_hypotheses: list[AlternativeHypothesis] = field(default_factory=list)
    boundary_tests: list[BoundaryTest] = field(default_factory=list)
    confounds: list[Confound] = field(default_factory=list)
    hypothesis_challenges: list[dict[str, Any]] = field(default_factory=list)  # Challenges to individual hypotheses

    # Metrics
    selectivity_score: float = 0.0  # 0-1, how specific is the neuron
    false_positive_rate: float = 0.0  # Rate of unexpected activations
    false_negative_rate: float = 0.0  # Rate of missed activations

    # Verdict
    verdict: str = ""  # "SUPPORTED", "WEAKENED", "REFUTED"
    confidence_adjustment: float = 0.0  # DEPRECATED: Use hypothesis_challenges instead
    revised_hypothesis: str | None = None

    # Summary
    key_challenges: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    agent_reasoning: str = ""

    # Metadata
    total_tests: int = 0
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "neuron_id": self.neuron_id,
            "original_hypothesis": self.original_hypothesis,
            "alternative_hypotheses": [h.to_dict() for h in self.alternative_hypotheses],
            "boundary_tests": [t.to_dict() for t in self.boundary_tests],
            "confounds": [c.to_dict() for c in self.confounds],
            "hypothesis_challenges": self.hypothesis_challenges,  # Individual hypothesis updates
            "metrics": {
                "selectivity_score": self.selectivity_score,
                "false_positive_rate": self.false_positive_rate,
                "false_negative_rate": self.false_negative_rate,
            },
            "verdict": self.verdict,
            "confidence_adjustment": self.confidence_adjustment,  # DEPRECATED
            "revised_hypothesis": self.revised_hypothesis,
            "key_challenges": self.key_challenges,
            "recommendations": self.recommendations,
            "agent_reasoning": self.agent_reasoning,
            "total_tests": self.total_tests,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkepticReport":
        """Reconstruct SkepticReport from saved JSON."""
        report = cls(
            neuron_id=data.get("neuron_id", ""),
            original_hypothesis=data.get("original_hypothesis", ""),
        )

        # Alternative hypotheses
        for h in data.get("alternative_hypotheses", []):
            report.alternative_hypotheses.append(AlternativeHypothesis(
                original_hypothesis=h.get("original_hypothesis", ""),
                alternative=h.get("alternative", ""),
                test_description=h.get("test_description", ""),
                results=h.get("results", []),
                verdict=h.get("verdict", ""),
                evidence=h.get("evidence", ""),
            ))

        # Boundary tests
        for t in data.get("boundary_tests", []):
            report.boundary_tests.append(BoundaryTest(
                description=t.get("description", ""),
                prompt=t.get("prompt", ""),
                expected_behavior=t.get("expected_behavior", ""),
                actual_activation=t.get("actual_activation", 0.0),
                passed=t.get("passed", False),
                notes=t.get("notes", ""),
            ))

        # Confounds
        for c in data.get("confounds", []):
            report.confounds.append(Confound(
                factor=c.get("factor", ""),
                description=c.get("description", ""),
                evidence=c.get("evidence", ""),
                severity=c.get("severity", ""),
                tested_prompts=c.get("tested_prompts", []),
            ))

        # Metrics
        metrics = data.get("metrics", {})
        report.selectivity_score = metrics.get("selectivity_score", 0.0)
        report.false_positive_rate = metrics.get("false_positive_rate", 0.0)
        report.false_negative_rate = metrics.get("false_negative_rate", 0.0)

        # Verdict
        report.verdict = data.get("verdict", "")
        report.confidence_adjustment = data.get("confidence_adjustment", 0.0)
        report.revised_hypothesis = data.get("revised_hypothesis")

        # Summary
        report.key_challenges = data.get("key_challenges", [])
        report.recommendations = data.get("recommendations", [])
        report.agent_reasoning = data.get("agent_reasoning", "")

        # Metadata
        report.total_tests = data.get("total_tests", 0)
        report.timestamp = data.get("timestamp", "")

        return report
