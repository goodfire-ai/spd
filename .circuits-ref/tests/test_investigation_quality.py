"""Investigation quality evaluation tests.

These tests verify:
- Investigation quality metrics are computed correctly
- Dashboard data is complete and well-formed
- Evidence quality scoring works correctly
- Hard gate enforcement on confidence claims
"""


from neuron_scientist.review_prompts import (
    apply_hard_gates,
    distill_investigation_for_review,
    parse_review_response,
)
from neuron_scientist.schemas import (
    AblationEffect,
    ActivationExample,
    ConnectivityNode,
    DashboardData,
    HypothesisRecord,
    NeuronInvestigation,
    PIResult,
    ReviewResult,
)
from neuron_scientist.tools import (
    ProtocolState,
)


class TestInvestigationCompleteness:
    """Test that investigations contain all required fields."""

    def test_minimal_investigation_valid(self):
        """Minimal investigation is still valid."""
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
        )

        d = inv.to_dict()

        # Core fields should exist
        assert d["neuron_id"] == "L4/N10555"
        assert d["layer"] == 4
        assert d["neuron_idx"] == 10555
        assert "characterization" in d
        assert "evidence" in d

    def test_full_investigation_all_fields(self):
        """Full investigation contains all expected fields."""
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            timestamp="2026-01-19T00:00:00",
            total_experiments=45,
            confidence=0.85,
            initial_label="dopamine neuron",
            initial_hypothesis="Activates on dopamine mentions",
            final_hypothesis="Encodes reward-related neurotransmitter concepts",
            input_function="Activates on dopamine and reward mentions",
            output_function="Promotes pleasure/reward tokens",
            function_type="semantic",
            activating_prompts=[
                {"prompt": "Dopamine is released", "activation": 2.5},
                {"prompt": "Reward pathway activation", "activation": 2.1},
            ],
            non_activating_prompts=[
                {"prompt": "The weather is nice", "activation": 0.1},
            ],
            ablation_effects=[
                {"promotes": [["pleasure", 0.5], ["reward", 0.3]]},
            ],
            connectivity={
                "upstream_neurons": [{"neuron_id": "L3/N1000", "label": "chemical"}],
                "downstream_targets": [{"neuron_id": "L5/N2000", "label": "benefit"}],
            },
            relp_results=[
                {"prompt": "Test", "neuron_found": True, "tau": 0.01},
            ],
            hypotheses_tested=[
                {"hypothesis_id": "H1", "hypothesis": "Test", "status": "confirmed"},
            ],
            key_findings=["Finding 1", "Finding 2"],
            open_questions=["Question 1"],
            agent_reasoning="Full reasoning...",
        )

        d = inv.to_dict()

        # All sections should be populated
        assert d["confidence"] == 0.85
        assert d["characterization"]["input_function"] != ""
        assert d["characterization"]["output_function"] != ""
        assert len(d["evidence"]["activating_prompts"]) == 2
        assert len(d["hypotheses_tested"]) == 1
        assert len(d["key_findings"]) == 2

    def test_investigation_from_dict_roundtrip(self):
        """Investigation survives to_dict -> from_dict roundtrip."""
        original = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            confidence=0.75,
            input_function="Test input",
            output_function="Test output",
            key_findings=["Finding 1", "Finding 2"],
        )

        d = original.to_dict()
        restored = NeuronInvestigation.from_dict(d)

        assert restored.neuron_id == original.neuron_id
        assert restored.confidence == original.confidence
        assert restored.input_function == original.input_function
        assert restored.key_findings == original.key_findings


class TestDashboardDataCompleteness:
    """Test that dashboard data is well-formed."""

    def test_minimal_dashboard_valid(self):
        """Minimal dashboard is valid."""
        dashboard = DashboardData(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
        )

        d = dashboard.to_dict()

        assert d["neuron_id"] == "L4/N10555"
        assert "summary_card" in d
        assert "activation_patterns" in d
        assert "ablation_effects" in d

    def test_full_dashboard_structure(self):
        """Full dashboard has expected structure."""
        dashboard = DashboardData(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            summary="Dopamine neuron",
            input_function="Activates on dopamine",
            output_function="Promotes reward tokens",
            function_type="semantic",
            confidence=0.85,
            total_experiments=45,
            positive_examples=[
                ActivationExample("Dopamine test", 2.5, 5, "dopamine", True),
            ],
            negative_examples=[
                ActivationExample("Weather test", 0.1, 3, "weather", False),
            ],
            ablation_effects=[
                AblationEffect("pleasure", 0.5, "promotes", "high"),
            ],
            hypotheses=[
                HypothesisRecord("Test hypothesis", 10, ["evidence1"], [], 0.8, "supported"),
            ],
            upstream_nodes=[
                ConnectivityNode("L3/N1000", "chemical", 0.5, "upstream"),
            ],
            downstream_nodes=[
                ConnectivityNode("L5/N2000", "benefit", 0.3, "downstream"),
            ],
            key_findings=["Finding 1"],
            open_questions=["Question 1"],
        )

        d = dashboard.to_dict()

        # Check all sections
        assert d["summary_card"]["confidence"] == 0.85
        assert len(d["activation_patterns"]["positive_examples"]) == 1
        assert len(d["ablation_effects"]["effects"]) == 1
        assert len(d["hypothesis_timeline"]["hypotheses"]) == 1
        assert len(d["connectivity"]["upstream"]) == 1

    def test_investigation_to_dashboard_conversion(self):
        """Investigation converts to dashboard correctly."""
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            confidence=0.75,
            input_function="Test input",
            output_function="Test output",
            activating_prompts=[
                {"prompt": "Test 1", "activation": 2.0, "position": 5, "token": "test"},
            ],
            ablation_effects=[
                {"promotes": [["good", 0.5]], "suppresses": [["bad", -0.3]]},
            ],
        )

        dashboard = inv.to_dashboard()

        assert dashboard.neuron_id == "L4/N10555"
        assert dashboard.confidence == 0.75
        assert len(dashboard.positive_examples) == 1
        assert len(dashboard.ablation_effects) >= 1


class TestDistillationQuality:
    """Test that distillation for review captures key info."""

    def test_distill_captures_neuron_id(self):
        """Distillation includes neuron ID."""
        inv_dict = {
            "neuron_id": "L4/N10555",
            "confidence": 0.75,
        }

        distilled = distill_investigation_for_review(inv_dict)

        assert distilled["neuron_id"] == "L4/N10555"

    def test_distill_captures_confidence(self):
        """Distillation includes confidence level."""
        inv_dict = {
            "neuron_id": "L4/N10555",
            "confidence": 0.85,
        }

        distilled = distill_investigation_for_review(inv_dict)

        assert distilled["confidence"] == 0.85

    def test_distill_includes_protocol_checklist(self):
        """Distillation includes protocol checklist."""
        inv_dict = {
            "neuron_id": "L4/N10555",
            "confidence": 0.75,
            "protocol_validation": {
                "baseline_comparison_done": True,
                "baseline_zscore": 3.0,
                "dose_response_done": True,
            },
        }

        distilled = distill_investigation_for_review(inv_dict)

        assert "protocol_checklist" in distilled
        assert distilled["protocol_checklist"]["baseline_done"] is True

    def test_distill_computes_auto_reject_reasons(self):
        """Distillation computes auto-reject reasons for high confidence without validation."""
        inv_dict = {
            "neuron_id": "L4/N10555",
            "confidence": 0.9,  # High confidence
            "protocol_validation": {
                "baseline_comparison_done": False,  # Missing baseline
                "baseline_zscore": None,
            },
        }

        distilled = distill_investigation_for_review(inv_dict)

        assert len(distilled["auto_reject_reasons"]) > 0
        assert any("baseline" in r.lower() for r in distilled["auto_reject_reasons"])

    def test_distill_no_auto_reject_for_low_confidence(self):
        """Distillation doesn't auto-reject low confidence investigations."""
        inv_dict = {
            "neuron_id": "L4/N10555",
            "confidence": 0.3,  # Low confidence
            "protocol_validation": {
                "baseline_comparison_done": False,
                "baseline_zscore": None,
            },
        }

        distilled = distill_investigation_for_review(inv_dict)

        # Low confidence with missing validation is acceptable
        # (you don't need strong validation for low confidence claims)
        high_conf_rejections = [r for r in distilled["auto_reject_reasons"]
                                if "high confidence" in r.lower()]
        assert len(high_conf_rejections) == 0


class TestHardGateEnforcement:
    """Test hard gate enforcement on confidence claims."""

    def test_hard_gate_overrides_lenient_approve(self):
        """Hard gates override lenient APPROVE verdict."""
        review_result = {
            "verdict": "APPROVE",
            "confidence_assessment": "appropriate",
            "gaps": [],
            "feedback": "Looks good",
        }

        distilled = {
            "auto_reject_reasons": ["High confidence but baseline z-score missing"],
            "protocol_checklist": {"baseline_passes": False},
        }

        result = apply_hard_gates(review_result, distilled)

        assert result["verdict"] == "REQUEST_CHANGES"
        assert result.get("hard_gate_override") is True
        assert len(result["gaps"]) > 0

    def test_hard_gate_preserves_valid_approve(self):
        """Hard gates don't affect valid APPROVE verdict."""
        review_result = {
            "verdict": "APPROVE",
            "confidence_assessment": "appropriate",
            "gaps": [],
            "feedback": "Excellent work",
        }

        distilled = {
            "auto_reject_reasons": [],  # No violations
            "protocol_checklist": {"baseline_passes": True},
        }

        result = apply_hard_gates(review_result, distilled)

        assert result["verdict"] == "APPROVE"
        assert result.get("hard_gate_override") is not True

    def test_hard_gate_adds_protocol_checklist(self):
        """Hard gates add protocol checklist to result."""
        review_result = {
            "verdict": "REQUEST_CHANGES",
            "confidence_assessment": "overconfident",
            "gaps": ["Gap 1"],
            "feedback": "Needs work",
        }

        distilled = {
            "auto_reject_reasons": [],
            "protocol_checklist": {
                "baseline_passes": True,
                "dose_response_done": True,
            },
        }

        result = apply_hard_gates(review_result, distilled)

        assert "protocol_checklist" in result
        assert result["protocol_checklist"]["baseline_passes"] is True


class TestEvidenceQualityScoring:
    """Test that evidence quality is scored correctly."""

    def test_empty_state_low_score(self):
        """Empty protocol state yields low confidence score."""
        state = ProtocolState()
        conf = state.compute_evidence_confidence()

        assert conf < 0.2

    def test_baseline_only_medium_score(self):
        """Baseline comparison alone yields medium score."""
        state = ProtocolState(
            baseline_comparison_done=True,
            baseline_zscore=2.5,
        )
        conf = state.compute_evidence_confidence()

        assert 0.2 <= conf < 0.6

    def test_full_validation_high_score(self):
        """Complete validation yields high confidence score."""
        state = ProtocolState(
            phase0_corpus_queried=True,
            phase0_graph_count=10,
            baseline_comparison_done=True,
            baseline_zscore=4.0,
            dose_response_done=True,
            dose_response_monotonic=True,
            dose_response_kendall_tau=0.85,
            relp_runs=5,
            relp_positive_control=True,
            relp_negative_control=True,
            hypotheses_registered=3,
            hypotheses_updated=3,
        )
        conf = state.compute_evidence_confidence()

        assert conf >= 0.85

    def test_zscore_affects_score_monotonically(self):
        """Higher z-score yields higher confidence."""
        scores = []
        for zscore in [1.0, 2.0, 3.0, 4.0, 5.0]:
            state = ProtocolState(
                baseline_comparison_done=True,
                baseline_zscore=zscore,
            )
            scores.append(state.compute_evidence_confidence())

        # Scores should be monotonically increasing
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], f"Score at z={i+1} should be <= z={i+2}"


class TestConfidenceConsistency:
    """Test that confidence claims are consistent with evidence."""

    def test_high_confidence_requires_evidence(self):
        """High confidence (0.8+) requires strong evidence."""
        # High confidence with no validation
        inv_dict = {
            "neuron_id": "L4/N10555",
            "confidence": 0.9,
            "protocol_validation": {},  # Empty validation
        }

        distilled = distill_investigation_for_review(inv_dict)

        # Should flag auto-reject
        assert len(distilled["auto_reject_reasons"]) > 0

    def test_medium_confidence_needs_hypothesis(self):
        """Medium confidence (0.5+) benefits from hypothesis registration."""
        # Medium confidence with no hypotheses
        inv_dict = {
            "neuron_id": "L4/N10555",
            "confidence": 0.6,
            "protocol_validation": {
                "hypotheses_registered": 0,
            },
        }

        distilled = distill_investigation_for_review(inv_dict)

        # Should flag missing pre-registration
        assert any("pre-registration" in r.lower() or "hypothes" in r.lower()
                  for r in distilled["auto_reject_reasons"])

    def test_low_confidence_always_acceptable(self):
        """Low confidence (<0.5) is always acceptable evidence-wise."""
        inv_dict = {
            "neuron_id": "L4/N10555",
            "confidence": 0.3,
            "protocol_validation": {},  # No validation
        }

        distilled = distill_investigation_for_review(inv_dict)

        # Should not have hard rejections for low confidence
        hard_rejections = [r for r in distilled["auto_reject_reasons"]
                          if "confidence" in r.lower()]
        assert len(hard_rejections) == 0


class TestReviewResponseParsing:
    """Test parsing of review responses."""

    def test_parse_approve_verdict(self):
        """Parses APPROVE verdict correctly."""
        response = """VERDICT: APPROVE
CONFIDENCE_ASSESSMENT: appropriate
GAPS:
- None
FEEDBACK:
Investigation is complete."""

        result = parse_review_response(response)

        assert result["verdict"] == "APPROVE"
        assert result["confidence_assessment"] == "appropriate"

    def test_parse_request_changes_verdict(self):
        """Parses REQUEST_CHANGES verdict correctly."""
        response = """VERDICT: REQUEST_CHANGES
CONFIDENCE_ASSESSMENT: overconfident
GAPS:
- Missing baseline
- No RelP verification
FEEDBACK:
Please add baseline comparison."""

        result = parse_review_response(response)

        assert result["verdict"] == "REQUEST_CHANGES"
        assert result["confidence_assessment"] == "overconfident"
        assert len(result["gaps"]) >= 2

    def test_parse_preserves_raw_response(self):
        """Parsing preserves raw response."""
        response = "Some response text"
        result = parse_review_response(response)

        assert result["raw_response"] == response

    def test_parse_handles_malformed(self):
        """Parsing handles malformed response gracefully."""
        response = "This has no structure at all."
        result = parse_review_response(response)

        # Should default to cautious
        assert result["verdict"] == "REQUEST_CHANGES"


class TestPIResultIntegration:
    """Test PIResult captures full pipeline state."""

    def test_pi_result_with_approve_path(self):
        """PIResult captures successful approval path."""
        result = PIResult(
            neuron_id="L4/N10555",
            iterations=1,
            final_verdict="APPROVE",
            review_history=[
                ReviewResult("APPROVE", "appropriate", [], "Good work", "", 1),
            ],
            investigation_path="/path/to/investigation.json",
            dashboard_path="/path/to/dashboard.html",
        )

        d = result.to_dict()

        assert d["final_verdict"] == "APPROVE"
        assert d["iterations"] == 1
        assert len(d["review_history"]) == 1

    def test_pi_result_with_revision_loop(self):
        """PIResult captures revision loop correctly."""
        result = PIResult(
            neuron_id="L4/N10555",
            iterations=3,
            final_verdict="APPROVE",
            review_history=[
                ReviewResult("REQUEST_CHANGES", "overconfident", ["Gap 1"], "Fix", "", 1),
                ReviewResult("REQUEST_CHANGES", "appropriate", ["Gap 2"], "More", "", 2),
                ReviewResult("APPROVE", "appropriate", [], "Good", "", 3),
            ],
        )

        d = result.to_dict()

        assert d["iterations"] == 3
        assert len(d["review_history"]) == 3
        assert d["review_history"][0]["verdict"] == "REQUEST_CHANGES"
        assert d["review_history"][2]["verdict"] == "APPROVE"

    def test_pi_result_with_max_iterations(self):
        """PIResult handles max iterations case."""
        result = PIResult(
            neuron_id="L4/N10555",
            iterations=3,
            final_verdict="MAX_ITERATIONS",
            review_history=[
                ReviewResult("REQUEST_CHANGES", "overconfident", ["Gap"], "Fix", "", 1),
                ReviewResult("REQUEST_CHANGES", "overconfident", ["Gap"], "Fix", "", 2),
                ReviewResult("REQUEST_CHANGES", "overconfident", ["Gap"], "Fix", "", 3),
            ],
        )

        d = result.to_dict()

        assert d["final_verdict"] == "MAX_ITERATIONS"
        assert d["iterations"] == 3

    def test_pi_result_with_error(self):
        """PIResult captures error state."""
        result = PIResult(
            neuron_id="L4/N10555",
            iterations=0,
            final_verdict="ERROR",
            error="CUDA out of memory",
        )

        d = result.to_dict()

        assert d["final_verdict"] == "ERROR"
        assert d["error"] == "CUDA out of memory"
