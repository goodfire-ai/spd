"""Schema validation tests for neuron investigation data structures.

These tests verify that data structures are correct, complete, and that
conversions preserve all information.
"""

import json
from pathlib import Path

import pytest
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


class TestActivationExample:
    """Test ActivationExample dataclass."""

    def test_to_dict_contains_all_fields(self):
        """to_dict includes all fields."""
        example = ActivationExample(
            prompt="Test prompt",
            activation=1.5,
            position=3,
            token="test",
            is_positive=True,
        )
        d = example.to_dict()

        assert d["prompt"] == "Test prompt"
        assert d["activation"] == 1.5
        assert d["position"] == 3
        assert d["token"] == "test"
        assert d["is_positive"] is True


class TestAblationEffect:
    """Test AblationEffect dataclass."""

    def test_to_dict_contains_all_fields(self):
        """to_dict includes all fields."""
        effect = AblationEffect(
            token="dopamine",
            shift=0.5,
            direction="promotes",
            consistency="high",
        )
        d = effect.to_dict()

        assert d["token"] == "dopamine"
        assert d["shift"] == 0.5
        assert d["direction"] == "promotes"
        assert d["consistency"] == "high"


class TestHypothesisRecord:
    """Test HypothesisRecord dataclass."""

    def test_to_dict_with_evidence(self):
        """to_dict includes evidence lists."""
        record = HypothesisRecord(
            hypothesis="Test hypothesis",
            formed_at_experiment=10,
            evidence_for=["Evidence 1", "Evidence 2"],
            evidence_against=["Counter 1"],
            confidence=0.75,
            status="supported",
        )
        d = record.to_dict()

        assert d["hypothesis"] == "Test hypothesis"
        assert d["formed_at_experiment"] == 10
        assert len(d["evidence_for"]) == 2
        assert len(d["evidence_against"]) == 1
        assert d["confidence"] == 0.75
        assert d["status"] == "supported"


class TestConnectivityNode:
    """Test ConnectivityNode dataclass."""

    def test_to_dict_for_neuron(self):
        """to_dict works for neuron nodes."""
        node = ConnectivityNode(
            neuron_id="L15/N7890",
            label="Dopamine detector",
            weight=0.35,
            direction="downstream",
            is_logit=False,
        )
        d = node.to_dict()

        assert d["neuron_id"] == "L15/N7890"
        assert d["label"] == "Dopamine detector"
        assert d["weight"] == 0.35
        assert d["direction"] == "downstream"
        assert d["is_logit"] is False

    def test_to_dict_for_logit(self):
        """to_dict works for logit nodes."""
        node = ConnectivityNode(
            neuron_id="logit_dopamine",
            label="dopamine",
            weight=0.5,
            direction="downstream",
            is_logit=True,
        )
        d = node.to_dict()

        assert d["is_logit"] is True


class TestNeuronInvestigation:
    """Test NeuronInvestigation dataclass."""

    def test_to_dict_contains_required_fields(self):
        """to_dict includes all required fields for downstream processing."""
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
        )
        d = inv.to_dict()

        required_fields = [
            "neuron_id", "layer", "neuron_idx", "timestamp",
            "total_experiments", "confidence", "characterization",
            "evidence", "key_findings", "open_questions"
        ]
        for field in required_fields:
            assert field in d, f"Missing required field: {field}"

    def test_to_dict_characterization_structure(self):
        """Characterization section has expected structure."""
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            input_function="Test input function",
            output_function="Test output function",
            function_type="semantic",
            final_hypothesis="Test hypothesis",
        )
        d = inv.to_dict()

        char = d["characterization"]
        assert char["input_function"] == "Test input function"
        assert char["output_function"] == "Test output function"
        assert char["function_type"] == "semantic"
        assert char["final_hypothesis"] == "Test hypothesis"

    def test_to_dict_evidence_structure(self):
        """Evidence section has expected structure."""
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            activating_prompts=[{"prompt": "test", "activation": 1.0}],
            non_activating_prompts=[{"prompt": "test2", "activation": 0.1}],
            ablation_effects=[{"promotes": [("token", 0.5)]}],
            connectivity={"upstream": [], "downstream": []},
            relp_results=[{"prompt": "test", "neuron_found": True}],
        )
        d = inv.to_dict()

        evidence = d["evidence"]
        assert "activating_prompts" in evidence
        assert "non_activating_prompts" in evidence
        assert "ablation_effects" in evidence
        assert "connectivity" in evidence
        assert "relp_results" in evidence

    def test_from_dict_roundtrip_preserves_data(self):
        """from_dict(to_dict(x)) preserves all data."""
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            timestamp="2026-01-19T00:00:00",
            total_experiments=45,
            confidence=0.75,
            initial_label="Initial label",
            initial_hypothesis="Initial hypothesis",
            input_function="Activates on dopamine",
            output_function="Promotes neurotransmitter tokens",
            function_type="semantic",
            final_hypothesis="Final hypothesis",
            key_findings=["Finding 1", "Finding 2"],
            open_questions=["Question 1"],
            agent_reasoning="Detailed reasoning...",
        )
        d = inv.to_dict()
        restored = NeuronInvestigation.from_dict(d)

        assert restored.neuron_id == inv.neuron_id
        assert restored.layer == inv.layer
        assert restored.neuron_idx == inv.neuron_idx
        assert restored.confidence == inv.confidence
        assert restored.input_function == inv.input_function
        assert restored.output_function == inv.output_function
        assert restored.function_type == inv.function_type
        assert restored.final_hypothesis == inv.final_hypothesis
        assert restored.key_findings == inv.key_findings
        assert restored.open_questions == inv.open_questions

    def test_from_dict_handles_nested_evidence(self):
        """from_dict correctly extracts nested evidence."""
        data = {
            "neuron_id": "L4/N10555",
            "layer": 4,
            "neuron_idx": 10555,
            "characterization": {
                "input_function": "Test",
                "output_function": "Test",
                "function_type": "semantic",
                "final_hypothesis": "Test",
            },
            "evidence": {
                "activating_prompts": [
                    {"prompt": "Prompt 1", "activation": 1.5},
                    {"prompt": "Prompt 2", "activation": 2.0},
                ],
                "non_activating_prompts": [
                    {"prompt": "Prompt 3", "activation": 0.1},
                ],
                "ablation_effects": [],
                "connectivity": {},
                "relp_results": [
                    {"prompt": "Test", "neuron_found": True, "tau": 0.01},
                ],
            },
            "relp_results": [
                {"prompt": "Test", "neuron_found": True, "tau": 0.01},
            ],
        }

        inv = NeuronInvestigation.from_dict(data)

        assert len(inv.activating_prompts) == 2
        assert len(inv.non_activating_prompts) == 1
        assert len(inv.relp_results) == 1

    def test_to_dashboard_creates_valid_dashboard(self):
        """to_dashboard creates valid DashboardData."""
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            confidence=0.8,
            input_function="Test input",
            output_function="Test output",
            activating_prompts=[
                {"prompt": "Test", "activation": 1.5, "position": 0, "token": "x"}
            ],
            key_findings=["Finding 1", "Finding 2"],
        )
        dashboard = inv.to_dashboard()

        assert isinstance(dashboard, DashboardData)
        assert dashboard.neuron_id == inv.neuron_id
        assert dashboard.layer == inv.layer
        assert dashboard.confidence == inv.confidence
        assert len(dashboard.positive_examples) == 1
        assert len(dashboard.key_findings) == 2


class TestDashboardData:
    """Test DashboardData dataclass."""

    def test_to_dict_structure(self):
        """Dashboard JSON has expected nested structure."""
        dashboard = DashboardData(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            confidence=0.8,
            input_function="Test input",
            output_function="Test output",
        )
        d = dashboard.to_dict()

        assert "summary_card" in d
        assert "activation_patterns" in d
        assert "ablation_effects" in d
        assert "hypothesis_timeline" in d
        assert "connectivity" in d
        assert "findings" in d
        assert "metadata" in d

    def test_to_dict_summary_card(self):
        """Summary card has expected fields."""
        dashboard = DashboardData(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            summary="Test summary",
            input_function="Test input",
            output_function="Test output",
            function_type="semantic",
            confidence=0.8,
            total_experiments=45,
            initial_label="Initial",
        )
        d = dashboard.to_dict()

        card = d["summary_card"]
        assert card["summary"] == "Test summary"
        assert card["input_function"] == "Test input"
        assert card["output_function"] == "Test output"
        assert card["function_type"] == "semantic"
        assert card["confidence"] == 0.8
        assert card["total_experiments"] == 45

    def test_to_dict_includes_examples(self):
        """Dashboard includes activation examples."""
        dashboard = DashboardData(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            positive_examples=[
                ActivationExample("Test 1", 1.5, 0, "x", True),
                ActivationExample("Test 2", 2.0, 1, "y", True),
            ],
            negative_examples=[
                ActivationExample("Test 3", 0.1, 0, "z", False),
            ],
        )
        d = dashboard.to_dict()

        patterns = d["activation_patterns"]
        assert len(patterns["positive_examples"]) == 2
        assert len(patterns["negative_examples"]) == 1


class TestReviewResult:
    """Test ReviewResult dataclass."""

    def test_to_dict_contains_all_fields(self):
        """to_dict includes all review fields."""
        result = ReviewResult(
            verdict="APPROVE",
            hypothesis_assessment="appropriate",
            gaps=["Gap 1", "Gap 2"],
            feedback="Good investigation",
            raw_response="Full response...",
            iteration=1,
        )
        d = result.to_dict()

        assert d["verdict"] == "APPROVE"
        assert d["hypothesis_assessment"] == "appropriate"
        assert len(d["gaps"]) == 2
        assert d["feedback"] == "Good investigation"
        assert d["iteration"] == 1


class TestPIResult:
    """Test PIResult dataclass."""

    def test_to_dict_contains_all_fields(self):
        """to_dict includes all PI result fields."""
        result = PIResult(
            neuron_id="L4/N10555",
            iterations=2,
            final_verdict="APPROVE",
            review_history=[
                ReviewResult("REQUEST_CHANGES", "overconfident", ["Gap 1"], "Fix", "", 1),
                ReviewResult("APPROVE", "appropriate", [], "Good", "", 2),
            ],
            dashboard_path="/path/to/dashboard.html",
            investigation_path="/path/to/investigation.json",
            timestamp="2026-01-19T00:00:00",
        )
        d = result.to_dict()

        assert d["neuron_id"] == "L4/N10555"
        assert d["iterations"] == 2
        assert d["final_verdict"] == "APPROVE"
        assert len(d["review_history"]) == 2
        assert d["dashboard_path"] == "/path/to/dashboard.html"


class TestLoadingRealInvestigations:
    """Test that real investigation files can be loaded and validated."""

    @pytest.fixture
    def sample_investigation_paths(self):
        """Paths to real investigation files."""
        base = Path("neuron_reports/json")
        if not base.exists():
            pytest.skip("Sample investigation directory not found")

        paths = list(base.glob("*_investigation.json"))[:5]  # Test first 5
        if not paths:
            pytest.skip("No investigation files found")
        return paths

    def test_load_real_investigations(self, sample_investigation_paths):
        """Real investigation JSON files load without error."""
        for path in sample_investigation_paths:
            with open(path) as f:
                data = json.load(f)

            inv = NeuronInvestigation.from_dict(data)
            assert inv.neuron_id is not None
            assert inv.layer >= 0
            assert inv.neuron_idx >= 0

    def test_real_investigations_have_evidence(self, sample_investigation_paths):
        """Real investigations contain expected evidence sections."""
        for path in sample_investigation_paths:
            with open(path) as f:
                data = json.load(f)

            inv = NeuronInvestigation.from_dict(data)

            # At least one type of evidence should be present
            has_evidence = (
                len(inv.activating_prompts) > 0 or
                len(inv.non_activating_prompts) > 0 or
                len(inv.ablation_effects) > 0 or
                len(inv.relp_results) > 0
            )
            assert has_evidence, f"No evidence in {path.name}"

    def test_real_investigations_convert_to_dashboard(self, sample_investigation_paths):
        """Real investigations can be converted to dashboard format."""
        for path in sample_investigation_paths:
            with open(path) as f:
                data = json.load(f)

            inv = NeuronInvestigation.from_dict(data)
            dashboard = inv.to_dashboard()

            assert dashboard.neuron_id == inv.neuron_id
            assert isinstance(dashboard.to_dict(), dict)
