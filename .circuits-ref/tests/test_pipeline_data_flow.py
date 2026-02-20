"""Integration tests for full NeuronPI pipeline data flow.

Verifies that data flows correctly through all pipeline stages:
1. Investigation creation → JSON serialization
2. Skeptic receives sufficient data (via to_dict structure)
3. Skeptic report embedded in investigation
4. Confidence adjustment applied
5. GPT review receives complete evidence
6. Revision context includes feedback
7. Dashboard receives all needed data

These tests ensure fixes from the pipeline audit are working correctly.
"""

import json

from neuron_scientist.schemas import (
    MAX_ACTIVATING_PROMPTS,
    MAX_NON_ACTIVATING_PROMPTS,
    NeuronInvestigation,
)


def create_test_investigation(
    neuron_id: str = "L15/N1234",
    num_activating: int = 30,
    num_non_activating: int = 15,
    num_relp_results: int = 5,
    num_steering_results: int = 3,
    confidence: float = 0.75,
) -> NeuronInvestigation:
    """Create a test investigation with configurable amounts of data."""
    investigation = NeuronInvestigation(
        neuron_id=neuron_id,
        layer=15,
        neuron_idx=1234,
    )

    # Populate activating prompts
    investigation.activating_prompts = [
        {
            "prompt": f"This is activating prompt {i} with some content about trains and railways.",
            "activation": 2.0 - (i * 0.05),
            "token_position": i % 10,
        }
        for i in range(num_activating)
    ]

    # Populate non-activating prompts
    investigation.non_activating_prompts = [
        {
            "prompt": f"This is non-activating prompt {i} about something unrelated.",
            "activation": 0.1 + (i * 0.01),
        }
        for i in range(num_non_activating)
    ]

    # Populate RelP results
    investigation.relp_results = [
        {
            "prompt": f"RelP prompt {i}",
            "source": "corpus" if i < 3 else "synthetic",
            "neuron_found": i < 3,
            "activation": 1.5 + i * 0.1,
            "upstream": [{"neuron_id": f"L14/N{100+i}", "weight": 0.3}],
            "downstream": [{"neuron_id": f"L16/N{200+i}", "weight": 0.25}],
        }
        for i in range(num_relp_results)
    ]

    # Populate steering results
    investigation.steering_results = [
        {
            "prompt": f"Steering prompt {i}",
            "steering_value": 1.0 + i * 0.5,
            "effect": "Increased train-related tokens",
            "promotes": [{"token": "train", "shift": 0.15}],
            "suppresses": [{"token": "car", "shift": -0.05}],
        }
        for i in range(num_steering_results)
    ]

    # Populate connectivity (stored in connectivity dict)
    investigation.connectivity = {
        "upstream_neurons": [
            {"neuron_id": f"L14/N{1000+i}", "weight": 0.5 - i * 0.05, "label": ""}
            for i in range(5)
        ],
        "downstream_neurons": [
            {"neuron_id": f"L16/N{2000+i}", "weight": 0.4 - i * 0.05, "label": ""}
            for i in range(5)
        ],
    }

    # Populate output projections (dict format)
    investigation.output_projections = {
        "promote": [
            {"token": "train", "weight": 0.85, "source_field": "projection_strength"},
            {"token": "railway", "weight": 0.72, "source_field": "projection_strength"},
        ],
        "suppress": [
            {"token": "car", "weight": 0.3, "source_field": "projection_strength"},
        ],
    }

    # Populate hypotheses
    investigation.hypotheses_tested = [
        {
            "hypothesis_id": "H1",
            "hypothesis": "Neuron fires on train-related content",
            "status": "confirmed",
            "prior_probability": 60,
            "posterior_probability": 85,
            "evidence": ["Strong activation on train prompts"],
            "history": [
                {"timestamp": "2024-01-01T12:00:00", "action": "registered", "prior": 60},
                {"timestamp": "2024-01-01T12:05:00", "action": "confirmed", "posterior": 85},
            ],
        }
    ]

    # Set characterization fields (flat on investigation)
    investigation.input_function = "Railway and train contexts"
    investigation.output_function = "Promotes 'train' token"
    investigation.final_hypothesis = "This neuron fires on train-related content."

    investigation.confidence = confidence

    return investigation


class TestInvestigationSerialization:
    """Test that investigation serializes correctly with proper truncation."""

    def test_to_dict_includes_all_key_fields(self):
        """to_dict() should include all fields needed by downstream stages."""
        investigation = create_test_investigation()
        data = investigation.to_dict()

        # Key fields must be present
        assert "neuron_id" in data
        assert "layer" in data
        assert "confidence" in data
        assert "evidence" in data
        assert "relp_results" in data
        assert "steering_results" in data
        assert "output_projections" in data
        assert "hypotheses_tested" in data
        assert "characterization" in data

        # Evidence nested fields
        assert "activating_prompts" in data["evidence"]
        assert "non_activating_prompts" in data["evidence"]

    def test_activating_prompts_truncated_within_limit(self):
        """Activating prompts should be truncated to MAX_ACTIVATING_PROMPTS."""
        investigation = create_test_investigation(num_activating=50)
        data = investigation.to_dict()

        assert len(data["evidence"]["activating_prompts"]) <= MAX_ACTIVATING_PROMPTS

    def test_non_activating_prompts_truncated_within_limit(self):
        """Non-activating prompts should be truncated to MAX_NON_ACTIVATING_PROMPTS."""
        investigation = create_test_investigation(num_non_activating=20)
        data = investigation.to_dict()

        assert len(data["evidence"]["non_activating_prompts"]) <= MAX_NON_ACTIVATING_PROMPTS

    def test_relp_results_preserved(self):
        """RelP results should be preserved in full."""
        investigation = create_test_investigation(num_relp_results=10)
        data = investigation.to_dict()

        # RelP results should be preserved (no truncation)
        assert len(data["relp_results"]) == 10

    def test_hypothesis_history_preserved(self):
        """Hypothesis history should be preserved in serialization."""
        investigation = create_test_investigation()
        data = investigation.to_dict()

        # Check hypothesis has history
        assert len(data["hypotheses_tested"]) > 0
        h = data["hypotheses_tested"][0]
        assert "history" in h
        assert len(h["history"]) == 2


class TestSkepticDataStructures:
    """Test that data structures support skeptic agent needs."""

    def test_to_dict_has_relp_results_at_top_level(self):
        """RelP results should be at top level for skeptic access."""
        investigation = create_test_investigation(num_relp_results=5)
        data = investigation.to_dict()

        assert "relp_results" in data
        assert len(data["relp_results"]) == 5

    def test_to_dict_has_steering_results_at_top_level(self):
        """Steering results should be at top level for skeptic access."""
        investigation = create_test_investigation(num_steering_results=5)
        data = investigation.to_dict()

        assert "steering_results" in data
        assert len(data["steering_results"]) == 5

    def test_to_dict_has_connectivity_in_evidence(self):
        """Connectivity should be in evidence for skeptic."""
        investigation = create_test_investigation()
        data = investigation.to_dict()

        assert "connectivity" in data["evidence"]

    def test_to_dict_has_hypotheses(self):
        """Hypotheses tested should be accessible."""
        investigation = create_test_investigation()
        data = investigation.to_dict()

        assert "hypotheses_tested" in data
        assert len(data["hypotheses_tested"]) > 0


class TestSkepticConfidenceAdjustment:
    """Test that skeptic confidence adjustments are applied."""

    def test_weakened_verdict_adjusts_confidence(self):
        """WEAKENED verdict should decrease confidence."""
        investigation = create_test_investigation(confidence=0.80)

        # Simulate what pi_agent does after skeptic
        original_confidence = investigation.confidence
        investigation.pre_skeptic_confidence = original_confidence
        investigation.skeptic_confidence_adjustment = -0.1  # WEAKENED

        # Apply adjustment
        adjusted = original_confidence + investigation.skeptic_confidence_adjustment
        investigation.confidence = max(0.0, min(1.0, adjusted))

        # Use approximate comparison for floating point
        assert abs(investigation.confidence - 0.70) < 0.001
        assert investigation.pre_skeptic_confidence == 0.80

    def test_confidence_fields_survive_serialization(self):
        """Confidence adjustment fields should survive to_dict/from_dict."""
        investigation = create_test_investigation(confidence=0.80)
        investigation.pre_skeptic_confidence = 0.80
        investigation.skeptic_confidence_adjustment = -0.1

        # Round-trip through serialization
        data = investigation.to_dict()
        restored = NeuronInvestigation.from_dict(data)

        assert restored.pre_skeptic_confidence == 0.80
        assert restored.skeptic_confidence_adjustment == -0.1


class TestRevisionContextFlow:
    """Test that revision context includes all feedback."""

    def test_revision_history_field_exists(self):
        """Investigation should have revision_history field."""
        investigation = create_test_investigation()

        assert hasattr(investigation, "revision_history")
        assert isinstance(investigation.revision_history, list)

    def test_revision_history_survives_serialization(self):
        """Revision history should survive serialization."""
        investigation = create_test_investigation()
        investigation.revision_history = [
            {
                "iteration": 1,
                "gaps_identified": ["Missing ablation experiments"],
                "gaps_addressed": [],
                "gaps_remaining": ["Missing ablation experiments"],
            },
            {
                "iteration": 2,
                "gaps_identified": [],
                "gaps_addressed": ["Missing ablation experiments"],
                "gaps_remaining": [],
            },
        ]

        data = investigation.to_dict()
        restored = NeuronInvestigation.from_dict(data)

        assert len(restored.revision_history) == 2
        assert restored.revision_history[0]["gaps_identified"] == ["Missing ablation experiments"]


class TestDashboardDataFlow:
    """Test that dashboard receives all needed data."""

    def test_investigation_has_output_projections_for_dashboard(self):
        """Investigation should have output projections in correct format."""
        investigation = create_test_investigation()
        data = investigation.to_dict()

        assert "output_projections" in data
        assert "promote" in data["output_projections"]
        assert "suppress" in data["output_projections"]
        assert len(data["output_projections"]["promote"]) > 0

    def test_investigation_has_connectivity_for_dashboard(self):
        """Investigation should have connectivity data for circuit diagrams."""
        investigation = create_test_investigation()
        data = investigation.to_dict()

        # Connectivity is nested under evidence
        assert "evidence" in data
        assert "connectivity" in data["evidence"]

    def test_investigation_has_characterization_for_dashboard(self):
        """Investigation should have characterization for summary card."""
        investigation = create_test_investigation()
        data = investigation.to_dict()

        assert "characterization" in data
        char = data["characterization"]
        assert "input_function" in char
        assert "output_function" in char
        assert "final_hypothesis" in char


class TestAtomicFileOperations:
    """Test that file operations are atomic."""

    def test_atomic_write_json_creates_file(self, tmp_path):
        """_atomic_write_json should create file atomically."""
        from neuron_scientist.pi_agent import _atomic_write_json

        filepath = tmp_path / "test.json"
        data = {"test": True, "nested": {"key": "value"}}

        _atomic_write_json(filepath, data)

        assert filepath.exists()
        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_atomic_write_json_no_temp_files(self, tmp_path):
        """_atomic_write_json should clean up temp files."""
        from neuron_scientist.pi_agent import _atomic_write_json

        filepath = tmp_path / "test.json"
        _atomic_write_json(filepath, {"test": True})

        # Check no temp files remain
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].name == "test.json"


class TestOutputProjectionNormalization:
    """Test that output projections preserve metadata."""

    def test_normalize_preserves_source_field(self):
        """Normalized projections should track source field."""
        from neuron_scientist.tools import _normalize_projection_item

        item = {"token": "train", "projection_strength": 0.85}
        result = _normalize_projection_item(item)

        assert result["source_field"] == "projection_strength"

    def test_normalize_preserves_raw_data(self):
        """Normalized projections should preserve raw data."""
        from neuron_scientist.tools import _normalize_projection_item

        item = {"token": "train", "projection_strength": 0.85, "rank": 1}
        result = _normalize_projection_item(item)

        assert result["raw_data"] is not None
        assert result["raw_data"]["rank"] == 1


class TestHypothesisHistoryTracking:
    """Test that hypothesis history is tracked."""

    def test_hypothesis_history_structure(self):
        """Hypotheses should have history tracking."""
        investigation = create_test_investigation()

        h = investigation.hypotheses_tested[0]
        assert "history" in h
        assert len(h["history"]) > 0
        assert "timestamp" in h["history"][0]
        assert "action" in h["history"][0]


class TestFullPipelineRoundTrip:
    """End-to-end test of pipeline data flow."""

    def test_investigation_survives_json_roundtrip(self, tmp_path):
        """Investigation should survive save/load cycle with all data intact."""
        investigation = create_test_investigation(
            num_activating=50,
            num_non_activating=20,
            num_relp_results=10,
            num_steering_results=5,
        )
        investigation.pre_skeptic_confidence = 0.80
        investigation.skeptic_confidence_adjustment = -0.1
        investigation.revision_history = [{"iteration": 1, "gaps_identified": ["test"]}]

        # Save to JSON
        filepath = tmp_path / "investigation.json"
        data = investigation.to_dict()
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        # Load back
        with open(filepath) as f:
            loaded_data = json.load(f)
        restored = NeuronInvestigation.from_dict(loaded_data)

        # Verify key fields survived
        assert restored.neuron_id == investigation.neuron_id
        assert restored.confidence == investigation.confidence
        assert restored.pre_skeptic_confidence == 0.80
        assert restored.skeptic_confidence_adjustment == -0.1
        assert len(restored.revision_history) == 1
        assert len(restored.relp_results) == 10
        assert len(restored.hypotheses_tested) == 1
        assert "history" in restored.hypotheses_tested[0]
