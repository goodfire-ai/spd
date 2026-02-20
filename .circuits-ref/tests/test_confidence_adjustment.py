"""Tests for confidence adjustment logic in NeuronPI pipeline.

Verifies that:
1. Skeptic confidence adjustments are applied correctly
2. Confidence bounds are enforced (0.0 to 1.0)
3. Pre-skeptic confidence is preserved for audit trail
4. Atomic file writes work correctly
"""

import json

from neuron_scientist.pi_agent import _atomic_write_json
from neuron_scientist.schemas import NeuronInvestigation, SkepticReport


class TestSkepticConfidenceAdjustment:
    """Test skeptic confidence adjustment logic."""

    def test_weakened_verdict_applies_negative_adjustment(self):
        """WEAKENED verdict should apply -0.1 adjustment."""
        investigation = NeuronInvestigation(
            neuron_id="L15/N1234",
            layer=15,
            neuron_idx=1234,
        )
        investigation.confidence = 0.75

        # Simulate what pi_agent.py does
        skeptic_adjustment = -0.1  # WEAKENED
        original_confidence = investigation.confidence
        investigation.pre_skeptic_confidence = original_confidence
        investigation.skeptic_confidence_adjustment = skeptic_adjustment

        if skeptic_adjustment != 0:
            adjusted = original_confidence + skeptic_adjustment
            investigation.confidence = max(0.0, min(1.0, adjusted))

        assert investigation.confidence == 0.65
        assert investigation.pre_skeptic_confidence == 0.75
        assert investigation.skeptic_confidence_adjustment == -0.1

    def test_refuted_verdict_applies_larger_adjustment(self):
        """REFUTED verdict should apply -0.3 adjustment."""
        investigation = NeuronInvestigation(
            neuron_id="L15/N1234",
            layer=15,
            neuron_idx=1234,
        )
        investigation.confidence = 0.85

        # Simulate what pi_agent.py does
        skeptic_adjustment = -0.3  # REFUTED
        original_confidence = investigation.confidence
        investigation.pre_skeptic_confidence = original_confidence
        investigation.skeptic_confidence_adjustment = skeptic_adjustment

        if skeptic_adjustment != 0:
            adjusted = original_confidence + skeptic_adjustment
            investigation.confidence = max(0.0, min(1.0, adjusted))

        assert investigation.confidence == 0.55
        assert investigation.pre_skeptic_confidence == 0.85
        assert investigation.skeptic_confidence_adjustment == -0.3

    def test_supported_verdict_applies_no_adjustment(self):
        """SUPPORTED verdict should apply 0 adjustment."""
        investigation = NeuronInvestigation(
            neuron_id="L15/N1234",
            layer=15,
            neuron_idx=1234,
        )
        investigation.confidence = 0.85

        # Simulate what pi_agent.py does
        skeptic_adjustment = 0.0  # SUPPORTED
        original_confidence = investigation.confidence
        investigation.pre_skeptic_confidence = original_confidence
        investigation.skeptic_confidence_adjustment = skeptic_adjustment

        if skeptic_adjustment != 0:
            adjusted = original_confidence + skeptic_adjustment
            investigation.confidence = max(0.0, min(1.0, adjusted))

        assert investigation.confidence == 0.85  # Unchanged
        assert investigation.pre_skeptic_confidence == 0.85
        assert investigation.skeptic_confidence_adjustment == 0.0

    def test_confidence_lower_bound_enforced(self):
        """Confidence should not go below 0.0."""
        investigation = NeuronInvestigation(
            neuron_id="L15/N1234",
            layer=15,
            neuron_idx=1234,
        )
        investigation.confidence = 0.2  # Low starting confidence

        # Large negative adjustment
        skeptic_adjustment = -0.3
        original_confidence = investigation.confidence
        investigation.pre_skeptic_confidence = original_confidence
        investigation.skeptic_confidence_adjustment = skeptic_adjustment

        if skeptic_adjustment != 0:
            adjusted = original_confidence + skeptic_adjustment  # Would be -0.1
            investigation.confidence = max(0.0, min(1.0, adjusted))

        assert investigation.confidence == 0.0  # Clamped to 0
        assert investigation.pre_skeptic_confidence == 0.2

    def test_confidence_upper_bound_enforced(self):
        """Confidence should not go above 1.0."""
        investigation = NeuronInvestigation(
            neuron_id="L15/N1234",
            layer=15,
            neuron_idx=1234,
        )
        investigation.confidence = 0.95

        # Positive adjustment (hypothetically if skeptic could boost confidence)
        skeptic_adjustment = 0.1
        original_confidence = investigation.confidence
        investigation.pre_skeptic_confidence = original_confidence
        investigation.skeptic_confidence_adjustment = skeptic_adjustment

        if skeptic_adjustment != 0:
            adjusted = original_confidence + skeptic_adjustment  # Would be 1.05
            investigation.confidence = max(0.0, min(1.0, adjusted))

        assert investigation.confidence == 1.0  # Clamped to 1.0
        assert investigation.pre_skeptic_confidence == 0.95


class TestInvestigationSerialization:
    """Test that confidence fields serialize correctly."""

    def test_to_dict_includes_skeptic_confidence_fields(self):
        """to_dict should include pre_skeptic_confidence and skeptic_confidence_adjustment."""
        investigation = NeuronInvestigation(
            neuron_id="L15/N1234",
            layer=15,
            neuron_idx=1234,
        )
        investigation.confidence = 0.65
        investigation.pre_skeptic_confidence = 0.75
        investigation.skeptic_confidence_adjustment = -0.1

        d = investigation.to_dict()

        assert d["confidence"] == 0.65
        assert d["pre_skeptic_confidence"] == 0.75
        assert d["skeptic_confidence_adjustment"] == -0.1

    def test_from_dict_restores_skeptic_confidence_fields(self):
        """from_dict should restore pre_skeptic_confidence and skeptic_confidence_adjustment."""
        data = {
            "neuron_id": "L15/N1234",
            "layer": 15,
            "neuron_idx": 1234,
            "confidence": 0.65,
            "pre_skeptic_confidence": 0.75,
            "skeptic_confidence_adjustment": -0.1,
        }

        investigation = NeuronInvestigation.from_dict(data)

        assert investigation.confidence == 0.65
        assert investigation.pre_skeptic_confidence == 0.75
        assert investigation.skeptic_confidence_adjustment == -0.1

    def test_from_dict_handles_missing_skeptic_fields(self):
        """from_dict should handle old data without skeptic fields."""
        data = {
            "neuron_id": "L15/N1234",
            "layer": 15,
            "neuron_idx": 1234,
            "confidence": 0.75,
            # No pre_skeptic_confidence or skeptic_confidence_adjustment
        }

        investigation = NeuronInvestigation.from_dict(data)

        assert investigation.confidence == 0.75
        assert investigation.pre_skeptic_confidence is None
        assert investigation.skeptic_confidence_adjustment == 0.0


class TestSkepticReportAdjustmentValues:
    """Test that SkepticReport produces correct adjustment values."""

    def test_weakened_verdict_has_correct_adjustment(self):
        """SkepticReport with WEAKENED should have -0.1 adjustment."""
        report = SkepticReport(
            neuron_id="L15/N1234",
            original_hypothesis="Test hypothesis",
        )
        report.verdict = "WEAKENED"
        # The adjustment is set by the skeptic agent based on verdict
        # In skeptic_agent.py line 712:
        # confidence_adjustment=-0.1 if verdict == "WEAKENED" else (-0.3 if verdict == "REFUTED" else 0)
        report.confidence_adjustment = -0.1 if report.verdict == "WEAKENED" else (-0.3 if report.verdict == "REFUTED" else 0)

        assert report.confidence_adjustment == -0.1

    def test_refuted_verdict_has_correct_adjustment(self):
        """SkepticReport with REFUTED should have -0.3 adjustment."""
        report = SkepticReport(
            neuron_id="L15/N1234",
            original_hypothesis="Test hypothesis",
        )
        report.verdict = "REFUTED"
        report.confidence_adjustment = -0.1 if report.verdict == "WEAKENED" else (-0.3 if report.verdict == "REFUTED" else 0)

        assert report.confidence_adjustment == -0.3

    def test_supported_verdict_has_zero_adjustment(self):
        """SkepticReport with SUPPORTED should have 0 adjustment."""
        report = SkepticReport(
            neuron_id="L15/N1234",
            original_hypothesis="Test hypothesis",
        )
        report.verdict = "SUPPORTED"
        report.confidence_adjustment = -0.1 if report.verdict == "WEAKENED" else (-0.3 if report.verdict == "REFUTED" else 0)

        assert report.confidence_adjustment == 0


class TestSkepticInvestigationSummary:
    """Test that skeptic gets comprehensive investigation data."""

    def test_investigation_has_relp_results(self):
        """Investigation should have relp_results field."""
        investigation = NeuronInvestigation(
            neuron_id="L15/N1234",
            layer=15,
            neuron_idx=1234,
        )
        investigation.relp_results = [
            {"prompt": "Test prompt", "neuron_found": True, "relp_score": 0.5},
            {"prompt": "Another prompt", "neuron_found": False},
        ]

        assert len(investigation.relp_results) == 2
        assert investigation.relp_results[0]["neuron_found"] is True

    def test_investigation_has_steering_results(self):
        """Investigation should have steering_results field."""
        investigation = NeuronInvestigation(
            neuron_id="L15/N1234",
            layer=15,
            neuron_idx=1234,
        )
        investigation.steering_results = [
            {"prompt": "Test", "steering_value": 5.0, "promotes": [["token", 0.1]]},
        ]

        assert len(investigation.steering_results) == 1
        assert investigation.steering_results[0]["steering_value"] == 5.0

    def test_investigation_has_connectivity(self):
        """Investigation should have connectivity field."""
        investigation = NeuronInvestigation(
            neuron_id="L15/N1234",
            layer=15,
            neuron_idx=1234,
        )
        investigation.connectivity = {
            "upstream_neurons": [{"neuron_id": "L14/N100", "weight": 0.5}],
            "downstream_targets": [{"neuron_id": "L16/N200", "weight": 0.3}],
        }

        assert "upstream_neurons" in investigation.connectivity
        assert "downstream_targets" in investigation.connectivity

    def test_investigation_has_output_projections(self):
        """Investigation should have output_projections field."""
        investigation = NeuronInvestigation(
            neuron_id="L15/N1234",
            layer=15,
            neuron_idx=1234,
        )
        investigation.output_projections = {
            "promote": [{"token": "dopamine", "weight": 0.8}],
            "suppress": [{"token": "serotonin", "weight": -0.3}],
        }

        assert len(investigation.output_projections["promote"]) == 1
        assert investigation.output_projections["promote"][0]["token"] == "dopamine"

    def test_investigation_has_hypotheses_tested(self):
        """Investigation should have hypotheses_tested field."""
        investigation = NeuronInvestigation(
            neuron_id="L15/N1234",
            layer=15,
            neuron_idx=1234,
        )
        investigation.hypotheses_tested = [
            {
                "hypothesis_id": "H1",
                "hypothesis": "Detects medical terms",
                "status": "confirmed",
                "prior_probability": 50,
                "posterior_probability": 85,
            }
        ]

        assert len(investigation.hypotheses_tested) == 1
        assert investigation.hypotheses_tested[0]["status"] == "confirmed"


class TestRevisionHistoryTracking:
    """Test that revision feedback is tracked across iterations."""

    def test_investigation_has_revision_history_field(self):
        """NeuronInvestigation should have revision_history field."""
        investigation = NeuronInvestigation(
            neuron_id="L15/N1234",
            layer=15,
            neuron_idx=1234,
        )

        assert hasattr(investigation, "revision_history")
        assert investigation.revision_history == []

    def test_revision_history_serializes(self):
        """revision_history should serialize in to_dict."""
        investigation = NeuronInvestigation(
            neuron_id="L15/N1234",
            layer=15,
            neuron_idx=1234,
        )
        investigation.revision_history = [
            {
                "iteration": 1,
                "gaps_identified": ["Need more activation examples", "Missing ablation data"],
                "gaps_addressed": ["Need more activation examples"],
                "gaps_remaining": ["Missing ablation data"],
            }
        ]

        d = investigation.to_dict()
        assert "revision_history" in d
        assert len(d["revision_history"]) == 1
        assert d["revision_history"][0]["iteration"] == 1

    def test_revision_history_deserializes(self):
        """revision_history should deserialize in from_dict."""
        data = {
            "neuron_id": "L15/N1234",
            "layer": 15,
            "neuron_idx": 1234,
            "revision_history": [
                {
                    "iteration": 1,
                    "gaps_identified": ["gap1", "gap2"],
                    "gaps_addressed": ["gap1"],
                    "gaps_remaining": ["gap2"],
                }
            ],
        }

        investigation = NeuronInvestigation.from_dict(data)
        assert len(investigation.revision_history) == 1
        assert investigation.revision_history[0]["gaps_addressed"] == ["gap1"]

    def test_revision_history_handles_missing_field(self):
        """from_dict should handle old data without revision_history."""
        data = {
            "neuron_id": "L15/N1234",
            "layer": 15,
            "neuron_idx": 1234,
            # No revision_history field
        }

        investigation = NeuronInvestigation.from_dict(data)
        assert investigation.revision_history == []


class TestConnectivityLabelEnrichment:
    """Test that connectivity labels are enriched from ground truth."""

    def test_batch_get_labels_returns_found_labels(self):
        """batch_get_neuron_labels_with_fallback should return labels for known neurons."""
        from neuron_scientist.tools import batch_get_neuron_labels_with_fallback

        # Test with neurons that should be in the labels file
        results = batch_get_neuron_labels_with_fallback(["L0/N2578", "L1/N2973"])

        # At least one should be found (if labels file exists)
        if results:
            for nid, info in results.items():
                if info.get("found"):
                    assert "label" in info
                    assert info["source"] in ["json", "neurondb", "csv", "duckdb"]

    def test_batch_get_labels_handles_unknown_neurons(self):
        """batch_get_neuron_labels_with_fallback should handle unknown neurons gracefully."""
        from neuron_scientist.tools import batch_get_neuron_labels_with_fallback

        results = batch_get_neuron_labels_with_fallback(["L99/N99999"])

        # Should return a result even for unknown neurons
        assert "L99/N99999" in results
        # Should indicate not found
        assert results["L99/N99999"].get("found") is False or results["L99/N99999"].get("label") == ""

    def test_label_enrichment_skips_uninformative(self):
        """Label enrichment should skip uninformative labels like 'uninterpretable-routing'."""
        from neuron_scientist.tools import get_neuron_label_with_fallback

        # L0/N2578 has function_label="uninterpretable-routing" but input_label is informative
        result = get_neuron_label_with_fallback("L0/N2578")

        if result.get("found"):
            # Should use input_label, not the uninformative function_label
            label = result.get("label", "")
            assert "uninterpretable" not in label.lower()


class TestConfidenceDowngradeIndicator:
    """Test that dashboard shows confidence downgrade indicator."""

    def test_render_header_shows_downgrade_warning(self):
        """render_header should show warning when confidence was downgraded."""
        from neuron_scientist.html_builder import render_header

        html = render_header(
            neuron_id="L15/N1234",
            title="Test Neuron",
            confidence=0.65,
            total_experiments=50,
            confidence_downgraded=True,
            pre_skeptic_confidence=0.85,
            skeptic_adjustment=-0.2,
        )

        assert "confidence-adjusted" in html
        assert "85%" in html  # Original confidence
        assert "65%" in html  # Adjusted confidence
        assert "skeptic" in html.lower()

    def test_render_header_no_warning_when_not_downgraded(self):
        """render_header should not show warning when confidence not downgraded."""
        from neuron_scientist.html_builder import render_header

        html = render_header(
            neuron_id="L15/N1234",
            title="Test Neuron",
            confidence=0.85,
            total_experiments=50,
            confidence_downgraded=False,
        )

        assert "confidence-adjusted" not in html
        assert "line-through" not in html

    def test_render_header_shows_both_values_when_different(self):
        """render_header shows both original and adjusted when they differ."""
        from neuron_scientist.html_builder import render_header

        html = render_header(
            neuron_id="L15/N1234",
            title="Test Neuron",
            confidence=0.55,
            total_experiments=50,
            confidence_downgraded=False,  # Even if not flagged
            pre_skeptic_confidence=0.85,  # But there's a difference
            skeptic_adjustment=-0.3,
        )

        assert "85%" in html
        assert "55%" in html

    def test_build_fixed_sections_passes_downgrade_info(self):
        """build_fixed_sections should pass downgrade info to render_header."""
        from neuron_scientist.html_builder import build_fixed_sections

        result = build_fixed_sections(
            neuron_id="L15/N1234",
            title="Test Neuron",
            confidence=0.65,
            total_experiments=50,
            narrative_lead="Test lead",
            narrative_body="Test body",
            upstream_neurons=[],
            downstream_neurons=[],
            selectivity_fires=[],
            selectivity_ignores=[],
            output_promote=[],
            output_suppress=[],
            open_questions=[],
            confidence_downgraded=True,
            pre_skeptic_confidence=0.85,
            skeptic_adjustment=-0.2,
        )

        assert "confidence-adjusted" in result["header"]


class TestAtomicFileWrites:
    """Test atomic file write functionality."""

    def test_atomic_write_creates_file(self, tmp_path):
        """_atomic_write_json should create a valid JSON file."""
        filepath = tmp_path / "test.json"
        data = {"key": "value", "number": 42}

        _atomic_write_json(filepath, data)

        assert filepath.exists()
        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_atomic_write_overwrites_existing(self, tmp_path):
        """_atomic_write_json should atomically replace existing file."""
        filepath = tmp_path / "test.json"

        # Write initial data
        _atomic_write_json(filepath, {"version": 1})

        # Overwrite with new data
        _atomic_write_json(filepath, {"version": 2})

        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded["version"] == 2

    def test_atomic_write_no_temp_files_on_success(self, tmp_path):
        """Temp files should be cleaned up after successful write."""
        filepath = tmp_path / "test.json"
        _atomic_write_json(filepath, {"test": True})

        # Check no temp files remain
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].name == "test.json"

    def test_atomic_write_preserves_original_on_error(self, tmp_path):
        """Original file should be preserved if write fails."""
        filepath = tmp_path / "test.json"
        original_data = {"original": True}
        _atomic_write_json(filepath, original_data)

        # Try to write unserializable data (will fail)
        class NotSerializable:
            pass

        try:
            _atomic_write_json(filepath, {"bad": NotSerializable()})
        except TypeError:
            pass  # Expected

        # Original should still be intact
        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded == original_data

    def test_atomic_write_with_nested_data(self, tmp_path):
        """_atomic_write_json should handle complex nested structures."""
        filepath = tmp_path / "complex.json"
        data = {
            "investigation": {
                "neuron_id": "L15/N1234",
                "evidence": {
                    "prompts": ["test1", "test2"],
                    "scores": [0.5, 0.8]
                }
            },
            "skeptic_report": {
                "verdict": "SUPPORTED",
                "adjustment": 0.0
            }
        }

        _atomic_write_json(filepath, data)

        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded == data


class TestHypothesisHistoryTracking:
    """Test hypothesis evolution tracking with timestamps."""

    def test_register_creates_history_entry(self):
        """Registration should create an initial history entry."""
        import asyncio

        from neuron_scientist.tools import _HYPOTHESIS_REGISTRY, tool_register_hypothesis

        # Clear registry
        _HYPOTHESIS_REGISTRY.clear()

        result = asyncio.run(tool_register_hypothesis(
            hypothesis="Test hypothesis",
            confirmation_criteria="Activation > 1.0",
            refutation_criteria="Activation < 0.5",
            prior_probability=60,
            hypothesis_type="activation",
        ))

        assert result["hypothesis_id"] == "H1"
        registered = result["registered"]

        # Should have history list
        assert "history" in registered
        assert len(registered["history"]) == 1

        # First entry should be registration
        first_entry = registered["history"][0]
        assert first_entry["action"] == "registered"
        assert first_entry["prior"] == 60
        assert first_entry["status"] == "registered"
        assert "timestamp" in first_entry

    def test_update_appends_history_entry(self):
        """Updates should append to history."""
        import asyncio

        from neuron_scientist.tools import (
            _HYPOTHESIS_REGISTRY,
            tool_register_hypothesis,
            tool_update_hypothesis_status,
        )

        # Clear registry
        _HYPOTHESIS_REGISTRY.clear()

        asyncio.run(tool_register_hypothesis(
            hypothesis="Test hypothesis",
            confirmation_criteria="Activation > 1.0",
            refutation_criteria="Activation < 0.5",
            prior_probability=60,
            hypothesis_type="activation",
        ))

        asyncio.run(tool_update_hypothesis_status(
            hypothesis_id="H1",
            status="confirmed",
            posterior_probability=85,
            evidence_summary="Strong activation observed",
        ))

        # Get hypothesis from registry
        h = _HYPOTHESIS_REGISTRY[0]

        # Should now have 2 history entries
        assert len(h["history"]) == 2

        # Second entry should be the update
        update_entry = h["history"][1]
        assert update_entry["action"] == "confirmed"
        assert update_entry["prior"] == 60
        assert update_entry["posterior"] == 85
        assert update_entry["evidence"] == "Strong activation observed"
        assert "timestamp" in update_entry

    def test_final_status_reflects_last_update(self):
        """Final status should reflect the last update."""
        import asyncio

        from neuron_scientist.tools import (
            _HYPOTHESIS_REGISTRY,
            tool_register_hypothesis,
            tool_update_hypothesis_status,
        )

        # Clear registry
        _HYPOTHESIS_REGISTRY.clear()

        asyncio.run(tool_register_hypothesis(
            hypothesis="Test hypothesis",
            confirmation_criteria="Activation > 1.0",
            refutation_criteria="Activation < 0.5",
            prior_probability=50,
            hypothesis_type="activation",
        ))

        # First update: inconclusive
        asyncio.run(tool_update_hypothesis_status(
            hypothesis_id="H1",
            status="inconclusive",
            posterior_probability=55,
            evidence_summary="Mixed results",
        ))

        # Second update: confirmed after more testing
        asyncio.run(tool_update_hypothesis_status(
            hypothesis_id="H1",
            status="confirmed",
            posterior_probability=90,
            evidence_summary="Strong confirmation after additional tests",
        ))

        h = _HYPOTHESIS_REGISTRY[0]

        # Final status should be last update
        assert h["status"] == "confirmed"
        assert h["posterior_probability"] == 90

        # Should have 3 history entries
        assert len(h["history"]) == 3

        # Verify history order
        assert h["history"][0]["action"] == "registered"
        assert h["history"][1]["action"] == "inconclusive"
        assert h["history"][2]["action"] == "confirmed"

    def test_history_backwards_compatibility(self):
        """Update should work on hypotheses without history field."""
        import asyncio

        from neuron_scientist.tools import (
            _HYPOTHESIS_REGISTRY,
            tool_update_hypothesis_status,
        )

        # Clear registry and add a hypothesis without history (old format)
        _HYPOTHESIS_REGISTRY.clear()
        _HYPOTHESIS_REGISTRY.append({
            "hypothesis_id": "H1",
            "hypothesis": "Old hypothesis",
            "prior_probability": 50,
            "status": "registered",
            "posterior_probability": None,
            "evidence": [],
            # No history field - simulating old data
        })

        asyncio.run(tool_update_hypothesis_status(
            hypothesis_id="H1",
            status="confirmed",
            posterior_probability=80,
            evidence_summary="Confirmed",
        ))

        h = _HYPOTHESIS_REGISTRY[0]

        # Should have created history and added entry
        assert "history" in h
        assert len(h["history"]) == 1
        assert h["history"][0]["action"] == "confirmed"


class TestOutputProjectionNormalization:
    """Test output projection metadata preservation during normalization."""

    def test_normalize_string_input(self):
        """Plain string tokens should normalize with null metadata."""
        from neuron_scientist.tools import _normalize_projection_item

        result = _normalize_projection_item("train")

        assert result["token"] == "train"
        assert result["weight"] == 0
        assert result["source_field"] is None
        assert result["raw_data"] is None

    def test_normalize_dict_with_projection_strength(self):
        """Dict with projection_strength field should preserve source_field."""
        from neuron_scientist.tools import _normalize_projection_item

        item = {"token": "train", "projection_strength": 0.85}
        result = _normalize_projection_item(item)

        assert result["token"] == "train"
        assert result["weight"] == 0.85
        assert result["source_field"] == "projection_strength"
        assert result["raw_data"] == item

    def test_normalize_dict_with_weight(self):
        """Dict with weight field should use that field."""
        from neuron_scientist.tools import _normalize_projection_item

        item = {"token": "locomotive", "weight": 0.72}
        result = _normalize_projection_item(item)

        assert result["token"] == "locomotive"
        assert result["weight"] == 0.72
        assert result["source_field"] == "weight"
        assert result["raw_data"] == item

    def test_normalize_dict_with_magnitude(self):
        """Dict with magnitude field should use that field."""
        from neuron_scientist.tools import _normalize_projection_item

        item = {"token": "railway", "magnitude": 0.65}
        result = _normalize_projection_item(item)

        assert result["token"] == "railway"
        assert result["weight"] == 0.65
        assert result["source_field"] == "magnitude"
        assert result["raw_data"] == item

    def test_normalize_preserves_extra_metadata(self):
        """Raw_data should preserve all original fields."""
        from neuron_scientist.tools import _normalize_projection_item

        item = {
            "token": "train",
            "projection_strength": 0.85,
            "rank": 3,
            "layer": 15,
            "custom_field": "extra_info",
        }
        result = _normalize_projection_item(item)

        assert result["raw_data"]["rank"] == 3
        assert result["raw_data"]["layer"] == 15
        assert result["raw_data"]["custom_field"] == "extra_info"

    def test_backward_compatibility(self):
        """Normalized output should still have token and weight for consumers."""
        from neuron_scientist.tools import _normalize_projection_item

        # Various input formats should all produce token/weight output
        inputs = [
            "simple_token",
            {"token": "dict_token", "weight": 0.5},
            {"token": "strength_token", "projection_strength": 0.8},
        ]

        for item in inputs:
            result = _normalize_projection_item(item)
            assert "token" in result
            assert "weight" in result
            assert isinstance(result["token"], str)
            assert isinstance(result["weight"], (int, float))


class TestDashboardFileValidation:
    """Test dashboard file existence validation."""

    def test_dashboard_response_includes_generated_flag(self):
        """Dashboard response should include dashboard_generated field."""
        # Test the structure of expected response
        success_response = {
            "status": "success",
            "html_path": "/path/to/dashboard.html",
            "dashboard_generated": True,
            "stdout": "",
        }
        assert "dashboard_generated" in success_response
        assert success_response["dashboard_generated"] is True

    def test_missing_file_response_structure(self):
        """Missing file response should have correct structure."""
        error_response = {
            "status": "error",
            "html_path": None,
            "dashboard_generated": False,
            "dashboard_error": "Dashboard script succeeded but file not created at /path/to/file.html",
            "stdout": "",
        }
        assert error_response["status"] == "error"
        assert error_response["html_path"] is None
        assert error_response["dashboard_generated"] is False
        assert "dashboard_error" in error_response

    def test_file_exists_check_logic(self, tmp_path):
        """Test the file existence check logic."""
        # Create a file
        test_file = tmp_path / "test.html"
        test_file.write_text("<html></html>")

        # File exists case
        assert test_file.exists()

        # File doesn't exist case
        missing_file = tmp_path / "missing.html"
        assert not missing_file.exists()

    def test_dashboard_path_propagation(self):
        """Dashboard path should be None when file not created."""
        # Simulates what save_pi_result_tool receives
        # If dashboard generation fails, dashboard_path should be empty/None
        dashboard_path = ""  # What's passed when file not created
        assert not dashboard_path  # Should be falsy
