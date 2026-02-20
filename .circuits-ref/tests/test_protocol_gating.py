"""Protocol gating tests for neuron scientist.

These tests verify that the investigation protocol is enforced:
- Investigations cannot be saved without required validation
- Confidence is auto-downgraded when protocol gaps exist
- Hard gates block high-confidence claims without evidence
"""

from neuron_scientist.schemas import NeuronInvestigation
from neuron_scientist.tools import (
    ProtocolState,
    get_protocol_state,
    init_protocol_state,
    update_protocol_state,
)


class TestProtocolGating:
    """Test that protocol validation gates are enforced."""

    def test_missing_baseline_flags_validation(self):
        """Missing baseline comparison is flagged."""
        state = ProtocolState(
            baseline_comparison_done=False,
        )
        missing = state.get_missing_validation()

        assert any("baseline" in m.lower() for m in missing)

    def test_low_zscore_flags_validation(self):
        """Low z-score (<2.0) is flagged even if baseline was run."""
        state = ProtocolState(
            baseline_comparison_done=True,
            baseline_zscore=1.5,  # Below threshold
        )
        missing = state.get_missing_validation()

        assert any("z-score" in m.lower() for m in missing)

    def test_missing_dose_response_flags_validation(self):
        """Missing dose-response test is flagged."""
        state = ProtocolState(
            dose_response_done=False,
        )
        missing = state.get_missing_validation()

        assert any("dose-response" in m.lower() for m in missing)

    def test_non_monotonic_dose_response_flags_validation(self):
        """Non-monotonic dose-response is flagged."""
        state = ProtocolState(
            dose_response_done=True,
            dose_response_monotonic=False,
        )
        missing = state.get_missing_validation()

        assert any("monotonic" in m.lower() for m in missing)

    def test_missing_relp_flags_validation(self):
        """Missing RelP attribution is flagged."""
        state = ProtocolState(
            relp_runs=0,
        )
        missing = state.get_missing_validation()

        assert any("relp" in m.lower() for m in missing)

    def test_missing_positive_control_flags_validation(self):
        """Missing RelP positive control is flagged."""
        state = ProtocolState(
            relp_runs=3,
            relp_positive_control=False,
        )
        missing = state.get_missing_validation()

        assert any("positive control" in m.lower() for m in missing)

    def test_missing_negative_control_flags_validation(self):
        """Missing RelP negative control is flagged."""
        state = ProtocolState(
            relp_runs=3,
            relp_positive_control=True,
            relp_negative_control=False,
        )
        missing = state.get_missing_validation()

        assert any("negative control" in m.lower() for m in missing)

    def test_missing_preregistration_flags_validation(self):
        """Missing hypothesis pre-registration is flagged."""
        state = ProtocolState(
            hypotheses_registered=0,
        )
        missing = state.get_missing_validation()

        assert any("pre-registration" in m.lower() for m in missing)

    def test_complete_validation_no_flags(self):
        """Complete validation has no missing items."""
        state = ProtocolState(
            phase0_corpus_queried=True,
            phase0_graph_count=10,
            baseline_comparison_done=True,
            baseline_zscore=3.5,
            dose_response_done=True,
            dose_response_monotonic=True,
            relp_runs=5,
            relp_positive_control=True,
            relp_negative_control=True,
            hypotheses_registered=3,
        )
        missing = state.get_missing_validation()

        assert len(missing) == 0


class TestConfidenceAutoDowngrade:
    """Test that confidence is auto-downgraded based on validation."""

    def test_empty_state_low_confidence(self):
        """Empty protocol state yields low confidence."""
        state = ProtocolState()
        conf = state.compute_evidence_confidence()

        assert conf < 0.2

    def test_partial_validation_medium_confidence(self):
        """Partial validation yields medium confidence."""
        state = ProtocolState(
            baseline_comparison_done=True,
            baseline_zscore=2.5,
            hypotheses_registered=1,
        )
        conf = state.compute_evidence_confidence()

        assert 0.2 <= conf < 0.7

    def test_full_validation_high_confidence(self):
        """Complete validation yields high confidence."""
        state = ProtocolState(
            phase0_corpus_queried=True,
            phase0_graph_count=10,
            baseline_comparison_done=True,
            baseline_zscore=4.0,
            dose_response_done=True,
            dose_response_monotonic=True,
            dose_response_kendall_tau=0.9,
            relp_runs=5,
            relp_positive_control=True,
            relp_negative_control=True,
            hypotheses_registered=3,
            hypotheses_updated=3,
        )
        conf = state.compute_evidence_confidence()

        assert conf >= 0.9

    def test_high_zscore_contributes_more(self):
        """Higher z-score contributes more to confidence."""
        state_low = ProtocolState(baseline_comparison_done=True, baseline_zscore=2.0)
        state_high = ProtocolState(baseline_comparison_done=True, baseline_zscore=4.0)

        assert state_high.compute_evidence_confidence() > state_low.compute_evidence_confidence()

    def test_monotonic_response_contributes_more(self):
        """Monotonic dose-response contributes more than non-monotonic."""
        state_non = ProtocolState(dose_response_done=True, dose_response_monotonic=False)
        state_mono = ProtocolState(
            dose_response_done=True,
            dose_response_monotonic=True,
            dose_response_kendall_tau=0.8,
        )

        assert state_mono.compute_evidence_confidence() > state_non.compute_evidence_confidence()


class TestProtocolStateUpdates:
    """Test that protocol state is properly updated during investigation."""

    def test_init_resets_state(self):
        """init_protocol_state creates fresh state."""
        # Set some state
        update_protocol_state(baseline_comparison_done=True)

        # Reset
        init_protocol_state()
        state = get_protocol_state()

        assert state.baseline_comparison_done is False

    def test_update_modifies_state(self):
        """update_protocol_state modifies existing state."""
        init_protocol_state()

        update_protocol_state(
            baseline_comparison_done=True,
            baseline_zscore=3.0,
        )

        state = get_protocol_state()
        assert state.baseline_comparison_done is True
        assert state.baseline_zscore == 3.0

    def test_update_is_cumulative(self):
        """Multiple updates are cumulative."""
        init_protocol_state()

        update_protocol_state(baseline_comparison_done=True)
        update_protocol_state(dose_response_done=True)
        update_protocol_state(relp_runs=5)

        state = get_protocol_state()
        assert state.baseline_comparison_done is True
        assert state.dose_response_done is True
        assert state.relp_runs == 5


class TestHardGateRequirements:
    """Test hard gate requirements for different confidence levels."""

    def test_high_confidence_requires_strong_zscore(self):
        """High confidence (0.8+) requires z-score >= 3."""
        # Strong z-score allows high confidence
        state_strong = ProtocolState(
            baseline_comparison_done=True,
            baseline_zscore=3.5,
            dose_response_done=True,
            dose_response_monotonic=True,
            dose_response_kendall_tau=0.8,
            relp_runs=5,
            relp_positive_control=True,
            relp_negative_control=True,
            hypotheses_registered=2,
            hypotheses_updated=2,
            phase0_corpus_queried=True,
            phase0_graph_count=10,
        )

        # Weak z-score caps confidence
        state_weak = ProtocolState(
            baseline_comparison_done=True,
            baseline_zscore=1.5,
            dose_response_done=True,
            dose_response_monotonic=True,
            relp_runs=5,
            relp_positive_control=True,
            relp_negative_control=True,
            hypotheses_registered=2,
            hypotheses_updated=2,
            phase0_corpus_queried=True,
            phase0_graph_count=10,
        )

        assert state_strong.compute_evidence_confidence() > state_weak.compute_evidence_confidence()

    def test_medium_confidence_requires_preregistration(self):
        """Medium confidence (0.5+) benefits from pre-registration."""
        state_with_prereg = ProtocolState(
            baseline_comparison_done=True,
            baseline_zscore=2.5,
            hypotheses_registered=2,
            hypotheses_updated=2,
        )

        state_without_prereg = ProtocolState(
            baseline_comparison_done=True,
            baseline_zscore=2.5,
            hypotheses_registered=0,
        )

        assert state_with_prereg.compute_evidence_confidence() > state_without_prereg.compute_evidence_confidence()

    def test_relp_controls_contribute_to_confidence(self):
        """RelP positive and negative controls contribute to confidence."""
        state_no_controls = ProtocolState(relp_runs=5)
        state_positive_only = ProtocolState(relp_runs=5, relp_positive_control=True)
        state_both = ProtocolState(relp_runs=5, relp_positive_control=True, relp_negative_control=True)

        conf_no = state_no_controls.compute_evidence_confidence()
        conf_pos = state_positive_only.compute_evidence_confidence()
        conf_both = state_both.compute_evidence_confidence()

        assert conf_pos > conf_no
        assert conf_both > conf_pos


class TestInvestigationValidation:
    """Test validation of investigation output before saving."""

    def test_investigation_includes_protocol_validation(self):
        """Investigation to_dict can include protocol validation info."""
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
        )
        d = inv.to_dict()

        # Protocol validation can be added to output
        d["protocol_validation"] = {
            "baseline_comparison_done": True,
            "baseline_zscore": 3.0,
            "dose_response_done": True,
            "relp_runs": 5,
        }

        # Should be serializable
        import json
        json_str = json.dumps(d)
        assert "protocol_validation" in json_str

    def test_confidence_consistency_with_evidence(self):
        """Investigation confidence should be consistent with evidence quality."""
        # Investigation with high claimed confidence
        inv_high = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            confidence=0.9,
        )

        # Protocol state should support high confidence
        state_for_high = ProtocolState(
            baseline_comparison_done=True,
            baseline_zscore=3.5,
            dose_response_done=True,
            dose_response_monotonic=True,
            relp_runs=5,
            relp_positive_control=True,
            hypotheses_registered=2,
            phase0_corpus_queried=True,
            phase0_graph_count=10,
        )

        evidence_confidence = state_for_high.compute_evidence_confidence()

        # Evidence should support claimed confidence
        assert evidence_confidence >= 0.7  # Some threshold
