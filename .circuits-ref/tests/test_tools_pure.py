"""Pure function tests for neuron scientist tools.

These tests don't require GPU access and test the non-model-dependent
functionality of the tools module.
"""

import random

import numpy as np
import torch
from neuron_scientist.tools import (
    ProtocolState,
    clear_hypothesis_registry,
    format_prompt,
    get_hypothesis_registry,
    get_next_hypothesis_id,
    get_protocol_state,
    get_seed,
    init_hypothesis_registry,
    init_protocol_state,
    set_seed,
    update_protocol_state,
)


class TestFormatPrompt:
    """Test prompt formatting for Llama 3.1 chat template."""

    def test_wraps_in_chat_template(self):
        """format_prompt wraps in Llama 3.1 chat template."""
        result = format_prompt("Hello world")

        assert "<|begin_of_text|>" in result
        assert "<|start_header_id|>" in result
        assert "<|end_header_id|>" in result
        assert "<|eot_id|>" in result
        assert "Hello world" in result

    def test_includes_system_header(self):
        """Chat template includes system header."""
        result = format_prompt("Test")
        assert "system" in result

    def test_includes_user_header(self):
        """Chat template includes user header."""
        result = format_prompt("Test")
        assert "user" in result

    def test_includes_assistant_header(self):
        """Chat template includes assistant header for completion."""
        result = format_prompt("Test")
        assert "assistant" in result

    def test_preserves_prompt_content(self):
        """Original prompt content is preserved."""
        prompt = "What is the capital of France?"
        result = format_prompt(prompt)
        assert prompt in result


class TestSeedManagement:
    """Test reproducibility seed management."""

    def test_set_get_seed_roundtrip(self):
        """set_seed stores value retrievable by get_seed."""
        set_seed(123)
        assert get_seed() == 123

        set_seed(456)
        assert get_seed() == 456

    def test_seed_affects_random(self):
        """Seed affects Python random module."""
        set_seed(42)
        a = random.random()

        set_seed(42)
        b = random.random()

        assert a == b

    def test_seed_affects_numpy(self):
        """Seed affects NumPy random."""
        set_seed(42)
        a = np.random.rand()

        set_seed(42)
        b = np.random.rand()

        assert a == b

    def test_seed_affects_torch(self):
        """Seed affects PyTorch random."""
        set_seed(42)
        a = torch.rand(1).item()

        set_seed(42)
        b = torch.rand(1).item()

        assert a == b

    def test_different_seeds_different_results(self):
        """Different seeds produce different random results."""
        set_seed(42)
        a = random.random()

        set_seed(123)
        b = random.random()

        assert a != b


class TestProtocolState:
    """Test protocol state tracking."""

    def test_init_creates_fresh_state(self):
        """init_protocol_state creates new state object."""
        state = init_protocol_state()

        assert isinstance(state, ProtocolState)
        assert state.baseline_comparison_done is False
        assert state.dose_response_done is False
        assert state.relp_runs == 0
        assert state.hypotheses_registered == 0

    def test_get_protocol_state_returns_current(self):
        """get_protocol_state returns current state."""
        init_protocol_state()
        state = get_protocol_state()

        assert isinstance(state, ProtocolState)

    def test_update_protocol_state(self):
        """update_protocol_state modifies state."""
        init_protocol_state()
        update_protocol_state(baseline_comparison_done=True, baseline_zscore=3.0)

        state = get_protocol_state()
        assert state.baseline_comparison_done is True
        assert state.baseline_zscore == 3.0

    def test_get_missing_validation_fresh_state(self):
        """Fresh state has many missing validations."""
        state = ProtocolState()
        missing = state.get_missing_validation()

        assert isinstance(missing, list)
        assert len(missing) > 0
        assert any("baseline" in m.lower() for m in missing)
        assert any("dose-response" in m.lower() for m in missing)
        assert any("relp" in m.lower() for m in missing)

    def test_get_missing_validation_complete_state(self):
        """Complete state has no missing validations."""
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

    def test_low_zscore_flagged(self):
        """Low z-score is flagged as missing validation."""
        state = ProtocolState(
            baseline_comparison_done=True,
            baseline_zscore=1.5,  # Below threshold
        )
        missing = state.get_missing_validation()

        assert any("z-score" in m.lower() for m in missing)


class TestProtocolStateConfidence:
    """Test evidence-based confidence calculation."""

    def test_confidence_in_valid_range(self):
        """Confidence is between 0 and 1."""
        state = ProtocolState()
        conf = state.compute_evidence_confidence()

        assert 0.0 <= conf <= 1.0

    def test_empty_state_low_confidence(self):
        """Fresh state has low confidence."""
        state = ProtocolState()
        conf = state.compute_evidence_confidence()

        assert conf < 0.2

    def test_full_validation_high_confidence(self):
        """Complete validation yields high confidence."""
        state = ProtocolState(
            phase0_corpus_queried=True,
            phase0_graph_count=10,
            baseline_comparison_done=True,
            baseline_zscore=3.5,
            dose_response_done=True,
            dose_response_monotonic=True,
            dose_response_kendall_tau=0.8,
            relp_runs=5,
            relp_positive_control=True,
            relp_negative_control=True,
            hypotheses_registered=3,
            hypotheses_updated=3,
        )
        conf = state.compute_evidence_confidence()

        assert conf >= 0.9

    def test_high_zscore_gives_more_points(self):
        """Higher z-score contributes more to confidence."""
        state_low = ProtocolState(baseline_comparison_done=True, baseline_zscore=2.0)
        state_high = ProtocolState(baseline_comparison_done=True, baseline_zscore=3.5)

        conf_low = state_low.compute_evidence_confidence()
        conf_high = state_high.compute_evidence_confidence()

        assert conf_high > conf_low

    def test_dose_response_monotonic_bonus(self):
        """Monotonic dose-response gives bonus confidence."""
        state_non_mono = ProtocolState(dose_response_done=True, dose_response_monotonic=False)
        state_mono = ProtocolState(dose_response_done=True, dose_response_monotonic=True, dose_response_kendall_tau=0.7)

        conf_non_mono = state_non_mono.compute_evidence_confidence()
        conf_mono = state_mono.compute_evidence_confidence()

        assert conf_mono > conf_non_mono

    def test_relp_controls_contribute(self):
        """RelP positive and negative controls contribute to confidence."""
        state_no_controls = ProtocolState(relp_runs=3)
        state_positive = ProtocolState(relp_runs=3, relp_positive_control=True)
        state_both = ProtocolState(relp_runs=3, relp_positive_control=True, relp_negative_control=True)

        conf_no = state_no_controls.compute_evidence_confidence()
        conf_pos = state_positive.compute_evidence_confidence()
        conf_both = state_both.compute_evidence_confidence()

        assert conf_pos > conf_no
        assert conf_both > conf_pos


class TestHypothesisRegistry:
    """Test hypothesis pre-registration system."""

    def test_clear_hypothesis_registry(self):
        """clear_hypothesis_registry resets the registry."""
        clear_hypothesis_registry()
        registry = get_hypothesis_registry()

        assert isinstance(registry, list)
        assert len(registry) == 0

    def test_get_next_id_returns_h1_on_empty(self):
        """get_next_hypothesis_id returns H1 when registry is empty."""
        clear_hypothesis_registry()

        # get_next_hypothesis_id just predicts the next ID based on registry length
        # It doesn't modify the registry
        next_id = get_next_hypothesis_id()
        assert next_id == "H1"

    def test_get_next_id_based_on_registry_length(self):
        """get_next_hypothesis_id is based on registry length."""
        clear_hypothesis_registry()

        # Empty registry -> H1
        assert get_next_hypothesis_id() == "H1"

        # Add some hypotheses to the registry
        prior = [
            {"hypothesis_id": "H1", "hypothesis": "First"},
            {"hypothesis_id": "H2", "hypothesis": "Second"},
        ]
        init_hypothesis_registry(prior)

        # Now should be H3
        assert get_next_hypothesis_id() == "H3"

    def test_init_with_prior_hypotheses(self):
        """Can initialize registry with prior hypotheses."""
        clear_hypothesis_registry()

        prior = [
            {"hypothesis_id": "H1", "hypothesis": "First hypothesis"},
            {"hypothesis_id": "H2", "hypothesis": "Second hypothesis"},
        ]
        init_hypothesis_registry(prior)

        # Registry should contain the priors
        registry = get_hypothesis_registry()
        assert len(registry) == 2

        # Next ID should be H3
        next_id = get_next_hypothesis_id()
        assert next_id == "H3"

    def test_clear_after_init_resets_to_h1(self):
        """Clearing registry after init resets next ID to H1."""
        # Init with some hypotheses
        init_hypothesis_registry([
            {"hypothesis_id": "H1", "hypothesis": "Test"},
        ])

        # Clear
        clear_hypothesis_registry()

        # Should be back to H1
        assert get_next_hypothesis_id() == "H1"


class TestProtocolStateEdgeCases:
    """Test edge cases in protocol state."""

    def test_none_zscore_handling(self):
        """None z-score is handled correctly."""
        state = ProtocolState(
            baseline_comparison_done=True,
            baseline_zscore=None,
        )

        # Should not crash
        conf = state.compute_evidence_confidence()
        assert 0.0 <= conf <= 1.0

        missing = state.get_missing_validation()
        # Should not flag z-score issue when zscore is None (test hasn't been run properly)
        assert isinstance(missing, list)

    def test_negative_zscore(self):
        """Negative z-score is handled (shouldn't happen but be defensive)."""
        state = ProtocolState(
            baseline_comparison_done=True,
            baseline_zscore=-1.0,
        )

        conf = state.compute_evidence_confidence()
        assert conf >= 0.0  # Should not go negative

    def test_zero_relp_runs(self):
        """Zero RelP runs flagged appropriately."""
        state = ProtocolState(relp_runs=0)
        missing = state.get_missing_validation()

        assert any("relp" in m.lower() for m in missing)

    def test_partial_relp_controls(self):
        """Partial RelP controls handled correctly."""
        state = ProtocolState(
            relp_runs=3,
            relp_positive_control=True,
            relp_negative_control=False,
        )
        missing = state.get_missing_validation()

        # Should flag missing negative control
        assert any("negative control" in m.lower() for m in missing)
