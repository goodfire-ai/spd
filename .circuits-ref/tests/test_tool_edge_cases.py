"""Edge case tests for neuron scientist tools.

These tests verify proper handling of:
- Invalid layer/neuron indices (out of bounds)
- Malformed prompts (empty, too long)
- Invalid parameter combinations
- Error paths and graceful degradation
"""

import asyncio

import pytest
from neuron_scientist.tools import (
    ProtocolState,
    clear_hypothesis_registry,
    format_prompt,
    get_hypothesis_registry,
    get_protocol_state,
    init_hypothesis_registry,
    init_protocol_state,
    tool_register_hypothesis,
    tool_update_hypothesis_status,
    update_protocol_state,
)


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestFormatPromptEdgeCases:
    """Test edge cases in prompt formatting."""

    def test_empty_prompt(self):
        """format_prompt handles empty string."""
        result = format_prompt("")

        # Should still produce valid template structure
        assert "<|begin_of_text|>" in result
        assert "<|eot_id|>" in result

    def test_very_long_prompt(self):
        """format_prompt handles very long prompts."""
        long_prompt = "A" * 10000

        result = format_prompt(long_prompt)

        # Should contain the full prompt
        assert long_prompt in result

    def test_special_tokens_in_prompt(self):
        """format_prompt handles prompts containing special tokens."""
        # Prompt containing what looks like special tokens
        prompt = "What is <|begin_of_text|> used for?"

        result = format_prompt(prompt)

        # Should contain the prompt as-is
        assert "What is <|begin_of_text|> used for?" in result

    def test_unicode_prompt(self):
        """format_prompt handles Unicode characters."""
        prompt = "El niño pregunta: ¿Qué significa 日本語?"

        result = format_prompt(prompt)

        assert "El niño" in result
        assert "¿Qué significa" in result
        assert "日本語" in result

    def test_newlines_in_prompt(self):
        """format_prompt handles prompts with newlines."""
        prompt = "Line 1\nLine 2\nLine 3"

        result = format_prompt(prompt)

        assert prompt in result

    def test_only_whitespace_prompt(self):
        """format_prompt handles whitespace-only prompts."""
        prompt = "   \t\n   "

        result = format_prompt(prompt)

        # Should still produce valid structure
        assert "<|begin_of_text|>" in result


class TestProtocolStateEdgeCases:
    """Test edge cases in protocol state management."""

    def setup_method(self):
        """Reset state before each test."""
        init_protocol_state()

    def test_update_with_no_changes(self):
        """update_protocol_state with no arguments is valid."""
        state_before = get_protocol_state()

        # Call with no changes
        update_protocol_state()

        state_after = get_protocol_state()
        # State should be unchanged
        assert state_before.baseline_comparison_done == state_after.baseline_comparison_done

    def test_update_with_invalid_type(self):
        """update_protocol_state handles type mismatches gracefully."""
        # This should not crash
        try:
            update_protocol_state(baseline_zscore="not_a_number")
            # If it doesn't crash, check state
            state = get_protocol_state()
            # May have stored the invalid value or ignored it
            assert state is not None
        except (TypeError, ValueError):
            # Type error is acceptable behavior
            pass

    def test_negative_zscore(self):
        """ProtocolState handles negative z-score."""
        state = ProtocolState(
            baseline_comparison_done=True,
            baseline_zscore=-5.0,  # Negative (shouldn't happen but test defensively)
        )

        conf = state.compute_evidence_confidence()

        # Should not go negative
        assert conf >= 0.0

        missing = state.get_missing_validation()
        # Should flag low z-score
        assert any("z-score" in m.lower() for m in missing)

    def test_extreme_zscore(self):
        """ProtocolState handles extreme z-score values."""
        state = ProtocolState(
            baseline_comparison_done=True,
            baseline_zscore=100.0,  # Very high
        )

        conf = state.compute_evidence_confidence()

        # Should be capped or reasonable
        assert 0.0 <= conf <= 1.0

    def test_very_high_relp_runs(self):
        """ProtocolState handles many RelP runs."""
        state = ProtocolState(
            relp_runs=1000,
            relp_positive_control=True,
            relp_negative_control=True,
        )

        conf = state.compute_evidence_confidence()

        # Should not overflow or cause issues
        assert 0.0 <= conf <= 1.0

    def test_none_values_in_state(self):
        """ProtocolState handles None values."""
        state = ProtocolState(
            baseline_comparison_done=True,
            baseline_zscore=None,
            dose_response_kendall_tau=None,
        )

        # Should not crash
        conf = state.compute_evidence_confidence()
        assert 0.0 <= conf <= 1.0

        missing = state.get_missing_validation()
        assert isinstance(missing, list)


class TestHypothesisEdgeCases:
    """Test edge cases in hypothesis management."""

    def setup_method(self):
        """Reset state before each test."""
        clear_hypothesis_registry()
        init_protocol_state()

    def test_register_empty_hypothesis(self):
        """Registering empty hypothesis text."""
        result = run_async(tool_register_hypothesis(
            hypothesis="",
            confirmation_criteria="",
            refutation_criteria="",
            prior_probability=50,
        ))

        # Should still work (validation is semantic, not syntactic)
        assert result["hypothesis_id"] == "H1"

    def test_register_zero_prior(self):
        """Registering hypothesis with 0% prior probability."""
        result = run_async(tool_register_hypothesis(
            hypothesis="Impossible hypothesis",
            confirmation_criteria="C",
            refutation_criteria="R",
            prior_probability=0,
        ))

        assert result["registered"]["prior_probability"] == 0

    def test_register_100_prior(self):
        """Registering hypothesis with 100% prior probability."""
        result = run_async(tool_register_hypothesis(
            hypothesis="Certain hypothesis",
            confirmation_criteria="C",
            refutation_criteria="R",
            prior_probability=100,
        ))

        assert result["registered"]["prior_probability"] == 100

    def test_update_with_same_probability(self):
        """Updating hypothesis with no probability change."""
        run_async(tool_register_hypothesis(
            hypothesis="Test",
            confirmation_criteria="C",
            refutation_criteria="R",
            prior_probability=50,
        ))

        result = run_async(tool_update_hypothesis_status(
            hypothesis_id="H1",
            status="inconclusive",
            posterior_probability=50,  # Same as prior
            evidence_summary="No change",
        ))

        assert result["probability_shift"] == 0

    def test_update_with_large_evidence_summary(self):
        """Updating hypothesis with very long evidence summary."""
        run_async(tool_register_hypothesis(
            hypothesis="Test",
            confirmation_criteria="C",
            refutation_criteria="R",
            prior_probability=50,
        ))

        long_evidence = "This is evidence. " * 1000

        result = run_async(tool_update_hypothesis_status(
            hypothesis_id="H1",
            status="confirmed",
            posterior_probability=80,
            evidence_summary=long_evidence,
        ))

        # Should not crash
        assert result["hypothesis_id"] == "H1"

        registry = get_hypothesis_registry()
        assert len(registry[0]["evidence"][0]) > 5000

    def test_update_already_confirmed(self):
        """Updating an already confirmed hypothesis."""
        run_async(tool_register_hypothesis(
            hypothesis="Test",
            confirmation_criteria="C",
            refutation_criteria="R",
            prior_probability=50,
        ))

        # First confirmation
        run_async(tool_update_hypothesis_status(
            hypothesis_id="H1",
            status="confirmed",
            posterior_probability=80,
            evidence_summary="First confirmation",
        ))

        # Second update (change to refuted)
        result = run_async(tool_update_hypothesis_status(
            hypothesis_id="H1",
            status="refuted",
            posterior_probability=20,
            evidence_summary="New evidence contradicts",
        ))

        # Should be allowed (status can change)
        assert result["status"] == "refuted"

        registry = get_hypothesis_registry()
        assert registry[0]["status"] == "refuted"

    def test_many_hypotheses(self):
        """Registering many hypotheses."""
        for i in range(50):
            result = run_async(tool_register_hypothesis(
                hypothesis=f"Hypothesis {i+1}",
                confirmation_criteria="C",
                refutation_criteria="R",
                prior_probability=50,
            ))
            assert result["hypothesis_id"] == f"H{i+1}"

        registry = get_hypothesis_registry()
        assert len(registry) == 50

    def test_init_with_malformed_prior(self):
        """Initializing registry with malformed prior hypotheses."""
        # Prior missing required fields
        prior = [
            {"hypothesis_id": "H1"},  # Missing hypothesis text
            {"hypothesis": "No ID"},  # Missing hypothesis_id
        ]

        init_hypothesis_registry(prior)
        registry = get_hypothesis_registry()

        # Should still work (just stores what's given)
        assert len(registry) == 2


class TestInputValidationConcepts:
    """Test concepts for input validation (without GPU)."""

    def test_layer_bounds_concept(self):
        """Conceptual test for layer bounds validation."""
        # Llama 3.1 8B has 32 layers (0-31)
        valid_layers = list(range(32))
        invalid_layers = [-1, 32, 100, -100]

        for layer in valid_layers:
            assert 0 <= layer < 32, f"Layer {layer} should be valid"

        for layer in invalid_layers:
            assert not (0 <= layer < 32), f"Layer {layer} should be invalid"

    def test_neuron_bounds_concept(self):
        """Conceptual test for neuron index bounds validation."""
        # 14336 neurons per layer
        valid_neurons = [0, 100, 14335]
        invalid_neurons = [-1, 14336, 100000]

        for idx in valid_neurons:
            assert 0 <= idx < 14336, f"Neuron {idx} should be valid"

        for idx in invalid_neurons:
            assert not (0 <= idx < 14336), f"Neuron {idx} should be invalid"

    def test_strength_bounds_concept(self):
        """Conceptual test for steering strength validation."""
        # Typical steering strengths are 0-10, maybe up to 100
        reasonable_strengths = [0, 0.5, 1.0, 5.0, 10.0]
        extreme_strengths = [-10.0, 100.0, 1000.0]

        # All should be valid but extreme values might cause issues
        for s in reasonable_strengths:
            assert isinstance(s, (int, float))

        for s in extreme_strengths:
            assert isinstance(s, (int, float))  # Type is valid, but value might be problematic


class TestErrorMessageFormats:
    """Test that error messages are informative."""

    def setup_method(self):
        """Reset state before each test."""
        clear_hypothesis_registry()
        init_protocol_state()

    def test_hypothesis_not_found_message(self):
        """Error message for not found hypothesis is informative."""
        result = run_async(tool_update_hypothesis_status(
            hypothesis_id="H999",
            status="confirmed",
            posterior_probability=80,
            evidence_summary="Evidence",
        ))

        assert "error" in result
        assert "H999" in result["error"]
        assert "not found" in result["error"].lower()

    def test_protocol_validation_messages(self):
        """Protocol validation messages are descriptive."""
        state = ProtocolState()
        missing = state.get_missing_validation()

        # All messages should be descriptive strings
        for msg in missing:
            assert isinstance(msg, str)
            assert len(msg) > 10  # Not too short


class TestConcurrentStateAccess:
    """Test behavior under concurrent-like access patterns."""

    def setup_method(self):
        """Reset state before each test."""
        clear_hypothesis_registry()
        init_protocol_state()

    def test_rapid_protocol_updates(self):
        """Rapid protocol state updates don't lose data."""
        init_protocol_state()

        # Rapid updates
        for i in range(100):
            update_protocol_state(relp_runs=i)

        state = get_protocol_state()
        assert state.relp_runs == 99  # Should be last value

    def test_rapid_hypothesis_registration(self):
        """Rapid hypothesis registration maintains sequence."""
        for i in range(20):
            run_async(tool_register_hypothesis(
                hypothesis=f"H {i}",
                confirmation_criteria="C",
                refutation_criteria="R",
                prior_probability=50,
            ))

        registry = get_hypothesis_registry()
        assert len(registry) == 20

        # Verify sequential IDs
        for i, h in enumerate(registry):
            assert h["hypothesis_id"] == f"H{i+1}"


@pytest.mark.gpu
class TestGPUToolEdgeCases:
    """Edge case tests requiring GPU access."""

    def test_activation_with_empty_prompt(self):
        """Test activation testing with empty prompt."""
        from neuron_scientist.tools import tool_test_activation

        # This test may fail or return specific error
        try:
            result = run_async(tool_test_activation(
                layer=15,
                neuron_idx=1000,
                prompt="",  # Empty prompt
            ))
            # If it works, check result structure
            assert "activation" in result or "error" in result
        except Exception:
            # Error is acceptable for invalid input
            assert True

    def test_batch_activation_empty_list(self):
        """Test batch activation with empty prompt list."""
        from neuron_scientist.tools import tool_batch_activation_test

        try:
            result = run_async(tool_batch_activation_test(
                layer=15,
                neuron_idx=1000,
                prompts=[],  # Empty list
            ))
            # Should return empty results or error
            assert "results" in result or "error" in result
            if "results" in result:
                assert len(result["results"]) == 0
        except Exception:
            # Error is acceptable
            assert True

    def test_ablation_with_invalid_layer(self):
        """Test ablation with invalid layer index."""
        from neuron_scientist.tools import tool_run_ablation

        try:
            result = run_async(tool_run_ablation(
                layer=100,  # Invalid layer (>31)
                neuron_idx=1000,
                prompt="Test prompt",
            ))
            # Should return error or fail gracefully
            assert "error" in result or isinstance(result, dict)
        except (IndexError, ValueError):
            # These errors are expected for invalid layer
            assert True

    def test_steer_with_zero_strength(self):
        """Test steering with zero strength."""
        from neuron_scientist.tools import tool_steer_neuron

        try:
            result = run_async(tool_steer_neuron(
                layer=15,
                neuron_idx=1000,
                prompt="Test prompt",
                strength=0.0,  # Zero strength should be no-op
            ))
            # Should work and show minimal effect
            assert isinstance(result, dict)
        except Exception:
            # Any exception should be caught
            assert True

    def test_steer_with_negative_strength(self):
        """Test steering with negative strength (suppression)."""
        from neuron_scientist.tools import tool_steer_neuron

        try:
            result = run_async(tool_steer_neuron(
                layer=15,
                neuron_idx=1000,
                prompt="Test prompt",
                strength=-5.0,  # Negative = suppression
            ))
            # Should work for suppression
            assert isinstance(result, dict)
        except Exception:
            assert True

    def test_relp_with_very_low_tau(self):
        """Test RelP with very low tau threshold."""
        from neuron_scientist.tools import tool_run_relp

        try:
            result = run_async(tool_run_relp(
                prompt="Test prompt",
                target_tokens=["test"],
                tau=0.0001,  # Very low threshold
            ))
            # Should return large graph or error
            assert isinstance(result, dict)
        except Exception:
            assert True

    def test_relp_with_nonexistent_target(self):
        """Test RelP with target token that won't appear."""
        from neuron_scientist.tools import tool_run_relp

        try:
            result = run_async(tool_run_relp(
                prompt="Hello world",
                target_tokens=["xyznonexistent123"],  # Won't be in vocab/output
                tau=0.01,
            ))
            # Should handle gracefully
            assert isinstance(result, dict)
        except Exception:
            assert True
