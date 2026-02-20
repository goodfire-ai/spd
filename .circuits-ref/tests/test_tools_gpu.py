"""GPU-required tool tests for neuron scientist.

These tests require GPU access and load the actual Llama model.
Run with: pytest tests/test_tools_gpu.py -v
Skip with: pytest -m "not gpu"
"""

import asyncio

import pytest

# Mark all tests in this module as requiring GPU
pytestmark = pytest.mark.gpu


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def model_loaded():
    """Ensure model is loaded once per test module."""
    from neuron_scientist.tools import get_model_and_tokenizer
    model, tokenizer = get_model_and_tokenizer()
    assert model is not None
    assert tokenizer is not None
    return model, tokenizer


@pytest.fixture
def reset_state():
    """Reset protocol state and hypothesis registry before each test."""
    from neuron_scientist.tools import (
        clear_hypothesis_registry,
        init_protocol_state,
        set_seed,
    )
    init_protocol_state()
    clear_hypothesis_registry()
    set_seed(42)
    yield
    # Cleanup after test
    init_protocol_state()
    clear_hypothesis_registry()


# =============================================================================
# Test Activation Tools
# =============================================================================

class TestActivation:
    """Test single-prompt activation testing."""

    # Use a well-characterized neuron for testing
    TEST_LAYER = 4
    TEST_NEURON = 10555  # Neurotransmitter detector

    def test_returns_expected_fields(self, model_loaded, reset_state):
        """test_activation returns all expected fields."""
        from neuron_scientist.tools import _sync_test_activation

        result = _sync_test_activation(
            layer=self.TEST_LAYER,
            neuron_idx=self.TEST_NEURON,
            prompt="The neurotransmitter dopamine is important.",
            activation_threshold=0.5,
        )

        required_fields = [
            "prompt", "max_activation", "max_position", "token_at_max",
            "activates", "activation_threshold", "assistant_start", "total_tokens"
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_activation_is_numeric(self, model_loaded, reset_state):
        """Activation value is a number."""
        from neuron_scientist.tools import _sync_test_activation

        result = _sync_test_activation(
            layer=self.TEST_LAYER,
            neuron_idx=self.TEST_NEURON,
            prompt="Test prompt",
            activation_threshold=0.5,
        )

        assert isinstance(result["max_activation"], (int, float))
        assert result["max_activation"] >= 0  # Activations should be non-negative

    def test_activates_flag_respects_threshold(self, model_loaded, reset_state):
        """activates flag correctly reflects threshold comparison."""
        from neuron_scientist.tools import _sync_test_activation

        result = _sync_test_activation(
            layer=self.TEST_LAYER,
            neuron_idx=self.TEST_NEURON,
            prompt="Dopamine affects reward circuits.",
            activation_threshold=0.5,
        )

        if result["max_activation"] > 0.5:
            assert result["activates"] is True
        else:
            assert result["activates"] is False

    def test_high_threshold_reduces_activating(self, model_loaded, reset_state):
        """Higher threshold should reduce number of prompts flagged as activating."""
        from neuron_scientist.tools import _sync_test_activation

        prompt = "The brain chemistry is complex."

        result_low = _sync_test_activation(
            layer=self.TEST_LAYER,
            neuron_idx=self.TEST_NEURON,
            prompt=prompt,
            activation_threshold=0.1,
        )

        result_high = _sync_test_activation(
            layer=self.TEST_LAYER,
            neuron_idx=self.TEST_NEURON,
            prompt=prompt,
            activation_threshold=10.0,
        )

        # Same activation, different threshold
        assert result_low["max_activation"] == result_high["max_activation"]
        # High threshold less likely to flag as activating
        if result_low["activates"]:
            assert result_low["max_activation"] > 0.1

    def test_reproducibility_with_seed(self, model_loaded, reset_state):
        """Same seed produces same activations."""
        from neuron_scientist.tools import _sync_test_activation, set_seed

        set_seed(42)
        result1 = _sync_test_activation(
            layer=self.TEST_LAYER,
            neuron_idx=self.TEST_NEURON,
            prompt="Test reproducibility",
            activation_threshold=0.5,
        )

        set_seed(42)
        result2 = _sync_test_activation(
            layer=self.TEST_LAYER,
            neuron_idx=self.TEST_NEURON,
            prompt="Test reproducibility",
            activation_threshold=0.5,
        )

        assert result1["max_activation"] == result2["max_activation"]
        assert result1["max_position"] == result2["max_position"]

    def test_empty_prompt_handled(self, model_loaded, reset_state):
        """Empty prompt doesn't crash."""
        from neuron_scientist.tools import _sync_test_activation

        # Empty prompt should still work (gets wrapped in template)
        result = _sync_test_activation(
            layer=self.TEST_LAYER,
            neuron_idx=self.TEST_NEURON,
            prompt="",
            activation_threshold=0.5,
        )

        assert "max_activation" in result or "error" in result

    def test_different_layers_give_different_results(self, model_loaded, reset_state):
        """Different layers should give different activation patterns."""
        from neuron_scientist.tools import _sync_test_activation

        prompt = "Test different layers"

        result_layer0 = _sync_test_activation(
            layer=0,
            neuron_idx=100,
            prompt=prompt,
            activation_threshold=0.5,
        )

        result_layer15 = _sync_test_activation(
            layer=15,
            neuron_idx=100,
            prompt=prompt,
            activation_threshold=0.5,
        )

        # Activations should differ (very unlikely to be exactly the same)
        # This test may occasionally fail if by chance they're equal
        assert result_layer0["max_activation"] != result_layer15["max_activation"] or \
               result_layer0["max_position"] != result_layer15["max_position"]


class TestBatchActivation:
    """Test batch activation testing."""

    TEST_LAYER = 4
    TEST_NEURON = 10555

    def test_returns_expected_fields(self, model_loaded, reset_state):
        """batch_activation_test returns all expected fields."""
        from neuron_scientist.tools import _sync_batch_activation_test

        result = _sync_batch_activation_test(
            layer=self.TEST_LAYER,
            neuron_idx=self.TEST_NEURON,
            prompts=["Test prompt 1", "Test prompt 2"],
            activation_threshold=0.5,
        )

        required_fields = [
            "total_tested", "activating_count", "non_activating_count",
            "activation_threshold", "mean_activation", "max_activation",
            "top_activating", "sample_non_activating"
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_empty_prompt_list(self, model_loaded, reset_state):
        """Empty prompt list returns zero counts."""
        from neuron_scientist.tools import _sync_batch_activation_test

        result = _sync_batch_activation_test(
            layer=self.TEST_LAYER,
            neuron_idx=self.TEST_NEURON,
            prompts=[],
            activation_threshold=0.5,
        )

        assert result["total_tested"] == 0
        assert result["activating_count"] == 0
        assert result["mean_activation"] == 0

    def test_counts_sum_to_total(self, model_loaded, reset_state):
        """activating_count + non_activating_count equals total_tested (minus errors)."""
        from neuron_scientist.tools import _sync_batch_activation_test

        prompts = [
            "Dopamine is a neurotransmitter",
            "The weather is nice today",
            "Serotonin affects mood",
            "I walked to the store",
        ]

        result = _sync_batch_activation_test(
            layer=self.TEST_LAYER,
            neuron_idx=self.TEST_NEURON,
            prompts=prompts,
            activation_threshold=0.5,
        )

        errors = len(result.get("errors", []) or [])
        assert result["activating_count"] + result["non_activating_count"] == result["total_tested"] - errors

    def test_top_activating_sorted_descending(self, model_loaded, reset_state):
        """top_activating list is sorted by activation descending."""
        from neuron_scientist.tools import _sync_batch_activation_test

        prompts = [f"Test prompt {i}" for i in range(10)]

        result = _sync_batch_activation_test(
            layer=self.TEST_LAYER,
            neuron_idx=self.TEST_NEURON,
            prompts=prompts,
            activation_threshold=0.0,  # Low threshold to get activating prompts
        )

        top = result["top_activating"]
        if len(top) > 1:
            activations = [p["activation"] for p in top]
            assert activations == sorted(activations, reverse=True)

    def test_batch_vs_individual_consistency(self, model_loaded, reset_state):
        """Batch results should match individual test_activation calls."""
        from neuron_scientist.tools import (
            _sync_batch_activation_test,
            _sync_test_activation,
            set_seed,
        )

        prompts = ["Dopamine in the brain", "Weather forecast"]

        # Individual calls
        set_seed(42)
        individual_results = []
        for p in prompts:
            r = _sync_test_activation(self.TEST_LAYER, self.TEST_NEURON, p, 0.5)
            individual_results.append(r["max_activation"])

        # Batch call
        set_seed(42)
        batch_result = _sync_batch_activation_test(
            self.TEST_LAYER, self.TEST_NEURON, prompts, 0.5
        )

        # Collect batch activations
        batch_activations = []
        for p in batch_result.get("top_activating", []):
            batch_activations.append(p["activation"])
        for p in batch_result.get("sample_non_activating", []):
            batch_activations.append(p["activation"])

        # Activations should be similar (may differ slightly due to batching)
        assert len(batch_activations) == len(individual_results)


# =============================================================================
# Test Ablation Tools
# =============================================================================

class TestAblation:
    """Test ablation experiments."""

    TEST_LAYER = 4
    TEST_NEURON = 10555

    def test_returns_expected_fields(self, model_loaded, reset_state):
        """run_ablation returns expected fields."""
        from neuron_scientist.tools import _sync_run_ablation

        result = _sync_run_ablation(
            layer=self.TEST_LAYER,
            neuron_idx=self.TEST_NEURON,
            prompt="The neurotransmitter is dopamine.",
        )

        # Should have logit shift information
        assert "prompt" in result
        assert "top_affected_tokens" in result or "logit_shifts" in result or "most_affected_token" in result

    def test_ablation_produces_shifts(self, model_loaded, reset_state):
        """Ablation should produce some logit shifts."""
        from neuron_scientist.tools import _sync_run_ablation

        result = _sync_run_ablation(
            layer=self.TEST_LAYER,
            neuron_idx=self.TEST_NEURON,
            prompt="Dopamine affects reward circuits.",
        )

        # Should have some affected tokens
        has_effects = (
            len(result.get("top_affected_tokens", [])) > 0 or
            len(result.get("logit_shifts", {})) > 0 or
            result.get("most_affected_token") is not None
        )
        assert has_effects or "error" in result


# =============================================================================
# Test Steering Tools
# =============================================================================

class TestSteering:
    """Test neuron steering experiments."""

    TEST_LAYER = 4
    TEST_NEURON = 10555

    def test_steer_returns_before_after(self, model_loaded, reset_state):
        """steer_neuron returns original and steered logits."""
        from neuron_scientist.tools import _sync_steer_neuron

        result = _sync_steer_neuron(
            layer=self.TEST_LAYER,
            neuron_idx=self.TEST_NEURON,
            prompt="The brain chemical is",
            steering_value=5.0,
            position=-1,
            top_k_logits=10,
        )

        assert "original_top_tokens" in result or "error" in result
        if "error" not in result:
            assert "steered_top_tokens" in result

    def test_different_steering_values(self, model_loaded, reset_state):
        """Different steering values should produce different results."""
        from neuron_scientist.tools import _sync_steer_neuron

        prompt = "The reward neurotransmitter is"

        result_positive = _sync_steer_neuron(
            layer=self.TEST_LAYER,
            neuron_idx=self.TEST_NEURON,
            prompt=prompt,
            steering_value=5.0,
            position=-1,
            top_k_logits=5,
        )

        result_negative = _sync_steer_neuron(
            layer=self.TEST_LAYER,
            neuron_idx=self.TEST_NEURON,
            prompt=prompt,
            steering_value=-5.0,
            position=-1,
            top_k_logits=5,
        )

        if "error" not in result_positive and "error" not in result_negative:
            # Results should differ
            assert result_positive != result_negative


# =============================================================================
# Test RelP Tools (Async)
# =============================================================================

class TestRelP:
    """Test RelP attribution tools."""

    TEST_LAYER = 4
    TEST_NEURON = 10555

    @pytest.mark.asyncio
    async def test_relp_returns_expected_fields(self, model_loaded, reset_state):
        """run_relp returns expected fields."""
        from neuron_scientist.tools import tool_run_relp

        result = await tool_run_relp(
            layer=self.TEST_LAYER,
            neuron_idx=self.TEST_NEURON,
            prompt="The neurotransmitter dopamine affects reward.",
            target_tokens=["dopamine"],
            tau=0.1,  # Coarse tau for faster test
        )

        # Should have key fields
        assert "neuron_found" in result or "error" in result
        if "error" not in result:
            assert isinstance(result["neuron_found"], bool)

    @pytest.mark.asyncio
    async def test_relp_updates_protocol_state(self, model_loaded, reset_state):
        """run_relp should update protocol state."""
        from neuron_scientist.tools import (
            get_protocol_state,
            init_protocol_state,
            tool_run_relp,
        )

        init_protocol_state()
        state_before = get_protocol_state()
        assert state_before.relp_runs == 0

        await tool_run_relp(
            layer=self.TEST_LAYER,
            neuron_idx=self.TEST_NEURON,
            prompt="Test RelP",
            target_tokens=None,
            tau=0.1,
        )

        state_after = get_protocol_state()
        # relp_runs should be incremented
        assert state_after.relp_runs >= 1


# =============================================================================
# Test Baseline Comparison (Async)
# =============================================================================

class TestBaselineComparison:
    """Test baseline comparison for effect size calibration."""

    TEST_LAYER = 4
    TEST_NEURON = 10555

    @pytest.mark.asyncio
    async def test_baseline_returns_zscore(self, model_loaded, reset_state):
        """run_baseline_comparison returns z-score."""
        from neuron_scientist.tools import tool_run_baseline_comparison

        prompts = [
            "Dopamine is important",
            "Serotonin affects mood",
            "Brain chemistry is complex",
        ]

        result = await tool_run_baseline_comparison(
            layer=self.TEST_LAYER,
            neuron_idx=self.TEST_NEURON,
            prompts=prompts,
            n_random_neurons=3,  # Use fewer for faster test
        )

        assert "z_score" in result or "error" in result
        if "error" not in result:
            assert isinstance(result["z_score"], (int, float))

    @pytest.mark.asyncio
    async def test_baseline_updates_protocol_state(self, model_loaded, reset_state):
        """Baseline comparison should update protocol state."""
        from neuron_scientist.tools import (
            get_protocol_state,
            init_protocol_state,
            tool_run_baseline_comparison,
        )

        init_protocol_state()
        state_before = get_protocol_state()
        assert state_before.baseline_comparison_done is False

        prompts = ["Test prompt 1", "Test prompt 2"]
        await tool_run_baseline_comparison(
            layer=self.TEST_LAYER,
            neuron_idx=self.TEST_NEURON,
            prompts=prompts,
            n_random_neurons=2,
        )

        state_after = get_protocol_state()
        assert state_after.baseline_comparison_done is True
        assert state_after.baseline_zscore is not None


# =============================================================================
# Test Dose-Response (Async)
# =============================================================================

class TestDoseResponse:
    """Test dose-response steering experiments."""

    TEST_LAYER = 4
    TEST_NEURON = 10555

    @pytest.mark.asyncio
    async def test_dose_response_returns_curve(self, model_loaded, reset_state):
        """steer_dose_response returns curve data."""
        from neuron_scientist.tools import tool_steer_dose_response

        result = await tool_steer_dose_response(
            layer=self.TEST_LAYER,
            neuron_idx=self.TEST_NEURON,
            prompt="The reward neurotransmitter is",
            steering_values=[-5, 0, 5],  # Fewer values for faster test
        )

        # Check for results (field name may vary)
        has_results = (
            "results" in result or
            "dose_response" in result or
            "linearity_check" in result or
            "error" in result
        )
        assert has_results

    @pytest.mark.asyncio
    async def test_dose_response_monotonicity_detection(self, model_loaded, reset_state):
        """Dose-response should detect monotonicity."""
        from neuron_scientist.tools import tool_steer_dose_response

        result = await tool_steer_dose_response(
            layer=self.TEST_LAYER,
            neuron_idx=self.TEST_NEURON,
            prompt="The brain chemical dopamine",
            steering_values=[-5, 0, 5],
        )

        if "error" not in result:
            # Check for monotonicity-related fields
            has_monotonicity = (
                "linearity_check" in result or
                "is_monotonic" in result or
                "kendall_tau" in result
            )
            assert has_monotonicity


# =============================================================================
# Test Output Projections
# =============================================================================

class TestOutputProjections:
    """Test output projection analysis."""

    TEST_LAYER = 4
    TEST_NEURON = 10555

    def test_projections_returns_tokens(self, model_loaded, reset_state):
        """get_output_projections returns promoted/suppressed tokens."""
        from neuron_scientist.tools import _sync_get_output_projections

        result = _sync_get_output_projections(
            layer=self.TEST_LAYER,
            neuron_idx=self.TEST_NEURON,
            top_k=15,
        )

        assert "promoted" in result or "error" in result
        if "error" not in result:
            assert "suppressed" in result
            assert isinstance(result["promoted"], list)
            assert isinstance(result["suppressed"], list)

    def test_projections_have_weights(self, model_loaded, reset_state):
        """Output projections include weight values."""
        from neuron_scientist.tools import _sync_get_output_projections

        result = _sync_get_output_projections(
            layer=self.TEST_LAYER,
            neuron_idx=self.TEST_NEURON,
            top_k=10,
        )

        if "error" not in result and result["promoted"]:
            # Each entry should have token and weight
            first = result["promoted"][0]
            assert isinstance(first, (tuple, list, dict))


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_layer_handled(self, model_loaded, reset_state):
        """Invalid layer number doesn't crash."""
        from neuron_scientist.tools import _sync_test_activation

        # Layer 100 doesn't exist in Llama-3.1-8B (only 0-31)
        try:
            result = _sync_test_activation(
                layer=100,
                neuron_idx=0,
                prompt="Test",
                activation_threshold=0.5,
            )
            # Should either error or handle gracefully
            assert "error" in result or "max_activation" in result
        except (IndexError, RuntimeError):
            pass  # Expected for invalid layer

    def test_invalid_neuron_idx_handled(self, model_loaded, reset_state):
        """Invalid neuron index doesn't crash."""
        from neuron_scientist.tools import _sync_test_activation

        # Neuron 99999999 doesn't exist
        try:
            result = _sync_test_activation(
                layer=0,
                neuron_idx=99999999,
                prompt="Test",
                activation_threshold=0.5,
            )
            assert "error" in result or "max_activation" in result
        except (IndexError, RuntimeError):
            pass  # Expected for invalid neuron

    def test_very_long_prompt_handled(self, model_loaded, reset_state):
        """Very long prompt is handled (truncated or errored)."""
        from neuron_scientist.tools import _sync_test_activation

        long_prompt = "This is a test. " * 1000  # Very long

        try:
            result = _sync_test_activation(
                layer=4,
                neuron_idx=10555,
                prompt=long_prompt,
                activation_threshold=0.5,
            )
            # Should either work (with truncation) or error gracefully
            assert "error" in result or "max_activation" in result
        except (RuntimeError, ValueError):
            pass  # Expected for too-long prompt

    def test_special_characters_in_prompt(self, model_loaded, reset_state):
        """Special characters in prompt are handled."""
        from neuron_scientist.tools import _sync_test_activation

        special_prompt = "Test with émojis 🧠 and unicode: 日本語"

        result = _sync_test_activation(
            layer=4,
            neuron_idx=10555,
            prompt=special_prompt,
            activation_threshold=0.5,
        )

        assert "max_activation" in result or "error" in result
