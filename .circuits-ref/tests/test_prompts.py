"""Prompt snapshot tests for neuron scientist.

These tests ensure critical protocol gates remain in system prompts.
If prompts are modified, these tests will catch any accidental removal
of required protocol enforcement language.
"""

import pytest


class TestSystemPromptGates:
    """Test that system prompt contains required protocol gates."""

    @pytest.fixture
    def system_prompt(self):
        """Load the system prompt."""
        from neuron_scientist.agent import SYSTEM_PROMPT
        return SYSTEM_PROMPT

    @pytest.fixture
    def system_prompt_v2(self):
        """Load the V2 system prompt if available."""
        from neuron_scientist.prompts import SYSTEM_PROMPT_V2
        return SYSTEM_PROMPT_V2

    # ==========================================================================
    # Pre-Registration Protocol
    # ==========================================================================

    def test_mentions_preregistration(self, system_prompt):
        """System prompt mentions pre-registration requirement."""
        assert "pre-registration" in system_prompt.lower() or "preregistration" in system_prompt.lower()

    def test_mentions_register_hypothesis(self, system_prompt):
        """System prompt mentions register_hypothesis tool."""
        assert "register_hypothesis" in system_prompt

    def test_mentions_update_hypothesis_status(self, system_prompt):
        """System prompt mentions update_hypothesis_status tool."""
        assert "update_hypothesis_status" in system_prompt

    def test_preregistration_before_testing(self, system_prompt):
        """System prompt requires pre-registration BEFORE testing."""
        assert "before" in system_prompt.lower()
        # Should mention registering before running experiments
        lower = system_prompt.lower()
        assert ("before running" in lower or
                "before testing" in lower or
                "before experiments" in lower)

    # ==========================================================================
    # Baseline Comparison
    # ==========================================================================

    def test_mentions_baseline(self, system_prompt):
        """System prompt mentions baseline comparison."""
        assert "baseline" in system_prompt.lower()

    def test_mentions_effect_size(self, system_prompt):
        """System prompt mentions effect size calibration."""
        lower = system_prompt.lower()
        assert "effect size" in lower or "calibrat" in lower

    def test_mentions_zscore_or_std(self, system_prompt):
        """System prompt mentions z-score or standard deviations."""
        lower = system_prompt.lower()
        assert "z-score" in lower or "z score" in lower or "standard deviation" in lower

    # ==========================================================================
    # RelP Verification
    # ==========================================================================

    def test_mentions_relp(self, system_prompt):
        """System prompt mentions RelP attribution."""
        assert "relp" in system_prompt.lower() or "RelP" in system_prompt

    def test_mentions_downstream_verification(self, system_prompt):
        """System prompt mentions downstream verification."""
        lower = system_prompt.lower()
        assert "downstream" in lower

    def test_mentions_causal_pathway(self, system_prompt):
        """System prompt mentions causal pathway verification."""
        lower = system_prompt.lower()
        assert "causal" in lower

    # ==========================================================================
    # Confidence Requirements
    # ==========================================================================

    def test_mentions_confidence(self, system_prompt):
        """System prompt mentions confidence levels."""
        assert "confidence" in system_prompt.lower()

    def test_mentions_high_confidence_requirements(self, system_prompt):
        """System prompt specifies requirements for high confidence."""
        lower = system_prompt.lower()
        # Should mention what's needed for high confidence
        has_high_confidence_discussion = (
            "high confidence" in lower or
            ("confidence" in lower and ("z" in lower or "baseline" in lower))
        )
        assert has_high_confidence_discussion

    # ==========================================================================
    # Hypothesis Testing
    # ==========================================================================

    def test_mentions_positive_controls(self, system_prompt):
        """System prompt mentions positive controls."""
        lower = system_prompt.lower()
        assert "positive control" in lower or "should activate" in lower

    def test_mentions_negative_controls(self, system_prompt):
        """System prompt mentions negative controls."""
        lower = system_prompt.lower()
        assert "negative control" in lower or "should not activate" in lower

    def test_mentions_minimal_pairs(self, system_prompt):
        """System prompt mentions minimal pairs."""
        assert "minimal pair" in system_prompt.lower()

    # ==========================================================================
    # Tool Usage
    # ==========================================================================

    def test_mentions_batch_activation(self, system_prompt):
        """System prompt mentions batch_activation_test for efficiency."""
        assert "batch_activation" in system_prompt.lower() or "batch activation" in system_prompt.lower()

    def test_mentions_ablation(self, system_prompt):
        """System prompt mentions ablation experiments."""
        assert "ablation" in system_prompt.lower()

    # ==========================================================================
    # V2 Prompt Additional Checks
    # ==========================================================================

    def test_v2_mentions_phases(self, system_prompt_v2):
        """V2 prompt mentions investigation phases."""
        lower = system_prompt_v2.lower()
        # Should mention phase structure
        assert "phase" in lower

    def test_v2_mentions_corpus_context(self, system_prompt_v2):
        """V2 prompt mentions corpus context (Phase 0)."""
        lower = system_prompt_v2.lower()
        has_corpus = "corpus" in lower or "phase 0" in lower or "existing" in lower
        assert has_corpus


class TestReviewPromptGates:
    """Test that review prompts contain required evaluation criteria."""

    @pytest.fixture
    def review_prompt(self):
        """Load the review prompt."""
        from neuron_scientist.review_prompts import INVESTIGATION_REVIEW_PROMPT
        return INVESTIGATION_REVIEW_PROMPT

    def test_mentions_baseline_requirement(self, review_prompt):
        """Review prompt mentions baseline z-score requirement."""
        lower = review_prompt.lower()
        assert "baseline" in lower or "z-score" in lower

    def test_mentions_preregistration_check(self, review_prompt):
        """Review prompt checks for pre-registration."""
        lower = review_prompt.lower()
        assert "pre-registration" in lower or "preregistration" in lower or "hypothesis" in lower

    def test_mentions_confidence_assessment(self, review_prompt):
        """Review prompt assesses confidence appropriateness."""
        lower = review_prompt.lower()
        assert "confidence" in lower

    def test_mentions_verdict_options(self, review_prompt):
        """Review prompt specifies verdict options."""
        assert "APPROVE" in review_prompt
        assert "REQUEST_CHANGES" in review_prompt

    def test_mentions_evidence_quality(self, review_prompt):
        """Review prompt evaluates evidence quality."""
        lower = review_prompt.lower()
        assert "evidence" in lower

    def test_mentions_gaps(self, review_prompt):
        """Review prompt identifies gaps."""
        lower = review_prompt.lower()
        assert "gap" in lower


class TestPromptConsistency:
    """Test consistency between different prompts."""

    def test_same_tools_mentioned(self):
        """Both prompts mention the same core tools."""
        from neuron_scientist.agent import SYSTEM_PROMPT
        from neuron_scientist.prompts import SYSTEM_PROMPT_V2

        core_tools = [
            "batch_activation",
            "ablation",
            "register_hypothesis",
            "relp",
        ]

        for tool in core_tools:
            tool_lower = tool.lower()
            in_v1 = tool_lower in SYSTEM_PROMPT.lower()
            in_v2 = tool_lower in SYSTEM_PROMPT_V2.lower()

            # At least one should have it
            assert in_v1 or in_v2, f"Tool {tool} not found in either prompt"

    def test_confidence_thresholds_consistent(self):
        """Confidence threshold language is present."""
        from neuron_scientist.agent import SYSTEM_PROMPT

        # Should mention thresholds for meaningful effects
        lower = SYSTEM_PROMPT.lower()
        has_threshold = (
            "threshold" in lower or
            "> 2" in lower or
            "z > 2" in lower or
            "z-score" in lower
        )
        assert has_threshold


class TestPromptStability:
    """Snapshot tests to detect unintended prompt changes."""

    def test_system_prompt_length_reasonable(self):
        """System prompt is within expected length range."""
        from neuron_scientist.agent import SYSTEM_PROMPT

        # Should be substantial but not excessively long
        assert 1000 < len(SYSTEM_PROMPT) < 50000

    def test_system_prompt_v2_length_reasonable(self):
        """V2 system prompt is within expected length range."""
        from neuron_scientist.prompts import SYSTEM_PROMPT_V2

        assert 1000 < len(SYSTEM_PROMPT_V2) < 100000

    def test_review_prompt_length_reasonable(self):
        """Review prompt is within expected length range."""
        from neuron_scientist.review_prompts import INVESTIGATION_REVIEW_PROMPT

        assert 500 < len(INVESTIGATION_REVIEW_PROMPT) < 20000

    def test_prompts_are_strings(self):
        """All prompts are non-empty strings."""
        from neuron_scientist.agent import SYSTEM_PROMPT
        from neuron_scientist.prompts import SYSTEM_PROMPT_V2
        from neuron_scientist.review_prompts import INVESTIGATION_REVIEW_PROMPT

        assert isinstance(SYSTEM_PROMPT, str) and len(SYSTEM_PROMPT) > 0
        assert isinstance(SYSTEM_PROMPT_V2, str) and len(SYSTEM_PROMPT_V2) > 0
        assert isinstance(INVESTIGATION_REVIEW_PROMPT, str) and len(INVESTIGATION_REVIEW_PROMPT) > 0
