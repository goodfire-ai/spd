"""Hypothesis lifecycle tests for neuron scientist.

These tests verify the pre-registration protocol enforcement:
- Hypothesis registration before experiments
- Status updates after testing
- Multiple hypotheses and conflicts
- Protocol state tracking integration
"""

import asyncio

from neuron_scientist.tools import (
    clear_hypothesis_registry,
    get_hypothesis_registry,
    get_next_hypothesis_id,
    get_protocol_state,
    init_hypothesis_registry,
    init_protocol_state,
    tool_register_hypothesis,
    tool_update_hypothesis_status,
)


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestHypothesisRegistration:
    """Test hypothesis registration functionality."""

    def setup_method(self):
        """Reset state before each test."""
        clear_hypothesis_registry()
        init_protocol_state()

    def test_register_single_hypothesis(self):
        """Can register a single hypothesis."""
        result = run_async(tool_register_hypothesis(
            hypothesis="Neuron activates on mentions of dopamine",
            confirmation_criteria="Activation >1.0 on 80%+ of dopamine prompts",
            refutation_criteria="Activation <0.5 on dopamine prompts",
            prior_probability=60,
            hypothesis_type="activation",
        ))

        assert result["hypothesis_id"] == "H1"
        assert "registered" in result["message"].lower()
        assert result["registered"]["status"] == "registered"
        assert result["registered"]["prior_probability"] == 60

    def test_register_multiple_hypotheses(self):
        """Can register multiple hypotheses with sequential IDs."""
        h1 = run_async(tool_register_hypothesis(
            hypothesis="First hypothesis",
            confirmation_criteria="Criteria 1",
            refutation_criteria="Refutation 1",
            prior_probability=50,
        ))
        h2 = run_async(tool_register_hypothesis(
            hypothesis="Second hypothesis",
            confirmation_criteria="Criteria 2",
            refutation_criteria="Refutation 2",
            prior_probability=70,
        ))
        h3 = run_async(tool_register_hypothesis(
            hypothesis="Third hypothesis",
            confirmation_criteria="Criteria 3",
            refutation_criteria="Refutation 3",
            prior_probability=30,
        ))

        assert h1["hypothesis_id"] == "H1"
        assert h2["hypothesis_id"] == "H2"
        assert h3["hypothesis_id"] == "H3"

        registry = get_hypothesis_registry()
        assert len(registry) == 3

    def test_registration_updates_protocol_state(self):
        """Registering hypotheses updates protocol state counter."""
        init_protocol_state()

        state_before = get_protocol_state()
        assert state_before.hypotheses_registered == 0

        run_async(tool_register_hypothesis(
            hypothesis="Test hypothesis",
            confirmation_criteria="Criteria",
            refutation_criteria="Refutation",
            prior_probability=50,
        ))

        state_after = get_protocol_state()
        assert state_after.hypotheses_registered == 1

    def test_hypothesis_types(self):
        """Can register different hypothesis types."""
        types = ["activation", "output", "causal", "connectivity"]

        for i, h_type in enumerate(types):
            result = run_async(tool_register_hypothesis(
                hypothesis=f"Hypothesis {i+1}",
                confirmation_criteria="Criteria",
                refutation_criteria="Refutation",
                prior_probability=50,
                hypothesis_type=h_type,
            ))
            assert result["registered"]["hypothesis_type"] == h_type

        registry = get_hypothesis_registry()
        assert len(registry) == 4
        assert [h["hypothesis_type"] for h in registry] == types


class TestHypothesisUpdates:
    """Test hypothesis status updates after testing."""

    def setup_method(self):
        """Reset state before each test."""
        clear_hypothesis_registry()
        init_protocol_state()

    def test_update_to_confirmed(self):
        """Can update hypothesis to confirmed status."""
        run_async(tool_register_hypothesis(
            hypothesis="Neuron activates on dopamine",
            confirmation_criteria="Activation >1.0",
            refutation_criteria="Activation <0.5",
            prior_probability=60,
        ))

        result = run_async(tool_update_hypothesis_status(
            hypothesis_id="H1",
            status="confirmed",
            posterior_probability=85,
            evidence_summary="Observed activation >1.5 on 90% of dopamine prompts",
        ))

        assert result["hypothesis_id"] == "H1"
        assert result["status"] == "confirmed"
        assert result["posterior_probability"] == 85
        assert result["probability_shift"] == 25  # 85 - 60

    def test_update_to_refuted(self):
        """Can update hypothesis to refuted status."""
        run_async(tool_register_hypothesis(
            hypothesis="Neuron activates on serotonin",
            confirmation_criteria="Activation >1.0",
            refutation_criteria="Activation <0.5",
            prior_probability=70,
        ))

        result = run_async(tool_update_hypothesis_status(
            hypothesis_id="H1",
            status="refuted",
            posterior_probability=15,
            evidence_summary="Activation near zero on all serotonin prompts",
        ))

        assert result["status"] == "refuted"
        assert result["posterior_probability"] == 15
        assert result["probability_shift"] == -55  # 15 - 70

    def test_update_to_inconclusive(self):
        """Can update hypothesis to inconclusive status."""
        run_async(tool_register_hypothesis(
            hypothesis="Neuron activates on neurotransmitters",
            confirmation_criteria="Activation >1.0",
            refutation_criteria="Activation <0.5",
            prior_probability=50,
        ))

        result = run_async(tool_update_hypothesis_status(
            hypothesis_id="H1",
            status="inconclusive",
            posterior_probability=50,
            evidence_summary="Mixed results - some activation but inconsistent",
        ))

        assert result["status"] == "inconclusive"
        assert result["probability_shift"] == 0

    def test_update_nonexistent_hypothesis(self):
        """Updating nonexistent hypothesis returns error."""
        result = run_async(tool_update_hypothesis_status(
            hypothesis_id="H99",
            status="confirmed",
            posterior_probability=90,
            evidence_summary="This should fail",
        ))

        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_update_accumulates_evidence(self):
        """Multiple updates accumulate evidence."""
        run_async(tool_register_hypothesis(
            hypothesis="Test hypothesis",
            confirmation_criteria="Criteria",
            refutation_criteria="Refutation",
            prior_probability=50,
        ))

        run_async(tool_update_hypothesis_status(
            hypothesis_id="H1",
            status="confirmed",
            posterior_probability=70,
            evidence_summary="First evidence batch",
        ))

        run_async(tool_update_hypothesis_status(
            hypothesis_id="H1",
            status="confirmed",
            posterior_probability=85,
            evidence_summary="Second evidence batch",
        ))

        registry = get_hypothesis_registry()
        h1 = registry[0]
        assert len(h1["evidence"]) == 2
        assert "First evidence" in h1["evidence"][0]
        assert "Second evidence" in h1["evidence"][1]

    def test_update_tracks_protocol_state(self):
        """Updating hypotheses increments protocol state counter."""
        run_async(tool_register_hypothesis(
            hypothesis="Test",
            confirmation_criteria="C",
            refutation_criteria="R",
            prior_probability=50,
        ))

        state_before = get_protocol_state()
        assert state_before.hypotheses_updated == 0

        run_async(tool_update_hypothesis_status(
            hypothesis_id="H1",
            status="confirmed",
            posterior_probability=80,
            evidence_summary="Evidence",
        ))

        state_after = get_protocol_state()
        assert state_after.hypotheses_updated == 1


class TestHypothesisLifecycle:
    """Test full hypothesis lifecycle scenarios."""

    def setup_method(self):
        """Reset state before each test."""
        clear_hypothesis_registry()
        init_protocol_state()

    def test_full_investigation_lifecycle(self):
        """Test complete lifecycle: register -> test -> update."""
        # Phase 1: Register hypotheses
        h1 = run_async(tool_register_hypothesis(
            hypothesis="Neuron activates on dopamine mentions",
            confirmation_criteria="Activation >1.0 on 80%+ dopamine prompts",
            refutation_criteria="Activation <0.5 on dopamine prompts",
            prior_probability=60,
            hypothesis_type="activation",
        ))

        h2 = run_async(tool_register_hypothesis(
            hypothesis="Neuron promotes reward-related tokens",
            confirmation_criteria="Ablation reduces reward token probability",
            refutation_criteria="No change on ablation",
            prior_probability=50,
            hypothesis_type="output",
        ))

        # Verify registration
        registry = get_hypothesis_registry()
        assert len(registry) == 2
        assert all(h["status"] == "registered" for h in registry)

        # Phase 2: Update after testing
        run_async(tool_update_hypothesis_status(
            hypothesis_id="H1",
            status="confirmed",
            posterior_probability=90,
            evidence_summary="Strong activation on dopamine prompts, z-score=3.5",
        ))

        run_async(tool_update_hypothesis_status(
            hypothesis_id="H2",
            status="refuted",
            posterior_probability=20,
            evidence_summary="No significant effect on ablation",
        ))

        # Verify final state
        registry = get_hypothesis_registry()
        assert registry[0]["status"] == "confirmed"
        assert registry[0]["posterior_probability"] == 90
        assert registry[1]["status"] == "refuted"
        assert registry[1]["posterior_probability"] == 20

    def test_hypothesis_refinement_flow(self):
        """Test refining a hypothesis based on evidence."""
        # Initial broad hypothesis
        run_async(tool_register_hypothesis(
            hypothesis="Neuron activates on neurotransmitters",
            confirmation_criteria="Activation on any neurotransmitter mention",
            refutation_criteria="No activation on neurotransmitters",
            prior_probability=70,
        ))

        # Update to inconclusive
        run_async(tool_update_hypothesis_status(
            hypothesis_id="H1",
            status="inconclusive",
            posterior_probability=40,
            evidence_summary="Activates on dopamine but not serotonin or norepinephrine",
        ))

        # Register refined hypothesis
        run_async(tool_register_hypothesis(
            hypothesis="Neuron specifically activates on dopamine",
            confirmation_criteria="Activation >1.0 on dopamine, <0.3 on other neurotransmitters",
            refutation_criteria="Equal activation on multiple neurotransmitters",
            prior_probability=80,
        ))

        # Confirm refined hypothesis
        run_async(tool_update_hypothesis_status(
            hypothesis_id="H2",
            status="confirmed",
            posterior_probability=95,
            evidence_summary="Strong dopamine specificity confirmed with minimal pairs",
        ))

        registry = get_hypothesis_registry()
        assert len(registry) == 2
        assert registry[0]["status"] == "inconclusive"
        assert registry[1]["status"] == "confirmed"

    def test_competing_hypotheses(self):
        """Test multiple competing hypotheses for same phenomenon."""
        # Hypothesis A: Semantic function
        run_async(tool_register_hypothesis(
            hypothesis="Neuron encodes 'reward' concept",
            confirmation_criteria="Activation on reward-related words across contexts",
            refutation_criteria="No consistent semantic pattern",
            prior_probability=50,
        ))

        # Hypothesis B: Positional function
        run_async(tool_register_hypothesis(
            hypothesis="Neuron fires at sentence boundaries",
            confirmation_criteria="Activation peaks at periods and sentence ends",
            refutation_criteria="No positional pattern",
            prior_probability=30,
        ))

        # Hypothesis C: Formatting function
        run_async(tool_register_hypothesis(
            hypothesis="Neuron promotes list formatting",
            confirmation_criteria="Ablation disrupts list structure",
            refutation_criteria="No effect on formatting",
            prior_probability=20,
        ))

        # Testing reveals H1 is correct
        run_async(tool_update_hypothesis_status(
            hypothesis_id="H1",
            status="confirmed",
            posterior_probability=85,
            evidence_summary="Strong semantic pattern confirmed",
        ))

        run_async(tool_update_hypothesis_status(
            hypothesis_id="H2",
            status="refuted",
            posterior_probability=5,
            evidence_summary="No positional correlation found",
        ))

        run_async(tool_update_hypothesis_status(
            hypothesis_id="H3",
            status="refuted",
            posterior_probability=10,
            evidence_summary="No formatting effect on ablation",
        ))

        registry = get_hypothesis_registry()
        confirmed = [h for h in registry if h["status"] == "confirmed"]
        refuted = [h for h in registry if h["status"] == "refuted"]

        assert len(confirmed) == 1
        assert len(refuted) == 2


class TestPriorIterationInitialization:
    """Test initialization from prior investigation iteration."""

    def setup_method(self):
        """Reset state before each test."""
        clear_hypothesis_registry()
        init_protocol_state()

    def test_init_from_prior_iteration(self):
        """Can initialize from prior hypotheses list."""
        prior = [
            {
                "hypothesis_id": "H1",
                "hypothesis": "Prior hypothesis 1",
                "status": "confirmed",
                "prior_probability": 60,
                "posterior_probability": 85,
            },
            {
                "hypothesis_id": "H2",
                "hypothesis": "Prior hypothesis 2",
                "status": "refuted",
                "prior_probability": 50,
                "posterior_probability": 15,
            },
        ]

        init_hypothesis_registry(prior)
        registry = get_hypothesis_registry()

        assert len(registry) == 2
        assert registry[0]["hypothesis_id"] == "H1"
        assert registry[0]["from_prior_iteration"] is True
        assert registry[1]["hypothesis_id"] == "H2"

    def test_next_id_continues_from_prior(self):
        """Next hypothesis ID continues from prior iteration."""
        prior = [
            {"hypothesis_id": "H1", "hypothesis": "First"},
            {"hypothesis_id": "H2", "hypothesis": "Second"},
            {"hypothesis_id": "H3", "hypothesis": "Third"},
        ]

        init_hypothesis_registry(prior)
        next_id = get_next_hypothesis_id()

        assert next_id == "H4"

    def test_new_hypothesis_after_prior(self):
        """New hypothesis gets correct ID after loading prior."""
        prior = [
            {"hypothesis_id": "H1", "hypothesis": "Prior"},
            {"hypothesis_id": "H2", "hypothesis": "Prior 2"},
        ]

        init_hypothesis_registry(prior)

        result = run_async(tool_register_hypothesis(
            hypothesis="New hypothesis",
            confirmation_criteria="New criteria",
            refutation_criteria="New refutation",
            prior_probability=50,
        ))

        assert result["hypothesis_id"] == "H3"

        registry = get_hypothesis_registry()
        assert len(registry) == 3
        # The new hypothesis should NOT have from_prior_iteration set (or False)
        new_h = registry[2]
        assert new_h.get("from_prior_iteration", False) is False or "from_prior_iteration" not in new_h

    def test_clear_after_prior_resets_properly(self):
        """Clearing after prior init resets completely."""
        prior = [
            {"hypothesis_id": "H1", "hypothesis": "Prior"},
            {"hypothesis_id": "H2", "hypothesis": "Prior 2"},
        ]

        init_hypothesis_registry(prior)
        assert len(get_hypothesis_registry()) == 2

        clear_hypothesis_registry()
        assert len(get_hypothesis_registry()) == 0
        assert get_next_hypothesis_id() == "H1"


class TestBayesFactorCalculation:
    """Test Bayes factor calculation in hypothesis updates."""

    def setup_method(self):
        """Reset state before each test."""
        clear_hypothesis_registry()
        init_protocol_state()

    def test_bayes_factor_on_confirmation(self):
        """Bayes factor calculated correctly on confirmation."""
        run_async(tool_register_hypothesis(
            hypothesis="Test hypothesis",
            confirmation_criteria="Criteria",
            refutation_criteria="Refutation",
            prior_probability=50,  # 0.5 prior
        ))

        result = run_async(tool_update_hypothesis_status(
            hypothesis_id="H1",
            status="confirmed",
            posterior_probability=90,  # 0.9 posterior
            evidence_summary="Strong evidence",
        ))

        # Bayes factor should be positive and substantial
        # BF = (0.9/0.5) / ((1-0.9)/(1-0.5)) = 1.8 / 0.2 = 9
        assert "bayes_factor" in result
        assert result["bayes_factor"] is not None
        assert result["bayes_factor"] > 1  # Evidence supports confirmation

    def test_bayes_factor_on_refutation(self):
        """Bayes factor calculated correctly on refutation."""
        run_async(tool_register_hypothesis(
            hypothesis="Test hypothesis",
            confirmation_criteria="Criteria",
            refutation_criteria="Refutation",
            prior_probability=70,  # 0.7 prior
        ))

        result = run_async(tool_update_hypothesis_status(
            hypothesis_id="H1",
            status="refuted",
            posterior_probability=10,  # 0.1 posterior
            evidence_summary="Counter evidence",
        ))

        # Bayes factor should indicate strong refutation
        assert "bayes_factor" in result
        assert result["bayes_factor"] is not None

    def test_bayes_factor_edge_cases(self):
        """Bayes factor handles edge cases gracefully."""
        # Prior = 0% (should handle gracefully)
        run_async(tool_register_hypothesis(
            hypothesis="Very unlikely",
            confirmation_criteria="Criteria",
            refutation_criteria="Refutation",
            prior_probability=0,
        ))

        result = run_async(tool_update_hypothesis_status(
            hypothesis_id="H1",
            status="confirmed",
            posterior_probability=50,
            evidence_summary="Surprise evidence",
        ))

        # Should not crash, may return None or special value
        assert "bayes_factor" in result


class TestProtocolIntegration:
    """Test integration with protocol state tracking."""

    def setup_method(self):
        """Reset state before each test."""
        clear_hypothesis_registry()
        init_protocol_state()

    def test_complete_protocol_flow(self):
        """Test that hypothesis lifecycle updates protocol state correctly."""
        state = get_protocol_state()
        assert state.hypotheses_registered == 0
        assert state.hypotheses_updated == 0

        # Register 3 hypotheses
        for i in range(3):
            run_async(tool_register_hypothesis(
                hypothesis=f"Hypothesis {i+1}",
                confirmation_criteria="C",
                refutation_criteria="R",
                prior_probability=50,
            ))

        state = get_protocol_state()
        assert state.hypotheses_registered == 3
        assert state.hypotheses_updated == 0

        # Update 2 of them
        run_async(tool_update_hypothesis_status(
            hypothesis_id="H1",
            status="confirmed",
            posterior_probability=80,
            evidence_summary="Evidence 1",
        ))
        run_async(tool_update_hypothesis_status(
            hypothesis_id="H2",
            status="refuted",
            posterior_probability=20,
            evidence_summary="Evidence 2",
        ))

        state = get_protocol_state()
        assert state.hypotheses_registered == 3
        assert state.hypotheses_updated == 2

    def test_protocol_state_affects_validation(self):
        """Protocol state validation checks hypothesis registration."""
        state = get_protocol_state()
        missing = state.get_missing_validation()

        # Should flag missing pre-registration
        assert any("pre-registration" in m.lower() for m in missing)

        # Register a hypothesis
        run_async(tool_register_hypothesis(
            hypothesis="Test",
            confirmation_criteria="C",
            refutation_criteria="R",
            prior_probability=50,
        ))

        state = get_protocol_state()
        missing = state.get_missing_validation()

        # Should no longer flag pre-registration
        assert not any("pre-registration" in m.lower() for m in missing)


class TestEdgeCases:
    """Test edge cases in hypothesis management."""

    def setup_method(self):
        """Reset state before each test."""
        clear_hypothesis_registry()
        init_protocol_state()

    def test_extreme_probability_values(self):
        """Test with extreme probability values."""
        run_async(tool_register_hypothesis(
            hypothesis="Certain hypothesis",
            confirmation_criteria="C",
            refutation_criteria="R",
            prior_probability=99,  # Almost certain
        ))

        result = run_async(tool_update_hypothesis_status(
            hypothesis_id="H1",
            status="confirmed",
            posterior_probability=100,  # Certain
            evidence_summary="Definitive evidence",
        ))

        assert result["posterior_probability"] == 100
        assert result["probability_shift"] == 1

    def test_long_hypothesis_text(self):
        """Test with very long hypothesis text."""
        long_hypothesis = "This is a very detailed hypothesis about " + "the neuron's function " * 50

        result = run_async(tool_register_hypothesis(
            hypothesis=long_hypothesis,
            confirmation_criteria="Detailed criteria...",
            refutation_criteria="Detailed refutation...",
            prior_probability=50,
        ))

        assert result["hypothesis_id"] == "H1"
        assert len(result["registered"]["hypothesis"]) > 500

    def test_special_characters_in_hypothesis(self):
        """Test with special characters in hypothesis text."""
        run_async(tool_register_hypothesis(
            hypothesis="Neuron activates on α-amino acids & β-blockers <50%",
            confirmation_criteria="Activation on α/β compounds",
            refutation_criteria="No pattern with α/β",
            prior_probability=60,
        ))

        registry = get_hypothesis_registry()
        assert "α" in registry[0]["hypothesis"]
        assert "β" in registry[0]["hypothesis"]
        assert "&" in registry[0]["hypothesis"]

    def test_empty_prior_initialization(self):
        """Test initialization with empty prior list."""
        init_hypothesis_registry([])

        assert len(get_hypothesis_registry()) == 0
        assert get_next_hypothesis_id() == "H1"
