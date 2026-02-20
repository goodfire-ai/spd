"""Data flow and information preservation tests.

These tests verify that information is not lost as it flows through
the investigation pipeline: tools -> investigation -> dashboard -> HTML.
"""

import json

from neuron_scientist.schemas import (
    AblationEffect,
    ActivationExample,
    ConnectivityNode,
    DashboardData,
    HypothesisRecord,
    NeuronInvestigation,
)


class TestActivatingPromptsPreservation:
    """Verify activating prompts are fully preserved."""

    def test_all_activating_prompts_in_dashboard(self):
        """All activating prompts appear in dashboard positive_examples."""
        num_prompts = 25
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            activating_prompts=[
                {"prompt": f"Prompt {i}", "activation": 1.0 + i * 0.1, "position": i, "token": f"tok{i}"}
                for i in range(num_prompts)
            ],
        )
        dashboard = inv.to_dashboard()

        assert len(dashboard.positive_examples) == num_prompts

    def test_prompt_content_preserved(self):
        """Prompt text is preserved in conversion."""
        prompts = [
            {"prompt": "Dopamine affects reward circuits", "activation": 1.5, "position": 3, "token": "amine"},
            {"prompt": "Serotonin regulates mood", "activation": 2.0, "position": 2, "token": "onin"},
        ]
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            activating_prompts=prompts,
        )
        dashboard = inv.to_dashboard()

        prompt_texts = [ex.prompt for ex in dashboard.positive_examples]
        assert "Dopamine affects reward circuits" in prompt_texts
        assert "Serotonin regulates mood" in prompt_texts

    def test_activation_values_preserved(self):
        """Activation values are preserved exactly."""
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            activating_prompts=[
                {"prompt": "Test", "activation": 1.2890625, "position": 0, "token": "x"},
            ],
        )
        dashboard = inv.to_dashboard()

        assert dashboard.positive_examples[0].activation == 1.2890625


class TestNonActivatingPromptsPreservation:
    """Verify non-activating prompts are preserved."""

    def test_all_non_activating_prompts_in_dashboard(self):
        """All non-activating prompts appear in dashboard negative_examples."""
        num_prompts = 15
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            non_activating_prompts=[
                {"prompt": f"Non-activating {i}", "activation": 0.1 + i * 0.01, "position": i, "token": f"tok{i}"}
                for i in range(num_prompts)
            ],
        )
        dashboard = inv.to_dashboard()

        assert len(dashboard.negative_examples) == num_prompts

    def test_is_positive_flag_correct(self):
        """Negative examples have is_positive=False."""
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            non_activating_prompts=[
                {"prompt": "Non-activating", "activation": 0.1, "position": 0, "token": "x"},
            ],
        )
        dashboard = inv.to_dashboard()

        assert dashboard.negative_examples[0].is_positive is False


class TestHypothesesPreservation:
    """Verify hypotheses are preserved in timeline."""

    def test_all_hypotheses_in_dashboard(self):
        """All hypotheses appear in dashboard timeline."""
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            hypotheses_tested=[
                {"hypothesis": "H1: First hypothesis", "status": "confirmed", "confidence": 0.8},
                {"hypothesis": "H2: Second hypothesis", "status": "refuted", "confidence": 0.2},
                {"hypothesis": "H3: Third hypothesis", "status": "testing", "confidence": 0.5},
            ],
        )
        dashboard = inv.to_dashboard()

        assert len(dashboard.hypotheses) == 3

    def test_hypothesis_status_preserved(self):
        """Hypothesis status is preserved."""
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            hypotheses_tested=[
                {"hypothesis": "Test", "status": "confirmed", "confidence": 0.9},
            ],
        )
        dashboard = inv.to_dashboard()

        assert dashboard.hypotheses[0].status == "confirmed"

    def test_final_hypothesis_preserved(self):
        """Final hypothesis is preserved in dashboard."""
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            final_hypothesis="Neurotransmitter detector with graded activation",
        )
        dashboard = inv.to_dashboard()

        assert dashboard.final_hypothesis == "Neurotransmitter detector with graded activation"


class TestKeyFindingsPreservation:
    """Verify key findings are not truncated."""

    def test_all_findings_preserved(self):
        """All key findings are preserved."""
        findings = [f"Finding {i}: detailed explanation of discovery" for i in range(10)]
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            key_findings=findings,
        )

        d = inv.to_dict()
        assert d["key_findings"] == findings
        assert len(d["key_findings"]) == 10

    def test_findings_in_dashboard(self):
        """Key findings appear in dashboard."""
        findings = ["Finding 1", "Finding 2", "Finding 3"]
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            key_findings=findings,
        )
        dashboard = inv.to_dashboard()

        assert dashboard.key_findings == findings

    def test_open_questions_preserved(self):
        """Open questions are preserved."""
        questions = ["Question 1?", "Question 2?"]
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            open_questions=questions,
        )
        dashboard = inv.to_dashboard()

        assert dashboard.open_questions == questions


class TestAblationEffectsPreservation:
    """Verify ablation effects are correctly extracted."""

    def test_promotes_extracted(self):
        """Promoted tokens are extracted from ablation effects."""
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            ablation_effects=[
                {
                    "prompt": "test",
                    "promotes": [("dopamine", 0.5), ("serotonin", 0.3)],
                    "suppresses": [],
                },
            ],
        )
        dashboard = inv.to_dashboard()

        promotes = [e.token for e in dashboard.ablation_effects if e.direction == "promotes"]
        assert "dopamine" in promotes
        assert "serotonin" in promotes

    def test_suppresses_extracted(self):
        """Suppressed tokens are extracted from ablation effects."""
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            ablation_effects=[
                {
                    "prompt": "test",
                    "promotes": [],
                    "suppresses": [("random", -0.3), ("noise", -0.2)],
                },
            ],
        )
        dashboard = inv.to_dashboard()

        suppresses = [e.token for e in dashboard.ablation_effects if e.direction == "suppresses"]
        assert "random" in suppresses
        assert "noise" in suppresses

    def test_consistent_lists_populated(self):
        """Consistent promotes/suppresses lists are populated."""
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            ablation_effects=[
                {"promotes": [("tok1", 0.5)], "suppresses": [("tok2", -0.3)]},
            ],
        )
        dashboard = inv.to_dashboard()

        assert "tok1" in dashboard.consistent_promotes
        assert "tok2" in dashboard.consistent_suppresses


class TestRelPResultsPreservation:
    """Verify RelP results are properly preserved."""

    def test_relp_results_in_output(self):
        """RelP results appear in to_dict output."""
        relp_results = [
            {"prompt": "Dopamine test", "neuron_found": True, "tau": 0.01, "edges": [("L4_10555_5", "L15_7890_5", 0.3)]},
            {"prompt": "Serotonin test", "neuron_found": True, "tau": 0.01, "edges": []},
            {"prompt": "Weather test", "neuron_found": False, "tau": 0.01, "edges": []},
        ]
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            relp_results=relp_results,
        )
        d = inv.to_dict()

        # relp_results is nested under evidence (not top-level)
        assert "relp_results" in d.get("evidence", {})
        assert len(d["evidence"]["relp_results"]) == 3

    def test_relp_in_evidence_section(self):
        """RelP results also appear in evidence section."""
        relp_results = [
            {"prompt": "Test", "neuron_found": True, "tau": 0.01},
        ]
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            relp_results=relp_results,
        )
        d = inv.to_dict()

        assert "relp_results" in d["evidence"]

    def test_can_extract_positive_control(self):
        """Can determine if positive control was achieved from RelP results."""
        relp_results = [
            {"prompt": "activating", "neuron_found": True},
            {"prompt": "non-activating", "neuron_found": False},
        ]

        positive_control = any(r.get("neuron_found") for r in relp_results)
        negative_control = any(not r.get("neuron_found") for r in relp_results)

        assert positive_control is True
        assert negative_control is True


class TestConnectivityPreservation:
    """Verify connectivity information is preserved."""

    def test_connectivity_in_output(self):
        """Connectivity dict is preserved in output."""
        connectivity = {
            "upstream_neurons": [
                {"neuron_id": "L3/N9778", "label": "Technical terms", "weight": 0.15},
            ],
            "downstream_neurons": [
                {"neuron_id": "L15/N7890", "label": "Reward", "weight": 0.35},
            ],
        }
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            connectivity=connectivity,
        )
        d = inv.to_dict()

        assert d["evidence"]["connectivity"] == connectivity


class TestAgentReasoningPreservation:
    """Verify agent reasoning is preserved."""

    def test_reasoning_preserved(self):
        """Agent reasoning text is preserved."""
        reasoning = """This is a detailed explanation of the agent's
        reasoning process spanning multiple lines with various
        observations and conclusions."""

        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            agent_reasoning=reasoning,
        )
        d = inv.to_dict()

        assert d["agent_reasoning"] == reasoning

    def test_reasoning_in_dashboard(self):
        """Agent reasoning appears in dashboard."""
        reasoning = "Detailed reasoning..."
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            agent_reasoning=reasoning,
        )
        dashboard = inv.to_dashboard()

        assert dashboard.agent_reasoning == reasoning


class TestStatisticsCalculation:
    """Verify statistics are correctly calculated."""

    def test_stats_in_dashboard(self):
        """Dashboard includes computed statistics."""
        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            activating_prompts=[{"prompt": f"P{i}"} for i in range(20)],
            non_activating_prompts=[{"prompt": f"N{i}"} for i in range(10)],
            ablation_effects=[{} for _ in range(5)],
            hypotheses_tested=[{} for _ in range(3)],
        )
        dashboard = inv.to_dashboard()

        assert dashboard.stats["activating_count"] == 20
        assert dashboard.stats["non_activating_count"] == 10
        assert dashboard.stats["ablation_count"] == 5
        assert dashboard.stats["hypotheses_count"] == 3


class TestDashboardToDict:
    """Verify Dashboard to_dict produces correct JSON structure."""

    def test_complete_dashboard_structure(self):
        """Complete dashboard produces valid nested structure."""
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
            positive_examples=[
                ActivationExample("Test", 1.5, 0, "x", True),
            ],
            negative_examples=[
                ActivationExample("Test", 0.1, 0, "y", False),
            ],
            ablation_effects=[
                AblationEffect("token", 0.5, "promotes", "high"),
            ],
            hypotheses=[
                HypothesisRecord("Test hypothesis", 10, [], [], 0.8, "supported"),
            ],
            final_hypothesis="Final hypothesis",
            upstream_nodes=[
                ConnectivityNode("L3/N1000", "Upstream", 0.2, "upstream", False),
            ],
            downstream_nodes=[
                ConnectivityNode("L15/N5000", "Downstream", 0.3, "downstream", False),
            ],
            key_findings=["Finding 1"],
            open_questions=["Question 1"],
            agent_reasoning="Reasoning...",
            timestamp="2026-01-19T00:00:00",
        )
        d = dashboard.to_dict()

        # Verify structure can be JSON serialized
        json_str = json.dumps(d)
        assert len(json_str) > 0

        # Verify nested structure
        assert d["summary_card"]["confidence"] == 0.8
        assert len(d["activation_patterns"]["positive_examples"]) == 1
        assert len(d["ablation_effects"]["effects"]) == 1
        assert len(d["hypothesis_timeline"]["hypotheses"]) == 1
        assert len(d["connectivity"]["upstream"]) == 1
        assert len(d["connectivity"]["downstream"]) == 1
        assert d["findings"]["key_findings"] == ["Finding 1"]
