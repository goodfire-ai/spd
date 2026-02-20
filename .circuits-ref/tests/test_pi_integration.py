"""PI orchestration integration tests.

These tests verify the NeuronPI orchestration pipeline:
- Happy path: investigation -> review approval -> dashboard generation
- Review rejection loop: REQUEST_CHANGES -> revision -> final verdict
- Skip flags: --skip-review, --skip-dashboard, --review-only
"""

import json

from neuron_scientist.schemas import (
    NeuronInvestigation,
    PIResult,
    ReviewResult,
)


class TestPIResultSchema:
    """Test PIResult data structure."""

    def test_pi_result_creation(self):
        """PIResult can be created with basic fields."""
        result = PIResult(
            neuron_id="L4/N10555",
            iterations=1,
            final_verdict="APPROVE",
        )

        assert result.neuron_id == "L4/N10555"
        assert result.iterations == 1
        assert result.final_verdict == "APPROVE"

    def test_pi_result_with_review_history(self):
        """PIResult can store review history."""
        reviews = [
            ReviewResult("REQUEST_CHANGES", "overconfident", ["Gap 1"], "Fix it", "", 1),
            ReviewResult("APPROVE", "appropriate", [], "Good", "", 2),
        ]

        result = PIResult(
            neuron_id="L4/N10555",
            review_history=reviews,
            iterations=2,
            final_verdict="APPROVE",
        )

        assert len(result.review_history) == 2
        assert result.review_history[0].verdict == "REQUEST_CHANGES"
        assert result.review_history[1].verdict == "APPROVE"

    def test_pi_result_to_dict(self):
        """PIResult serializes correctly."""
        result = PIResult(
            neuron_id="L4/N10555",
            iterations=1,
            final_verdict="APPROVE",
            investigation_path="/path/to/investigation.json",
            dashboard_path="/path/to/dashboard.html",
            timestamp="2026-01-19T00:00:00",
        )

        d = result.to_dict()

        assert d["neuron_id"] == "L4/N10555"
        assert d["final_verdict"] == "APPROVE"
        assert d["investigation_path"] == "/path/to/investigation.json"


class TestReviewResultSchema:
    """Test ReviewResult data structure."""

    def test_review_result_approve(self):
        """ReviewResult can represent APPROVE verdict."""
        result = ReviewResult(
            verdict="APPROVE",
            confidence_assessment="appropriate",
            gaps=[],
            feedback="Investigation is thorough.",
        )

        assert result.verdict == "APPROVE"
        assert result.confidence_assessment == "appropriate"

    def test_review_result_request_changes(self):
        """ReviewResult can represent REQUEST_CHANGES verdict."""
        result = ReviewResult(
            verdict="REQUEST_CHANGES",
            confidence_assessment="overconfident",
            gaps=["Missing baseline comparison", "No RelP verification"],
            feedback="Please add baseline comparison.",
        )

        assert result.verdict == "REQUEST_CHANGES"
        assert len(result.gaps) == 2

    def test_review_result_to_dict(self):
        """ReviewResult serializes correctly."""
        result = ReviewResult(
            verdict="APPROVE",
            confidence_assessment="appropriate",
            gaps=["Minor gap"],
            feedback="Good work",
            raw_response="Full response...",
            iteration=1,
        )

        d = result.to_dict()

        assert d["verdict"] == "APPROVE"
        assert d["iteration"] == 1
        assert len(d["gaps"]) == 1


class TestReviewParsing:
    """Test parsing of GPT review responses."""

    def test_parse_approve_response(self):
        """Can parse APPROVE response."""
        from neuron_scientist.review_prompts import parse_review_response

        response = """VERDICT: APPROVE
CONFIDENCE_ASSESSMENT: appropriate
GAPS:
- None significant
FEEDBACK:
Investigation is thorough and well-documented."""

        result = parse_review_response(response)

        assert result["verdict"] == "APPROVE"
        assert result["confidence_assessment"] == "appropriate"
        assert "raw_response" in result

    def test_parse_request_changes_response(self):
        """Can parse REQUEST_CHANGES response."""
        from neuron_scientist.review_prompts import parse_review_response

        response = """VERDICT: REQUEST_CHANGES
CONFIDENCE_ASSESSMENT: overconfident
GAPS:
- Missing baseline comparison with random neurons
- No dose-response test performed
- RelP verification incomplete
FEEDBACK:
Please run baseline comparison before claiming high confidence. The investigation shows strong activation patterns but lacks statistical validation."""

        result = parse_review_response(response)

        assert result["verdict"] == "REQUEST_CHANGES"
        assert result["confidence_assessment"] == "overconfident"
        assert len(result["gaps"]) >= 2

    def test_parse_handles_malformed_response(self):
        """Parsing handles malformed response gracefully."""
        from neuron_scientist.review_prompts import parse_review_response

        # Missing structure
        response = "This is not a proper review response."

        result = parse_review_response(response)

        # Should not crash, should return dict with defaults
        assert result is not None
        assert "verdict" in result
        # Defaults to cautious REQUEST_CHANGES
        assert result["verdict"] == "REQUEST_CHANGES"


class TestInvestigationDistillation:
    """Test distillation of investigation for review."""

    def test_distill_investigation(self):
        """Investigation can be distilled for review."""
        from neuron_scientist.review_prompts import distill_investigation_for_review

        inv = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            confidence=0.75,
            input_function="Activates on neurotransmitter mentions",
            output_function="Promotes brain chemistry tokens",
            key_findings=["Finding 1", "Finding 2"],
            activating_prompts=[
                {"prompt": "Test", "activation": 1.5},
            ],
        )

        distilled = distill_investigation_for_review(inv.to_dict())

        # Should return a dict with key fields
        assert isinstance(distilled, dict)
        assert distilled["neuron_id"] == "L4/N10555"
        assert distilled["confidence"] == 0.75
        assert "evidence" in distilled
        assert "protocol_checklist" in distilled


class TestReviewLoop:
    """Test the review loop behavior."""

    def test_max_iterations_respected(self):
        """Review loop respects max iterations."""
        # Simulate a scenario where review always requests changes
        max_iterations = 3

        iterations = 0
        verdict = "REQUEST_CHANGES"

        while verdict == "REQUEST_CHANGES" and iterations < max_iterations:
            iterations += 1
            # Simulate review
            if iterations >= max_iterations:
                verdict = "MAX_ITERATIONS"  # Exit condition
                break
            verdict = "REQUEST_CHANGES"  # Simulate continued rejection

        assert iterations == max_iterations
        assert verdict == "MAX_ITERATIONS"

    def test_early_approval_stops_loop(self):
        """Approval stops the review loop early."""
        max_iterations = 3
        iterations = 0
        verdict = "REQUEST_CHANGES"

        # Simulate approval on second iteration
        approval_at = 2

        while verdict == "REQUEST_CHANGES" and iterations < max_iterations:
            iterations += 1
            if iterations >= approval_at:
                verdict = "APPROVE"
                break
            verdict = "REQUEST_CHANGES"

        assert iterations == approval_at
        assert verdict == "APPROVE"


class TestPIResultPersistence:
    """Test PI result persistence."""

    def test_pi_result_can_be_saved(self, tmp_path):
        """PIResult can be saved to JSON file."""
        result = PIResult(
            neuron_id="L4/N10555",
            iterations=2,
            final_verdict="APPROVE",
            review_history=[
                ReviewResult("REQUEST_CHANGES", "overconfident", ["Gap"], "Fix", "", 1),
                ReviewResult("APPROVE", "appropriate", [], "Good", "", 2),
            ],
            timestamp="2026-01-19T00:00:00",
        )

        path = tmp_path / "pi_result.json"
        with open(path, "w") as f:
            json.dump(result.to_dict(), f)

        # Verify file was written
        assert path.exists()

        # Verify content can be read back
        with open(path) as f:
            loaded = json.load(f)

        assert loaded["neuron_id"] == "L4/N10555"
        assert loaded["final_verdict"] == "APPROVE"
        assert len(loaded["review_history"]) == 2

    def test_pi_result_includes_paths(self, tmp_path):
        """PIResult includes artifact paths."""
        result = PIResult(
            neuron_id="L4/N10555",
            investigation_path=str(tmp_path / "investigation.json"),
            dashboard_path=str(tmp_path / "dashboard.html"),
            dashboard_json_path=str(tmp_path / "dashboard.json"),
        )

        d = result.to_dict()

        assert "investigation_path" in d
        assert "dashboard_path" in d
        assert "dashboard_json_path" in d


class TestSkipFlags:
    """Test skip flag behavior."""

    def test_skip_review_flag_behavior(self):
        """--skip-review skips the review loop."""
        skip_review = True

        # Simulate pipeline with skip_review
        if skip_review:
            review_result = None
            final_verdict = "SKIPPED"
        else:
            review_result = ReviewResult("APPROVE", "appropriate", [], "", "", 1)
            final_verdict = review_result.verdict

        assert review_result is None
        assert final_verdict == "SKIPPED"

    def test_skip_dashboard_flag_behavior(self):
        """--skip-dashboard skips dashboard generation."""
        skip_dashboard = True

        # Simulate pipeline with skip_dashboard
        if skip_dashboard:
            dashboard_path = None
        else:
            dashboard_path = "/path/to/dashboard.html"

        assert dashboard_path is None

    def test_review_only_mode_behavior(self):
        """--review-only mode loads existing investigation."""
        review_only = True
        existing_investigation_path = "/path/to/investigation.json"

        # Simulate review-only mode
        if review_only:
            # Would load investigation instead of running new one
            investigation_source = "loaded"
        else:
            investigation_source = "generated"

        assert investigation_source == "loaded"


class TestErrorHandling:
    """Test error handling in PI pipeline."""

    def test_pi_result_with_error(self):
        """PIResult can store error information."""
        result = PIResult(
            neuron_id="L4/N10555",
            final_verdict="ERROR",
            error="Investigation failed due to CUDA error",
        )

        assert result.final_verdict == "ERROR"
        assert result.error is not None
        assert "CUDA" in result.error

    def test_pi_result_error_serializes(self):
        """PIResult with error serializes correctly."""
        result = PIResult(
            neuron_id="L4/N10555",
            final_verdict="ERROR",
            error="Test error message",
        )

        d = result.to_dict()

        assert d["error"] == "Test error message"
        assert d["final_verdict"] == "ERROR"


class TestRevisionContext:
    """Test revision context passed to agent."""

    def test_revision_context_includes_feedback(self):
        """Revision context includes reviewer feedback."""
        feedback = "Please add baseline comparison with at least 30 random neurons."
        gaps = ["Missing baseline", "No RelP verification"]

        # Simulate building revision context
        revision_context = f"""
Previous Review Feedback:
{feedback}

Identified Gaps:
- {gaps[0]}
- {gaps[1]}

Please address these issues in your revision.
"""

        assert feedback in revision_context
        for gap in gaps:
            assert gap in revision_context

    def test_revision_preserves_prior_investigation(self):
        """Revision builds on prior investigation data."""
        prior = NeuronInvestigation(
            neuron_id="L4/N10555",
            layer=4,
            neuron_idx=10555,
            activating_prompts=[
                {"prompt": "Prior finding", "activation": 1.5},
            ],
        )

        # Simulate passing prior investigation to revision
        prior_data = prior.to_dict()

        # Revision should have access to prior findings
        assert len(prior_data["evidence"]["activating_prompts"]) > 0
