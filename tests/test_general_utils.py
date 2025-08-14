"""Test p-annealing functionality."""

from spd.utils.general_utils import get_linear_annealed_p


class TestLinearAnnealedP:
    """Test get_linear_annealed_p function."""

    def test_no_annealing_cases(self):
        """Test all edge cases where annealing should be a no-op."""
        initial_p = 2.0
        steps = 1000

        # Case 1: final_p is None
        for step in [0, 500, 1000]:
            p = get_linear_annealed_p(
                step=step,
                steps=steps,
                initial_p=initial_p,
                p_anneal_start_frac=0.5,
                p_anneal_final_p=None,
                p_anneal_end_frac=1.0,
            )
            assert p == initial_p

        # Case 2: start_frac >= 1.0
        for step in [0, 500, 1000]:
            p = get_linear_annealed_p(
                step=step,
                steps=steps,
                initial_p=initial_p,
                p_anneal_start_frac=1.0,
                p_anneal_final_p=0.9,
                p_anneal_end_frac=1.0,
            )
            assert p == initial_p

    def test_annealing_scenarios(self):
        """Test various annealing scenarios."""
        initial_p = 2.0
        final_p = 0.5
        steps = 1000

        test_cases = [
            # (start_frac, end_frac, test_points)
            # start=0, end=1: anneal throughout
            (0.0, 1.0, [(0, initial_p), (500, 1.25), (1000, final_p)]),
            # start=0.25, end=1: skip first quarter
            (0.25, 1.0, [(0, initial_p), (250, initial_p), (625, 1.25), (1000, final_p)]),
            # start=0.25, end=0.75: anneal in middle
            (
                0.25,
                0.75,
                [(0, initial_p), (250, initial_p), (500, 1.25), (750, final_p), (1000, final_p)],
            ),
            # start=0, end=0.75: anneal then plateau
            (0.0, 0.75, [(0, initial_p), (375, 1.25), (750, final_p), (1000, final_p)]),
        ]

        for start_frac, end_frac, test_points in test_cases:
            for step, expected in test_points:
                p = get_linear_annealed_p(
                    step=step,
                    steps=steps,
                    initial_p=initial_p,
                    p_anneal_start_frac=start_frac,
                    p_anneal_final_p=final_p,
                    p_anneal_end_frac=end_frac,
                )
                assert abs(p - expected) < 1e-6, (
                    f"start={start_frac}, end={end_frac}, step={step}: expected {expected}, got {p}"
                )
