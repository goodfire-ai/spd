import math

import pytest

from spd.configs import CosineSchedule, LinearSchedule
from spd.scheduling import get_cosine_schedule_value, get_linear_schedule_value


class TestLinearSchedule:
    def test_before_schedule_starts(self):
        d = {"start_value": 1.0, "end_value": 0.0, "start_frac": 0.3, "end_frac": 0.7}
        schedule = LinearSchedule(type="linear", **d)
        result = get_linear_schedule_value(schedule, current_frac_of_training=0.1)
        assert result == 1.0

    def test_at_schedule_start(self):
        d = {"start_value": 1.0, "end_value": 0.0, "start_frac": 0.3, "end_frac": 0.7}
        schedule = LinearSchedule(type="linear", **d)
        result = get_linear_schedule_value(schedule, current_frac_of_training=0.3)
        assert result == 1.0

    def test_at_schedule_end(self):
        d = {"start_value": 1.0, "end_value": 0.0, "start_frac": 0.3, "end_frac": 0.7}
        schedule = LinearSchedule(type="linear", **d)
        result = get_linear_schedule_value(schedule, current_frac_of_training=0.7)
        assert result == 0.0

    def test_after_schedule_ends(self):
        d = {"start_value": 1.0, "end_value": 0.0, "start_frac": 0.3, "end_frac": 0.7}
        schedule = LinearSchedule(type="linear", **d)
        result = get_linear_schedule_value(schedule, current_frac_of_training=0.9)
        assert result == 0.0

    def test_midpoint_interpolation(self):
        d = {"start_value": 1.0, "end_value": 0.0, "start_frac": 0.3, "end_frac": 0.7}
        schedule = LinearSchedule(type="linear", **d)
        # At 0.5, halfway between 0.3 and 0.7
        result = get_linear_schedule_value(schedule, current_frac_of_training=0.5)
        assert math.isclose(result, 0.5, rel_tol=1e-9)

    @pytest.mark.parametrize(
        "current_frac,expected",
        [
            (0.3, 1.0),  # start
            (0.4, 0.75),  # 25% through
            (0.5, 0.5),  # 50% through
            (0.6, 0.25),  # 75% through
            (0.7, 0.0),  # end
        ],
    )
    def test_decreasing_schedule_interpolation(self, current_frac: float, expected: float):
        d = {"start_value": 1.0, "end_value": 0.0, "start_frac": 0.3, "end_frac": 0.7}
        schedule = LinearSchedule(type="linear", **d)
        result = get_linear_schedule_value(schedule, current_frac_of_training=current_frac)
        assert math.isclose(result, expected, rel_tol=1e-9)

    @pytest.mark.parametrize(
        "current_frac,expected",
        [
            (0.0, 0.1),  # before start
            (0.2, 0.1),  # at start
            (0.4, 0.55),  # 50% through
            (0.6, 1.0),  # at end
            (1.0, 1.0),  # after end
        ],
    )
    def test_increasing_schedule(self, current_frac: float, expected: float):
        d = {"start_value": 0.1, "end_value": 1.0, "start_frac": 0.2, "end_frac": 0.6}
        schedule = LinearSchedule(type="linear", **d)
        result = get_linear_schedule_value(schedule, current_frac_of_training=current_frac)
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_full_range_schedule(self):
        # Schedule that covers full training range
        d = {"start_value": 10.0, "end_value": 1.0, "start_frac": 0.0, "end_frac": 1.0}
        schedule = LinearSchedule(type="linear", **d)
        result_start = get_linear_schedule_value(schedule, current_frac_of_training=0.0)
        result_mid = get_linear_schedule_value(schedule, current_frac_of_training=0.5)
        result_end = get_linear_schedule_value(schedule, current_frac_of_training=1.0)

        assert result_start == 10.0
        assert math.isclose(result_mid, 5.5, rel_tol=1e-9)
        assert result_end == 1.0

    def test_negative_values(self):
        # Test with negative values
        d = {"start_value": -1.0, "end_value": -5.0, "start_frac": 0.0, "end_frac": 1.0}
        schedule = LinearSchedule(type="linear", **d)
        result = get_linear_schedule_value(schedule, current_frac_of_training=0.5)
        assert math.isclose(result, -3.0, rel_tol=1e-9)


class TestCosineSchedule:
    def test_before_schedule_starts(self):
        d = {"start_value": 1.0, "end_value": 0.0, "start_frac": 0.3, "end_frac": 0.7}
        schedule = CosineSchedule(type="cosine", **d)
        result = get_cosine_schedule_value(schedule, current_frac_of_training=0.1)
        assert result == 1.0

    def test_after_schedule_ends(self):
        d = {"start_value": 1.0, "end_value": 0.0, "start_frac": 0.3, "end_frac": 0.7}
        schedule = CosineSchedule(type="cosine", **d)
        result = get_cosine_schedule_value(schedule, current_frac_of_training=0.9)
        assert result == 0.0

    def test_at_full_range_start(self):
        # At current_frac=0, cos(0) = 1, formula gives start_value
        d = {"start_value": 1.0, "end_value": 0.0, "start_frac": 0.0, "end_frac": 1.0}
        schedule = CosineSchedule(type="cosine", **d)
        result = get_cosine_schedule_value(schedule, current_frac_of_training=0.0)
        # cos(0) = 1, so: 0.0 + 0.5 * (1.0 - 0.0) * (1 + 1) = 1.0
        assert math.isclose(result, 1.0, rel_tol=1e-9)

    def test_at_full_range_end(self):
        # At current_frac=1, cos(pi) = -1, formula gives end_value
        d = {"start_value": 1.0, "end_value": 0.0, "start_frac": 0.0, "end_frac": 1.0}
        schedule = CosineSchedule(type="cosine", **d)
        result = get_cosine_schedule_value(schedule, current_frac_of_training=1.0)
        # cos(pi) = -1, so: 0.0 + 0.5 * (1.0 - 0.0) * (1 + (-1)) = 0.0
        assert math.isclose(result, 0.0, rel_tol=1e-9)

    def test_at_half_training(self):
        # At current_frac=0.5, cos(pi/2) = 0
        d = {"start_value": 1.0, "end_value": 0.0, "start_frac": 0.0, "end_frac": 1.0}
        schedule = CosineSchedule(type="cosine", **d)
        result = get_cosine_schedule_value(schedule, current_frac_of_training=0.5)
        # cos(pi/2) = 0, so: 0.0 + 0.5 * (1.0 - 0.0) * (1 + 0) = 0.5
        assert math.isclose(result, 0.5, rel_tol=1e-9)

    @pytest.mark.parametrize(
        "current_frac,expected_value",
        [
            (0.0, 1.0),  # cos(0) = 1
            (0.25, 0.8535533905932737),  # cos(pi/4) ≈ 0.707
            (0.5, 0.5),  # cos(pi/2) = 0
            (0.75, 0.14644660940672624),  # cos(3pi/4) ≈ -0.707
            (1.0, 0.0),  # cos(pi) = -1
        ],
    )
    def test_cosine_curve_full_range(self, current_frac: float, expected_value: float):
        d = {"start_value": 1.0, "end_value": 0.0, "start_frac": 0.0, "end_frac": 1.0}
        schedule = CosineSchedule(type="cosine", **d)
        result = get_cosine_schedule_value(schedule, current_frac_of_training=current_frac)
        assert math.isclose(result, expected_value, rel_tol=1e-9)

    def test_increasing_cosine_schedule(self):
        # Test with increasing values (start < end)
        d = {"start_value": 0.0, "end_value": 1.0, "start_frac": 0.0, "end_frac": 1.0}
        schedule = CosineSchedule(type="cosine", **d)
        result_start = get_cosine_schedule_value(schedule, current_frac_of_training=0.0)
        result_mid = get_cosine_schedule_value(schedule, current_frac_of_training=0.5)
        result_end = get_cosine_schedule_value(schedule, current_frac_of_training=1.0)

        assert math.isclose(result_start, 0.0, rel_tol=1e-9)
        assert math.isclose(result_mid, 0.5, rel_tol=1e-9)
        assert math.isclose(result_end, 1.0, rel_tol=1e-9)

    def test_partial_range_schedule(self):
        # Test behavior when start_frac and end_frac don't cover full range
        # This documents the potentially unexpected behavior mentioned in the WARNING
        d = {"start_value": 1.0, "end_value": 0.0, "start_frac": 0.25, "end_frac": 0.75}
        schedule = CosineSchedule(type="cosine", **d)

        # Within the schedule range, but cosine uses current_frac directly
        result_at_start_frac = get_cosine_schedule_value(schedule, current_frac_of_training=0.25)
        # cos(0.25*pi) ≈ 0.707, so: 0.0 + 0.5 * 1.0 * (1 + 0.707) ≈ 0.8535
        expected = 0.0 + 0.5 * (1.0 - 0.0) * (1 + math.cos(math.pi * 0.25))
        assert math.isclose(result_at_start_frac, expected, rel_tol=1e-9)

        # Just before end_frac (since at end_frac it returns end_value directly)
        result_just_before_end = get_cosine_schedule_value(schedule, current_frac_of_training=0.74)
        expected = 0.0 + 0.5 * (1.0 - 0.0) * (1 + math.cos(math.pi * 0.74))
        assert math.isclose(result_just_before_end, expected, rel_tol=1e-9)

        # At end_frac, should return end_value directly
        result_at_end_frac = get_cosine_schedule_value(schedule, current_frac_of_training=0.75)
        assert result_at_end_frac == 0.0

    def test_negative_values(self):
        # Test with negative values
        d = {"start_value": -1.0, "end_value": -5.0, "start_frac": 0.0, "end_frac": 1.0}
        schedule = CosineSchedule(type="cosine", **d)

        result_start = get_cosine_schedule_value(schedule, current_frac_of_training=0.0)
        result_end = get_cosine_schedule_value(schedule, current_frac_of_training=1.0)

        assert math.isclose(result_start, -1.0, rel_tol=1e-9)
        assert math.isclose(result_end, -5.0, rel_tol=1e-9)
