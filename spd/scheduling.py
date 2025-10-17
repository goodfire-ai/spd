from math import cos, pi

from spd.configs import CosineSchedule, LinearSchedule


def get_linear_schedule_value(
    schedule: LinearSchedule,
    current_frac_of_training: float,
) -> float:
    if current_frac_of_training < schedule.start_frac:
        return schedule.start_value
    elif current_frac_of_training >= schedule.end_frac:
        return schedule.end_value
    else:
        return schedule.start_value + (schedule.end_value - schedule.start_value) * (
            current_frac_of_training - schedule.start_frac
        ) / (schedule.end_frac - schedule.start_frac)


def get_cosine_schedule_value(
    schedule: CosineSchedule,
    current_frac_of_training: float,
) -> float:
    if current_frac_of_training < schedule.start_frac:
        return schedule.start_value
    elif current_frac_of_training >= schedule.end_frac:
        return schedule.end_value
    else:
        return schedule.end_value + 0.5 * (schedule.start_value - schedule.end_value) * (
            1 + cos(pi * current_frac_of_training)
        )
