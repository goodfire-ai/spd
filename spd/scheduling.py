from math import cos, pi

from spd.configs import CoeffSchedule, CosineSchedule, LinearSchedule


def _get_linear_schedule_value(
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


def _get_cosine_schedule_value(
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

def get_value(
    value: CoeffSchedule | float,
    current_frac_of_training: float,
) -> float:
    match value:
        case float() | int():
            return value
        case LinearSchedule():
            return _get_linear_schedule_value(value, current_frac_of_training)
        case CosineSchedule():
            return _get_cosine_schedule_value(value, current_frac_of_training)