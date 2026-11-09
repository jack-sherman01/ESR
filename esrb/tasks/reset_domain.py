"""Validation shared by task-native structured resets."""

from math import isfinite


def validate_disturbance(bounds, point):
    if set(point) != set(bounds):
        raise ValueError("reset disturbance fields do not match the task domain")
    for field, (low, high) in bounds.items():
        value = point[field]
        if isinstance(value, bool) or not isfinite(value) or not low <= value <= high:
            raise ValueError(f"reset disturbance {field} must be in [{low}, {high}]")
    return {field: float(point[field]) or 0.0 for field in bounds}
