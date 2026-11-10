"""Canonical structured cases and task-native operating-domain edges."""

from esrb.tasks.reset_domain import validate_disturbance
from esrb.utils.hashing import hash_dict


def gate_case(task, seed, phase, disturbance=None, **metadata):
    point = validate_disturbance(
        task.reset_bounds, task.disturbance_for_seed(seed) if disturbance is None else disturbance,
    )
    return {
        **metadata, "phase": phase, "seed": seed, "disturbance": point,
        "reset_key": f"{task.name}/reset/" + hash_dict(point),
    }


def boundary_catalog(task, primary_test):
    """Vary one coordinate at a time from the task's nominal reset."""
    catalog = []
    for field in getattr(task, "boundary_fields", task.reset_bounds):
        low, high = task.reset_bounds[field]
        for label, value in (("low", low), ("high", high)):
            catalog.append({
                "id": f"{task.name}_{field}_{label}", "tests": [primary_test],
                "disturbance": {**task.reset_reference, field: value},
            })
    return catalog


def control_steps(trace):
    """Use the runner's step count, falling back to legacy telemetry lengths."""
    if "control_step_count" in trace:
        return trace["control_step_count"]
    if "ee_positions" in trace:
        return len(trace["ee_positions"])
    return len(trace["contact_forces"])
