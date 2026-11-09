"""Budgeted coordinate reduction of a failing environment toward a reference.

The reduction keeps only measured violations. The search does not assume
monotonicity or guarantee global minimality.
"""

import math


def minimize(environment, reference, evaluate, budget=8, tolerance=1e-6):
    """Try removing each offset, then bisect when removal clears the violation."""
    if type(budget) is not int or budget < 1:
        raise ValueError("budget must be a positive integer")
    if set(environment) != set(reference) or not environment:
        raise ValueError("reference must specify the same environment fields")
    for point in (environment, reference):
        if not all(type(value) in (int, float) and math.isfinite(value) for value in point.values()):
            raise ValueError("environment values must be finite numbers")

    current = dict(environment)
    history = []

    def trial(point):
        result = evaluate(dict(point))
        if not isinstance(result, dict) or type(result.get("unsafe")) is not bool:
            raise ValueError("evaluator must return a boolean unsafe label")
        history.append({"environment": dict(point), "result": result})
        return result

    original_result = trial(current)
    if not original_result["unsafe"]:
        raise ValueError("initial counterexample is not a verified violation")
    retained_result = original_result
    for field in sorted(current):
        if len(history) >= budget:
            break
        if abs(current[field] - reference[field]) <= tolerance:
            continue

        endpoint = {**current, field: reference[field]}
        result = trial(endpoint)
        if result["unsafe"]:
            current, retained_result = endpoint, result
            continue

        safe, failing = reference[field], current[field]
        while len(history) < budget and abs(safe - failing) > tolerance:
            point = {**current, field: (safe + failing) / 2}
            result = trial(point)
            if result["unsafe"]:
                current, retained_result = point, result
                failing = current[field]
            else:
                safe = point[field]

    return {
        "original": environment, "reference": reference, "reduced": current,
        "original_result": original_result, "reduced_result": retained_result,
        "evaluations": len(history), "history": history,
    }
