"""Reproducible product-uniform sampling for the risk gate."""

import numpy as np

from esrb.detection.gate_cases import gate_case
from esrb.detection.risk_selection import CandidateEvaluation, ucb_safety_risk
from esrb.utils.hashing import hash_dict


def sampling_protocol(task, seed, samples, bounds):
    """Resolve configured coordinates against the task's supported reset domain."""
    unknown = bounds.keys() - task.reset_bounds.keys()
    if unknown:
        raise ValueError(f"{task.name}: unknown risk coordinates {', '.join(sorted(unknown))}")
    domain = {}
    for field, (low, high) in task.reset_bounds.items():
        left, right = bounds.get(field, (low, high))
        if not low <= left <= right <= high:
            raise ValueError(f"{task.name}.{field}: risk bounds must be ordered within [{low}, {high}]")
        domain[field] = [float(left), float(right)]
    return {"distribution": "uniform", "seed": seed, "samples": samples, "bounds": domain}


def risk_cases(task, protocol, instance_name, generation):
    cases = []
    for index in range(protocol["samples"]):
        seed = int(hash_dict({
            "purpose": "independent_risk", "seed": protocol["seed"],
            "instance": instance_name, "generation": generation, "index": index,
        })[:32], 16)
        rng = np.random.default_rng(seed)
        point = {field: float(rng.uniform(*bounds)) for field, bounds in sorted(protocol["bounds"].items())}
        cases.append(gate_case(task, seed, "risk", point))
    return cases


def risk_summary(executed, generation, alpha):
    """Summarize risk draws and targeted regressions from verified rollout labels."""
    samples = [case for case in executed if case["phase"] == "risk"]
    labels = [case["safe"] for case in samples]
    unknown = labels.count(None)
    violations = [label is False for label in labels]
    upper_bound = None
    if labels and not unknown:
        upper_bound = ucb_safety_risk(CandidateEvaluation(None, [], violations, generation), alpha)
    return {
        "sample_count": len(labels), "violations": sum(violations), "unknown": unknown,
        "upper_bound": upper_bound, "evidence_ids": [case["evidence_id"] for case in samples],
        "regression_violations": [
            case["evidence_id"] for case in executed
            if case["phase"] in {"replay", "boundary", "reduction"} and case["safe"] is False
        ],
    }
