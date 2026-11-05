from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class CandidateEvaluation:
    policy: Any
    task_scores: list[float]        # repeated trial scores
    safety_violations: list[bool]   # per-trial oracle verdicts (True = unsafe)
    generation: int


def ucb_safety_risk(eval: CandidateEvaluation, alpha: float = 0.1) -> float:
    """Upper confidence bound on safety violation probability at level 1-alpha."""
    n = len(eval.safety_violations)
    if n == 0:
        return 1.0
    p_hat = sum(eval.safety_violations) / n
    # Wilson score upper bound approximation
    z = _z_score(1.0 - alpha)
    denom = 1 + z**2 / n
    centre = p_hat + z**2 / (2 * n)
    spread = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    return (centre + spread) / denom


def _z_score(confidence: float) -> float:
    """Rough normal quantile for common confidence levels."""
    table = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
    return table.get(round(confidence, 2), 1.645)


class RiskSensitiveSelector:
    """
    Selects the best candidate from the Pareto set over task performance
    and safety risk, subject to UCB constraint:

        max J_task(π)  s.t.  UCB_{1-α}(R_safety(π)) ≤ ε
    """

    def __init__(self, epsilon: float = 0.05, alpha: float = 0.1) -> None:
        self.epsilon = epsilon
        self.alpha = alpha

    def select(self, candidates: list[CandidateEvaluation]) -> CandidateEvaluation | None:
        """
        Return the highest-mean-task-score candidate that satisfies the UCB
        safety constraint. Returns None if all candidates are unsafe.
        """
        feasible = [
            c for c in candidates
            if ucb_safety_risk(c, self.alpha) <= self.epsilon
        ]
        if not feasible:
            return None
        return max(feasible, key=lambda c: sum(c.task_scores) / len(c.task_scores))
