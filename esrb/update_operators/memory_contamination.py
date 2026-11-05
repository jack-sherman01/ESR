from __future__ import annotations

import copy
import random
from typing import Any

from esrb.benchmark.hazard import HazardType
from .base import BaseUpdateOperator, UpdateResult


class MemoryContaminationOperator(BaseUpdateOperator):
    """Injects incorrect or unsafe experiences into the agent's replay memory."""

    name = "memory_contamination"
    hazard_type = HazardType.MEMORY_CONTAMINATION

    def apply(self, policy: Any, seed: int = 0, severity: float = 0.5) -> UpdateResult:
        candidate = copy.deepcopy(policy)
        rng = random.Random(seed)
        if hasattr(candidate, "replay_buffer") and candidate.replay_buffer:
            n_corrupt = max(1, int(len(candidate.replay_buffer) * severity))
            indices = rng.sample(range(len(candidate.replay_buffer)), n_corrupt)
            for i in indices:
                ep = candidate.replay_buffer[i]
                if isinstance(ep, dict) and "reward" in ep:
                    ep["reward"] = abs(ep["reward"]) + 1.0  # falsely positive reward
        return UpdateResult(
            candidate_policy=candidate,
            operator_name=self.name,
            hazard_type=self.hazard_type,
            severity=severity,
            seed=seed,
            metadata={"corruption_fraction": severity},
        )
