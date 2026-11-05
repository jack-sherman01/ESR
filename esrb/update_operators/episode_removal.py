from __future__ import annotations

import copy
import random
from typing import Any

from esrb.benchmark.hazard import HazardType
from .base import BaseUpdateOperator, UpdateResult


class EpisodeRemovalOperator(BaseUpdateOperator):
    """Removes rare failure episodes from the agent-visible summary."""

    name = "episode_removal"
    hazard_type = HazardType.SAFETY_FORGETTING

    def apply(self, policy: Any, seed: int = 0, severity: float = 0.5) -> UpdateResult:
        candidate = copy.deepcopy(policy)
        rng = random.Random(seed)
        removed = 0
        if hasattr(candidate, "episode_summary"):
            failures = [e for e in candidate.episode_summary if not e.get("success", True)]
            n_remove = int(len(failures) * severity)
            to_remove = set(id(e) for e in rng.sample(failures, min(n_remove, len(failures))))
            candidate.episode_summary = [
                e for e in candidate.episode_summary if id(e) not in to_remove
            ]
            removed = len(to_remove)
        return UpdateResult(
            candidate_policy=candidate,
            operator_name=self.name,
            hazard_type=self.hazard_type,
            severity=severity,
            seed=seed,
            metadata={"episodes_removed": removed},
        )
