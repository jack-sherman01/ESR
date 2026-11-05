from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from esrb.benchmark.hazard import HazardType


@dataclass(frozen=True)
class UpdateResult:
    candidate_policy: Any
    operator_name: str
    hazard_type: HazardType
    severity: float
    seed: int
    metadata: dict


class BaseUpdateOperator(ABC):
    """
    Adversarial update operator that injects a controlled fault into a policy
    or its supporting components. Each operator targets one HazardType.
    """

    name: str = "base_operator"
    hazard_type: HazardType = HazardType.REWARD_HACKING

    @abstractmethod
    def apply(self, policy: Any, seed: int = 0, severity: float = 0.5) -> UpdateResult:
        """Return a modified candidate policy that embeds the fault."""
        ...
