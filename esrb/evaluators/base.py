from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluatorVerdict:
    task_score: float
    success: bool
    metadata: dict


class BaseEvaluator(ABC):
    """
    Task-side evaluator. Unlike oracles, evaluators may be adapted during the
    improvement loop — but all changes are audited and version-tracked.
    """

    name: str = "base_evaluator"
    version: int = 0

    @abstractmethod
    def evaluate(self, episode_data: dict) -> EvaluatorVerdict:
        ...

    def bump_version(self) -> None:
        object.__setattr__(self, "version", self.version + 1)
