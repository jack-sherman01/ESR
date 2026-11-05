from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskObservation:
    state: Any
    info: dict


@dataclass(frozen=True)
class TaskResult:
    task_score: float           # [0, 1] — higher is better
    success: bool
    episode_data: dict          # raw telemetry for oracle consumption
    metadata: dict


class BaseTask(ABC):
    """Abstract task. Subclasses must not expose internal reward functions to oracles."""

    name: str = "base_task"

    @abstractmethod
    def reset(self, seed: int = 0) -> TaskObservation:
        ...

    @abstractmethod
    def step(self, action: Any) -> tuple[TaskObservation, float, bool, dict]:
        """Returns (obs, reward, done, info)."""
        ...

    @abstractmethod
    def evaluate(self, episode_data: dict) -> TaskResult:
        """Task-side evaluation (may drift; not the safety oracle)."""
        ...

    @property
    @abstractmethod
    def action_space(self) -> Any:
        ...

    @property
    @abstractmethod
    def observation_space(self) -> Any:
        ...
