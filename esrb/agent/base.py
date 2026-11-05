from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Policy agent that participates in the ESRB improvement loop."""

    name: str = "base_agent"

    @abstractmethod
    def act(self, observation: Any) -> Any:
        """Return an action given an observation."""
        ...

    @abstractmethod
    def update(self, episode_data: dict) -> None:
        """Update policy parameters from one episode of experience."""
        ...

    @abstractmethod
    def state_dict(self) -> dict:
        """Serialisable snapshot for versioning and lineage tracking."""
        ...

    @abstractmethod
    def load_state_dict(self, state: dict) -> None:
        ...
