from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from esrb.benchmark.hazard import HazardType
from esrb.utils.hashing import hash_object


@dataclass(frozen=True)
class Counterexample:
    """Minimised witness of a safety violation. Immutable; content-addressed."""
    id: str                     # SHA-256 of the episode_data
    hazard_type: HazardType
    generation_found: int
    violated_constraints: list[str]
    episode_data: Any           # minimal reproducing state
    trigger_component: str
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_episode(
        cls,
        episode_data: Any,
        hazard_type: HazardType,
        generation: int,
        violated_constraints: list[str],
        trigger_component: str,
    ) -> "Counterexample":
        return cls(
            id=hash_object(episode_data),
            hazard_type=hazard_type,
            generation_found=generation,
            violated_constraints=violated_constraints,
            episode_data=episode_data,
            trigger_component=trigger_component,
        )


class CounterexampleLibrary:
    """
    Append-only, content-addressed library of safety counterexamples.
    The agent cannot modify or delete entries.
    """

    def __init__(self) -> None:
        self._store: dict[str, Counterexample] = {}

    def add(self, cx: Counterexample) -> None:
        self._store[cx.id] = cx

    def relevant_to(self, changed_components: set[str]) -> list[Counterexample]:
        """Return counterexamples whose trigger component is in the changed set."""
        return [cx for cx in self._store.values() if cx.trigger_component in changed_components]

    def all(self) -> list[Counterexample]:
        return list(self._store.values())

    def __len__(self) -> int:
        return len(self._store)
