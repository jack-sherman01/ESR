from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .hashing import hash_object


@dataclass(frozen=True)
class VersionedArtifact:
    """Immutable snapshot of any serialisable artifact."""
    name: str
    generation: int
    artifact: Any
    digest: str = field(init=False)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        object.__setattr__(self, "digest", hash_object(self.artifact))

    def __repr__(self) -> str:
        return f"VersionedArtifact(name={self.name!r}, gen={self.generation}, digest={self.digest[:8]})"
