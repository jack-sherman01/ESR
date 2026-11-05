from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from esrb.utils.hashing import hash_object


@dataclass(frozen=True)
class GenerationRecord:
    """Immutable record of one policy generation."""
    generation: int
    policy_digest: str
    data_digest: str
    evaluator_digest: str
    update_operator: str
    task_score: float
    safety_violated: bool
    oracle_verdicts: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class LineageTracker:
    """Append-only log of generation records; each entry is content-addressed."""

    def __init__(self) -> None:
        self._records: list[GenerationRecord] = []

    def record(self, rec: GenerationRecord) -> None:
        if rec.generation != len(self._records):
            raise ValueError(
                f"Expected generation {len(self._records)}, got {rec.generation}"
            )
        self._records.append(rec)

    def get(self, generation: int) -> GenerationRecord:
        return self._records[generation]

    def all(self) -> list[GenerationRecord]:
        return list(self._records)

    def digest(self) -> str:
        """Hash of the entire lineage (tamper detection)."""
        return hash_object(self._records)

    def __len__(self) -> int:
        return len(self._records)
