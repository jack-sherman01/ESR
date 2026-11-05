from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OracleVerdict:
    """Immutable verdict from an independent safety oracle."""
    oracle_name: str
    safe: bool
    signal_values: dict[str, float]   # raw sensor readings
    violated_constraints: list[str]
    generation: int


class BaseOracle(ABC):
    """
    Read-only independent safety oracle.

    Oracle code and thresholds are immutable to the agent; they cannot be
    modified during an improvement loop. Subclasses must enforce this.
    """

    name: str = "base_oracle"

    @abstractmethod
    def evaluate(self, episode_data: dict, generation: int) -> OracleVerdict:
        """Return a safety verdict from independent sensor signals."""
        ...

    @abstractmethod
    def calibration_check(self) -> bool:
        """Return True if the oracle's sensors pass a self-calibration test."""
        ...

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_threshold") or name.startswith("_rule"):
            raise AttributeError("Oracle thresholds and rules are immutable.")
        super().__setattr__(name, value)
