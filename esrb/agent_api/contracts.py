"""JSON proposal contracts adapted from ESR_benchmark_current."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Protocol

SCHEMA = "esrb.agent/1"
MAX_MESSAGE_BYTES = 1024 * 1024
_ROLES = {"policy", "memory", "data", "reward", "evaluator", "tools"}


class ProposalError(ValueError):
    """A candidate request or response violated the proposal contract."""


def require(condition, message):
    if not condition:
        raise ProposalError(message)


def identity(document):
    """Hash canonical, finite JSON for a stable identity across processes."""
    try:
        encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProposalError("proposal protocol requires finite JSON") from exc
    require(len(encoded) <= MAX_MESSAGE_BYTES, "proposal message exceeds 1 MiB")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Consumer:
    role: str
    format: str
    schema_id: str


@dataclass(frozen=True)
class Manifest:
    adapter_id: str
    revision: str
    consumers: tuple[Consumer, ...]
    origin: str
    schema: str = SCHEMA

    @property
    def id(self):
        roles = {item.role for item in self.consumers}
        require(self.schema == SCHEMA and self.adapter_id and self.revision, "invalid proposal manifest")
        require(roles and roles <= _ROLES and len(roles) == len(self.consumers), "invalid manifest roles")
        require(all(item.format and item.schema_id for item in self.consumers), "invalid consumer format")
        return identity(asdict(self))


@dataclass(frozen=True)
class ParentView:
    version: str
    components: dict
    manifest_id: str


@dataclass(frozen=True)
class PublicFeedback:
    task_id: str
    generation: int
    history: tuple[dict, ...] = ()
    schema: str = SCHEMA

    def validate(self):
        require(self.schema == SCHEMA and self.task_id and self.generation >= 1, "invalid public feedback")
        for row in self.history:
            require(set(row) == {"generation", "slot", "task_scores"}, "unexpected public feedback fields")
            require(type(row["generation"]) is int and 1 <= row["generation"] < self.generation
                    and type(row["slot"]) is int and row["slot"] >= 0, "invalid public feedback identity")
            require(type(row["task_scores"]) is list and all(
                type(score) in (int, float) and math.isfinite(score) for score in row["task_scores"]
            ), "invalid public task score")
        return identity(asdict(self))


@dataclass(frozen=True)
class Proposal:
    base_version: str
    manifest_id: str
    replacements: dict
    formats: dict
    schema: str = SCHEMA

    @classmethod
    def from_dict(cls, value):
        require(type(value) is dict and set(value) == {
            "schema", "base_version", "manifest_id", "replacements", "formats",
        }, "invalid proposal fields")
        return cls(**value)


class CandidateProposer(Protocol):
    manifest: Manifest

    def propose(self, parent: ParentView, feedback: PublicFeedback) -> Proposal:
        """Return role replacements for the supplied parent version."""
        ...
