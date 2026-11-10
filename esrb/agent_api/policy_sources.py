"""Shared proposal sources for policies backed by numeric parameters."""

import json
import math
from dataclasses import dataclass

from esrb.agent.base import BaseAgent
from esrb.update_operators.base import BaseUpdateOperator
from esrb.utils.hashing import hash_object
from .contracts import Consumer, Manifest, MAX_MESSAGE_BYTES, Proposal, ProposalError, identity, require


@dataclass(frozen=True)
class ParameterPolicy:
    label: str
    policy_type: type[BaseAgent]
    consumer: Consumer
    json_adapter_id: str

    @property
    def json_manifest(self):
        return Manifest(self.json_adapter_id, "1", (self.consumer,), "deterministic_json_fixture")

    @property
    def validators(self):
        return {"policy": (self.consumer, self.validate)}

    def snapshot(self, policy):
        return {"policy": policy.state_dict()}

    def policy_digest(self, components):
        return hash_object(self.build(components["policy"]).state_dict())

    def validate(self, state):
        """Validate structure and values, leaving safety verdicts to the gate."""
        fields = self.policy_type().state_dict()
        require(type(state) is dict and set(state) == set(fields), f"{self.label} policy fields must match the schema")
        try:
            valid = all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                        for value in state.values())
        except OverflowError:
            valid = False
        require(valid, f"{self.label} policy parameters must be finite nonnegative numbers")
        return {key: float(state[key]) for key in fields}

    def build(self, state):
        policy = self.policy_type()
        policy.load_state_dict(self.validate(state))
        return policy

    def load_proposal_file(self, path, generations):
        with path.open("rb") as source:
            raw = source.read(MAX_MESSAGE_BYTES + 1)
        require(len(raw) <= MAX_MESSAGE_BYTES, "proposal file exceeds 1 MiB")
        try:
            document = json.loads(raw, object_pairs_hook=_unique_object)
        except ProposalError:
            raise
        except (UnicodeError, ValueError) as exc:
            raise ProposalError(f"invalid proposal JSON: {exc}") from exc
        require(type(document) is dict and set(document) in ({"proposals"}, {"initial_policy", "proposals"}),
                "proposal file must contain proposals and optionally initial_policy")
        identity(document)
        rows = document["proposals"]
        require(type(rows) is list and len(rows) == generations, "provide one proposal per generation")
        initial = self.build(document.get("initial_policy", self.policy_type().state_dict()))
        return JsonProposer(tuple(Proposal.from_dict(row) for row in rows), self.json_manifest), initial


@dataclass(frozen=True)
class OperatorProposer:
    operator: BaseUpdateOperator
    seed: int
    severity: float
    policy: ParameterPolicy

    @property
    def manifest(self):
        return Manifest(self.operator.name, "1", (self.policy.consumer,), "controlled_operator")

    def propose(self, parent, feedback):
        policy = self.policy.build(parent.components["policy"])
        candidate = self.operator.apply(
            policy, seed=self.seed + feedback.generation - 1, severity=self.severity,
        ).candidate_policy
        return Proposal(parent.version, parent.manifest_id,
                        {"policy": candidate.state_dict()}, {"policy": self.policy.consumer.format})


@dataclass(frozen=True)
class JsonProposer:
    proposals: tuple[Proposal, ...]
    manifest: Manifest

    def propose(self, parent, feedback):
        require(feedback.generation <= len(self.proposals), "missing proposal for this generation")
        return self.proposals[feedback.generation - 1]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result
