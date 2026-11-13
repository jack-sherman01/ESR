"""Proposal adapters for trusted in-process fault injection operators."""

from dataclasses import dataclass, field
from functools import partial
from math import isfinite

from esrb.agent.base import BaseAgent
from esrb.agent.mlp_policy import MLPInsertionPolicy
from esrb.update_operators.base import BaseUpdateOperator
from esrb.utils.hashing import hash_object

from .contracts import Consumer, Manifest, Proposal, identity, require


@dataclass(frozen=True)
class ControlledProfile:
    policy_type: type[BaseAgent]
    mutable_roles: tuple[str, ...]
    fields_by_role: dict[str, tuple[str, ...]] = field(default_factory=dict)
    nominal_task_config: dict[str, float] | None = None
    task_config_metadata: dict[str, str] = field(default_factory=dict)

    @property
    def validators(self):
        return {
            role: (
                Consumer(role, f"esrb.{self.policy_type.name}.{role}/1", f"{self.policy_type.name}-{role}/1"),
                partial(self.validate, role),
            )
            for role in self.mutable_roles
        }

    def snapshot(self, policy, task_config=None):
        require(type(policy) is self.policy_type, "unexpected operator policy type")
        state = policy.state_dict()
        components = {
            role: {name: state.pop(name) for name in fields}
            for role, fields in self.fields_by_role.items()
        }
        if state:
            components["policy"] = state
        if isinstance(policy, MLPInsertionPolicy):
            components["memory"] = policy.training_identity()
        if self.nominal_task_config is not None:
            components["data"] = dict(self.nominal_task_config if task_config is None else task_config)
        return components

    def candidate_config(self, metadata):
        if self.nominal_task_config is not None:
            return {name: metadata[key] for name, key in self.task_config_metadata.items()}
        return None

    def validate(self, role, document):
        require(type(document) is dict and bool(document), f"invalid {role} component")
        identity(document)
        if role == "memory":
            require(set(document) == {"replay_buffer", "episode_summary", "optimizer", "rng"}
                    and all(type(value) is str and len(value) == 64 for value in document.values()),
                    "invalid MLP training identity")
        elif role == "data":
            require(set(document) == set(self.nominal_task_config)
                    and all(type(value) in (int, float) and isfinite(value) for value in document.values())
                    and document.get("true_force", 0.0) >= 0
                    and 0 <= document.get("fault_prob", 0.0) <= 1, "invalid task configuration")
        elif role != "policy" or self.policy_type is not MLPInsertionPolicy:
            if role == "policy":
                fields = set(self.policy_type().state_dict()).difference(
                    name for names in self.fields_by_role.values() for name in names
                )
            else:
                fields = set(self.fields_by_role[role])
            require(set(document) == fields, f"invalid {role} fields")
            for name, value in document.items():
                if name == "tool_permissions":
                    require(type(value) is list and all(type(item) is str for item in value)
                            and value == sorted(set(value)), "invalid tool permissions")
                else:
                    require(type(value) in (int, float) and isfinite(value) and value >= 0,
                            f"invalid {role} parameter: {name}")
        else:
            require(set(document) == {"net_state", "lr", "buffer_len", "summary_len"}, "invalid MLP policy fields")
        return document

    def policy_digest(self, components):
        state = dict(components.get("policy", {}))
        for role in self.fields_by_role:
            state.update(components[role])
        if self.policy_type is MLPInsertionPolicy:
            # The receipt contains the state_dict, with large training state bound separately.
            return hash_object(state)
        policy = self.policy_type()
        policy.load_state_dict(state)
        return hash_object(policy.state_dict())


class TrustedOperatorProposer:
    """Keep full operator output in memory while binding it to a typed proposal."""

    def __init__(self, profile: ControlledProfile, operator: BaseUpdateOperator, parent, seed, severity):
        self.profile = profile
        self.operator = operator
        self.parent = parent
        self.seed = seed
        self.severity = severity
        self.manifest = Manifest(operator.name, "1", tuple(
            consumer for consumer, _ in profile.validators.values()
        ), "controlled_operator")
        self._candidate = None
        self._components = None

    def propose(self, parent, feedback):
        result = self.operator.apply(self.parent, seed=self.seed + feedback.generation - 1,
                                     severity=self.severity)
        require(result.candidate_policy is not self.parent, "operator must produce a separate candidate")
        self._candidate = result.candidate_policy
        self._components = self.profile.snapshot(self._candidate, self.profile.candidate_config(result.metadata))
        replacements = {role: self._components[role] for role in self.profile.mutable_roles}
        return Proposal(parent.version, parent.manifest_id, replacements,
                        {item.role: item.format for item in self.manifest.consumers})

    def consume(self, child):
        require(self._candidate is not None and child == self._components,
                "operator output does not match the validated proposal")
        return self._candidate, child.get("data")
