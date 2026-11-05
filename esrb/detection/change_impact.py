from __future__ import annotations

from enum import Enum
from typing import Any

import networkx as nx


class ComponentType(str, Enum):
    POLICY = "policy"
    MEMORY = "memory"
    DATA = "data"
    REWARD = "reward"
    EVALUATOR = "evaluator"
    TOOLS = "tools"
    ROBOT_INTERFACE = "robot_interface"


# Canonical data-flow edges in the agentic improvement loop.
_DEFAULT_EDGES: list[tuple[ComponentType, ComponentType]] = [
    (ComponentType.DATA, ComponentType.POLICY),
    (ComponentType.DATA, ComponentType.REWARD),
    (ComponentType.REWARD, ComponentType.POLICY),
    (ComponentType.MEMORY, ComponentType.POLICY),
    (ComponentType.MEMORY, ComponentType.DATA),
    (ComponentType.EVALUATOR, ComponentType.POLICY),
    (ComponentType.EVALUATOR, ComponentType.REWARD),
    (ComponentType.POLICY, ComponentType.TOOLS),
    (ComponentType.POLICY, ComponentType.ROBOT_INTERFACE),
    (ComponentType.TOOLS, ComponentType.ROBOT_INTERFACE),
]

# Maps each component to the safety test suites it triggers.
_REQUIRED_TESTS: dict[ComponentType, list[str]] = {
    ComponentType.POLICY:          ["dynamics_safety", "force_safety", "regression_suite"],
    ComponentType.EVALUATOR:       ["evaluator_integrity", "cross_view_agreement"],
    ComponentType.REWARD:          ["reward_hacking_scan", "force_safety"],
    ComponentType.MEMORY:          ["memory_integrity", "regression_suite"],
    ComponentType.DATA:            ["distribution_check", "regression_suite"],
    ComponentType.TOOLS:           ["permission_audit", "tool_timing_check"],
    ComponentType.ROBOT_INTERFACE: ["dynamics_safety", "force_safety"],
}


class ChangeImpactGraph:
    """
    Dependency graph over system components.
    Given a set of changed nodes, computes the downstream components
    and the union of required safety tests.
    """

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        for src, dst in _DEFAULT_EDGES:
            self._graph.add_edge(src, dst)

    def impacted_components(self, changed: set[ComponentType]) -> set[ComponentType]:
        """Return changed nodes plus all downstream dependents."""
        reachable: set[ComponentType] = set(changed)
        for node in changed:
            reachable |= nx.descendants(self._graph, node)
        return reachable

    def required_tests(self, changed: set[ComponentType]) -> list[str]:
        """Union of required test suites for all impacted components."""
        impacted = self.impacted_components(changed)
        tests: set[str] = set()
        for c in impacted:
            tests |= set(_REQUIRED_TESTS.get(c, []))
        return sorted(tests)
