from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HazardType(str, Enum):
    REWARD_HACKING = "reward_hacking"
    EVALUATOR_DRIFT = "evaluator_drift"
    MEMORY_CONTAMINATION = "memory_contamination"
    POLICY_REGRESSION = "policy_regression"
    DISTRIBUTION_SHIFT = "distribution_shift"
    TOOL_PERMISSION_MISUSE = "tool_permission_misuse"
    SAFETY_FORGETTING = "safety_forgetting"
    SENSOR_DRIFT = "sensor_drift"
    STOCHASTIC_FAULT = "stochastic_fault"


@dataclass(frozen=True)
class HazardModel:
    hazard_type: HazardType
    severity: float          # [0, 1]
    trigger_description: str
    observable_evidence: str
    physical_consequence: str
    expected_mitigation: str
