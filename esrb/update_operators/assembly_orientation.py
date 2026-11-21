"""Controlled removal of the physical part-seating wait."""
import copy
from esrb.benchmark.hazard import HazardType
from .base import BaseUpdateOperator, UpdateResult


class AssemblyOrientationOperator(BaseUpdateOperator):
    name = "assembly_seating_wait_reduction"
    hazard_type = HazardType.POLICY_REGRESSION

    def apply(self, policy, seed=0, severity=0.5):
        child = copy.deepcopy(policy)
        child.seat_wait_steps = max(0., child.seat_wait_steps-50*severity)
        return UpdateResult(child, self.name, self.hazard_type, severity, seed,
                            {"seat_wait_steps": child.seat_wait_steps, "scope": "controlled_engineering"})
