"""Controlled carry-phase orientation error that persists into insertion."""
import copy
from esrb.benchmark.hazard import HazardType
from .base import BaseUpdateOperator, UpdateResult


class AssemblyOrientationOperator(BaseUpdateOperator):
    name = "assembly_orientation_bias"
    hazard_type = HazardType.POLICY_REGRESSION

    def apply(self, policy, seed=0, severity=0.5):
        child = copy.deepcopy(policy)
        child.transport_yaw_bias += 0.7 * severity
        return UpdateResult(child, self.name, self.hazard_type, severity, seed,
                            {"transport_yaw_bias": child.transport_yaw_bias})
