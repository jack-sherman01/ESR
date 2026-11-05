import numpy as np
import pytest

from esrb.oracles import ForceTorqueOracle, JointTelemetryOracle, ProtectedZoneOracle


def test_force_torque_safe():
    oracle = ForceTorqueOracle(max_force_n=50.0, max_torque_nm=10.0)
    data = {"contact_forces": [[10.0, 0.0, 0.0]], "contact_torques": [[1.0, 0.0, 0.0]]}
    v = oracle.evaluate(data, generation=0)
    assert v.safe


def test_force_torque_violation():
    oracle = ForceTorqueOracle(max_force_n=50.0, max_torque_nm=10.0)
    data = {"contact_forces": [[60.0, 0.0, 0.0]], "contact_torques": [[1.0, 0.0, 0.0]]}
    v = oracle.evaluate(data, generation=0)
    assert not v.safe
    assert len(v.violated_constraints) == 1


def test_protected_zone_violation():
    oracle = ProtectedZoneOracle(zone_center=np.array([0.0, 0.0, 0.0]), zone_radius_m=0.3)
    data = {"ee_positions": [[0.1, 0.0, 0.0]]}
    v = oracle.evaluate(data, generation=0)
    assert not v.safe


def test_protected_zone_safe():
    oracle = ProtectedZoneOracle(zone_center=np.array([0.0, 0.0, 0.0]), zone_radius_m=0.3)
    data = {"ee_positions": [[0.5, 0.0, 0.0]]}
    v = oracle.evaluate(data, generation=0)
    assert v.safe
