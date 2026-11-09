import numpy as np
import pytest

from esrb.detection.gate_cases import gate_case
from esrb.tasks.agentic import GraspForceTask, SensorDriftTask
from esrb.tasks.contact_rich import PegInsertionTask
from esrb.tasks.dynamic_workspace import DynamicObstacleTask
from esrb.tasks.long_horizon import PickPlaceTask
from esrb.tasks.tool_handling import ToolTightenTask


@pytest.mark.parametrize("task_type", [PegInsertionTask, DynamicObstacleTask, PickPlaceTask, ToolTightenTask])
def test_seeded_and_structured_reset_match_and_can_be_replayed(task_type):
    task = task_type()
    case = gate_case(task, 42, "fresh")
    legacy = task.reset(42)
    replayed = task.reset(99, disturbance=case["disturbance"])
    np.testing.assert_array_equal(replayed.state, legacy.state)
    field = next(iter(task.reset_bounds))
    point = {**case["disturbance"], field: task.reset_bounds[field][1]}
    changed = task.reset(99, disturbance=point)
    assert not np.array_equal(changed.state, legacy.state)
    assert task._episode_log["reset_readback"] == point
    with pytest.raises(ValueError, match="must be in"):
        task.reset(disturbance={**point, field: 99.0})


@pytest.mark.parametrize("task_type", [GraspForceTask, SensorDriftTask])
def test_force_fixture_load_scale_is_an_executed_disturbance(task_type):
    task = task_type()
    if isinstance(task, SensorDriftTask):
        task.configure(true_force=30.0, sensor_bias=-20.0)
    task.reset(42, disturbance={"load_scale": 0.8})
    done = False
    while not done:
        _, _, done, episode = task.step(np.array([30.0]))
    assert episode["reset_readback"] == {"load_scale": 0.8}
    assert all(force == [24.0, 0.0, 0.0] for force in episode["contact_forces"])
    if isinstance(task, SensorDriftTask):
        assert all(value == 4.0 for value in episode["sensor_readings"])
        assert episode["applied_task_config"] == {"true_force": 30.0, "sensor_bias": -20.0}
