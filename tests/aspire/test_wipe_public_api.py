"""High-level Wipe motions never turn a fixed-tool placeholder into actuation."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest

pytest.importorskip("cap_harness", reason="Wipe API integration requires the optional ASPIRE checkout")

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from scripts.adapters.wipe_public_api import WipeCapApi
from cap_harness.api import CapApi
from cap_harness.contracts import PlanResult,Trajectory,RobotState,Pose


def test_planning_removes_implicit_gripper_column_before_return():
    state=RobotState(joint_positions={'primary':np.zeros(7)},joint_velocities={'primary':np.zeros(7)},
        end_effector_poses={'primary':Pose(position=[0.,0.,0.],quaternion_wxyz=[1.,0.,0.,0.],frame='robot0_base')},
        gripper_positions={'primary':1.},base_frame='robot0_base',joint_names={'primary':tuple(f'panda_joint{i}' for i in range(1,8))},
        timestamp_s=0.,embodiment='robosuite')
    trajectory=Trajectory(joint_positions=np.zeros((2,7)),dt_s=.05,
        joint_names=tuple(f'panda_joint{i}' for i in range(1,8)),planner='test',collision_aware=False,
        expected_start=state,arm='primary',gripper_positions=np.ones(2),embodiment='robosuite')
    api=object.__new__(WipeCapApi)
    with patch.object(CapApi,'plan_motion',return_value=PlanResult(ok=True,trajectory=trajectory)):
        result=api.plan_motion(np.zeros(7))
    assert result.ok and result.trajectory.gripper_positions is None
    np.testing.assert_array_equal(result.trajectory.joint_positions,trajectory.joint_positions)
    assert trajectory.gripper_positions is not None


def test_joint_composition_omits_gripper_without_changing_fixed_tool_state():
    api=object.__new__(WipeCapApi)
    api._commanded_gripper_positions={}
    state=SimpleNamespace(gripper_positions={'primary':1.})
    def inherited_joint_motion(target):
        assert api._commanded_gripper_position(state,'primary') is None
        return 'completed'
    with patch.object(CapApi,'move_to_joints',lambda self,target:inherited_joint_motion(target)):
        assert api.move_to_joints(np.zeros(7))=='completed'
    assert api._commanded_gripper_position(state,'primary')==1.
    assert api._fixed_tool_joint_motion is False


def test_joint_motion_context_restored_after_failure():
    api=object.__new__(WipeCapApi)
    with patch.object(CapApi,'move_to_joints',side_effect=ValueError('test')):
        try:api.move_to_joints(np.zeros(7))
        except ValueError:pass
    assert api._fixed_tool_joint_motion is False
