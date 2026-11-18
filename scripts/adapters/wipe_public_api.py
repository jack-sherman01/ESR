"""Wipe-only public compositions for a fixed tool without gripper actuation.

The upstream API adds a gripper hold column to every planned trajectory, even
for fixed tools. Remove that implicit column before public planning returns.
Explicit gripper commands retain the native unsupported-operation behavior.
"""
from dataclasses import replace
from cap_harness.api import CapApi
from cap_harness.contracts import Trajectory


class WipeCapApi(CapApi):
    def plan_motion(self, *args, **kwargs):
        result = super().plan_motion(*args, **kwargs)
        if result.ok and isinstance(result.trajectory, Trajectory):
            result = replace(result, trajectory=replace(result.trajectory, gripper_positions=None))
        return result

    def move_to_joints(self, *args, **kwargs):
        # The inherited joint convergence loop obtains its implicit hold value
        # through this hook. None omits the field from each ArmCommand while
        # preserving the original loop, tolerances, limits, and terminal checks.
        previous = getattr(self, '_fixed_tool_joint_motion', False)
        self._fixed_tool_joint_motion = True
        try:
            return super().move_to_joints(*args, **kwargs)
        finally:
            self._fixed_tool_joint_motion = previous

    def _commanded_gripper_position(self, state, arm):
        if getattr(self, '_fixed_tool_joint_motion', False):
            return None
        return super()._commanded_gripper_position(state, arm)
