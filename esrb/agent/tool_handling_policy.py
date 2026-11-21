"""Panda joint-target policy using measured end-effector Jacobians."""
import numpy as np
from esrb.agent.tool_waypoint_policy import ToolWaypointPolicy
from esrb.tasks.robots.panda import orientation_error, joint_target, yaw_rotation


class ToolHandlingPolicy(ToolWaypointPolicy):
    """Convert task waypoints to seven joint targets; the eighth action closes the hand."""
    name = 'tool_handling'

    def __init__(self):
        super().__init__()
        self.grasp_closure = .023
        self.transport_closure = .023

    def act(self, observation):
        target = super().act(observation)
        info = observation.info
        return joint_target(info, target[:3], yaw_rotation(target[3]), target[4])
