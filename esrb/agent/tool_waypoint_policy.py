"""Deterministic actuator-target policy for the friction-held tool task."""

import numpy as np

from esrb.tasks.base import TaskObservation
from .base import BaseAgent


class ToolWaypointPolicy(BaseAgent):
    name = "tool_handling"

    def __init__(self):
        self.max_speed = 0.0015
        self.angular_step = 0.025
        self.grasp_closure = 0.022
        self.transport_closure = 0.022

    def act(self, observation: TaskObservation):
        state, info = observation.state, observation.info
        stage = info["stage"]
        target = np.array(info["pick"])
        angle = info["initial_yaw"]
        closure = self.grasp_closure
        if stage in {"lift", "transport", "orient", "operate", "done"}:
            target[2] = info.get('lift_height', .20)
        if stage in {"transport", "orient", "operate", "done"}:
            target[:2] = info["target_xy"]
            closure = self.transport_closure
        if stage in {"orient", "operate", "done"}:
            angle = info["target_yaw"]
        if stage in {"operate", "done"}:
            target[2] = 0.126 + info["pick"][2] - 0.115
        previous = np.asarray(info["actuator_targets"])
        position = previous[:3] + np.clip(target - previous[:3], -self.max_speed, self.max_speed)
        yaw_target = previous[3] + np.clip(angle - previous[3], -self.angular_step, self.angular_step)
        return np.array([*position, yaw_target, closure])

    def update(self, episode_data):
        pass

    def state_dict(self):
        return {name: getattr(self, name) for name in (
            "max_speed", "angular_step", "grasp_closure", "transport_closure",
        )}

    def load_state_dict(self, state):
        for name in self.state_dict():
            setattr(self, name, state[name])
