"""State-driven deterministic policy for continuous keyed assembly."""
import numpy as np
from esrb.agent.tool_handling_policy import ToolHandlingPolicy


class AssemblyPolicy(ToolHandlingPolicy):
    name = "long_horizon_assembly"

    def __init__(self):
        super().__init__()
        self.transport_yaw_bias = 0.0

    def act(self, observation):
        info = observation.info
        stage = info["stage"]
        offset = info["grasp_offset_m"]
        target, angle = np.array(info["pick"]), info["initial_yaw"]
        closure = self.grasp_closure
        if stage != "grasp":
            target[2] = 0.23 + offset
        if stage in {"transport", "preposition", "insert", "verify", "done"}:
            target[:2] = info["target_xy"]
            angle += self.transport_yaw_bias
            closure = self.transport_closure
        if stage == "preposition":
            target[2] = 0.195 + offset
        if stage in {"insert", "verify", "done"}:
            previous = np.asarray(info["actuator_targets"])
            target[2] = previous[2] - self.max_speed if info["insertion_depth_m"] < 0.028 else previous[2]
        if stage in {"verify", "done"}:
            target[2] = info["actuator_targets"][2]
            closure = 0.0
        previous = np.asarray(info["actuator_targets"])
        position = previous[:3] + np.clip(target-previous[:3], -self.max_speed, self.max_speed)
        rotation = previous[3] + np.clip(angle-previous[3], -self.angular_step, self.angular_step)
        return np.array([*position, rotation, closure])

    def state_dict(self):
        return {**super().state_dict(), "transport_yaw_bias": self.transport_yaw_bias}
