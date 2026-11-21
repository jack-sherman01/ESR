"""Friction-held free-body tool manipulation with measured stage transitions."""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from esrb.tasks.base import BaseTask, TaskObservation, TaskResult
from esrb.tasks.reset_domain import validate_disturbance
from esrb.utils.hashing import hash_artifact, hash_dict

DT = 0.020
STAGES = ("grasp", "lift", "transport", "orient", "operate", "done")


def yaw(quaternion):
    w, x, y, z = quaternion
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class FrictionToolTask(BaseTask):
    """Shared free-tool physics and stage evidence, with robot-specific actuator hooks."""

    name = "friction_tool"
    MODEL_XML = None
    MODEL_SCHEMA = "friction_tool_handling/1"
    GRASP_HEIGHT = .115
    STAGES = STAGES
    MAX_STEPS = 600
    reset_reference = {
        "tool_mass_kg": 0.08, "friction": 0.9,
        "tool_x_m": -0.10, "tool_y_m": 0.0, "tool_yaw_rad": 0.0,
        "grasp_offset_m": 0.0,
    }
    reset_bounds = {
        "tool_mass_kg": (0.06, 0.10), "friction": (0.65, 1.0),
        "tool_x_m": (-0.103, -0.097), "tool_y_m": (-0.003, 0.003),
        "tool_yaw_rad": (-0.06, 0.06), "grasp_offset_m": (-0.008, 0.008),
    }
    boundary_fields = ("friction", "tool_mass_kg", "grasp_offset_m")

    def __init__(self):
        self._model = self._make_model()
        self._data = mujoco.MjData(self._model)
        self._tool = self._model.body("tool").id
        self._gripper = self._model.body("gripper").id
        self._handle = self._model.geom("tool_handle").id
        self._tool_geoms = {index for index in range(self._model.ngeom)
                            if self._model.geom_bodyid[index] == self._tool
                            and (self._model.geom_contype[index] or self._model.geom_conaffinity[index])}
        self._pads = [self._model.geom(name).id for name in ("left_pad", "right_pad")]
        self._pad_groups = [{pad} for pad in self._pads]
        self._surface = self._model.geom("work_surface").id
        self._tool_q = self._model.jnt_qposadr[self._model.joint("tool_free").id]
        self._original_inertia = self._model.body_inertia[self._tool].copy()
        tree = ET.parse(self.MODEL_XML).getroot()
        compiler = tree.find("compiler")
        mesh_dir = compiler.get("meshdir", ".") if compiler is not None else "."
        resources = [self.MODEL_XML, *(self.MODEL_XML.parent / mesh_dir / mesh.get("file")
                                     for mesh in tree.findall("asset/mesh") if mesh.get("file"))]
        self._model_resources = {str(path.relative_to(self.MODEL_XML.parent)): hash_artifact(path)
                                 for path in resources}
        self._renderer = None

    def _make_model(self):
        return mujoco.MjModel.from_xml_path(str(self.MODEL_XML))

    def _reset_robot(self):
        self._data.qpos[:6] = [*self._pick, self._initial_yaw, 0, 0]
        self._data.ctrl[:] = [*self._pick, self._initial_yaw, 0, 0]

    def _apply_action(self, action):
        self._data.ctrl[:] = [*action[:4], action[4], action[4]]

    def _before_substep(self):
        """Extension hook for externally prescribed obstacle motion."""

    def _after_substep(self):
        """Extension hook for physics-rate evidence, without resetting state."""

    @property
    def action_space(self):
        return {"shape": (5,), "low": np.array([-0.2, -0.15, 0.07, -1.5, 0.0]),
                "high": np.array([0.2, 0.15, 0.35, 1.5, 0.025])}

    @property
    def observation_space(self):
        return {"shape": (10,), "dtype": np.float64}

    @classmethod
    def disturbance_for_seed(cls, seed):
        rng = np.random.default_rng(seed)
        return {field: float(rng.uniform(*bounds)) for field, bounds in cls.reset_bounds.items()}

    def reset(self, seed=0, *, disturbance=None):
        point = self.disturbance_for_seed(seed) if disturbance is None else validate_disturbance(
            self.reset_bounds, disturbance,
        )
        self._model.body_mass[self._tool] = point["tool_mass_kg"]
        self._model.body_inertia[self._tool] = self._original_inertia * point["tool_mass_kg"] / 0.08
        for geom in [self._handle, *self._pads]:
            self._model.geom_friction[geom, 0] = point["friction"]
        mujoco.mj_setConst(self._model, self._data)
        mujoco.mj_resetData(self._model, self._data)
        self._pick = np.array([point["tool_x_m"], point["tool_y_m"], self.GRASP_HEIGHT + point["grasp_offset_m"]])
        self._initial_yaw = point["tool_yaw_rad"]
        self._reset_robot()
        q = self._tool_q
        self._data.qpos[q:q + 3] = [point["tool_x_m"], point["tool_y_m"], 0.095]
        self._data.qpos[q + 3:q + 7] = [np.cos(self._initial_yaw / 2), 0, 0, np.sin(self._initial_yaw / 2)]
        mujoco.mj_forward(self._model, self._data)
        self._steps = 0
        self._stage = 0
        self._stage_ticks = 0
        self._hold = 0
        self._surface_force = 0.0
        self._grasp_reference = None
        self._drop_active = False
        self._log = {field: [] for field in (
            "ee_positions", "gripper_quaternions", "tool_positions", "tool_quaternions",
            "tool_tip_positions", "grip_normal_forces", "surface_forces", "contact_forces",
            "retention_active", "contact_samples",
            "relative_positions", "relative_quaternions",
            "contact_torques", "stages", "times_s", "commands", "contacts",
        )}
        self._log.update(
            reset_readback=point, model=self.MODEL_SCHEMA, dt_s=DT,
            stage_events=[{"stage": "grasp", "step": 0}], success=False,
            measurement_frame="world", wrench_origin="tool_center",
            model_digest=hash_artifact(self.MODEL_XML), drop_events=[],
            model_resources=dict(self._model_resources),
            model_resource_digest=hash_dict(self._model_resources),
        )
        return self._observe()

    def _observe(self):
        return TaskObservation(
            np.array([*self._data.xpos[self._gripper], yaw(self._data.xquat[self._gripper]),
                      *self._data.xpos[self._tool], yaw(self._data.xquat[self._tool]),
                      self._stage, self._surface_force]),
            {"stage": self.STAGES[self._stage], "pick": self._pick.tolist(),
             "initial_yaw": self._initial_yaw, "target_yaw": 0.7, "target_xy": [0.10, 0.0],
             "actuator_targets": self._data.ctrl[:4].tolist()},
        )

    def _contacts(self):
        force, torque = np.zeros(3), np.zeros(3)
        grip, surface = np.zeros(2), 0.0
        records = []
        for index in range(self._data.ncon):
            contact = self._data.contact[index]
            pair = {contact.geom1, contact.geom2}
            tool_parts = pair.intersection(self._tool_geoms)
            if not tool_parts:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self._model, self._data, index, wrench)
            world_force = contact.frame.reshape(3, 3).T @ wrench[:3]
            world_torque = contact.frame.reshape(3, 3).T @ wrench[3:]
            # MuJoCo reports the force acting on geom2.
            sign = 1.0 if contact.geom2 in tool_parts else -1.0
            if not pair.intersection(self._pads):
                force += sign * world_force
                torque += sign * (world_torque + np.cross(contact.pos - self._data.xpos[self._tool], world_force))
            for pad_index, pads in enumerate(self._pad_groups):
                if pair.intersection(pads):
                    grip[pad_index] += wrench[0]
            if self._surface in pair:
                surface += wrench[0]
            records.append({
                "geom1": self._model.geom(contact.geom1).name,
                "geom2": self._model.geom(contact.geom2).name,
                "position_world": contact.pos.tolist(), "normal_force_n": float(wrench[0]),
                "force_on_tool_world": (sign * world_force).tolist(),
                "torque_on_tool_world": (sign * world_torque).tolist(),
            })
        return force, torque, grip, float(surface), records

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != self.action_space["shape"] or not np.isfinite(action).all():
            raise ValueError(f"tool action must contain {self.action_space['shape'][0]} finite actuator targets")
        action = np.clip(action, self.action_space["low"], self.action_space["high"])
        self._apply_action(action)
        peaks = None
        contact_samples = []
        for _ in range(10):
            self._before_substep()
            mujoco.mj_step(self._model, self._data)
            mujoco.mj_forward(self._model, self._data)
            self._after_substep()
            values = self._contacts()
            contact_samples.append([float(self._data.time), *values[0], *values[1], *values[2], values[3]])
            if peaks is None:
                peaks = values
            else:
                force = values[0] if np.linalg.norm(values[0]) > np.linalg.norm(peaks[0]) else peaks[0]
                torque = values[1] if np.linalg.norm(values[1]) > np.linalg.norm(peaks[1]) else peaks[1]
                peaks = (force, torque, np.maximum(values[2], peaks[2]),
                         max(values[3], peaks[3]), values[4] if np.linalg.norm(values[0]) >= np.linalg.norm(peaks[0]) else peaks[4])
        self._surface_force = peaks[3]
        self._steps += 1
        self._stage_ticks += 1
        self._log["ee_positions"].append(self._data.xpos[self._gripper].tolist())
        self._log["gripper_quaternions"].append(self._data.xquat[self._gripper].tolist())
        self._log["tool_positions"].append(self._data.xpos[self._tool].tolist())
        self._log["tool_quaternions"].append(self._data.xquat[self._tool].tolist())
        tip = self._data.site_xpos[self._model.site("tool_tip").id]
        self._log["tool_tip_positions"].append(tip.tolist())
        self._log["grip_normal_forces"].append(peaks[2].tolist())
        self._log["surface_forces"].append(peaks[3])
        self._log["contact_forces"].append(peaks[0].tolist())
        self._log["contact_torques"].append(peaks[1].tolist())
        relative = self._data.xmat[self._gripper].reshape(3, 3).T @ (
            self._data.xpos[self._tool] - self._data.xpos[self._gripper]
        )
        inverse, relative_quaternion = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(inverse, self._data.xquat[self._gripper])
        mujoco.mju_mulQuat(relative_quaternion, inverse, self._data.xquat[self._tool])
        self._log["relative_positions"].append(relative.tolist())
        self._log["relative_quaternions"].append(relative_quaternion.tolist())
        if self._retention_required():
            if self._grasp_reference is None:
                self._grasp_reference = relative.copy()
            dropped = np.min(peaks[2]) < 0.05 and np.linalg.norm(relative - self._grasp_reference) > 0.035
            if dropped and not self._drop_active:
                self._log["drop_events"].append({
                    "step": self._steps, "time_s": float(self._data.time), "stage": self.STAGES[self._stage],
                })
            self._drop_active = bool(dropped)
        self._log["retention_active"].append(int(self._retention_required()))
        self._log["contact_samples"].append(contact_samples)
        self._log["stages"].append(self.STAGES[self._stage])
        self._log["times_s"].append(float(self._data.time))
        self._log["commands"].append(action.tolist())
        self._log["contacts"].append(peaks[4])
        self._advance(peaks[2], tip)
        done = self._stage == len(self.STAGES) - 1 or self._steps >= self.MAX_STEPS
        self._log["success"] = self._stage == len(self.STAGES) - 1
        return self._observe(), float(self._stage) / (len(self.STAGES) - 1), done, dict(self._log)

    def _retention_required(self):
        return self._stage >= 1

    def _advance(self, grip, tip):
        tool = self._data.xpos[self._tool]
        gripper = self._data.xpos[self._gripper]
        if self._stage == 0:
            reached = self._stage_ticks >= 25 and np.min(grip) > 0.3
        elif self._stage == 1:
            reached = gripper[2] > 0.195 and tool[2] > 0.145
        elif self._stage == 2:
            reached = np.linalg.norm(tool[:2] - [0.10, 0.0]) < 0.004
        elif self._stage == 3:
            reached = abs(yaw(self._data.xquat[self._tool]) - 0.7) < 0.04
        else:
            aligned = abs(yaw(self._data.xquat[self._tool]) - 0.7) < 0.08
            reached = aligned and np.linalg.norm(tip[:2] - [0.10, 0.0]) < 0.01 and self._surface_force >= 2.0
            self._hold = self._hold + 1 if reached else 0
            reached = self._hold >= 10
        if reached:
            self._stage += 1
            self._stage_ticks = 0
            self._log["stage_events"].append({"stage": self.STAGES[self._stage], "step": self._steps})

    @staticmethod
    def evaluate(episode_data):
        completed = max(0, len(episode_data.get("stage_events", [])) - 1)
        passed = completed == 5 and episode_data.get("success", False)
        return TaskResult(float(completed) / 5, passed, episode_data,
                          {"stage_events": episode_data.get("stage_events", []),
                           "termination": "completed" if passed else "stage_timeout"})

    def render_frame(self, width=640, height=480):
        if self._renderer is not None and self._render_size != (width, height):
            self.close()
        if self._renderer is None:
            self._model.vis.global_.offwidth = max(width, self._model.vis.global_.offwidth)
            self._model.vis.global_.offheight = max(height, self._model.vis.global_.offheight)
            self._renderer = mujoco.Renderer(self._model, height=height, width=width)
            self._render_size = (width, height)
        options = mujoco.MjvOption()
        options.geomgroup[3] = 0  # Draw visual meshes; collision geometry remains active in physics.
        options.geomgroup[5] = 0
        self._renderer.update_scene(self._data, camera="overview", scene_option=options)
        self._decorate_render_scene(self._renderer.scene)
        return self._renderer.render()

    def _decorate_render_scene(self, scene):
        """Optional display-only arena additions; shared physics stays unchanged."""

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
