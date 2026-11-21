"""A single uninterrupted free-body grasp/transport/insert/verify episode."""
from pathlib import Path

import mujoco
import numpy as np

from esrb.tasks.base import TaskObservation, TaskResult
from esrb.tasks.tabletop_appearance import TabletopVisuals
from esrb.tasks.tool_handling.friction import DT, FrictionToolTask, yaw

HOLE_XY = np.array([0.10, 0.0])
HOLE_TOP_Z = 0.085
REQUIRED_DEPTH_M = 0.025
MAX_INSERTION_DEPTH_M = 0.041  # 40 mm socket depth plus 1 mm contact tolerance
POSITION_TOLERANCE_M = 0.002
YAW_TOLERANCE_RAD = 0.08
TILT_TOLERANCE_RAD = 0.04
HOLD_TIME_S = 0.4
RETENTION_STAGES = ("lift", "transport", "preposition", "insert")


def assembly_readback(trace):
    """Reconstruct task geometry and motion from measured world poses."""
    positions = np.asarray(trace["tool_positions"], dtype=float)
    tips = np.asarray(trace["tool_tip_positions"], dtype=float)
    rotations = np.asarray(trace["tool_quaternions"], dtype=float)
    times = np.asarray(trace["times_s"], dtype=float)
    angles = np.array([abs(yaw(q)) for q in rotations])
    tilt = np.arccos(np.clip(1 - 2*(rotations[:, 1]**2 + rotations[:, 2]**2), -1, 1))
    speed = np.r_[0., np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(times)]
    spin = np.r_[0., 2*np.arccos(np.clip(np.abs(np.sum(rotations[1:]*rotations[:-1], axis=1)), 0, 1)) / np.diff(times)]
    return {
        "insertion_depth_m": np.maximum(0., HOLE_TOP_Z - tips[:, 2]),
        "assembly_xy_error_m": np.linalg.norm(tips[:, :2] - np.asarray(trace.get("socket_xy_m", HOLE_XY)), axis=1),
        "assembly_yaw_error_rad": angles, "assembly_tilt_rad": tilt,
        "part_speed_ms": speed, "part_angular_speed_rads": spin,
    }


def verification_mask(values):
    return ((values["insertion_depth_m"] >= REQUIRED_DEPTH_M)
            & (values["insertion_depth_m"] <= MAX_INSERTION_DEPTH_M)
            & (values["assembly_xy_error_m"] <= POSITION_TOLERANCE_M)
            & (values["assembly_yaw_error_rad"] <= YAW_TOLERANCE_RAD)
            & (values["assembly_tilt_rad"] <= TILT_TOLERANCE_RAD)
            & (values["part_speed_ms"] <= 0.005)
            & (values["part_angular_speed_rads"] <= 0.05))


def stable_hold_times(trace, values):
    """Count only consecutive verified samples after the fingers are released."""
    good = verification_mask(values)
    released = ((np.asarray(trace["commands"])[:, 4] <= 0.001)
                & (np.max(trace["grip_normal_forces"], axis=1) < 0.05))
    held, result = 0, []
    for stage, valid, opened in zip(trace["stages"], good, released, strict=True):
        held = held + 1 if stage == "verify" and valid and opened else 0
        result.append(held * DT)
    return np.array(result)


class SingleInsertionTask(TabletopVisuals, FrictionToolTask):
    """Carry a friction-held rectangular part into a keyed physical socket.

    No state is reset at a stage boundary. Verification opens the fingers and
    requires the actual free part to remain assembled for 0.4 continuous seconds.
    """
    name = "long_horizon_assembly"
    MODEL_XML = Path(__file__).parent / "assets" / "single_insertion.xml"
    STAGES = ("grasp", "lift", "transport", "preposition", "insert", "verify", "done")
    MAX_STEPS = 850
    reset_reference = {**FrictionToolTask.reset_reference, "grasp_x_error_m": 0.0}
    reset_bounds = {**FrictionToolTask.reset_bounds, "tool_yaw_rad": (-0.03, 0.03),
                    "grasp_offset_m": (-0.006, 0.006), "grasp_x_error_m": (-0.001, 0.001)}
    boundary_fields = ("grasp_x_error_m", "tool_yaw_rad", "friction", "tool_mass_kg")
    STAGE_BUDGETS = (100, 150, 220, 120, 160, 120)

    def reset(self, seed=0, *, disturbance=None):
        super().reset(seed, disturbance=disturbance)
        self._pick[0] += self._log["reset_readback"]["grasp_x_error_m"]
        self._data.qpos[0] = self._pick[0]
        self._data.ctrl[0] = self._pick[0]
        mujoco.mj_forward(self._model, self._data)
        self._failure = None
        self._log.update(model="continuous_assembly/1", reset_count=1, failure_reason=None,
                         stage_records=[], retention_stages=list(RETENTION_STAGES))
        for field in ("insertion_depth_m", "assembly_xy_error_m", "assembly_yaw_error_rad",
                      "assembly_tilt_rad", "part_speed_ms", "part_angular_speed_rads", "verification_hold_s"):
            self._log[field] = []
        self._enter_stage()
        return self._observe()

    def _observe(self):
        observed = super()._observe()
        info = {**observed.info, "target_yaw": 0.0, "target_xy": HOLE_XY.tolist(),
                "insertion_depth_m": self._log.get("insertion_depth_m", [0.])[-1] if self._log.get("insertion_depth_m") else 0.,
                "contact_force_n": float(np.linalg.norm(self._log["contact_forces"][-1])) if self._log["contact_forces"] else 0.,
                "grasp_offset_m": self._log["reset_readback"]["grasp_offset_m"]}
        return TaskObservation(observed.state, info)

    def _retention_required(self):
        return self.STAGES[self._stage] in RETENTION_STAGES

    def _state(self):
        return {"qpos": self._data.qpos.tolist(), "qvel": self._data.qvel.tolist(),
                "part_position": self._data.xpos[self._tool].tolist(),
                "part_quaternion": self._data.xquat[self._tool].tolist()}

    def _enter_stage(self):
        self._log["stage_records"].append({
            "stage": self.STAGES[self._stage], "enter_step": self._steps,
            "enter_time_s": float(self._data.time), "entry_state": self._state(),
            "exit_step": None, "exit_time_s": None, "exit_condition": None,
            "failure_reason": None, "status": "active",
        })

    def _exit_stage(self, condition, failed=False):
        self._log["stage_records"][-1].update(
            exit_step=self._steps, exit_time_s=float(self._data.time),
            exit_condition=condition, exit_state=self._state(),
            failure_reason=condition if failed else None, status="failed" if failed else "completed",
        )

    def _advance(self, grip, tip):
        # The parent recorded post-physics poses for this very same MjData.
        values = assembly_readback(self._log)
        for field, array in values.items():
            self._log[field].append(float(array[-1]))
        stage = self.STAGES[self._stage]
        v = {key: value[-1] for key, value in values.items()}
        hold = stable_hold_times(self._log, values)[-1]
        self._log["verification_hold_s"].append(float(hold))
        if self._log["drop_events"]:
            failure = "part_dropped"
        elif stage == "insert" and np.linalg.norm(self._log["contact_forces"][-1]) > 35.0:
            failure = "insertion_force_limit"
        elif self._stage_ticks >= self.STAGE_BUDGETS[self._stage]:
            if stage == "insert" and v["assembly_yaw_error_rad"] > YAW_TOLERANCE_RAD:
                failure = "insertion_blocked_orientation"
            else:
                failure = f"{stage}_timeout"
        else:
            failure = None
        if failure:
            self._failure = failure
            self._log["failure_reason"] = failure
            self._exit_stage(failure, failed=True)
            return
        if stage == "grasp":
            reached, condition = self._stage_ticks >= 25 and np.min(grip) > 0.3, "bilateral_grasp_contact"
        elif stage == "lift":
            reached = self._data.xpos[self._tool, 2] > 0.17
            condition = "part_lifted_clear_of_rack"
        elif stage == "transport":
            reached = v["assembly_xy_error_m"] < 0.004 and tip[2] > HOLE_TOP_Z + 0.04
            condition = "part_above_socket"
        elif stage == "preposition":
            reached = v["assembly_xy_error_m"] < 0.004 and abs(tip[2] - (HOLE_TOP_Z+0.025)) < 0.005
            condition = "part_tip_at_preinsertion_pose"
        elif stage == "insert":
            reached = (v["insertion_depth_m"] >= REQUIRED_DEPTH_M and v["assembly_xy_error_m"] <= POSITION_TOLERANCE_M
                       and v["assembly_yaw_error_rad"] <= YAW_TOLERANCE_RAD and v["assembly_tilt_rad"] <= TILT_TOLERANCE_RAD)
            condition = "measured_depth_and_pose_reached"
        else:
            reached, condition = hold >= HOLD_TIME_S, "released_part_stable_for_0.4s"
        if reached:
            self._exit_stage(condition)
            self._stage += 1
            self._stage_ticks = 0
            self._log["stage_events"].append({"stage": self.STAGES[self._stage], "step": self._steps})
            if self.STAGES[self._stage] != "done":
                self._enter_stage()

    def step(self, action):
        if getattr(self, "_failure", None) or self._stage == len(self.STAGES)-1:
            raise RuntimeError("reset is required after assembly episode termination")
        observed, reward, done, _ = super().step(action)
        if done and not self._log["success"] and self._failure is None:
            self._failure = "episode_timeout"
            self._log["failure_reason"] = self._failure
            self._exit_stage(self._failure, failed=True)
        return observed, reward, done or self._failure is not None, dict(self._log)

    @staticmethod
    def evaluate(episode_data):
        if not episode_data.get("tool_positions"):
            return TaskResult(0., False, episode_data, {"failure_reason": "missing_pose_evidence"})
        try:
            values = assembly_readback(episode_data)
            hold = stable_hold_times(episode_data, values)
            completed = sum(row["status"] == "completed" for row in episode_data["stage_records"])
            passed = (completed == 6 and episode_data.get("failure_reason") is None and hold[-1] >= HOLD_TIME_S
                      and len(episode_data["stages"]) >= 20)
        except (KeyError, ValueError, TypeError, IndexError):
            return TaskResult(0., False, episode_data, {"failure_reason": "invalid_assembly_evidence"})
        return TaskResult(completed/6., bool(passed), episode_data, {
            "final_depth_m": float(values["insertion_depth_m"][-1]),
            "final_position_error_m": float(values["assembly_xy_error_m"][-1]),
            "stable_hold_s": float(hold[-1]), "failure_reason": episode_data.get("failure_reason"),
            "stage_records": episode_data["stage_records"],
        })
