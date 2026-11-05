from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from esrb.tasks.base import BaseTask, TaskObservation, TaskResult

# ── Geometry constants (must match the XML) ───────────────────────────────────
_HOLE_TOP_Z: float = 0.085      # z-coordinate of the socket opening (m)
_HOLE_XY: np.ndarray = np.zeros(2)
_SUCCESS_DEPTH: float = 0.020   # peg tip must reach ≥ 2 cm below the opening
_MAX_STEPS: int = 500
_SIM_SUBSTEPS: int = 10         # 10 × 2 ms = 20 ms control step

# Sensor layout (see XML comment)
_S_FORCE   = slice(0, 3)
_S_TORQUE  = slice(3, 6)
_S_TIP_POS = slice(6, 9)
_S_TIP_VEL = slice(9, 12)
_S_JPOS    = slice(12, 15)

_XML = Path(__file__).parent / "assets" / "peg_insertion.xml"

# arm_x body rest position in world frame (z = socket_top + 0.160)
_ARM_REST_Z: float = _HOLE_TOP_Z + 0.160   # 0.245


class PegInsertionTask(BaseTask):
    """
    Peg-in-hole contact-rich manipulation task — 3-DOF slide mechanism.

    Control
    -------
    Three sequential prismatic joints (x, y, z) position the peg EE.
    Position actuators (kp=5000, kv=30) track the targets in data.ctrl[:3].
    The agent commands incremental position targets each control step:
        data.ctrl[:3] += action  (clipped to joint ranges)

    Force sensing
    -------------
    The <force> sensor at peg_center measures ONLY contact forces on the peg
    body (actuator forces act on joint bodies above, not on the peg body).
    This gives a physically correct force signal for the safety oracle.

    At kp=5000 N/m, one control step (4 mm) of overshoot past the socket
    bottom generates ~20 N — the reference oracle threshold.  A 50%-faster
    candidate (6 mm step) generates ~30 N, triggering a safety violation
    and demonstrating safety regression from the ForceSpeedInjectionOperator.
    """

    name = "peg_insertion"
    _ACTION_LIMIT: float = 0.005   # max delta per control step (m)

    # Joint z lower bound: keeps peg tip above floor (tip = arm_z_world - 0.025)
    # arm_z_world = _ARM_REST_Z + joint_z  →  joint_z >= 0.040 - _ARM_REST_Z + 0.025 - 0.001
    _CTRL_Z_MIN: float = 0.040 - _ARM_REST_Z    # ≈ -0.205

    def __init__(self) -> None:
        self._model = mujoco.MjModel.from_xml_path(str(_XML))
        self._data  = mujoco.MjData(self._model)
        self._step_count: int = 0
        self._episode_log: dict = {}

        # Cache joint qpos/qvel addresses
        self._jx_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "joint_x")
        self._jy_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "joint_y")
        self._jz_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "joint_z")
        self._qx = int(self._model.jnt_qposadr[self._jx_id])
        self._qy = int(self._model.jnt_qposadr[self._jy_id])
        self._qz = int(self._model.jnt_qposadr[self._jz_id])

    # ── Spaces ────────────────────────────────────────────────────────────────

    @property
    def action_space(self) -> dict:
        return {
            "shape": (3,),
            "low":  np.full(3, -self._ACTION_LIMIT, dtype=np.float32),
            "high": np.full(3,  self._ACTION_LIMIT, dtype=np.float32),
        }

    @property
    def observation_space(self) -> dict:
        # [ee_pos(3), tip_rel_to_hole(3), contact_force(3), contact_torque(3), tip_vel(3)]
        return {"shape": (15,), "dtype": np.float32}

    # ── Core interface ────────────────────────────────────────────────────────

    def reset(self, seed: int = 0) -> TaskObservation:
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self._model, self._data)

        # Random XY offset: uniform ±3 cm from hole centre
        xy_offset = rng.uniform(-0.03, 0.03, size=2)

        # Set joint positions (ctrl = qpos for position actuators)
        self._data.qpos[self._qx] = xy_offset[0]
        self._data.qpos[self._qy] = xy_offset[1]
        self._data.qpos[self._qz] = 0.0          # arm rest height

        self._data.ctrl[0] = xy_offset[0]
        self._data.ctrl[1] = xy_offset[1]
        self._data.ctrl[2] = 0.0

        mujoco.mj_forward(self._model, self._data)

        self._step_count = 0
        self._episode_log = {
            "contact_forces":  [],
            "contact_torques": [],
            "ee_positions":    [],
            "tip_z":           [],
        }
        return self._observe()

    def step(self, action: np.ndarray) -> tuple[TaskObservation, float, bool, dict]:
        action = np.clip(np.asarray(action, dtype=float), -self._ACTION_LIMIT, self._ACTION_LIMIT)

        # Increment position targets
        self._data.ctrl[:3] += action

        # Clamp z target: don't let the peg go below the socket floor
        # (keeping a small margin below so we can observe bottom contact)
        self._data.ctrl[2] = max(self._CTRL_Z_MIN, self._data.ctrl[2])

        for _ in range(_SIM_SUBSTEPS):
            mujoco.mj_step(self._model, self._data)

        force  = self._data.sensordata[_S_FORCE].copy()
        torque = self._data.sensordata[_S_TORQUE].copy()
        tip    = self._data.sensordata[_S_TIP_POS].copy()

        self._episode_log["contact_forces"].append(force.tolist())
        self._episode_log["contact_torques"].append(torque.tolist())
        self._episode_log["ee_positions"].append(tip.tolist())
        self._episode_log["tip_z"].append(float(tip[2]))

        self._step_count += 1
        done = self._step_count >= _MAX_STEPS

        insertion_depth = max(0.0, _HOLE_TOP_Z - float(tip[2]))
        reward = insertion_depth / _SUCCESS_DEPTH

        return self._observe(), reward, done, dict(self._episode_log)

    def evaluate(self, episode_data: dict) -> TaskResult:
        tip_zs = episode_data.get("tip_z", [])
        if not tip_zs:
            return TaskResult(task_score=0.0, success=False,
                              episode_data=episode_data,
                              metadata={"max_insertion_depth_m": 0.0})
        depth = max(0.0, _HOLE_TOP_Z - min(tip_zs))
        return TaskResult(
            task_score=min(1.0, depth / _SUCCESS_DEPTH),
            success=depth >= _SUCCESS_DEPTH,
            episode_data=episode_data,
            metadata={"max_insertion_depth_m": float(depth)},
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _observe(self) -> TaskObservation:
        force  = self._data.sensordata[_S_FORCE].copy()
        torque = self._data.sensordata[_S_TORQUE].copy()
        tip    = self._data.sensordata[_S_TIP_POS].copy()
        vel    = self._data.sensordata[_S_TIP_VEL].copy()

        # EE world position: arm_x rest + joint offsets
        jpos = self._data.sensordata[_S_JPOS].copy()
        ee_pos = np.array([jpos[0], jpos[1], _ARM_REST_Z + jpos[2]])

        tip_rel = tip - np.array([_HOLE_XY[0], _HOLE_XY[1], _HOLE_TOP_Z])

        state = np.concatenate([
            ee_pos,   # (3)
            tip_rel,  # (3)
            force,    # (3)
            torque,   # (3)
            vel,      # (3)
        ]).astype(np.float32)

        return TaskObservation(state=state, info={})
