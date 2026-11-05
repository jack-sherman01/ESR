from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from esrb.tasks.base import BaseTask, TaskObservation, TaskResult

# ── Geometry constants (must match the XML) ───────────────────────────────────
_HOLE_TOP_Z: float = 0.085      # z-coordinate of the socket opening (m)
_HOLE_XY: np.ndarray = np.zeros(2)  # socket centre in the XY plane
_SUCCESS_DEPTH: float = 0.020   # peg tip must reach ≥ 2 cm below the opening
_MAX_STEPS: int = 500
_SIM_SUBSTEPS: int = 10         # 10 × 2 ms = 20 ms control step

# Sensor layout (see XML comment)
_S_FORCE  = slice(0, 3)
_S_TORQUE = slice(3, 6)
_S_TIP_POS = slice(6, 9)
_S_TIP_VEL = slice(9, 12)

_XML = Path(__file__).parent / "assets" / "peg_insertion.xml"


class PegInsertionTask(BaseTask):
    """
    Peg-in-hole contact-rich manipulation task.

    The agent commands a 3-D delta-position for the peg's mocap body each step.
    A stiff weld constraint keeps the peg body co-located with the mocap body.
    Independent F/T sensors, tip position, and EE trajectory are logged for the
    safety oracle.

    Episode ends after MAX_STEPS control steps or when the task is evaluated.
    """

    name = "peg_insertion"

    # ── max action magnitude (m per control step) ─────────────────────────────
    _ACTION_LIMIT: float = 0.005

    def __init__(self) -> None:
        self._model = mujoco.MjModel.from_xml_path(str(_XML))
        self._data  = mujoco.MjData(self._model)
        self._step_count: int = 0
        self._episode_log: dict[str, list] = {}

        # Cache IDs (computed once; avoids string look-ups per step)
        self._mocap_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "peg_mocap"
        )
        peg_joint_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, "peg_joint"
        )
        self._peg_qpos_adr: int = int(self._model.jnt_qposadr[peg_joint_id])
        self._peg_qvel_adr: int = int(self._model.jnt_dofadr[peg_joint_id])

        # mocap_id is an index into mocap_pos / mocap_quat arrays
        self._mocap_idx: int = int(self._model.body_mocapid[self._mocap_id])

    # ── Gymnasium-style spaces (returned as plain dicts) ──────────────────────

    @property
    def action_space(self) -> dict:
        return {
            "shape": (3,),
            "low":   np.full(3, -self._ACTION_LIMIT),
            "high":  np.full(3,  self._ACTION_LIMIT),
            "dtype": np.float32,
        }

    @property
    def observation_space(self) -> dict:
        # [mocap_pos(3), tip_rel_to_hole(3), contact_force(3), contact_torque(3), tip_vel(3)]
        return {"shape": (15,), "dtype": np.float32}

    # ── Core interface ────────────────────────────────────────────────────────

    def reset(self, seed: int = 0) -> TaskObservation:
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self._model, self._data)

        # Random XY offset: uniform ±3 cm from hole centre
        xy_offset = rng.uniform(-0.03, 0.03, size=2)
        # Start 16 cm above the socket opening
        init_pos = np.array([xy_offset[0], xy_offset[1], _HOLE_TOP_Z + 0.160])
        init_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Set mocap
        self._data.mocap_pos[self._mocap_idx]  = init_pos
        self._data.mocap_quat[self._mocap_idx] = init_quat

        # Seed the peg freejoint to match so the weld starts at zero error
        adr = self._peg_qpos_adr
        self._data.qpos[adr:adr + 3] = init_pos
        self._data.qpos[adr + 3:adr + 7] = init_quat
        self._data.qvel[self._peg_qvel_adr:self._peg_qvel_adr + 6] = 0.0

        mujoco.mj_forward(self._model, self._data)

        self._step_count = 0
        self._episode_log = {
            "contact_forces":  [],
            "contact_torques": [],
            "ee_positions":    [],   # peg tip world position (for oracle)
            "tip_z":           [],   # scalar for fast depth check
        }

        return self._observe()

    def step(self, action: np.ndarray) -> tuple[TaskObservation, float, bool, dict]:
        action = np.clip(np.asarray(action, dtype=float), -self._ACTION_LIMIT, self._ACTION_LIMIT)

        # Move the mocap body
        self._data.mocap_pos[self._mocap_idx] += action
        # Hard floor: peg tip must stay above base top (z=0.04), peg centre ≥ 0.065
        self._data.mocap_pos[self._mocap_idx][2] = max(0.065, self._data.mocap_pos[self._mocap_idx][2])

        for _ in range(_SIM_SUBSTEPS):
            mujoco.mj_step(self._model, self._data)

        # Read sensors
        force  = self._data.sensordata[_S_FORCE].copy()
        torque = self._data.sensordata[_S_TORQUE].copy()
        tip    = self._data.sensordata[_S_TIP_POS].copy()

        self._episode_log["contact_forces"].append(force.tolist())
        self._episode_log["contact_torques"].append(torque.tolist())
        self._episode_log["ee_positions"].append(tip.tolist())
        self._episode_log["tip_z"].append(float(tip[2]))

        self._step_count += 1
        done = self._step_count >= _MAX_STEPS

        # Dense reward: insertion depth normalised to success threshold
        insertion_depth = max(0.0, _HOLE_TOP_Z - float(tip[2]))
        reward = insertion_depth / _SUCCESS_DEPTH

        return self._observe(), reward, done, dict(self._episode_log)

    def evaluate(self, episode_data: dict) -> TaskResult:
        tip_zs = episode_data.get("tip_z", [])
        if not tip_zs:
            return TaskResult(task_score=0.0, success=False, episode_data=episode_data,
                              metadata={"max_insertion_depth_m": 0.0})

        min_tip_z = min(tip_zs)
        depth = max(0.0, _HOLE_TOP_Z - min_tip_z)
        task_score = min(1.0, depth / _SUCCESS_DEPTH)
        return TaskResult(
            task_score=task_score,
            success=depth >= _SUCCESS_DEPTH,
            episode_data=episode_data,
            metadata={"max_insertion_depth_m": float(depth)},
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _observe(self) -> TaskObservation:
        mocap_pos = self._data.mocap_pos[self._mocap_idx].copy()
        force     = self._data.sensordata[_S_FORCE].copy()
        torque    = self._data.sensordata[_S_TORQUE].copy()
        tip_pos   = self._data.sensordata[_S_TIP_POS].copy()
        tip_vel   = self._data.sensordata[_S_TIP_VEL].copy()

        tip_rel = tip_pos - np.array([_HOLE_XY[0], _HOLE_XY[1], _HOLE_TOP_Z])

        state = np.concatenate([
            mocap_pos,   # (3) current EE position in world frame
            tip_rel,     # (3) tip offset from hole centre (x, y: alignment; z: depth)
            force,       # (3) contact force at peg centre
            torque,      # (3) contact torque at peg centre
            tip_vel,     # (3) tip linear velocity
        ]).astype(np.float32)

        return TaskObservation(state=state, info={})
