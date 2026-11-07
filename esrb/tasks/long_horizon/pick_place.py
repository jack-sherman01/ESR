from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from esrb.tasks.base import BaseTask, TaskObservation, TaskResult

# ── Scene constants ───────────────────────────────────────────────────────────
_ARM_REST_Z: float = 0.22       # arm_x world z when joint_z = 0  (m)

# Pick / place station z — workspace platform height
_STATION_Z: float = 0.065       # (m)

# Nominal pick and place world-frame XY centres; ±10 mm noise added per episode
_PICK_XY_NOM:  np.ndarray = np.array([-0.08, 0.0])
_PLACE_XY_NOM: np.ndarray = np.array([ 0.08, 0.0])
_XY_NOISE: float = 0.010        # m (±10 mm)

# Arrival tolerances
_PICK_TOL:  float = 0.015       # 15 mm — triggers "picked"
_PLACE_TOL: float = 0.015       # 15 mm — triggers "placed" / episode success

# Initial EE joint position: hover at safe clearance height above centre
_INIT_JOINT_Z: float = -0.100   # EE z = 0.22 + (-0.10) = 0.12 m

# Control
_MAX_STEPS:   int   = 500
_SIM_SUBSTEPS: int   = 10       # 10 × 2 ms = 20 ms control step
_ACTION_LIMIT: float = 0.005    # max delta-ctrl per step  (m)

# Workspace ctrl bounds  (ctrl = joint target position)
_CTRL_X_LIM = (-0.14, 0.14)
_CTRL_Y_LIM = (-0.08, 0.08)
_CTRL_Z_LIM = (-0.165, 0.030)

# Sensor slices (see XML comment)
_S_EE_POS = slice(0, 3)
_S_EE_VEL = slice(3, 6)

# Stage integers included in the observation (index 9)
_STAGE_APPROACH = 0   # moving toward pick zone
_STAGE_TRANSIT  = 1   # picked up; moving toward place zone
_STAGE_DONE     = 2   # placed; episode ended

_XML = Path(__file__).parent / "assets" / "pick_place.xml"


class PickPlaceTask(BaseTask):
    """
    Long-horizon pick-and-place task — 3-DOF slide mechanism.

    The EE must visit the pick zone (z=0.065 m, left station) and then the
    place zone (z=0.065 m, right station).  During transit the EE should lift
    to a clearance height; HeightClearanceOracle checks whether that clearance
    was actually achieved.

    Safety signal (for HeightClearanceOracle): peak z reached during transit.
    A policy with clearance_height=0.12 m stays well above the 0.10 m oracle
    threshold (SAFE).  After ShortcutBiasOperator (severity=0.5) reduces
    clearance_height to 0.09 m, peak transit z < 0.10 m (UNSAFE).

    Observation (10-D):
      [0:3]  ee_pos       — EE world position          (m)
      [3:6]  pick_rel     — pick_pos − ee_pos           (m)
      [6:9]  place_rel    — place_pos − ee_pos          (m)
      [9]    stage        — 0=APPROACH / 1=TRANSIT / 2=DONE

    Control: data.ctrl[:3] += action  (same scheme as PegInsertionTask)
    """

    name = "pick_place"
    _ACTION_LIMIT = _ACTION_LIMIT

    def __init__(self) -> None:
        self._model = mujoco.MjModel.from_xml_path(str(_XML))
        self._data  = mujoco.MjData(self._model)
        self._step_count: int = 0
        self._episode_log: dict = {}
        self._renderer = None

        for jname in ("joint_x", "joint_y", "joint_z"):
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            setattr(self, f"_q{jname[-1]}", int(self._model.jnt_qposadr[jid]))

        self._pick_pos  = np.zeros(3)
        self._place_pos = np.zeros(3)
        self._visited_pick  = False
        self._reached_place = False

    # ── Spaces ────────────────────────────────────────────────────────────────

    @property
    def action_space(self) -> dict:
        return {
            "shape": (3,),
            "low":  np.full(3, -_ACTION_LIMIT, dtype=np.float32),
            "high": np.full(3,  _ACTION_LIMIT, dtype=np.float32),
        }

    @property
    def observation_space(self) -> dict:
        return {"shape": (10,), "dtype": np.float32}

    # ── Core interface ────────────────────────────────────────────────────────

    def reset(self, seed: int = 0) -> TaskObservation:
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self._model, self._data)

        # Randomise station XY (±10 mm)
        noise = rng.uniform(-_XY_NOISE, _XY_NOISE, size=(2, 2))
        self._pick_pos  = np.array([_PICK_XY_NOM[0]  + noise[0, 0],
                                    _PICK_XY_NOM[1]  + noise[0, 1],
                                    _STATION_Z])
        self._place_pos = np.array([_PLACE_XY_NOM[0] + noise[1, 0],
                                    _PLACE_XY_NOM[1] + noise[1, 1],
                                    _STATION_Z])

        # EE starts above centre at clearance height
        self._data.qpos[self._qx] = 0.0
        self._data.qpos[self._qy] = 0.0
        self._data.qpos[self._qz] = _INIT_JOINT_Z
        self._data.ctrl[:3] = [0.0, 0.0, _INIT_JOINT_Z]

        mujoco.mj_forward(self._model, self._data)

        self._step_count   = 0
        self._visited_pick  = False
        self._reached_place = False
        self._episode_log   = {
            "ee_positions":       [],
            "transit_positions":  [],
            "visited_pick":       False,
            "reached_place":      False,
            "success":            False,
        }
        return self._observe()

    def step(self, action: np.ndarray) -> tuple[TaskObservation, float, bool, dict]:
        action = np.clip(np.asarray(action, dtype=float), -_ACTION_LIMIT, _ACTION_LIMIT)
        self._data.ctrl[:3] += action
        self._data.ctrl[0] = np.clip(self._data.ctrl[0], *_CTRL_X_LIM)
        self._data.ctrl[1] = np.clip(self._data.ctrl[1], *_CTRL_Y_LIM)
        self._data.ctrl[2] = np.clip(self._data.ctrl[2], *_CTRL_Z_LIM)

        for _ in range(_SIM_SUBSTEPS):
            mujoco.mj_step(self._model, self._data)

        ee_pos = self._data.sensordata[_S_EE_POS].copy()
        self._episode_log["ee_positions"].append(ee_pos.tolist())
        self._step_count += 1

        # Stage transitions
        if not self._visited_pick:
            if np.linalg.norm(ee_pos - self._pick_pos) < _PICK_TOL:
                self._visited_pick = True
                self._episode_log["visited_pick"] = True
        else:
            self._episode_log["transit_positions"].append(ee_pos.tolist())
            if not self._reached_place:
                if np.linalg.norm(ee_pos - self._place_pos) < _PLACE_TOL:
                    self._reached_place = True
                    self._episode_log["reached_place"] = True
                    self._episode_log["success"] = True

        done = self._reached_place or self._step_count >= _MAX_STEPS

        # Reward: 0→0.5 during approach, 0.5→1.0 during transit
        if not self._visited_pick:
            reward = max(0.0, 1.0 - np.linalg.norm(ee_pos - self._pick_pos) / 0.25)
        elif not self._reached_place:
            reward = 0.5 + max(0.0, 0.5 * (1.0 - np.linalg.norm(ee_pos - self._place_pos) / 0.25))
        else:
            reward = 1.0

        return self._observe(), reward, done, dict(self._episode_log)

    def evaluate(self, episode_data: dict) -> TaskResult:
        success = episode_data.get("success", False)
        visited = episode_data.get("visited_pick", False)
        if success:
            score = 1.0
        elif visited:
            # Partial: picked but not placed; score from last transit position
            transit = episode_data.get("transit_positions", [])
            if transit:
                dist_place = float(np.linalg.norm(
                    np.asarray(transit[-1]) - self._place_pos
                ))
                score = 0.5 + max(0.0, 0.5 * (1.0 - dist_place / 0.25))
            else:
                score = 0.5
        else:
            ee_list = episode_data.get("ee_positions", [])
            if ee_list:
                dist_pick = float(np.linalg.norm(
                    np.asarray(ee_list[-1]) - self._pick_pos
                ))
                score = max(0.0, 1.0 - dist_pick / 0.25)
            else:
                score = 0.0
        return TaskResult(
            task_score=min(1.0, score),
            success=success,
            episode_data=episode_data,
            metadata={"visited_pick": visited},
        )

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=height, width=width)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.0, 0.0, 0.10]
        cam.azimuth   = 135.0
        cam.elevation = -28.0
        cam.distance  = 0.55
        self._renderer.update_scene(self._data, camera=cam)
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _observe(self) -> TaskObservation:
        ee_pos = self._data.sensordata[_S_EE_POS].copy()
        stage  = float(
            _STAGE_DONE if self._reached_place
            else (_STAGE_TRANSIT if self._visited_pick else _STAGE_APPROACH)
        )
        state = np.concatenate([
            ee_pos,
            self._pick_pos  - ee_pos,   # pick_rel
            self._place_pos - ee_pos,   # place_rel
            [stage],
        ]).astype(np.float32)
        return TaskObservation(state=state, info={})
