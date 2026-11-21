"""Full articulated Panda, joint-space actions and a freely held precision driver."""
from pathlib import Path
import mujoco
import numpy as np
from esrb.tasks.base import TaskObservation
from esrb.tasks.tool_handling.friction import FrictionToolTask, yaw
from esrb.tasks.robots.panda import orientation_error, PandaRobot, HOME


class ToolHandlingTask(FrictionToolTask):
    name = 'tool_handling'
    MODEL_XML = Path(__file__).parent/'assets/tool_handling.xml'
    MODEL_SCHEMA = 'panda_tool_workcell/3'
    GRASP_HEIGHT = .138  # Keep the top of the driver below the native Panda palm.

    def __init__(self):
        super().__init__()
        self._panda = PandaRobot(self._model, self._data)
        self._arm_joints = self._panda.joints
        self._arm_q, self._arm_v = self._panda.q, self._panda.v
        self._joint_limits = self._panda.limits
        self._home = HOME.copy()
        self._site = self._panda.site
        self._pad_groups = self._panda.pad_groups
        self._pads = sorted(self._panda.pads)
        self._robot_geoms = self._panda.geoms
        self._physics = []
        self._target_data = self._panda.command_data

    @property
    def action_space(self):
        return {'shape': (8,), 'low': np.r_[self._joint_limits[:, 0], 0.],
                'high': np.r_[self._joint_limits[:, 1], .025]}

    @property
    def observation_space(self):
        return {'shape': (24,), 'dtype': np.float64}

    def _reset_robot(self):
        q = self._home.copy()
        target = self._pick
        angle = self._initial_yaw
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                             [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        self._data.qpos[7:9] = .025
        for iteration in range(300):
            self._data.qpos[self._arm_q] = q
            mujoco.mj_kinematics(self._model, self._data)
            mujoco.mj_comPos(self._model, self._data)
            position_error = target - self._data.site_xpos[self._site]
            angular_error = orientation_error(self._data.site_xmat[self._site].reshape(3, 3), rotation)
            if np.linalg.norm(position_error) < 2e-5 and np.linalg.norm(angular_error) < 2e-4:
                break
            jp, jr = np.zeros((3, self._model.nv)), np.zeros((3, self._model.nv))
            mujoco.mj_jacSite(self._model, self._data, jp, jr, self._site)
            j = np.vstack((jp[:, self._arm_v], .25*jr[:, self._arm_v]))
            error = np.r_[position_error, .25*angular_error]
            dq = j.T @ np.linalg.solve(j @ j.T + .0001*np.eye(6), error)
            q = np.clip(q + np.clip(dq, -.12, .12), self._joint_limits[:, 0]+.03, self._joint_limits[:, 1]-.03)
        else:
            raise RuntimeError(f'Panda initial IK failed: position={np.linalg.norm(position_error):.6f} m, '
                               f'orientation={np.linalg.norm(angular_error):.6f}, q={q.tolist()}')
        self._data.qpos[self._arm_q] = q
        self._data.ctrl[:] = np.r_[q, .025/.04*255]
        self._reset_arm_q = q.copy()
        self._reset_ik = {'iterations': iteration+1, 'position_error_m': float(np.linalg.norm(position_error)),
                          'orientation_error_rad': float(np.linalg.norm(angular_error))}

    def reset(self, seed=0, *, disturbance=None):
        super().reset(seed, disturbance=disturbance)
        for field in ['arm_joint_positions', 'arm_joint_velocities', 'arm_joint_torques',
                      'robot_physics', 'robot_collisions']:
            self._log[field] = []
        self._log.update(robot='Franka Emika Panda', action_schema='panda-joint-targets-and-grip/1',
                         arm_joint_names=[f'joint{i}' for i in range(1, 8)],
                         arm_joint_limits=self._joint_limits.tolist(),
                         initial_arm_joints=self._reset_arm_q.tolist(), initial_ik=self._reset_ik,
                         gripper_servo={'opening_m_per_finger': '.025 - closure_m', 'kp_n_m': 700.,
                                        'tendon_force_limit_n': 20., 'opening_step_m': .0006},
                         gravity_compensation=True, unexpected_contact_min_force_n=.01)
        self._log.update(grasp_height_m=self.GRASP_HEIGHT, lift_height_m=.23,
                         grasp_settle_control_steps=45)
        return self._observe()

    def _observe(self):
        base = super()._observe()
        jp, jr = np.zeros((3, self._model.nv)), np.zeros((3, self._model.nv))
        self._target_data.qpos[:] = self._data.qpos
        self._target_data.qpos[self._arm_q] = self._data.ctrl[:7]
        mujoco.mj_kinematics(self._model, self._target_data)
        mujoco.mj_comPos(self._model, self._target_data)
        mujoco.mj_jacSite(self._model, self._target_data, jp, jr, self._site)
        q = self._data.qpos[self._arm_q]
        return TaskObservation(np.r_[base.state, q, self._data.qvel[self._arm_v]],
                               {**base.info, 'lift_height': .23, 'arm_joint_positions': q.tolist(),
                                'arm_home': self._reset_arm_q.tolist(), 'arm_joint_limits': self._joint_limits.tolist(),
                                'grip_position': self._data.site_xpos[self._site].tolist(),
                                'grip_rotation': self._data.site_xmat[self._site].reshape(3, 3).tolist(),
                                'arm_joint_targets': self._data.ctrl[:7].tolist(),
                                'target_grip_position': self._target_data.site_xpos[self._site].tolist(),
                                'target_grip_rotation': self._target_data.site_xmat[self._site].reshape(3, 3).tolist(),
                                'target_grip_jacobian': np.vstack((jp[:, self._arm_v], jr[:, self._arm_v])).tolist(),
                                'actuator_targets': [*self._target_data.xpos[self._gripper],
                                                     yaw(self._target_data.xquat[self._gripper])]})

    def _apply_action(self, action):
        opening = np.clip(.025-action[7], 0., .04)
        previous = self._data.ctrl[7]/255*.04
        opening = previous + np.clip(opening-previous, -.0006, .0006)
        self._data.ctrl[:] = np.r_[action[:7], opening/.04*255]

    def _advance(self, grip, tip):
        # The native fingers need time to finish their bounded closing motion.
        if self._stage == 0 and self._stage_ticks < 45:
            return
        super()._advance(grip, tip)

    def _after_substep(self):
        contacts = []
        for index, contact in enumerate(self._data.contact):
            pair = {contact.geom1, contact.geom2}
            robot = pair.intersection(self._robot_geoms)
            if not robot or (pair.intersection(self._tool_geoms) and robot.intersection(self._pads)):
                continue
            if contact.dist > 0:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self._model, self._data, index, force)
            if force[0] > .01:
                contacts.append({'geom1': self._model.geom(contact.geom1).name,
                                 'geom2': self._model.geom(contact.geom2).name,
                                 'normal_force_n': float(force[0]), 'time_s': float(self._data.time)})
        self._physics.append({'time_s': float(self._data.time), 'qpos': self._data.qpos[self._arm_q].tolist(),
                              'qvel': self._data.qvel[self._arm_v].tolist(),
                              'actuator_torque_nm': self._data.qfrc_actuator[self._arm_v].tolist(),
                              'joint_targets': self._data.ctrl[:7].tolist(), 'contacts': contacts})

    def step(self, action):
        self._physics = []
        observation, reward, done, _ = super().step(action)
        self._log['robot_physics'].append(self._physics)
        self._log['arm_joint_positions'].append(self._data.qpos[self._arm_q].tolist())
        self._log['arm_joint_velocities'].append(self._data.qvel[self._arm_v].tolist())
        self._log['arm_joint_torques'].append(self._data.qfrc_actuator[self._arm_v].tolist())
        self._log['robot_collisions'].append(int(any(row['contacts'] for row in self._physics)))
        return observation, reward, done, dict(self._log)

    def render_frame(self, width=640, height=480):
        from esrb.tasks.tool_handling.appearance import apply_appearance
        if self._renderer is None:
            apply_appearance(self._model)
        camera = self._model.camera('overview').id
        position, target = np.array([1.05, -1.10, .85]), np.array([.10, -.16, .28])
        backward = position-target
        backward /= np.linalg.norm(backward)
        right = np.cross([0., 0., 1.], backward)
        right /= np.linalg.norm(right)
        self._model.cam_pos[camera] = position
        mujoco.mju_mat2Quat(self._model.cam_quat[camera],
                           np.column_stack((right, np.cross(backward, right), backward)).ravel())
        mujoco.mj_camlight(self._model, self._data)
        return super().render_frame(width, height)

    def _decorate_render_scene(self, scene):
        from esrb.tasks.tool_handling.appearance import decorate_scene
        decorate_scene(scene)
