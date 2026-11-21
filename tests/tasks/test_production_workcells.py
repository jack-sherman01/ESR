"""Conveyor contact physics, replayable resets and failure/safety separation."""
import mujoco
import numpy as np
import pytest
from esrb.agent.production_policy import ProductionPolicy
from esrb.tasks.extensions.moving import MovingTransportTask,MovingAssemblyTask
from esrb.tasks.long_horizon.assembly import LongHorizonAssemblyTask
from esrb.tasks.extensions.settings import configured_task
from esrb.oracles.multiconstraint import scenario_oracle


@pytest.mark.parametrize('cls',[LongHorizonAssemblyTask,MovingTransportTask,MovingAssemblyTask])
def test_reset_identity_and_task_timeout_are_not_safety_violations(cls):
    task=configured_task(cls,{'max_steps':5})()
    try:
        task.reset(3); initial=task._log['initial_state_digest']; point=task._log['reset_readback']
        task.reset(4); assert initial!=task._log['initial_state_digest']
        obs=task.reset(99,disturbance=point); assert initial==task._log['initial_state_digest']
        policy=ProductionPolicy(); policy.max_speed=0.
        for _ in range(5): obs,_,done,trace=task.step(policy.act(obs))
        assert done and not task.evaluate(trace).success
        verdict=scenario_oracle(cls.name).evaluate(trace,0)
        assert verdict.status=='PASS', verdict
    finally: task.close()


def test_belt_moves_free_parts_by_friction_without_robot_attachment():
    task=MovingTransportTask()
    try:
        task.reset(0); start=task._pose('base')[0]
        for _ in range(500): mujoco.mj_step(task._model,task._data)
        mujoco.mj_forward(task._model,task._data)
        assert task._pose('base')[0][0]-start[0]>.02
        assert task._model.neq==1 and task._model.geom_surfacevel[task._belt,0]>0
        assert task._model.nmocap==0
    finally: task.close()


def test_moving_assembly_uses_real_pallet_and_relative_stability():
    task=MovingAssemblyTask()
    try:
        task.reset(0)
        assert task._order==('connector','cap') and task._speed>0
        assert task._model.jnt_type[task._model.joint('pallet_free').id]==mujoco.mjtJoint.mjJNT_FREE
        assert set(task.metrics())=={'base','connector','cap'}
    finally: task.close()


def test_assembly_metrics_follow_translated_rotated_and_spinning_pallet():
    from scipy.spatial.transform import Rotation
    from esrb.tasks.production.geometry import ASSEMBLED_Z,HALF_HEIGHT
    task=MovingAssemblyTask()
    try:
        task.reset(0)
        rotation=Rotation.from_euler('xyz',[.15,-.1,.4]).as_matrix()
        quat=np.zeros(4); mujoco.mju_mat2Quat(quat,rotation.ravel())
        origin=np.array([.03,.21,.027])
        pallet=task._model.body('pallet').id
        q=task._model.jnt_qposadr[task._model.joint('pallet_free').id]
        task._data.qpos[q:q+7]=np.r_[origin,quat]
        base=origin+rotation@np.array([0.,0.,.007+HALF_HEIGHT['base']])
        for name in ('base','connector','cap'):
            p=base+rotation@np.array([0.,0.,ASSEMBLED_Z[name]-ASSEMBLED_Z['base']])
            address=task._part_q[name]
            task._data.qpos[address:address+7]=np.r_[p,quat]
        mujoco.mj_forward(task._model,task._data)
        linear=np.array([.02,.01,-.003]); angular=np.array([.01,-.02,.04])
        for body,joint in [(pallet,task._model.joint('pallet_free').id)]+[(task._part_ids[n],task._model.joint(n+'_free').id) for n in ('base','connector','cap')]:
            dof=task._model.jnt_dofadr[joint]
            jp,jr=np.zeros((3,task._model.nv)),np.zeros((3,task._model.nv))
            mujoco.mj_jacBody(task._model,task._data,jp,jr,body)
            velocity=np.r_[linear+np.cross(angular,task._data.xpos[body]-origin),angular]
            task._data.qvel[dof:dof+6]=np.linalg.solve(np.vstack((jp,jr))[:,dof:dof+6],velocity)
        mujoco.mj_forward(task._model,task._data)
        for value in task.metrics().values():
            assert value['seated']
            assert np.linalg.norm(value['position_error_m'])<1e-10
            assert abs(value['yaw_error_rad'])<1e-10 and value['tilt_rad']<1e-7
            assert value['relative_speed_ms']<1e-10 and value['relative_spin_rads']<1e-10
    finally: task.close()
