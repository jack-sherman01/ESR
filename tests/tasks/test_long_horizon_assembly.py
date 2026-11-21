"""Three-piece continuity, native Panda and physically measured early release."""
import mujoco
import numpy as np
import pytest
from esrb.agent.assembly_policy import AssemblyPolicy
from esrb.oracles.multiconstraint import scenario_oracle
from esrb.tasks.long_horizon.assembly import LongHorizonAssemblyTask
from esrb.tasks.production.workcell import PARTS,PHASES
from esrb.update_operators.assembly_orientation import AssemblyOrientationOperator


def rollout(task,policy,seed=0):
    obs=task.reset(seed)
    try:
        for _ in range(task.MAX_STEPS):
            obs,_,done,trace=task.step(policy.act(obs))
            if done: return trace
        raise AssertionError('unbounded production episode')
    finally: task.close()


@pytest.fixture(scope='module')
def pair():
    return (rollout(LongHorizonAssemblyTask(),AssemblyPolicy()),
            rollout(LongHorizonAssemblyTask(),AssemblyOrientationOperator().apply(AssemblyPolicy()).candidate_policy))


def test_full_robot_three_free_parts_and_no_grasp_weld():
    task=LongHorizonAssemblyTask()
    try:
        obs=task.reset(0)
        assert task.action_space['shape']==(8,) and obs.state.shape==task.observation_space['shape']
        assert task._model.neq==1
        assert np.all(task._model.eq_type==mujoco.mjtEq.mjEQ_JOINT)
        assert len(task.robot.q)==7
        for part in PARTS:
            assert task._model.jnt_type[task._model.joint(part+'_free').id]==mujoco.mjtJoint.mjJNT_FREE
    finally: task.close()


def test_complete_continuous_operation_chain_and_stable_release(pair):
    good,_=pair
    assert LongHorizonAssemblyTask.evaluate(good).success
    assert [r['stage'] for r in good['stage_records']]==[n+':'+p for n in PARTS for p in PHASES]+['verify']
    for a,b in zip(good['stage_records'],good['stage_records'][1:]):
        assert a['exit_state']==b['entry_state'] and a['exit_step']==b['enter_step']
    assert scenario_oracle('long_horizon_assembly').evaluate(good,0).status=='PASS'


def test_early_release_falls_under_gravity_and_is_detected(pair):
    _,bad=pair
    assert not LongHorizonAssemblyTask.evaluate(bad).success
    assert bad['failure_reason']=='part_dropped' and bad['drop_events']
    verdict=scenario_oracle('long_horizon_assembly').evaluate(bad,0)
    assert verdict.status=='VIOLATION' and 'part_drop' in verdict.violated_constraints
    assert not verdict.missing_required


def test_task_requires_complete_release_and_consecutive_verification(pair):
    good,_=pair
    shortened={**good,'stages':good['stages'][:-24],'assembly_metrics':good['assembly_metrics'][:-24]}
    assert not LongHorizonAssemblyTask.evaluate(shortened).success


def test_raw_contact_clock_and_geometry_damage_are_unknown(pair):
    good,_=pair
    trace={**good,'physics_samples':list(good['physics_samples'])}
    trace['physics_samples'][0]={**trace['physics_samples'][0],'time_s':-1.}
    verdict=scenario_oracle('long_horizon_assembly').evaluate(trace,0)
    assert not verdict.safe and verdict.missing_required
    trace={**good,'physics_samples':good['physics_samples'][:-1]}
    assert scenario_oracle('long_horizon_assembly').evaluate(trace,0).status=='UNKNOWN'
