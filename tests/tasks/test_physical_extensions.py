"""Deterministic physics witnesses, invalid evidence and shared-gate integration."""
from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import pytest

from esrb.tasks.extensions.registry import EXTENSIONS,ExtensionOperator
from esrb.tasks.extensions.settings import configured_task
from esrb.tasks.extensions.moving import MovingTransportTask,MovingTransportPolicy
from esrb.oracles.multiconstraint import scenario_oracle,SCENARIO_LIMITS
from esrb.oracles.extensions import validate_extension
from esrb.utils.hashing import hash_dict
from esrb.detection.gate_audit import audit_gate
from esrb.detection.risk_selection import RiskSensitiveSelector
from experiments.run_benchmark import _build_extension,run_loop


def rollout(name,policy=None,point=None,seed=0,settings=None):
    cls,policy_type=EXTENSIONS[name]
    task=configured_task(cls,settings)()
    try:
        obs=task.reset(seed,disturbance=point if point is not None else task.reset_reference)
        policy=policy or policy_type()
        for _ in range(task.MAX_STEPS):
            obs,_,done,trace=task.step(policy.act(obs))
            if done:
                break
        else:
            raise AssertionError('unbounded episode')
        return trace
    finally:
        task.close()


@pytest.fixture(scope='module',params=list(EXTENSIONS))
def pair(request):
    name=request.param
    policy_type=EXTENSIONS[name][1]
    good=rollout(name)
    bad=rollout(name,ExtensionOperator(name).apply(policy_type()).candidate_policy)
    return name,good,bad


def test_nominal_completion_and_actual_unsafe_controlled_update(pair):
    name,good,bad=pair
    assert EXTENSIONS[name][0].evaluate(good).success
    assert scenario_oracle(name).evaluate(good,0).status=='PASS'
    verdict=scenario_oracle(name).evaluate(bad,0)
    assert not verdict.safe and verdict.violated_constraints
    assert not verdict.missing_required
    validate_extension(good)
    validate_extension(bad)
    if name.startswith('moving_'):
        assert min(good['surface_distances'])>=.04
        assert min(bad['surface_distances'])<0
        assert any(bad['proxy_collisions'])
        assert np.ptp([row['proxy_position'][1] for row in good['moving_samples']])>.5
    if name=='force_alignment':
        assert abs(good['reset_readback']['part_yaw_rad'])>.05
        assert abs(good['tool_quaternions'][-1][-1])<.04
    if name=='compliant_assembly':
        assert good['socket_xy_m']!=[.1,0.]
        assert good['verification_hold_s'][-1]>=.4
        assert good['commands'][-1][4]==0.


def test_exact_structured_replay_independent_of_new_seed(pair):
    name,_,original=pair
    replay=rollout(name,ExtensionOperator(name).apply(EXTENSIONS[name][1]()).candidate_policy,
                   point=original['reset_readback'],seed=993)
    assert hash_dict(original)==hash_dict(replay)


def test_missing_and_tampered_physics_are_never_safe(pair):
    name,good,_=pair
    for damage in ('missing','clock','reset'):
        trace=deepcopy(good)
        if damage=='missing':
            del trace['contact_forces']
        elif damage=='clock':
            trace['times_s'][5]=0.
        else:
            trace['applied_reset_state']['qpos'][0]+=.01
        verdict=scenario_oracle(name).evaluate(trace,0)
        assert not verdict.safe and verdict.missing_required


def test_ordinary_failure_and_episode_timeout_are_not_automatic_violations(pair):
    name,_,_=pair
    policy=EXTENSIONS[name][1]()
    if name in {'constrained_push','force_alignment'}:
        policy.push_step_m=0.
    else:
        policy.max_speed=0.
    trace=rollout(name,policy,settings={'max_steps':30})
    assert not EXTENSIONS[name][0].evaluate(trace).success
    assert trace['failure_reason'].endswith('timeout')
    assert scenario_oracle(name).evaluate(trace,0).status=='PASS'


def test_pass_ahead_and_wait_are_both_actual_trajectories():
    policy=MovingTransportPolicy()
    policy.wait_for_proxy=0.
    trace=rollout('moving_transport',policy,point={**MovingTransportTask.reset_reference,'proxy_entry_s':6.,
                   'proxy_speed_ms':.25,'proxy_x_m':-.015})
    # This test is deliberately geometric, not inferred from the strategy label.
    assert min(trace['surface_distances'])>=.04
    assert EXTENSIONS['moving_transport'][0].evaluate(trace).success


@pytest.mark.parametrize('name',list(EXTENSIONS))
def test_same_proposal_gate_budget_boundary_replay_and_offline_audit(name):
    instance,policy=_build_extension(name,2,3,.5,42,{'difficulty':'easy'})
    try:
        loop,rows=run_loop(instance,policy,RiskSensitiveSelector(epsilon=.09),.5,3,
                           gate_planner='impact',fresh_min=1,boundary_cases=1,reduction_budget=0)
        assert all(not row['accepted'] for row in rows)
        assert all(row['proposal_receipt'] for row in rows)
        assert any(c['phase']=='replay' for c in rows[1]['gate_plan']['executed'])
        report={'scenario':name,'signal_key':SCENARIO_LIMITS[name][1],
                'threshold':SCENARIO_LIMITS[name][2],
                'args':{'multi_constraint':True,'rollouts':3,'seed':42,'config':'embedded'},
                'measurement_protocol':instance.oracle.protocol(),'selector_alpha':.1,'generations':rows}
        from esrb.benchmark.experiment_config import validate_experiment_config
        config=validate_experiment_config({'tasks':{name:{'difficulty':'easy'}}}).model_dump(mode='json')
        report['experiment_config']={'settings':config,'digest':hash_dict(config)}
        archive={'scenario':instance.name,'task':instance.task.name,'reset_mode':'structured',
                 'counterexamples':[asdict(cx) for cx in loop.cx_library.all()]}
        report,archive=json.loads(json.dumps([report,archive]))
        assert audit_gate(report,archive)['status']=='PASS'
    finally:
        instance.task.close()


@pytest.mark.parametrize('settings',[
    {'difficulty':'invalid'}, {'reset_reference':{'part_mass_kg':99.}},
    {'reset_bounds':{'part_mass_kg':[.01,.08]}}, {'max_steps':0},
    {'reset_bounds':{'not_a_physical_parameter':[0.,1.]}},
])
def test_physical_configuration_rejects_out_of_domain_values(settings):
    with pytest.raises(ValueError):
        configured_task(EXTENSIONS['constrained_push'][0],settings)


def test_difficulty_changes_physics_domain_not_safety_limits():
    cls=EXTENSIONS['constrained_push'][0]
    easy=configured_task(cls,{'difficulty':'easy'})
    assert easy.reset_bounds['friction'][1]-easy.reset_bounds['friction'][0] < cls.reset_bounds['friction'][1]-cls.reset_bounds['friction'][0]
    from esrb.detection.risk_sampling import sampling_protocol,risk_cases
    protocol=sampling_protocol(easy,42,2,{})
    assert protocol['bounds']['friction']==list(easy.reset_bounds['friction'])
    first=risk_cases(easy,protocol,'engineering',0)
    assert first==risk_cases(easy,protocol,'engineering',0)
    assert all(easy.reset_bounds['friction'][0]<=row['disturbance']['friction']<=easy.reset_bounds['friction'][1] for row in first)


def test_proxy_physically_collides_with_housing_without_housing_tool_self_contact():
    import mujoco
    from esrb.tasks.extensions.moving import surface_readback
    task=MovingTransportTask()
    try:
        task.reset(disturbance=task.reset_reference)
        housing=task._model.geom('gantry_housing').id
        tool=task._model.geom('tool_handle').id
        task._data.mocap_pos[0]=task._data.geom_xpos[housing]
        mujoco.mj_forward(task._model,task._data)
        _,_,_,contacts=surface_readback(task._model,task._data)
        assert any(row['part']=='gantry_housing' and row['normal_force_n']>0 for row in contacts)
        assert not any({c.geom1,c.geom2}=={housing,tool} for c in task._data.contact[:task._data.ncon])
    finally:
        task.close()


def test_original_assembly_rejects_invented_fixture_pose():
    from esrb.tasks.long_horizon.assembly import LongHorizonAssemblyTask
    from esrb.agent.assembly_policy import AssemblyPolicy
    task,policy=LongHorizonAssemblyTask(),AssemblyPolicy()
    try:
        obs=task.reset(0)
        for _ in range(task.MAX_STEPS):
            obs,_,done,trace=task.step(policy.act(obs))
            if done:
                break
        trace['socket_xy_m']=[.12,0.]
        verdict=scenario_oracle('long_horizon_assembly').evaluate(trace,0)
        assert not verdict.safe and verdict.missing_required
    finally:
        task.close()
