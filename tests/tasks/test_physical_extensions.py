"""Current full-Panda contact tasks: physical evidence and public composition."""
from copy import deepcopy
import numpy as np
import pytest
from esrb.tasks.extensions.registry import EXTENSIONS,ExtensionOperator
from esrb.tasks.extensions.settings import configured_task
from esrb.oracles.multiconstraint import scenario_oracle
from esrb.agent_api.profiles import BY_SCENARIO

NAMES=('constrained_push','force_alignment','compliant_assembly')

@pytest.mark.parametrize('name',NAMES)
def test_full_robot_physical_reset_and_eight_actuators(name):
    task=EXTENSIONS[name][0]()
    try:
        obs=task.reset(0);point=task._log['reset_readback'];digest=task._log['initial_state_digest']
        assert task._model.nu==8 and task._model.nq==16 and task._model.neq==1 and task._model.nmocap==0
        assert task._model.nv==15 and task.action_space['shape']==(8,)
        task.reset(1);assert task._log['initial_state_digest']!=digest
        task.reset(993,disturbance=point);assert task._log['initial_state_digest']==digest
        assert np.isfinite(obs.state).all() and task._model.body('workpiece').id>0
    finally:task.close()

@pytest.mark.parametrize('name',NAMES)
def test_timeout_is_separate_from_safety_and_incomplete_evidence_is_unknown(name):
    task=configured_task(EXTENSIONS[name][0],{'max_steps':6})()
    try:
        policy=EXTENSIONS[name][1]();policy.max_speed=0.
        obs=task.reset(0)
        for _ in range(6):obs,_,done,trace=task.step(policy.act(obs))
        assert done and not task.evaluate(trace).success and trace['failure_reason'].endswith('_timeout')
        assert scenario_oracle(name).evaluate(trace,0).status=='PASS'
        broken=deepcopy(trace);broken['physics_samples'].pop()
        assert scenario_oracle(name).evaluate(broken,0).status=='UNKNOWN'
        forged=deepcopy(trace);forged['robot_collisions'][0]=1
        assert not scenario_oracle(name).evaluate(forged,0).safe
    finally:task.close()

@pytest.mark.parametrize('name',NAMES)
def test_policy_consumer_and_controlled_update_use_new_joint_policy(name):
    policy=EXTENSIONS[name][1]();profile=BY_SCENARIO[name]
    assert profile.consumer.schema_id.endswith('/2')
    state=policy.state_dict();clone=profile.build(state)
    assert clone.state_dict()==state
    child=ExtensionOperator(name).apply(policy,severity=.5).candidate_policy
    assert policy.state_dict()==state and child.state_dict()!=state

@pytest.mark.parametrize('settings',[
    {'difficulty':'invalid'},{'reset_reference':{'position_jitter_m':99.}},
    {'reset_bounds':{'position_jitter_m':[-1.,1.]}},{'max_steps':0},
    {'reset_bounds':{'not_a_physical_parameter':[0.,1.]}},
])
def test_physical_configuration_rejects_out_of_domain_values(settings):
    with pytest.raises(ValueError):configured_task(EXTENSIONS['constrained_push'][0],settings)

def test_difficulty_narrows_supported_physical_domain():
    cls=EXTENSIONS['constrained_push'][0];easy=configured_task(cls,{'difficulty':'easy'})
    assert np.diff(easy.reset_bounds['friction'])[0]<np.diff(cls.reset_bounds['friction'])[0]
