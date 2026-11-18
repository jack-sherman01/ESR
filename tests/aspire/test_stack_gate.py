"""Contact semantics, missing evidence, and parent retention for Stack."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from scripts.gates.stack_gate_rules import StackSafety, decide
from scripts.gates import stack_gate_state as state
SPEC=importlib.util.spec_from_file_location('gated_stack_batch',ROOT/'scripts/runners/run_aspire_stack_batch.py')
batch=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(batch)
RULES=json.loads((ROOT/'configs/experiment/stack_safety_gate.json').read_text())


def contact(a,b,force=1.):
    return {'geom1':1,'geom2':2,'groups':[a,b],'normal_force_n':force}


def sample(monitor,red=(0.,0.,.82),contacts=()):
    t=monitor.last_end
    return monitor.observe(t,t+.002,list(red),[.1,0.,.825],list(contacts))


def report(monitor):
    return monitor.report(elapsed_physics_steps=monitor.steps,records_written=monitor.steps)


@pytest.mark.parametrize('groups',RULES['forbidden_contact_pairs'])
def test_forbidden_contact_is_rejected_at_physics_rate(groups):
    m=StackSafety(RULES,.8)
    sample(m,contacts=[contact(*groups)])
    assert report(m)['status']=='VIOLATION'
    assert m.events[0]['time_start_s']==0.


def test_expected_grasp_support_and_release_contacts_remain_allowed():
    m=StackSafety(RULES,.8)
    sample(m,contacts=[contact('red_cube','table')])
    sample(m,red=(0.,0.,.9),contacts=[contact('red_cube','gripper')])
    sample(m,red=(.1,0.,.87),contacts=[contact('red_cube','green_cube')])
    assert report(m)['status']=='PASS'


def test_lifted_target_returned_to_table_without_grasp_is_drop():
    m=StackSafety(RULES,.8)
    sample(m,red=(0.,0.,.9),contacts=[contact('red_cube','gripper')])
    sample(m,contacts=[contact('red_cube','table')])
    assert m.events[-1]['kind']=='lifted_red_returned_to_table_without_grasp'


@pytest.mark.parametrize('cube',['red_cube','green_cube'])
def test_either_cube_on_floor_is_rejected(cube):
    m=StackSafety(RULES,.8);sample(m,contacts=[contact(cube,'floor')])
    assert m.events[0]['kind']=='cube_on_floor'


def test_zero_load_ghost_contact_does_not_count_as_collision():
    m=StackSafety(RULES,.8);sample(m,contacts=[contact('gripper','table',0.)])
    assert report(m)['status']=='PASS'


def test_missing_sample_cannot_be_safe():
    m=StackSafety(RULES,.8);sample(m)
    assert m.report(elapsed_physics_steps=2,records_written=1)['status']=='UNKNOWN'
    with pytest.raises(ValueError,match='missing or nonuniform'):
        m.observe(.004,.006,[0.,0.,.82],[.1,0.,.825],[])


def test_nonfinite_or_unclassified_contact_is_not_silently_allowed():
    m=StackSafety(RULES,.8)
    with pytest.raises(ValueError,match='unclassified'):
        sample(m,contacts=[contact('gripper','other')])
    with pytest.raises(ValueError,match='nonfinite'):
        sample(m,contacts=[contact('gripper','table',float('nan'))])


def valid_result(**changes):
    value={'status':'MEASURED','program_ok':True,'native_task_success':True,'settle_steps':20,'settle_success_steps':20,
           'safety':{'status':'PASS','coverage_complete':True}}
    value.update(changes);return value


def test_task_success_with_unsafe_contact_is_rejected_and_parent_retained():
    parent={'version':1}
    result=decide(valid_result(safety={'status':'VIOLATION','coverage_complete':True}),active=parent,binding_ok=True,expected_settle_steps=20)
    assert not result['accepted'] and result['retained_parent'] and result['observed_safety_regression']
    assert result['reason']=='observed_safety_violation'


@pytest.mark.parametrize('changes,reason',[
    ({'native_task_success':False},'task_not_qualified'),
    ({'safety':{'status':'UNKNOWN'}},'unknown_safety_evidence'),
    ({'settle_success_steps':19},'terminal_goal_not_maintained_or_missing_hold'),
    ({'settle_steps':0},'terminal_goal_not_maintained_or_missing_hold'),
    ({'program_ok':False},'task_not_qualified'),
])
def test_incomplete_task_or_evidence_never_promotes(changes,reason):
    result=decide(valid_result(**changes),active={'version':1},binding_ok=True,expected_settle_steps=20)
    assert not result['accepted'] and result['retained_parent'] and not result['observed_safety_regression']
    assert result['reason']==reason


def test_binding_failure_and_first_parent_bootstrap():
    assert not decide(valid_result(),active=None,binding_ok=False,expected_settle_steps=20)['accepted']
    assert decide(valid_result(),active=None,binding_ok=True,expected_settle_steps=20)['accepted']


@pytest.fixture
def campaign(tmp_path):
    root=tmp_path/'batch';batch.init(root,[17])
    state.save(root/'round_01/initial_reference.json',{'initial_state_id':'same-initial'})
    return root


def propose(root,version):
    source=root/f'source-{version}.py';source.write_text(f'result = {version}\n')
    return state.prepare(root,1,version,source,'synthetic unit-test candidate')


def measurement(root,folder,version,safety='PASS'):
    out=folder/'execution';out.mkdir()
    witness=out/'safety_physics.jsonl.gz';witness.write_bytes(b'synthetic witness')
    result=valid_result(round=1,version=version,seed=17,initial_state_id='same-initial',
        program_sha256=state.digest(folder/'program.py'),protocol_sha256=state.digest(root/'protocol.json'),
        safety={'status':safety,'coverage_complete':True,'rules_sha256':state.digest(root/'safety_rules.json'),
                'witness_sha256':state.digest(witness)})
    state.save(out/'result.json',result)


def commit(root,version):
    with state.round_lock(root/'round_01'):
        return state.commit(root,1,version)


def test_rejected_candidate_does_not_become_next_parent(campaign):
    root=campaign
    first=propose(root,1);measurement(root,first,1);assert commit(root,1)['accepted']
    second=propose(root,2);measurement(root,second,2,'VIOLATION')
    decision=commit(root,2)
    assert not decision['accepted'] and decision['parent_after']['version']==1
    third=propose(root,3)
    assert state.read(third/'proposal.json')['gate_base']['version']==1
    measurement(root,third,3);assert commit(root,3)['accepted']
    assert state.load_chain(root,1)['active']['version']==3


def test_stale_parent_rule_mutation_and_program_changes_block_execution(campaign):
    root=campaign;folder=propose(root,1)
    proposal=state.read(folder/'proposal.json');proposal['gate_base']={'version':99}
    state.save(folder/'proposal.json',proposal)
    with pytest.raises(ValueError,match='current active parent'):state.preflight(root,1,1)
    proposal['gate_base']=None;state.save(folder/'proposal.json',proposal)
    (folder/'program.py').write_text('changed')
    with pytest.raises(ValueError,match='differs'):state.preflight(root,1,1)
    with (root/'safety_rules.json').open('a') as f:f.write(' ')
    with pytest.raises(ValueError,match='rules changed'):state.load_rules(root)


def test_missing_witness_rejects_and_preserves_active_parent(campaign):
    root=campaign;first=propose(root,1);measurement(root,first,1);commit(root,1)
    second=propose(root,2);measurement(root,second,2)
    (second/'execution/safety_physics.jsonl.gz').unlink()
    decision=commit(root,2)
    assert decision['reason']=='binding_or_integrity_error' and decision['parent_after']['version']==1


def test_history_and_frozen_parent_are_bound_but_convenience_pointer_is_not_authoritative(campaign):
    root=campaign;folder=propose(root,1);measurement(root,folder,1);commit(root,1)
    state.save(root/'round_01/active_parent.json',{'version':99})
    assert state.load_chain(root,1)['active']['version']==1
    chain=state.read(root/'round_01/gate_chain.json');chain['events'][0]['accepted']=False
    state.save(root/'round_01/gate_chain.json',chain)
    with pytest.raises(ValueError,match='broken gate'):state.load_chain(root,1)


def test_historical_no_gate_batch_is_not_extended(tmp_path):
    state.save(tmp_path/'protocol.json',{'safety_gate':{'enabled':False}})
    with pytest.raises(ValueError,match='legacy'):state.load_rules(tmp_path)
