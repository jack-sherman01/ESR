"""Meaningful negative/positive temporal and integrity controls; no new episodes."""
from pathlib import Path
import numpy as np
import pytest
from esrb.runtime.io import read,digest,save
from esrb.detection.natural_audit import temporal_events,rotation,classify,audit_episode

RULES=read(Path(__file__).resolve().parents[2]/'configs/evaluation/natural_offline_v1.json')


def series(n=1000):
    t=np.arange(n)*.002;R=np.tile(np.eye(3),(n,1,1));z=np.ones(n)
    return {'time':t,'z':z,'vz':np.zeros(n),'initial_z':.8,
        'held0':np.r_[np.ones(200,bool),np.zeros(n-200,bool)],'held1':np.zeros(n,bool),
        'grasp_contact':np.r_[np.ones(200,bool),np.zeros(n-200,bool)],
        'support':np.zeros(n,bool),'ground':np.zeros(n,bool),'goal':np.zeros(n,bool),
        'grip_z0':np.ones(n),'grip_R0':R,'local0':np.zeros((n,3)),
        'grip_z1':np.ones(n),'grip_R1':R.copy(),'local1':np.zeros((n,3))}


def test_unheld_airborne_fall_then_upward_support_is_confirmed_drop():
    s=series();s['z'][200:400]=np.linspace(1,.8,200);s['z'][400:]=.8
    s['vz'][200:400]=-.5;s['support'][400:]=True;s['ground'][400:]=True
    events,stages,unknown,phases=temporal_events(s,'stack',RULES)
    drop=next(e for e in events if e['kind']=='drop')
    assert drop['loss_physics_index']<drop['physics_index']==400
    assert not unknown and 'lift' in stages


def test_supported_roll_or_settling_after_release_is_not_airborne_drop():
    s=series();s['z'][200:400]=np.linspace(1,.8,200);s['z'][400:]=.8
    s['vz'][200:400]=-.5;s['support'][200:]=True;s['ground'][200:]=True
    events,_,unknown,_=temporal_events(s,'pickplacecan',RULES)
    assert not events and not unknown


def test_geometric_goal_supported_release_is_not_a_drop():
    s=series();s['z'][200:]=.9;s['vz'][200:220]=-.5
    s['goal'][200:]=True;s['support'][200:]=True;s['ground'][200:]=True
    events,stages,unknown,_=temporal_events(s,'nut-assembly',RULES)
    assert not events and not unknown and 'stable_goal_release' in stages


def test_failed_task_without_grasp_does_not_become_a_drop():
    s=series();s['held0'][:]=False;s['grasp_contact'][:]=False
    s['z'][:]=.8;s['support'][:]=True;s['ground'][:]=True
    events,stages,unknown,_=temporal_events(s,'stack',RULES)
    assert not events and not unknown and 'lift' not in stages


def test_unconfirmed_airborne_loss_is_unknown_and_regrasp_cancels_it():
    s=series();s['z'][200:400]=np.linspace(1,.8,200);s['z'][400:]=.8;s['vz'][200:400]=-.5
    events,_,unknown,_=temporal_events(s,'stack',RULES)
    assert unknown and [e['kind'] for e in events]==['unconfirmed_drop']
    s['grasp_contact'][350:]=True;s['held0'][350:]=True
    events,_,unknown,_=temporal_events(s,'stack',RULES)
    assert not unknown and not events


def test_handover_stable_receiver_is_legal_and_premature_release_is_separate():
    s=series();s['held1'][120:]=True;s['grasp_contact'][:]=True
    events,stages,unknown,_=temporal_events(s,'two-arm-handover',RULES)
    assert not events and 'handover' in stages
    s['held1'][120:]=False;s['held1'][190:]=True
    events,stages,unknown,_=temporal_events(s,'two-arm-handover',RULES)
    assert 'premature_release' in [e['kind'] for e in events]


def test_grasp_contact_chatter_shorter_than_dwell_does_not_qualify_pickup():
    s=series();s['held0'][30:]=False;s['grasp_contact'][30:]=False
    s['vz'][30:]=-.5;s['z'][30:]=.8;s['support'][100:]=True;s['ground'][100:]=True
    events,stages,unknown,_=temporal_events(s,'stack',RULES)
    assert not events and 'lift' not in stages


@pytest.mark.parametrize('body,ancestors,name,task,expected',[
    ('peg1',['peg1'],None,'nut-assembly','square_peg'),
    ('bin2',['bin2'],None,'pickplacecan','target_bin'),
    ('gripper0_wiping_gripper',['gripper0_wiping_gripper','robot0_right_hand'],'gripper0_wiping_corner1','wipe','tool'),
    ('gripper0_leftfinger',['gripper0_leftfinger','robot0_right_hand'],'gripper0_finger1_collision','stack','gripper0'),
    ('mount0_base',['mount0_base','robot0_base'],'mount0_base','stack','mount')])
def test_parts_are_identified_by_ancestors_even_for_unnamed_geometry(body,ancestors,name,task,expected):
    assert classify({'body_name':body,'ancestors':ancestors,'geom_name':name},task)==expected


def test_contact_frame_rotation_uses_transpose():
    q=np.array([np.cos(np.pi/4),0,0,np.sin(np.pi/4)])
    R=rotation(q)
    np.testing.assert_allclose(R@np.array([1,0,0]),[0,1,0],atol=1e-12)
    # Contact axes are rows. First axis points upward; positive normal supports geom2.
    C=np.array([[0,0,1],[0,1,0],[-1,0,0]])
    np.testing.assert_array_equal(C.T@np.array([2,0,0]),[0,0,2])


def test_missing_and_corrupted_evidence_remain_unknown(tmp_path):
    row={'id':'synthetic','execution':str(tmp_path)}
    result=audit_episode(row,RULES)
    assert result['audit_status']=='UNKNOWN' and not result['coverage_complete']
    save(tmp_path/'artifacts.json',{'result.json':'not-the-current-hash'})
    save(tmp_path/'result.json',{'native_task_success':True})
    result=audit_episode(row,RULES)
    assert result['audit_status']=='UNKNOWN' and 'hash mismatch' in result['unknown'][0]


def test_parent_child_transitions_keep_existing_violations_and_unknown_separate():
    from esrb.tools.analysis.natural_audit import transition
    assert transition('PASS','VIOLATION')=='NEW_VIOLATION'
    assert transition('VIOLATION','VIOLATION')=='PERSISTENT_VIOLATION'
    assert transition('VIOLATION','PASS')=='RESOLVED'
    assert transition('UNKNOWN','VIOLATION')=='UNKNOWN'


def test_accepted_lineage_runs_measure_persistence_resolution_and_recurrence():
    from esrb.tools.analysis.natural_audit import lineage_intervals
    chain={'task':'stack','chain':1,'events':[{'version':v,'accepted':v!=3} for v in [1,2,3,4,5]]}
    results={}
    labels={1:'PASS',2:'VIOLATION',4:'PASS',5:'VIOLATION'}
    for v,label in labels.items():
        for seed in [1,2,3,4,5]:results[('stack',1,v,seed)]={'predicates':{'drop':label}}
    runs=lineage_intervals(chain,results)
    assert len(runs)==10
    first=next(r for r in runs if r['seed']==1 and r['start_version']==2)
    again=next(r for r in runs if r['seed']==1 and r['start_version']==5)
    assert first['accepted_parent_states']==1 and first['resolution_version']==4
    assert again['recurrence_after_known_pass'] and again['end_status']=='RIGHT_CENSORED_AT_FINAL_PARENT'
