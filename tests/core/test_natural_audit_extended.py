"""Independent analytic counterexamples for extended offline rules; no stepping."""
from types import SimpleNamespace
from pathlib import Path
import mujoco
import numpy as np
import pytest
from esrb.runtime.io import read
from esrb.detection.natural_audit_extended import (wrench_at_origin,independent_bodies,extra_contact_kind,
    range_summary,released_goal_collapse,interface_metrics,audit_episode)

R=read(Path(__file__).resolve().parents[2]/'configs/evaluation/natural_offline_v2.json')


def contact(force=(0,0,2),point=(1,0,0),moment=(0,0,0),frame=None):
    c=np.zeros((1,21));c[0,3:6]=point;c[0,6:15]=np.eye(3).ravel() if frame is None else np.asarray(frame).ravel()
    c[0,15:18]=force;c[0,18:21]=moment
    return c


def test_offset_and_intrinsic_moment_are_shifted_to_body_origin_with_body_sign():
    c=contact(moment=(0,0,3))
    value=wrench_at_origin(c,np.array([[0.,0.,0.]]),[1])
    np.testing.assert_array_equal(value,[[0,0,2,0,-2,3]])
    np.testing.assert_array_equal(wrench_at_origin(c,np.array([[1.,0.,0.]]),[-1]),[[0,0,-2,0,0,-3]])


def test_rotated_contact_force_and_moment_transform_before_origin_shift():
    c=contact(force=(2,0,0),point=(0,1,0),moment=(3,0,0),frame=[[0,0,1],[0,1,0],[-1,0,0]])
    np.testing.assert_array_equal(wrench_at_origin(c,np.zeros((1,3)),[1]),[[0,0,2,2,0,3]])


def model():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body name="a"><joint name="a"/><geom type="sphere" size=".1"/>
      <body name="weld" pos="0 0 .2"><geom type="sphere" size=".1"/>
        <body name="b" pos="0 0 .2"><joint name="b"/><geom type="sphere" size=".1"/>
          <body name="c" pos="0 0 .2"><joint name="c"/><geom type="sphere" size=".1"/></body>
        </body></body></body></worldbody></mujoco>''')


def test_same_arm_excludes_same_weld_and_adjacent_but_detects_remote_bodies():
    m=model();a,b,c,w=[m.body(n).id for n in ['a','b','c','weld']]
    assert not independent_bodies(m,a,a)
    assert not independent_bodies(m,a,w)
    assert not independent_bodies(m,a,b)
    assert independent_bodies(m,a,c)
    assert not independent_bodies(m,b,c)


def fake_geometry(groups,names=None,task='nut-assembly'):
    # Four body ids correspond to a, weld, b, c in the analytic model.
    m=model();m.geom_bodyid[:]=[m.body(n).id for n in ['a','weld','b','c']]
    return SimpleNamespace(model=m,groups=groups,robot=np.array([x.startswith(('gripper','link')) for x in groups]),
        arms=np.array([0 if x.startswith(('gripper','link')) else -1 for x in groups]),task=task,
        geoms=[{'geom_name':n} for n in (names or ['g0','g1','g2','g3'])])


@pytest.mark.parametrize('groups,pair,expected',[
    (['link0','link0','link0','link0'],(0,3),'same_arm_self_collision'),
    (['link0','link0','link0','link0'],(0,2),None),
    (['gripper0']*4,(0,3),None),
    (['target','round_peg','other_object','floor'],(0,1),'wrong_peg_contact'),
    (['target','round_peg','other_object','floor'],(0,2),'target_other_object_collision'),
    (['target','round_peg','other_object','floor'],(2,3),'non_target_floor_contact'),
    (['target','square_peg','other_object','table'],(0,1),None),
    (['target','square_peg','other_object','table'],(0,3),None),
    (['tool','table','tool','table'],(0,1),None),
])
def test_expected_contacts_and_additional_unexpected_parts(groups,pair,expected):
    assert extra_contact_kind(fake_geometry(groups),*pair)==expected


def test_housing_is_not_finger_grasp():
    groups=['gripper0','target','target','target']
    assert extra_contact_kind(fake_geometry(groups,['gripper0_hand_collision','obj','obj','obj']),0,1)=='gripper_housing_object_collision'
    assert extra_contact_kind(fake_geometry(groups,['gripper0_finger1_collision','obj','obj','obj']),0,1) is None


def test_operating_range_equality_and_short_soft_excursion_are_distinct():
    assert not range_summary(np.zeros(100),0,1,1e-6,.002,.1)['qualified_intervals']
    assert not range_summary(np.r_[np.full(49,-.1),np.zeros(51)],0,1,1e-6,.002,.1)['qualified_intervals']
    r=range_summary(np.r_[np.full(50,-.1),np.zeros(50)],0,1,1e-6,.002,.1)
    assert r['qualified_intervals']==[(0,50)] and r['longest_excursion_s']==.1


def collapse_series():
    n=300;t=np.arange(n)*.002;z=np.ones(n);vz=np.zeros(n);goal=np.ones(n,bool);grasp=np.zeros(n,bool);up=np.ones(n,bool)
    goal[100:]=False;up[100:220]=False;z[100:220]=np.linspace(1,.8,120);z[220:]=.8;vz[100:220]=-1
    return t,z,vz,goal,grasp,up


def test_stable_placement_then_later_airborne_collapse_is_witnessed():
    e=released_goal_collapse(*collapse_series(),R)
    assert e['physics_index']==220 and e['height_loss_m']==pytest.approx(.2)


def test_supported_roll_or_regrasp_is_not_post_release_collapse():
    values=list(collapse_series());values[-1][:]=True
    assert released_goal_collapse(*values,R) is None
    values=list(collapse_series());values[4][110:]=True
    assert released_goal_collapse(*values,R) is None


def test_unconfirmed_post_release_airborne_loss_is_not_fabricated_impact():
    values=list(collapse_series());values[-1][100:]=False
    assert released_goal_collapse(*values,R) is None


def test_interface_magnitude_does_not_cancel_opposing_forces_or_cross_phase_window():
    block=np.zeros((100,10));block[:,6]=4;block[:,8]=4
    # Force resultants cancel, but 2N on each side gives 4N magnitude sum.
    result=interface_metrics({'nut|peg':[(0,block)]},100,np.array(['insert']*50+['hold']*50),.002,50,.01)
    for v in result.values():
        assert v['peak_net_force_n']==0 and v['peak_force_magnitude_sum_n']==4
        assert v['peak_100ms_force_magnitude_sum_n']==4
        assert v['active_s']==.1
    result=interface_metrics({'nut|peg':[(0,block[:49])]},49,np.array(['insert']*49),.002,50,.01)
    assert next(iter(result.values()))['peak_100ms_force_magnitude_sum_n'] is None


def test_missing_source_is_unknown_and_never_clean(tmp_path):
    r=audit_episode({'base_audit_file':str(tmp_path/'absent'),'base_audit_sha256':'missing'},R)
    assert r['audit_status']=='UNKNOWN' and not r['coverage_complete']


def test_exact_native_disposal_marker_is_not_an_arbitrary_workspace_exclusion():
    from esrb.detection.natural_audit_extended import parked_objects
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="inactive"><freejoint/><geom size=".02"/></body>
    <body name="active"><freejoint/><geom size=".02"/></body></worldbody></mujoco>''')
    initial=np.r_[0.,m.qpos0,np.zeros(m.nv)]
    initial[1:8]=[10,10,10,1,0,0,0]
    assert parked_objects(m,initial,['other_object','target'])=={m.body('inactive').id}
    initial[1]=9.
    assert not parked_objects(m,initial,['other_object','target'])


@pytest.fixture
def passive_fixture(tmp_path):
    """Computational channel fixture only. No MuJoCo stepping or real episode."""
    from esrb.runtime.io import save,digest
    from esrb.detection.natural_audit import fingerprint_identity
    root=tmp_path/'execution';root.mkdir();(root/'physics').mkdir()
    xml='''<mujoco><option timestep=".002"/><worldbody>
      <body name="robot0_link1"><joint name="robot0_joint1" range="-1 1" limited="true"/>
      <geom name="link" size=".02"/><body name="gripper0_wiping_gripper" pos="0 0 .1">
      <geom name="gripper0_wiping_surface1a" size=".02"/><site name="force_site"/></body></body>
      </worldbody><actuator><motor name="robot0_motor" joint="robot0_joint1" ctrllimited="true" ctrlrange="-80 80"/></actuator>
      <sensor><force name="gripper0_force_ee" site="force_site"/></sensor></mujoco>'''
    (root/'model.xml').write_text(xml);m=mujoco.MjModel.from_xml_string(xml)
    geoms=[]
    for i in range(m.ngeom):
        b=int(m.geom_bodyid[i]);ancestors=[];cursor=b
        while cursor:ancestors.append(m.body(cursor).name);cursor=int(m.body_parentid[cursor])
        geoms.append({'geom_id':i,'geom_name':m.geom(i).name,'body_id':b,'body_name':m.body(b).name,
          'ancestors':ancestors,'type':int(m.geom_type[i]),'size':m.geom_size[i].tolist(),'contype':int(m.geom_contype[i]),'conaffinity':int(m.geom_conaffinity[i])})
    save(root/'physics/geometry.json',geoms)
    save(root/'physics/schema.json',{'object_body_ids':[],'object_body_names':[],
         'sensors':[{'name':'gripper0_force_ee','dimension':3,'address':0,'type':4}]})
    state=np.r_[0.,m.qpos0,np.zeros(m.nv)];np.save(root/'initial_state.npy',state);np.save(root/'final_state.npy',state)
    save(root/'model_resources.json',[]);save(root/'environment.json',{})
    result={'seed':1,'status':'MEASURED','program_sha256':'fixture','control_steps':20,'task_qualified':False,
        'program_ok':True,'native_task_success':False,'initial_state_id':fingerprint_identity(digest(root/'model.xml'),state)}
    save(root/'result.json',result)
    n=500;t=np.arange(n)*.002
    arrays={'time':np.c_[t,t+.002],'qpos_start':np.zeros((n,m.nq)),'qvel_start':np.zeros((n,m.nv)),
       'ctrl':np.zeros((n,m.nu)),'actuator_force':np.zeros((n,m.nu)),'sensordata_solve':np.zeros((n,3)),
       'object_pose_solve':np.zeros((n,0,7)),'contact_offsets':np.zeros(n+1,dtype=int),'contacts':np.zeros((0,21))}
    save(root/'physics/summary.json',{'steps':n,'coverage_complete':True,'chunks':[{'file':'chunk_000000.npz','steps':n,'contacts':0}]})
    item={'id':'synthetic/wipe','task':'wipe','chain':1,'version':1,'seed':1,'execution':str(root),
      'program_sha256':'fixture','result_sha256':digest(root/'result.json'),'model_sha256':digest(root/'model.xml'),
      'initial_state_id':result['initial_state_id'],'task_accepted':False,'task_qualified':False}
    base={**item,'audit_status':'PASS','events':[],'predicates':{'robot_fixture_contact':'PASS'},'unknown':[],
      'metrics':{'physics_steps':n},'coverage_complete':True,'phase_segments':[{'start_physics_index':0,'end_physics_index_exclusive':n,'phase':'wipe'}]}
    previous=tmp_path/'base.json';save(previous,base);item.update(base_audit_file=str(previous),base_audit_sha256=digest(previous))
    def finish():
        np.savez_compressed(root/'physics/chunk_000000.npz',**arrays)
        save(root/'artifacts.json',{str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file() and p.name!='artifacts.json'})
        return audit_episode(item,R)
    return arrays,finish


def test_native_wipe_explicit_force_limit_and_equality_using_full_reader(passive_fixture):
    arrays,finish=passive_fixture
    arrays['sensordata_solve'][100:150,0]=60
    clean=finish();assert clean['audit_status']=='PASS',clean['unknown']
    arrays['sensordata_solve'][100:150,0]=61
    r=finish();e=next(e for e in r['events'] if e['kind']=='native_wipe_eef_force_limit')
    assert e['physics_index']==100 and e['sensor_force_norm_n']==61
    assert r['audit_status']=='VIOLATION'
    assert r['unassessed']['external_moment_overload']=='UNKNOWN'


def test_native_control_clipping_is_separate_from_excess_actual_actuator_output(passive_fixture):
    arrays,finish=passive_fixture;arrays['ctrl'][100:150,0]=100
    r=finish();assert r['audit_status']=='PASS',r['unknown']
    assert r['extended_metrics']['actuator_saturation']['robot0_motor']['control_out_of_range_s']==.1
    arrays['actuator_force'][100:150,0]=81
    r=finish();assert r['predicates']['actuator_force_range:robot0_motor']=='VIOLATION'


def test_persistent_joint_operating_excursion_is_reconstructed(passive_fixture):
    arrays,finish=passive_fixture;arrays['qpos_start'][100:200,0]=1.1
    r=finish();assert r['audit_status']=='VIOLATION',r['unknown']
    assert r['predicates']['joint_range:robot0_joint1']=='VIOLATION'
    assert r['extended_metrics']['joint_bounds']['robot0_joint1']['longest_excursion_s']==.2


def test_invalid_contact_frame_is_unknown_before_any_force_verdict(passive_fixture):
    arrays,finish=passive_fixture
    arrays['contact_offsets'][101:]=1
    arrays['contacts']=contact(force=(61,0,0));arrays['contacts'][0,:2]=[0,1];arrays['contacts'][0,6:15]=0
    r=finish();assert r['audit_status']=='UNKNOWN' and 'invalid contact frame' in r['unknown'][0]
