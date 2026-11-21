"""Physical and computational acceptance of versioned grasp-loss rules."""
from pathlib import Path
import json
import numpy as np
import pytest

from esrb.measurement.rules import analyze
from esrb.measurement.reference import check
from esrb.measurement.spec import event_profile
from tests.core.test_measurement_validation import physical_trace


def physical_grasp_trace(folder,*,release=True,place=False):
    import mujoco
    from esrb.measurement.recording import Recorder
    model=mujoco.MjModel.from_xml_string('''<mujoco>
    <option timestep=".002" gravity="0 0 -9.81"/>
    <default><geom friction="2 .01 .001" solref=".005 1"/></default>
    <worldbody><geom name="floor" type="plane" size="1 1 .1" contype="1" conaffinity="2"/>
      <body name="piece" pos="0 0 .05"><freejoint/><geom name="part" type="box" size=".01 .01 .02" mass=".02" contype="2" conaffinity="5"/></body>
      <body name="left_finger" mocap="true" pos="-.024 0 .06">
        <geom name="left_pad" type="box" size=".015 .015 .012" contype="4" conaffinity="2"/>
        <geom name="left_lip" type="box" pos=".016 0 -.035" size=".009 .015 .005" contype="4" conaffinity="2"/>
      </body>
      <body name="right_finger" mocap="true" pos=".024 0 .06">
        <geom name="right_pad" type="box" size=".015 .015 .012" contype="4" conaffinity="2"/>
        <geom name="right_lip" type="box" pos="-.016 0 -.035" size=".009 .015 .005" contype="4" conaffinity="2"/>
      </body>
      <body name="hand" mocap="true" pos="0 0 .06"><site name="grip_origin"/></body>
    </worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    body=model.body('piece').id;groups=[[model.geom(side+'_'+kind).id for kind in ('pad','lip')] for side in ('left','right')];pads=sum(groups,[])
    roles=dict(scene='tool_handling',objects=[body],robot=pads,pads=pads,pad_groups=groups,
        tool=[],fixtures=[0],part_geoms={str(body):[model.geom('part').id]},initial_positions={str(body):[0.,0.,.05]},
        tcp_sites=[model.site('grip_origin').id],grasp_required=True,control_dt_s=.02,
        ground_support_geoms=[0],analytic_fixture=True)
    if place:roles['goals']={str(body):dict(position=[0.,0.,.021],tolerance_m=[.01,.01,.008],yaw_tolerance_rad=.12)}
    folder.mkdir();actions=[]
    # The only stimulus is explicit mocap commands to this small engineering
    # gripper; object qpos/qvel are never assigned after recording begins.
    end=740 if place else 420
    with Recorder(model,data,folder/'raw',roles) as recorder:
        for step in range(end):
            if step<50:z=.06
            elif step<250:z=.06+.10*(step-50)/200
            elif place and step<500:z=.16-.129*(step-250)/250
            elif place:z=.031
            else:z=.16
            opening=release and step>=(550 if place else 350)
            half_gap=.065 if opening else .024
            data.mocap_pos[:]=[[-half_gap,0.,z],[half_gap,0.,z],[0.,0.,z]]
            recorder.control=step//10;recorder.phase='grasp_fixture'
            if step%10==0:actions.append(dict(step=step//10,time_s=float(data.time),action=[]))
            mujoco.mj_step(model,data)
    (folder/'actions.json').write_text(json.dumps(actions))
    return folder


@pytest.mark.parametrize('release,place,expected',[(True,False,'VIOLATION'),(False,False,'PASS'),(True,True,'PASS')])
def test_physical_loss_vs_hold_vs_qualified_release(tmp_path,release,place,expected):
    folder=physical_grasp_trace(tmp_path/'trace',release=release,place=place)
    measured=analyze(folder)
    assert measured['primary_status']==expected,measured
    if release and not place:
        loss=next(e for e in measured['events'] if e['kind']=='grasp_loss')
        assert loss['duration_s']+1e-10>=.02
        assert not any(e['kind']=='drop' for e in measured['events'])
        assert 'NOT_OBSERVED_AFTER_CONFIRMED_LOSS' in measured['landing_outcomes'].values()
        assert not measured['missing']
        old=analyze(folder,profile=event_profile(5))
        assert old['primary_status']=='UNKNOWN'
    if place:assert any(e['kind']=='seated_release' for e in measured['events'])
    reference=check(folder,measured,max_samples=8)
    assert reference['status']=='PASS',reference
    assert all(c['status']=='PASS' for c in reference['comparisons'])


@pytest.mark.parametrize('offset,expected',[(-.002,'VIOLATION'),(0.,'VIOLATION'),(.002,'UNKNOWN')])
def test_sustained_loss_duration_boundary(tmp_path,offset,expected):
    from esrb.measurement.recording import chunks
    folder=physical_grasp_trace(tmp_path/'trace')
    base=analyze(folder);loss=next(e for e in base['events'] if e['kind']=='grasp_loss')
    final=next(reversed(list(chunks(folder/'raw'))))['time'][-1,0]
    profile=event_profile();profile['grasp_loss_confirmation_control_intervals']=(final-loss['loss_start_time_s']+offset)/.02
    measured=analyze(folder,profile=profile)
    assert measured['primary_status']==expected
    if expected=='UNKNOWN':assert measured['unknown_reasons'][0]['kind']=='UNRESOLVED_GRASP_LOSS'


@pytest.mark.parametrize('case,expected',[
    ('single_pad','PASS'),('early_regrasp','PASS'),('late_regrasp','VIOLATION'),
    ('qualified_release','PASS'),('receiver_grasp','PASS'),('no_prior_lift','PASS')])
def test_computational_release_and_recovery_cases(physical_trace,monkeypatch,case,expected):
    """Prescribed geometry/contact stream for temporal and phase boundaries."""
    from esrb.measurement import rules
    from esrb.measurement.recording import chunks
    batch=next(chunks(physical_trace/'raw'));geometry=json.loads((physical_trace/'raw/geometry.json').read_text())
    geometry['geoms'] += [dict(geometry['geoms'][1],id=i,name=f'pad{i}') for i in range(2,6)]
    roles=geometry['roles'];roles.update(scene='two-arm-handover' if case=='receiver_grasp' else 'tool_handling',
        robot=list(range(2,6)),pads=list(range(2,6)),pad_groups=[[i] for i in range(2,6)],
        tcp_sites=[0,1],grasp_required=True,control_dt_s=.02,ground_support_geoms=[0])
    n=100;batch={k:np.repeat(v[:1],n,axis=0) for k,v in batch.items() if k!='contacts'}
    batch['time']=np.c_[np.arange(n)*.002,(np.arange(n)+1)*.002]
    batch['end_qpos']=batch['start_qpos'].copy();batch['end_qvel']=batch['start_qvel'].copy()
    positions=np.full(n,.15);positions[0]=.05;positions[3:]=.10-np.arange(n-3)*.0005
    if case=='qualified_release':
        positions[3:]=.05;roles['goals']={'1':dict(position=[0.,0.,.05],tolerance_m=[.01,.01,.005])}
    if case=='no_prior_lift':roles['initial_positions']['1'][2]=.2
    batch['body_pos']=np.zeros((n,1,3));batch['body_pos'][:,0,2]=positions
    batch['site_pos']=np.tile([[[0.,0.,.15],[0.,0.,.15]]],(n,1,1));batch['site_pos'][0,:,2]=.05
    batch['site_rot']=np.tile(np.eye(3).reshape(1,1,9),(n,2,1))
    contacts=[];ranges=[]
    def contact(g):return np.r_[g,1,0.,[0.,0.,0.],np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]).reshape(-1),[1.,0.,0.,0.,0.,0.]]
    for i in range(n):
        start=len(contacts)
        if i<3 or case=='early_regrasp' and i>=8 or case=='late_regrasp' and i>=20:
            contacts += [contact(2),contact(3)]
        elif case=='single_pad':contacts.append(contact(2))
        elif case=='receiver_grasp':contacts += [contact(4),contact(5)]
        elif case=='qualified_release':contacts.append(contact(0))
        ranges.append([start,len(contacts)])
    batch['contacts']=np.asarray(contacts);batch['contact_range']=np.asarray(ranges)
    original=Path.read_text
    def read(path,*args,**kwargs):
        if path==physical_trace/'raw/geometry.json':return json.dumps(geometry)
        if path==physical_trace/'raw/manifest.json':
            m=json.loads(original(path));m.update(physics_steps=n,control_steps=1,site_indices=[0,1],chunks=[dict(file='computed',steps=n)])
            m.pop('physics_steps_per_control',None);return json.dumps(m)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read);monkeypatch.setattr(rules,'chunks',lambda folder:iter([batch]))
    measured=analyze(physical_trace)
    assert measured['primary_status']==expected,measured
    if case=='early_regrasp':
        assert any(e['kind']=='grasp_recovered' for e in measured['events'])
        assert not any(e['kind']=='grasp_loss' for e in measured['events'])
    if case=='late_regrasp':
        assert any(e['kind']=='grasp_loss' for e in measured['events'])
        assert any(e['kind']=='grasp_recovered' and e['confirmed_loss'] for e in measured['events'])
    if case in ('qualified_release','receiver_grasp'):
        assert any(e['kind']=='seated_release' for e in measured['events'])
