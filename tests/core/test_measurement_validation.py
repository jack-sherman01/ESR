"""Controlled evidence must not get a free PASS from a program name or missing data."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import pytest

from esrb.measurement.spec import SCENES,cases,protocol
from esrb.measurement.reporting import legacy_metrics


def episode(unsafe=False,score=1.):
    return dict(task_success=True,task_score=score,measurement=dict(primary_status='VIOLATION' if unsafe else 'PASS',
        signals={'drop_height_loss_m':.1 if unsafe else 0.}))


def test_legacy_fixed_input_parity():
    from esrb.benchmark.lineage import GenerationRecord
    from esrb.metrics.core import (safety_regression_rate,unsafe_false_acceptance_rate,
        detection_recall_precision,regression_discovery_cost)
    groups=[];records=[]
    for g in range(3):
        groups.append(dict(generation=g,parent_sha256='p',accepted=g==0,
            parents=[episode(g>0)]*5,children=[episode(True)]*5))
        records.append(GenerationRecord(g,'p','','','controlled',1.,g>0,{'primary_signal':.1 if g else 0.},timestamp=0.))
    result=legacy_metrics(groups,12.)
    assert result['srr']==safety_regression_rate(records)==.5
    assert result['ufar']==unsafe_false_acceptance_rate([True]*3,[True,False,False])
    expected=detection_recall_precision([False]*5+[True]*5+[False]*5+[True]*5+[False]*5+[True]*5,
        [False]*5+[True]*25)
    assert {k:result[k] for k in ('recall','precision','f1')}==asdict(expected)
    expected_cost=regression_discovery_cost(records,5,4.)
    assert result['rdc_rollouts']==expected_cost.n_rollouts==10
    assert result['rdc_wall_s']==expected_cost.wall_clock_s==8.


def test_five_safe_trials_still_rejected():
    from esrb.detection.risk_selection import CandidateEvaluation,RiskSensitiveSelector,ucb_safety_risk
    evaluation=CandidateEvaluation(None,[1.]*5,[False]*5,0)
    assert .24<ucb_safety_risk(evaluation,.1)<.25
    assert RiskSensitiveSelector(.09,.10).select([evaluation]) is None


def test_plan_budget_and_scope():
    assert len(SCENES)==12 and len(set(SCENES))==12
    assert 'grasp_pressure' not in SCENES
    assert 12*(3*2*5+2*12)==648
    assert all(cases(s).count('primary')==1 and cases(s).count('baseline')==1 for s in SCENES)
    assert protocol(SCENES,8)['calibration_seeds']==[42,43,44]


@pytest.fixture
def physical_trace(tmp_path):
    mujoco=pytest.importorskip('mujoco');np=pytest.importorskip('numpy')
    from esrb.measurement.recording import Recorder
    folder=tmp_path/'episode';folder.mkdir()
    model=mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".002"/><worldbody>
    <geom name="floor" type="plane" size="1 1 .1"/><body name="piece" pos="0 0 .05">
    <freejoint/><geom name="piece_collision" type="box" size=".05 .05 .05" mass="1"/>
    </body></worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    roles=dict(scene='constrained_push',robot=[],pads=[],pad_groups=[],tool=[],objects=[1],
        part_geoms={'1':[1]},fixtures=[0],initial_positions={'1':[0.,0.,.05]},grasp_required=False,control_dt_s=.12,analytic_fixture=True)
    with Recorder(model,data,folder/'raw',roles,chunk_size=13) as rec:
        rec.control=0;rec.phase='support'
        for _ in range(60):mujoco.mj_step(model,data)
    (folder/'actions.json').write_text(json.dumps([dict(step=0,forward_intent=False,time_s=0.,action=[])]))
    return folder


def test_physics_time_and_force_replay(physical_trace):
    from esrb.measurement.rules import analyze
    from esrb.measurement.reference import check
    from esrb.measurement.recording import chunks
    measurement=analyze(physical_trace)
    assert measurement['primary_status']=='PASS'
    assert measurement['physics_steps']==60
    assert check(physical_trace,measurement,max_samples=5)['status']=='PASS'
    batches=list(chunks(physical_trace/'raw'))
    assert len(batches)==5
    assert batches[0]['time'][0].tolist()==[0.,.002]


def test_unavailable_load_limit_is_not_all_rules_safe(physical_trace):
    from esrb.measurement.rules import analyze
    measured=analyze(physical_trace,dict(status='INCOMPLETE',interfaces={}))
    assert measured['primary_status']=='PASS'
    assert measured['status']=='UNKNOWN'
    assert measured['unqualified_load_channels']


@pytest.mark.parametrize('shared_body',[False,True])
def test_opposing_finger_loads_do_not_cancel(tmp_path,shared_body):
    mujoco=pytest.importorskip('mujoco');np=pytest.importorskip('numpy')
    from esrb.measurement.recording import Recorder,chunks
    from esrb.measurement.rules import analyze
    from esrb.measurement.reference import check
    folder=tmp_path/'pinch';folder.mkdir()
    import xml.etree.ElementTree as ET
    xml=ET.Element('mujoco');ET.SubElement(xml,'option',timestep='.002');world=ET.SubElement(xml,'worldbody')
    ET.SubElement(world,'geom',name='floor',type='plane',size='1 1 .1')
    fixture=ET.SubElement(world,'body',name='fixture') if shared_body else None
    for side,x in (('left',-.095),('right',.095)):
        owner=fixture if shared_body else ET.SubElement(world,'body',name=side)
        ET.SubElement(owner,'geom',name=side+'_pad',type='box',size='.05 .1 .1',pos=f'{x} 0 .05')
    piece=ET.SubElement(world,'body',name='piece',pos='0 0 .05');ET.SubElement(piece,'freejoint')
    ET.SubElement(piece,'geom',name='part',type='box',size='.05 .05 .05',mass='1')
    model=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    left=model.geom('left_pad').id;right=model.geom('right_pad').id
    part=model.geom('part').id;body=model.body('piece').id
    roles=dict(scene='constrained_push',robot=[left,right],pads=[left,right],pad_groups=[[left],[right]],
        tool=[],objects=[body],part_geoms={str(body):[part]},fixtures=[0],ground_support_geoms=[0],
        initial_positions={str(body):[0.,0.,.05]},grasp_required=False,control_dt_s=.12,analytic_fixture=True)
    if shared_body:roles.update(robot=[],pads=[],pad_groups=[],fixtures=[0,left,right])
    with Recorder(model,data,folder/'raw',roles) as rec:
        rec.control=0;rec.phase='pinch'
        for _ in range(60):mujoco.mj_step(model,data)
    (folder/'actions.json').write_text(json.dumps([dict(step=0,time_s=0.,action=[],forward_intent=False)]))
    measured=analyze(folder)
    name=f':object_fixture@body{model.body("fixture").id}:' if shared_body else ':grasp@'
    grip=[v['peak_force_n'] for k,v in measured['signals']['interfaces'].items() if name in k]
    assert len(grip)==(1 if shared_body else 2) and min(grip)>1.
    # A symmetric pinch has nearly zero net horizontal force despite both
    # nonzero interface loads; neither source can disappear through summation.
    assert abs(float(data.qvel[0]))<1e-6
    assert check(folder,measured,max_samples=5)['status']=='PASS'
    if shared_body:
        key=next(k for k in measured['signals']['interfaces'] if name in k)
        signal=measured['signals']['interfaces'][key]
        assert signal['peak_force_n']>signal['net_peak_force_n']+1.
        bound=(signal['peak_force_n']+signal['net_peak_force_n'])/2
        limits=dict(interfaces={key:dict(peak_force_n=bound)})
        detected=analyze(folder,limits)
        assert any(e['kind']=='overload' for e in detected['events'])
        assert check(folder,detected,max_samples=5,limits=limits)['status']=='PASS'
    else:
        from copy import deepcopy
        keys=[k for k in measured['signals']['interfaces'] if ':grasp@' in k]
        limits=dict(interfaces={key:dict(peak_force_n=measured['signals']['interfaces'][key]['peak_force_n']/2) for key in keys})
        complete=analyze(folder,limits)
        assert check(folder,complete,max_samples=5,limits=limits)['status']=='PASS'
        incomplete=deepcopy(complete)
        incomplete['events']=[e for e in complete['events'] if not (e['kind']=='overload' and e.get('interface')==keys[-1])]
        audit=check(folder,incomplete,max_samples=5,limits=limits)
        assert audit['status']=='FAIL'
        assert next(c for c in audit['comparisons'] if c['rule']=='overload')['false_negative']


def test_one_coordinator_per_frozen_run(tmp_path):
    from experiments.studies.measurement_validation import execution_lock
    with execution_lock(tmp_path):
        with pytest.raises(RuntimeError,match='another coordinator'):
            with execution_lock(tmp_path):pytest.fail('duplicate writer acquired the lock')
    with execution_lock(tmp_path):pass


def test_known_load_crossing_survives_unrelated_incomplete_channels():
    from esrb.measurement.spec import case_status
    partial=dict(status='INCOMPLETE')
    measured=dict(status='VIOLATION',missing=[],events=[dict(kind='overload',limit=2.,value=3.)])
    assert case_status('stack','overload',measured,partial)=='TRIGGERED'
    measured['events']=[]
    assert case_status('stack','overload',measured,partial)=='UNKNOWN'
    assert case_status('stack','primary',measured,partial)=='NOT_TRIGGERED'


def test_qualification_worker_creates_output_directory(tmp_path,monkeypatch):
    pytest.importorskip('mujoco')
    from esrb.measurement import qualification
    from esrb.measurement.episodes import worker
    monkeypatch.setattr(qualification,'qualify',lambda:dict(status='PASS',scope='worker output contract fixture'))
    output=tmp_path/'new-calibration'/'qualification.json'
    request=tmp_path/'request.json';request.write_text(json.dumps(dict(kind='qualification',output=str(output))))
    worker(request)
    assert json.loads(output.read_text())['status']=='PASS'
    assert not output.with_suffix('.json.tmp').exists()


def test_signed_command_uses_applied_motion_not_rotation_or_requested_target():
    mujoco=pytest.importorskip('mujoco');np=pytest.importorskip('numpy')
    from esrb.measurement.commands import CommandProjection
    import xml.etree.ElementTree as ET
    xml=ET.Element('mujoco');world=ET.SubElement(xml,'worldbody');parent=world
    for i in range(1,8):
        body=ET.SubElement(parent,'body',name=f'link{i}')
        ET.SubElement(body,'joint',name=f'joint{i}',type='slide' if i in (1,3) else 'hinge',
                      axis='1 0 0' if i==1 else '0 1 0' if i==3 else '0 0 1')
        ET.SubElement(body,'geom',type='sphere',size='.01',mass='1',contype='0',conaffinity='0')
        parent=body
    ET.SubElement(parent,'site',name='grip_origin',size='.001')
    model=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
    reader=CommandProjection(model,'constrained_push',dict(goal_bodies={}))
    initial=np.zeros(7);control=np.zeros(8);requested=np.ones(8)
    control[0]=.001
    assert reader.project('push',initial,control,requested)==pytest.approx(.001)
    control[0]=-.001
    assert reader.project('push',initial,control,requested)==pytest.approx(-.001)
    control[:]=0.;control[1]=.2
    assert reader.project('push',initial,control,requested)==pytest.approx(0.)
    assert reader.project('lower',initial,control,requested)==0.
    control[:]=0.;control[2]=.001
    y_reader=CommandProjection(model,'force_alignment',dict(goal_bodies={}))
    assert y_reader.project('insert',initial,control,requested)==pytest.approx(.001)
    assert reader.project('push',initial,control,requested)==pytest.approx(0.)
    assert np.array_equal(initial,np.zeros(7))


def test_native_insertion_distinguishes_backoff_and_lateral_alignment():
    pytest.importorskip('mujoco')
    from esrb.measurement.commands import CommandProjection
    reader=CommandProjection(None,'nut-assembly',{},'OSC_POSE')
    assert reader.project('lower',None,None,[0.,0.,-.2,0.,0.,0.,1.])==pytest.approx(.01)
    assert reader.project('lower',None,None,[0.,0.,.2,0.,0.,0.,1.])==pytest.approx(-.01)
    assert reader.project('lower',None,None,[.2,0.,0.,0.,0.,0.,1.])==0.
    unsupported=CommandProjection(None,'nut-assembly',{},'UNKNOWN')
    with pytest.raises(ValueError,match='unverified'):unsupported.project('lower',None,None,[0.]*7)


def test_reference_rejects_wrong_finger_registration_without_contact(tmp_path):
    mujoco=pytest.importorskip('mujoco');np=pytest.importorskip('numpy')
    from esrb.measurement.recording import Recorder,sha
    from esrb.measurement.rules import analyze
    from esrb.measurement.reference import check
    folder=tmp_path/'identity';folder.mkdir()
    model=mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".002"/><worldbody>
    <geom name="floor" type="plane" size="2 2 .1"/>
    <body name="robot" pos="1 0 .1"><geom name="panda_base" type="sphere" size=".01"/><site name="grip_origin" size=".001"/></body>
    <body name="left_finger" pos="1 .1 .1"><geom name="left_pad" type="sphere" size=".01"/></body>
    <body name="right_finger" pos="1 -.1 .1"><geom name="right_pad" type="sphere" size=".01"/></body>
    <body name="tool" pos="0 0 .05"><freejoint/><geom name="part" type="box" size=".05 .05 .05" mass="1"/></body>
    </worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    a=model.geom('left_pad').id;b=model.geom('right_pad').id;body=model.body('tool').id
    roles=dict(scene='tool_handling',robot=[model.geom('panda_base').id,a,b],pads=[a,b],pad_groups=[[a],[b]],
        tool=[],objects=[body],part_geoms={str(body):[model.geom('part').id]},fixtures=[0],ground_support_geoms=[0],
        initial_positions={str(body):[0.,0.,.05]},grasp_required=True,control_dt_s=.12,tcp_sites=[model.site('grip_origin').id])
    with Recorder(model,data,folder/'raw',roles) as rec:
        rec.control=0;rec.phase='support'
        for _ in range(60):mujoco.mj_step(model,data)
    (folder/'actions.json').write_text(json.dumps([dict(step=0,time_s=0.,action=[])]))
    assert check(folder,analyze(folder),max_samples=3)['status']=='PASS'
    # A hash-consistent registration typo must fail even without any finger
    # touching the object; replaying contact forces alone cannot find it.
    path=folder/'raw/geometry.json';geometry=json.loads(path.read_text())
    geometry['roles']['pad_groups']=[[a,b],[]];path.write_text(json.dumps(geometry))
    manifest=folder/'raw/manifest.json';value=json.loads(manifest.read_text())
    value['geometry_sha256']=sha(path);manifest.write_text(json.dumps(value))
    result=check(folder,analyze(folder),max_samples=3)
    assert result['status']!='PASS' and 'finger registration' in result['error']


def test_legal_hold_needs_real_loaded_unseated_duration(physical_trace):
    from esrb.measurement.reporting import holding_evidence
    from esrb.measurement.recording import sha
    from esrb.measurement.rules import analyze
    root=physical_trace.parent
    row=dict(status='RECORDED',file=physical_trace.name,measurement=analyze(physical_trace))
    assert holding_evidence(root,row,.1)['status']=='NOT_EXERCISED'

    geometry=physical_trace/'raw/geometry.json';g=json.loads(geometry.read_text());body=g['roles']['objects'][0]
    g['roles']['goals']={str(body):dict(position=[.5,0.,.05],tolerance_m=[.01,.01,.01])}
    geometry.write_text(json.dumps(g));manifest=physical_trace/'raw/manifest.json';m=json.loads(manifest.read_text())
    m['geometry_sha256']=sha(geometry);manifest.write_text(json.dumps(m))
    actions=physical_trace/'actions.json';a=json.loads(actions.read_text());a[0]['holding']=True;actions.write_text(json.dumps(a))
    row['measurement']=analyze(physical_trace)
    assert holding_evidence(root,row,.1)['status']=='EXERCISED'
    row['measurement']['command_projections']['0']['projection_m']=.001
    assert holding_evidence(root,row,.1)['status']=='NOT_EXERCISED'


def test_legal_hold_serializes_numpy_boolean_without_losing_actions(tmp_path,monkeypatch):
    mujoco=pytest.importorskip('mujoco');np=pytest.importorskip('numpy')
    from types import SimpleNamespace
    from esrb.measurement import episodes
    model=mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".002"/><worldbody>
    <geom type="plane" size="1 1 .1"/><body name="piece" pos="0 0 .05"><freejoint/>
    <geom type="box" size=".05 .05 .05" mass="1"/></body></worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    observation=SimpleNamespace(info={'stage':'approach'})
    class Task:
        MAX_STEPS=1;stage='approach';_part=1;_model=model;_data=data
        robot=SimpleNamespace(q=np.arange(7))
        def reset(self,seed):return observation
        def step(self,action):
            for _ in range(10):mujoco.mj_step(model,data)
            return observation,0.,True,{}
        def evaluate(self,trace):return SimpleNamespace(success=False,task_score=0.)
        def close(self):pass
    policy=SimpleNamespace(act=lambda obs:np.zeros(8),state_dict=lambda:{})
    monkeypatch.setattr(episodes,'selfbuilt',lambda scene:(Task(),policy))
    roles=dict(scene='compliant_assembly',robot=[],pads=[],pad_groups=[],tool=[],objects=[1],part_geoms={'1':[1]},
        fixtures=[0],initial_positions={'1':[0.,0.,.05]},grasp_required=False,control_dt_s=.02)
    monkeypatch.setattr(episodes,'roles_for',lambda *args:roles)
    folder=tmp_path/'boolean-regression'
    result=episodes.run(dict(folder=str(folder),scene='compliant_assembly',case='legal_hold',seed=42,
        program={'scope':'serialization contract fixture'},program_sha256='fixture'))
    assert result['status']=='RECORDED',result.get('error')
    actions=json.loads((folder/'actions.json').read_text())
    assert len(actions)==1 and actions[0]['holding'] is False


@pytest.mark.parametrize('outcome,expected',[
    ('single_pad','PASS'),('regrasp','PASS'),('impact','VIOLATION'),('airborne','UNKNOWN')])
def test_drop_requires_total_grasp_loss_and_resolved_outcome(physical_trace,monkeypatch,outcome,expected):
    """Computational event fixture: a sliding finger contact is not a dropped part.

    The input positions/contacts are prescribed, not a new physical rollout or
    an independent solver check. Original physical fixture files stay intact.
    """
    import numpy as np
    from esrb.measurement import rules
    from esrb.measurement.recording import chunks
    batch=next(chunks(physical_trace/'raw'))
    geometry=json.loads((physical_trace/'raw/geometry.json').read_text())
    geometry['geoms'] += [dict(geometry['geoms'][1],id=i,name=f'pad{i}') for i in (2,3)]
    roles=geometry['roles'];roles.update(scene='tool_handling',robot=[2,3],pads=[2,3],
        pad_groups=[[2],[3]],tcp_sites=[0],grasp_required=True,ground_support_geoms=[0])
    n=60
    batch={key:np.repeat(value[:1],n,axis=0) for key,value in batch.items() if key!='contacts'}
    batch['end_qpos']=batch['start_qpos'].copy();batch['end_qvel']=batch['start_qvel'].copy()
    batch['time']=np.c_[np.arange(n)*.002,(np.arange(n)+1)*.002]
    batch['body_pos']=np.tile([[[0.,0.,.10]]],(n,1,1));batch['body_pos'][0,0,2]=.05
    batch['body_pos'][3:,0,2]=.05
    batch['site_pos']=np.tile([[[0.,0.,.10]]],(n,1,1));batch['site_pos'][0,0,2]=.05
    batch['site_rot']=np.tile(np.eye(3).reshape(1,1,9),(n,1,1))
    contacts=[];ranges=[]
    def contact(g1,g2):
        basis=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        return np.r_[g1,g2,0.,[0.,0.,0.],basis.reshape(-1),[1.,0.,0.,0.,0.,0.]]
    for i in range(n):
        start=len(contacts)
        if i<3 or outcome=='regrasp' and i==4:contacts.extend([contact(2,1),contact(3,1)])
        elif outcome=='single_pad':contacts.append(contact(2,1))
        if i>=5 and outcome!='airborne':contacts.append(contact(0,1))
        ranges.append([start,len(contacts)])
    batch['contacts']=np.array(contacts);batch['contact_range']=np.array(ranges)
    original_read=Path.read_text
    def read(path,*args,**kwargs):
        if path==physical_trace/'raw/geometry.json':return json.dumps(geometry)
        if path==physical_trace/'raw/manifest.json':
            manifest=json.loads(original_read(path));manifest['site_indices']=[0];return json.dumps(manifest)
        return original_read(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    monkeypatch.setattr(rules,'chunks',lambda folder:iter([batch]))
    measured=rules.analyze(physical_trace)
    assert measured['primary_status']==expected, measured['missing']
    if outcome=='impact':
        event=next(e for e in measured['events'] if e['kind']=='drop')
        assert event['physics_step']==5 and event['loss_physics_step']==3
    if outcome=='airborne':assert any('unconfirmed drop' in issue for issue in measured['missing'])


def test_corrupt_copies_unknown_preserve_source(physical_trace,tmp_path):
    from esrb.measurement.fixtures import validate_copies
    from esrb.measurement.recording import sha
    from esrb.measurement.rules import analyze
    before=sha(physical_trace/'raw/manifest.json')
    results=validate_copies(tmp_path,[dict(scene='constrained_push',case='baseline',seed=0,status='RECORDED',file='episode')])
    assert len(results)==9
    assert all(r['status']=='PASS' for r in results)
    assert sha(physical_trace/'raw/manifest.json')==before
    assert analyze(physical_trace)['primary_status']=='PASS'


def test_real_contact_peak_boundary_is_inclusive(physical_trace):
    from esrb.measurement.rules import analyze
    normal=analyze(physical_trace)
    key=next(iter(normal['signals']['interfaces']))
    peak=normal['signals']['interfaces'][key]['peak_force_n']
    assert peak>9.
    for multiplier,violates in ((1.-1e-7,True),(1.,False),(1.+1e-7,False)):
        measured=analyze(physical_trace,{'interfaces':{key:{'peak_force_n':peak*multiplier}}})
        assert any(e['kind']=='overload' for e in measured['events'])==violates


def test_valid_but_wrong_component_fails_reconstruction(physical_trace,tmp_path):
    import numpy as np
    from esrb.measurement.fixtures import _copy_first,_write
    from esrb.measurement.rules import analyze
    from esrb.measurement.reference import check
    target=tmp_path/'wrong';batch,manifest=_copy_first(physical_trace,target)
    batch['contacts'][0,1]=0
    _write(target,batch,manifest)
    actual=analyze(target)
    assert check(target,actual)['status']!='PASS'


def test_analytic_load_and_moment_reference():
    pytest.importorskip('mujoco')
    from esrb.measurement.qualification import qualify
    assert qualify()['status']=='PASS'


def test_new_help_has_no_heavy_imports_and_cross_directory(tmp_path):
    root=Path(__file__).resolve().parents[2]
    import os
    code="""import sys
from esrb.cli.main import main
try:main(['benchmark','study','measurement-validation','--help'])
except SystemExit as e:assert e.code==0
assert not ({'numpy','mujoco','torch','cap_harness'} & sys.modules.keys())
"""
    result=subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env={**os.environ,'PYTHONPATH':str(root)},capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_report_never_starts_an_episode(tmp_path,monkeypatch):
    from esrb.measurement import reporting
    (tmp_path/'protocol.json').write_text(json.dumps(protocol(['stack'],1)))
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:pytest.fail('report started an external job'))
    monkeypatch.setattr(reporting,'plot',lambda *a,**k:None)
    result=reporting.report(tmp_path)
    assert result['summary']['accounting']['basic']==0
    assert result['scenes'][0]['status']=='PENDING'
    assert result['scenes'][0]['metrics'] is None


def test_report_cli_uses_read_only_reporter_without_frozen_execution(tmp_path,monkeypatch):
    from argparse import Namespace
    from experiments.studies import measurement_validation as study
    from esrb.measurement import reporting
    monkeypatch.setattr(study,'Study',lambda args:Namespace(root=tmp_path,protocol={'source_root':str(tmp_path/'snapshot')}))
    monkeypatch.setattr(subprocess,'call',lambda *a,**k:pytest.fail('report delegated to execution'))
    seen=[]
    monkeypatch.setattr(reporting,'report',lambda root:(seen.append(root) or dict(summary={'status':'PASS'})))
    assert study.main(Namespace(phase='report'))==0
    assert seen==[tmp_path]


@pytest.mark.parametrize('mismatch',[False,True])
def test_scene_gate_pairs_seeds_binds_programs_and_retains_parent(tmp_path,mismatch):
    from argparse import Namespace
    from experiments.studies.measurement_validation import Study,digest,save
    from esrb.measurement.reporting import gate_audit
    study=Study.__new__(Study);study.root=tmp_path;study.scenes=['stack'];study.workers=1
    study.protocol={'source_sha256':'frozen'};study._verify_limits=lambda:None
    save(tmp_path/'calibration/complete.json',{})
    calls=[]
    def substitute(phase,scene,case,seed,tag,program,reference):
        calls.append((case,seed,tag,reference))
        r=episode(False)
        r.update(file=f'gate/{scene}/{tag}-seed{seed}',status='RECORDED',scene=scene,
                 program_sha256=digest(program),initial_state_sha256=f'initial-{seed}',model_sha256='m',coordinator_wall_s=1.)
        if mismatch and case=='primary' and seed==42:r['initial_state_sha256']='different-actual-initial-state'
        save(tmp_path/r['file']/'result.json',r)
        return r
    study.episode=substitute;study.gate()
    assert len(calls)==30
    assert [s for _,s,_,_ in calls[:10]]==[42,43,44,45,46]*2
    assert [s for _,s,_,_ in calls[10:20]]==[1042,1043,1044,1045,1046]*2
    complete=json.loads((tmp_path/'gate/stack/complete.json').read_text())
    assert all(not g['accepted'] and g['parent_preserved'] for g in complete['generations'])
    assert len({g['active_sha256'] for g in complete['generations']})==1
    assert complete['elapsed_s']==30.
    audit=gate_audit(tmp_path,complete,verify_evidence=False)[0]
    assert audit['status']=='PASS'
    assert audit['eligible_for_metrics'] is not mismatch
    assert bool(audit['protocol_issues']) is mismatch
    if mismatch:assert complete['generations'][0]['decision']=='UNKNOWN'


def test_changed_source_blocks_before_simulation(tmp_path):
    from esrb.measurement.episodes import worker
    from experiments.studies.measurement_validation import digest
    source=tmp_path/'source';source.mkdir();(source/'actor.py').write_text('changed')
    files={'actor.py':'wrong'};manifest=tmp_path/'source_hashes.json';manifest.write_text(json.dumps(files))
    folder=tmp_path/'run';request=tmp_path/'request.json'
    request.write_text(json.dumps(dict(scene='stack',case='baseline',seed=0,folder=str(folder),
        frozen_source_manifest=str(manifest),frozen_source_sha256=digest(files),project_root=str(source))))
    worker(request)
    result=json.loads((folder/'result.json').read_text())
    assert result['status']=='BLOCKED'
    assert not (folder/'raw').exists()


def test_gate_runs_scenes_concurrently_and_keeps_generation_order(tmp_path):
    import threading
    from experiments.studies.measurement_validation import Study,digest,save
    study=Study.__new__(Study);study.root=tmp_path;study.scenes=['stack','wipe','nut-assembly'];study.workers=2
    study.protocol={'source_sha256':'frozen'};study._verify_limits=lambda:None
    save(tmp_path/'calibration/complete.json',{})
    rendezvous=threading.Barrier(2);lock=threading.Lock();active=set();peak=0;calls=[]
    def substitute(phase,scene,case,seed,tag,program,reference):
        nonlocal peak
        with lock:
            assert scene not in active
            active.add(scene);peak=max(peak,len(active));calls.append((scene,seed,tag))
        try:
            if tag=='g0-parent' and seed==42 and scene in ('stack','wipe'):
                rendezvous.wait(timeout=10)
            gen=(seed-42)//1000
            if gen:assert (tmp_path/f'gate/{scene}/generation-{gen-1}.json').is_file()
            row=episode(False)
            row.update(file=f'gate/{scene}/{tag}-seed{seed}',status='RECORDED',scene=scene,
                program_sha256=digest(program),initial_state_sha256=f'initial-{seed}',
                model_sha256='model',coordinator_wall_s=1.)
            save(tmp_path/row['file']/'result.json',row)
            return row
        finally:
            with lock:active.remove(scene)
    study.episode=substitute;study.gate()
    assert peak==2 and len(calls)==90
    for scene in study.scenes:
        assert [(seed,tag) for name,seed,tag in calls if name==scene]==[
            (42+1000*gen+i,f'g{gen}-{role}') for gen in range(3) for role in ('parent','child') for i in range(5)]
    execution=json.loads((tmp_path/'gate/execution.json').read_text())
    assert execution['workers']==2 and execution['episode_wall_sum_s']==90.
    before={str(p):p.read_bytes() for p in (tmp_path/'gate').glob('*/generation-*.json')}
    study.episode=lambda *a,**k:pytest.fail('completed scene was rerun')
    study.gate()
    assert all(Path(p).read_bytes()==value for p,value in before.items())


def test_reprocessing_keeps_real_violations_and_structural_unknown(tmp_path):
    from experiments.studies.measurement_reprocessing import repair_kind
    from esrb.measurement.reference import check
    assert repair_kind(dict(status='RECORDED',measurement={'primary_status':'VIOLATION'}))=='reuse'
    assert repair_kind(dict(status='BLOCKED',error='less than 5 GiB free; no rollout started'))=='rerun_not_started'
    assert repair_kind(dict(status='BLOCKED',case='legal_hold',error='Object of type bool is not JSON serializable'))=='rerun_serialization'
    with pytest.raises(ValueError,match='investigation'):repair_kind(dict(status='BLOCKED',error='unexplained crash'))
    assert repair_kind(dict(status='RECORDED',reference={'status':'UNKNOWN'}))=='reference_only'
    reference=check(tmp_path,dict(missing=['physics duration mismatch'],signals={}))
    assert reference['status']=='UNKNOWN' and 'physics duration mismatch' in reference['error']


def test_repaired_capsule_preserves_exact_compiled_asset_hash(tmp_path):
    import hashlib
    import mujoco
    from esrb.measurement.physical_sources import bind
    old=tmp_path/'original';new=tmp_path/'repair';old.mkdir();new.mkdir()
    vertices='v 0 0 0\nv .1 0 0\nv 0 .1 0\nv 0 0 .1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n'
    for root in (old,new):(root/'part.obj').write_text(vertices)
    manifest=tmp_path/'hashes.json';manifest.write_text(json.dumps({'part.obj':hashlib.sha256((old/'part.obj').read_bytes()).hexdigest()}))
    def xml(root):return f'<mujoco><asset><mesh name="part" file="{root}/part.obj"/></asset><worldbody><body><freejoint/><geom type="mesh" mesh="part"/></body></worldbody></mujoco>'
    original_factory=mujoco.MjModel.from_xml_string
    expected=mujoco.MjModel.from_xml_string(xml(old));different=mujoco.MjModel.from_xml_string(xml(new))
    assert expected.paths!=different.paths
    try:
        bind(new,old,manifest)
        actual=mujoco.MjModel.from_xml_string(xml(new))
        a=tmp_path/'a.mjb';b=tmp_path/'b.mjb'
        mujoco.mj_saveModel(expected,str(a));mujoco.mj_saveModel(actual,str(b))
        assert a.read_bytes()==b.read_bytes()
    finally:mujoco.MjModel.from_xml_string=staticmethod(original_factory)


def test_reuse_repair_requires_exact_raw_and_request_binding(physical_trace,tmp_path):
    import shutil
    from experiments.studies.measurement_validation import save,digest
    from experiments.studies.measurement_reprocessing import reusable_hold_repair,sha
    origin=tmp_path/'old';cache=tmp_path/'cached';relative=Path('validation/constrained_push/legal_hold-seed0000')
    for root in (origin,cache):
        (root/relative).parent.mkdir(parents=True);shutil.copytree(physical_trace,root/relative)
    limits={'frozen':True};program={'case':'legal_hold'};request={'program':program,'limits':limits}
    save(origin/'calibration/constrained_push/limits.json',limits)
    save(cache/'requests'/relative.with_suffix('.json'),request)
    row=dict(status='RECORDED',case='legal_hold',scene='constrained_push',file=str(relative),
        reference={'status':'PASS'},raw_manifest_sha256=sha(cache/relative/'raw/manifest.json'),
        program_sha256=digest(program),request_sha256=digest(request),limits_sha256=digest(limits))
    save(cache/relative/'result.json',row)
    assert reusable_hold_repair(cache,origin,relative)
    partial=dict(row);partial.pop('file');save(cache/relative/'result.json',partial)
    assert not reusable_hold_repair(cache,origin,relative)
    save(cache/relative/'result.json',row)
    request['program']={'case':'altered'};save(cache/'requests'/relative.with_suffix('.json'),request)
    with pytest.raises(ValueError,match='request binding'):reusable_hold_repair(cache,origin,relative)


def test_reprocessing_new_capsule_reuses_evidence_without_mutating_original(tmp_path):
    from argparse import Namespace
    from experiments.studies.measurement_validation import save,digest
    from experiments.studies.measurement_reprocessing import prepare,sha
    old=tmp_path/'original';source=old/'source';scene='long_horizon_assembly'
    actor=source/'esrb/measurement/episodes.py';actor.parent.mkdir(parents=True)
    actor.write_text("# Frozen task implementation\nholding=case=='legal_hold' and lifted and transport and hold_ticks<60\n")
    hashes={'esrb/measurement/episodes.py':sha(actor)};save(old/'source_hashes.json',hashes)
    save(old/'protocol.json',dict(source_root=str(source),source_sha256=digest(hashes),scenes=[scene]))
    save(old/'gate'/scene/'complete.json',dict(generations=[dict(accepted=False)]))
    save(old/'validation/complete.json',dict(replay=[],fixtures=[]))
    save(old/'calibration'/scene/'limits.json',dict(frozen=True,interfaces={}))
    for case,status in [('baseline','RECORDED'),('legal_hold','BLOCKED')]:
        relative=Path('validation')/scene/(case+'-seed0000')
        save(old/relative/'result.json',dict(status=status,scene=scene,case=case,seed=0,
             error='Object of type bool is not JSON serializable' if status=='BLOCKED' else None,
             measurement={'primary_status':'VIOLATION'}))
        save(old/'requests'/relative.with_suffix('.json'),dict(program={'source_sha256':digest(hashes)}))
        raw=old/relative/'raw';raw.mkdir();(raw/'immutable.bin').write_bytes(b'evidence')
    baseline=old/'validation'/scene/'baseline-seed0000/result.json';before=baseline.read_bytes()
    target=tmp_path/'new-revision'
    prepare(Namespace(from_run=old,out_dir=target,workers=2,scenes=None))
    plan=json.loads((target/'repair_plan.json').read_text())
    assert plan['physical_attempts_expected']==1 and plan['counts']=={'reuse':1,'rerun_serialization':1}
    assert baseline.read_bytes()==before and actor.read_text().startswith('# Frozen task implementation')
    assert 'holding=bool(' not in actor.read_text()
    assert 'holding=bool(' in (target/'source/esrb/measurement/episodes.py').read_text()
    assert not (target/'validation'/scene/'legal_hold-seed0000').exists()
    assert (target/'validation'/scene/'baseline-seed0000/raw/immutable.bin').read_bytes()==b'evidence'
    # Derived reports/results use atomic replacement even when evidence is linked.
    save(target/'validation'/scene/'baseline-seed0000/result.json',{'new':'receipt'})
    assert baseline.read_bytes()==before


def test_frozen_interpreter_preserves_virtualenv_symlink(tmp_path,monkeypatch):
    from argparse import Namespace
    from experiments.studies import measurement_validation as study
    repo=tmp_path/'repo';repo.mkdir();(repo/'pyproject.toml').write_text('[project]\nname="fixture"\n')
    virtualenv=repo/'.venv/bin';virtualenv.mkdir(parents=True)
    python=virtualenv/'python';python.symlink_to(sys.executable)
    monkeypatch.setattr(study,'project_root',lambda:repo)
    args=Namespace(out_dir=tmp_path/'run',phase='calibrate',workers=1,scenes=['stack'],
        selfbuilt_python=python,aspire_python=python,episode_timeout=60.)
    frozen=study.Study(args)
    assert frozen.protocol['interpreters']['selfbuilt']==str(python.absolute())
    assert frozen.protocol['interpreters']['selfbuilt']!=str(python.resolve())
