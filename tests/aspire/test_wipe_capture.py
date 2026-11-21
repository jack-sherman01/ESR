"""Every native mj_step writes loads and all contact points, including same pair."""
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from esrb.backends.aspire.capture import wipe as capture

RULES=json.loads((Path(__file__).resolve().parents[2]/'configs/experiment/wipe_safety_gate.json').read_text())


def environment():
    data=SimpleNamespace(time=0.,ncon=2,_data=object(),sensordata=np.array([1.,2.,3.,.1,.2,.3]),
        contact=[SimpleNamespace(geom1=0,geom2=1,pos=np.zeros(3),frame=np.eye(3)) for _ in range(2)],
        get_site_xmat=lambda name:np.eye(3))
    model=SimpleNamespace(_model=object(),sensor_dim=[3,3],sensor_adr=[0,3],sensor_name2id=lambda name:0 if name=='force' else 1)
    sim=SimpleNamespace(data=data,model=model)
    def step():data.time+=.002
    sim.step=step
    env=SimpleNamespace(sim=sim,robots=[SimpleNamespace(gripper=SimpleNamespace(important_sensors={'force_ee':'force','torque_ee':'torque'}))])
    return env


def test_capture_preserves_every_contact_and_physics_interval(tmp_path,monkeypatch):
    env=environment()
    def wrench(model,data,index,result):result[:]=[10.*(index+1),1.,2.,3.,4.,5.]
    monkeypatch.setattr(capture.mujoco,'mj_contactForce',wrench)
    original=env.sim.step
    witness=capture.WipeSafetyCapture(env,['tool','table','robot_link','floor'],RULES,tmp_path)
    env.sim.step();env.sim.step();report=witness.finish()
    assert env.sim.step is original
    with gzip.open(tmp_path/'safety_physics.jsonl.gz','rt') as stream:rows=[json.loads(x) for x in stream]
    assert len(rows)==2 and report['physics_steps']==2 and report['coverage_complete']
    assert rows[0]['tool_table_normal_force_n']==30.
    assert [c['normal_force_n'] for c in rows[0]['contacts']]==[10.,20.]
    assert rows[0]['tool_sensor_force_raw_n']==[1.,2.,3.]
    assert rows[0]['tool_sensor_torque_raw_nm']==[.1,.2,.3]


def test_failed_sensor_sample_stops_and_is_unknown(tmp_path,monkeypatch):
    env=environment();env.sim.data.sensordata[0]=float('nan')
    monkeypatch.setattr(capture.mujoco,'mj_contactForce',lambda model,data,index,result:result.fill(0.))
    witness=capture.WipeSafetyCapture(env,['tool','table','robot_link','floor'],RULES,tmp_path)
    with pytest.raises(capture.SafetyStop):env.sim.step()
    report=witness.finish()
    assert report['status']=='UNKNOWN' and not report['coverage_complete'] and report['errors']
