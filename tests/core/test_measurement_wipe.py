"""Native Wipe joint proximity is separate from its primary contact rule."""
import json
import pytest

from esrb.measurement.spec import event_profile


def test_future_wipe_protocol_enables_joint_rule_without_changing_other_scenes():
    from esrb.measurement.spec import protocol
    mixed=protocol(['wipe','stack'],1)
    assert mixed['scene_event_profiles']['wipe']['schema']=='measurement-event-profile/7'
    assert 'stack' not in mixed['scene_event_profiles']
    assert mixed['event_profile']['schema']=='measurement-event-profile/6'


@pytest.mark.parametrize('position,expected',[(.89,False),(.90,True),(.91,True)])
def test_wipe_joint_proximity_boundary_and_reference(tmp_path,position,expected):
    import mujoco
    from esrb.measurement.recording import Recorder
    from esrb.measurement.rules import analyze
    from esrb.measurement.reference import check
    m=mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".002" gravity="0 0 0"/>
      <worldbody><body name="arm" pos="0 0 1"><joint name="robot0_joint1" range="-1 1" limited="true"/>
      <geom name="robot0_link" type="sphere" size=".01" mass="1"/><site name="grip_origin"/></body></worldbody></mujoco>'''.replace('<mujoco>','<mujoco><compiler angle="radian"/>'))
    d=mujoco.MjData(m);d.qpos[0]=position;mujoco.mj_forward(m,d)
    roles=dict(scene='wipe',objects=[1],robot=[0],pads=[],pad_groups=[],tool=[],fixtures=[],part_geoms={'1':[0]},
        initial_positions={'1':[0.,0.,1.]},tcp_sites=[0],grasp_required=False,control_dt_s=.02,analytic_fixture=True)
    folder=tmp_path/'trace';folder.mkdir()
    with Recorder(m,d,folder/'raw',roles) as rec:
        rec.control=0;rec.phase='hold'
        for _ in range(10):mujoco.mj_step(m,d)
    (folder/'actions.json').write_text(json.dumps([dict(step=0,time_s=0.,action=[])]))
    result=analyze(folder,profile=event_profile(7))
    events=[e for e in result['events'] if e['kind']=='joint_limit_proximity']
    assert bool(events)==expected
    assert result['primary_status']=='PASS'
    assert result['status']==('VIOLATION' if expected else 'PASS')
    if events:assert events[0]['joint']=='robot0_joint1' and events[0]['state_timing']=='end of physics step'
    ref=check(folder,result,max_samples=4)
    assert ref['status']=='PASS',ref
    assert analyze(folder,profile=event_profile(6))['status']=='PASS'
