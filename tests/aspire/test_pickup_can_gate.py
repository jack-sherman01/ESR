import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/'scripts'))
from scripts.gates.pickup_can_gate_rules import PickupCanSafety,decide,classify_geometry
from scripts.feedback.pickup_can_task_feedback import TASK_PROMPT,validate_prompt,task_feedback
RULES=json.loads((Path(__file__).parents[2]/'configs/experiment/pickup_can_safety_gate.json').read_text())

def c(a,b,force=1.):return {'geom1':1,'geom2':2,'groups':[a,b],'normal_force_n':force}
def step(m,can=(0.,0.,.88),contacts=()):return m.observe(m.last_end,m.last_end+.002,list(can),list(contacts))
def report(m):return m.report(elapsed_physics_steps=m.steps,records_written=m.steps)

@pytest.mark.parametrize('pair',RULES['forbidden_contact_pairs'])
def test_forbidden(pair):
 m=PickupCanSafety(RULES,.82);step(m,contacts=[c(*pair)]);assert report(m)['status']=='VIOLATION'

def test_expected_pickup_and_placement():
 m=PickupCanSafety(RULES,.82)
 step(m,contacts=[c('can','source_bin')]);step(m,(0.,0.,1.),[c('can','gripper')]);step(m,(0.,.4,.88),[c('can','target_bin')])
 assert report(m)['status']=='PASS'

def test_drop_to_source_after_lift():
 m=PickupCanSafety(RULES,.82);step(m,(0.,0.,1.));step(m,contacts=[c('can','source_bin')]);assert report(m)['status']=='VIOLATION'

def test_floor_and_missing_data():
 m=PickupCanSafety(RULES,.82);step(m,contacts=[c('can','floor')]);assert report(m)['status']=='VIOLATION'
 m=PickupCanSafety(RULES,.82);assert report(m)['status']=='UNKNOWN'
 with pytest.raises(ValueError,match='missing'):m.observe(.01,.012,[0.,0.,.9],[])

def test_unsafe_success_retains_parent():
 d=decide({'status':'MEASURED','native_task_success':True,'program_ok':True,'safety':{'status':'VIOLATION'}},active={'version':1},binding_ok=True,expected_settle_steps=20)
 assert d['retained_parent'] and d['observed_safety_regression']

def test_geometry_identities():
 assert classify_geometry('', ['bin1'])=='source_bin'
 assert classify_geometry('', ['Can_main'])=='can'
 assert classify_geometry('VisualCan_g0',['VisualCan_main'])=='other'

def test_prompt_and_feedback_remove_all_host_fields_and_errors():
 validate_prompt(TASK_PROMPT)
 result={'native_task_success':False,'program_ok':True,'control_steps':12,'safety':{'status':'VIOLATION'},'gate':'REJECT'}
 feedback=task_feedback(result,'grasp_descent False ESRB safety interlock\nsecret gate information\nlocalized can pixels 9 top_center [0.5, -0.2, 0.1]\n')
 validate_prompt(json.dumps(feedback));assert feedback['motion_stages']==[{'stage':'grasp_descent','completed':False}]
 assert len(feedback['public_rgbd_observations'])==1
 with pytest.raises(ValueError):validate_prompt('parent passed safety gate')

def test_prepare_archives_only_task_inputs_and_enforces_hashes(tmp_path):
 import importlib.util
 from scripts.gates import pickup_can_gate_state as state
 path=Path(__file__).parents[2]/'scripts/runners/run_aspire_pickup_can_batch.py'
 spec=importlib.util.spec_from_file_location('pickup_batch',path);batch=importlib.util.module_from_spec(spec);spec.loader.exec_module(batch)
 root=tmp_path/'batch';batch.init(root,[21417],1)
 program=tmp_path/'program.py';program.write_text('result = {"revision": 1}\n')
 folder=state.prepare(root,1,1,program,'Initial task program.',generation_base=program)
 validate_prompt((folder/'candidate_prompt.txt').read_text())
 state.preflight(root,1,1)
 (folder/'candidate_prompt.txt').write_text('Changed prompt.')
 with pytest.raises(ValueError,match='input changed'):state.preflight(root,1,1)
