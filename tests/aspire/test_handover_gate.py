import importlib.util,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
from esrb.detection.handover_gate import HandoverSafety, classify_geometry
from esrb.detection.admission import decide
from esrb.backends.aspire.feedback.handover import validate_prompt,feedback
RULES=json.loads((ROOT/'configs/experiment/handover_safety_gate.json').read_text())

def contact(a,b,ga,gb):return {'names':[a,b],'groups':[ga,gb],'normal_force_n':1.0}
def grasp(arm):
 return [contact(f'gripper{arm}_finger{f}_pad_collision','hammer_handle','primary_gripper' if arm==0 else 'secondary_gripper','hammer_handle') for f in (1,2)]
def advance(m,n,contacts,z=.05):
 for _ in range(n):m.observe(m.last_end,m.last_end+.002,[0.,0.,z],contacts)
def monitor():return HandoverSafety(RULES,0.)
def report(m):return m.report(elapsed_physics_steps=m.steps,records_written=m.steps)

def test_stable_handover():
 m=monitor();advance(m,60,grasp(0));advance(m,60,grasp(0)+grasp(1));advance(m,60,grasp(1))
 assert report(m)['status']=='PASS';assert m.stages['giver_release']>m.stages['overlap_start']
def test_open_precursor_is_not_actual_release():
 m=monitor();advance(m,60,grasp(0));m.command(m.last_end,{'secondary':0,'primary':1},3)
 advance(m,60,grasp(0)+grasp(1));advance(m,60,grasp(1))
 assert m.precursors and report(m)['status']=='PASS'
def test_actual_premature_release_continues_to_observe_drop():
 m=monitor();advance(m,60,grasp(0));advance(m,60,[])
 assert [e['kind'] for e in m.events]==['premature_release']
 e=m.events[0];assert e['onset_physics_index']==60 and e['physics_index']==109
 assert e['kind'] not in RULES['stop_on']
 advance(m,1,[contact('hammer_handle','table_collision','hammer_handle','table')],0.)
 assert [e['kind'] for e in m.events]==['premature_release','hammer_drop']
def test_drop_requires_sustained_pickup():
 m=monitor();advance(m,60,[contact('hammer_handle','table_collision','hammer_handle','table')],0.)
 assert not m.events
@pytest.mark.parametrize('groups,kind',[(('primary_gripper','secondary_link'),'inter_arm_collision'),(('secondary_gripper','table'),'robot_support_collision'),(('primary_link','floor'),'robot_support_collision')])
def test_loaded_contacts(groups,kind):
 m=monitor();advance(m,1,[contact('a','b',*groups)]);assert m.events[0]['kind']==kind
@pytest.mark.parametrize('groups',[('primary_gripper','primary_link'),('secondary_gripper','hammer_handle')])
def test_expected_or_same_arm_contact(groups):
 m=monitor();advance(m,1,[contact('a','b',*groups)]);assert not m.events

def test_missing_sampling():
 m=monitor()
 with pytest.raises(ValueError,match='missing'):m.observe(.002,.004,[0.,0.,0.],[])
 assert report(m)['status']=='UNKNOWN'
def test_unknown_loaded_identity():
 m=monitor()
 with pytest.raises(ValueError,match='unclassified'):advance(m,1,[contact('a','b','primary_link','other')])
def test_only_receiver_handle_counts():
 m=monitor();advance(m,60,[dict(c,names=[c['names'][0],'hammer_head']) for c in grasp(1)])
 assert m.receiver_dwell==0
@pytest.mark.parametrize('text',['门控结果','gate result','ESR decision'])
def test_no_host_information(text):
 with pytest.raises(ValueError):validate_prompt(text)
def test_task_feedback_filters_host_error_and_keeps_both_arms(tmp_path):
 native=tmp_path/'native';(native/'source').mkdir(parents=True)
 (tmp_path/'execution').mkdir();(tmp_path/'execution/result.json').write_text(json.dumps({'native_run':str(native),'native_task_success':False,'program_ok':False,'control_steps':10,'safety':{'status':'VIOLATION'}}))
 (native/'source/program-result.json').write_text(json.dumps({'error':'ESRB safety interlock','calls':[{'tool':'set_grippers','ok':False,'error_message':'ESRB safety interlock'}]}))
 d=feedback(tmp_path);validate_prompt(json.dumps(d));assert d['tool_results']==[{'tool':'set_grippers','returned':False}]
