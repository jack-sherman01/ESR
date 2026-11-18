"""Native tool/table load is expected; unclassified evidence cannot qualify."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts'))
from scripts.gates.wipe_gate_rules import WipeSafety, classify_geometry, decide
from scripts.gates.wipe_gate_state import prepare, load_chain, preflight
SPEC=importlib.util.spec_from_file_location('wipe_batch',REPO/'scripts/runners/run_aspire_wipe_batch.py')
BATCH=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(BATCH)
RULES=json.loads((REPO/'configs/experiment/wipe_safety_gate.json').read_text())


def contact(a,b,force=50.):
    return {'groups':[a,b],'normal_force_n':force}


def test_tool_table_allowed_even_when_load_exceeds_native_reward_threshold():
    witness=WipeSafety(RULES)
    assert witness.observe(0.,.002,[contact('tool','table',1000.)],[0.,0.,1000.],[0.,1.,0.]) is None
    report=witness.report(elapsed_physics_steps=1,records_written=1)
    assert report['status']=='PASS' and report['coverage_complete']
    assert report['peak_tool_table_normal_force_n']==1000.
    assert report['load_overload_assessed'] is False


@pytest.mark.parametrize('pair',[('robot_link','table'),('robot_link','floor'),('tool','floor')])
def test_forbidden_contacts(pair):
    witness=WipeSafety(RULES)
    assert witness.observe(0.,.002,[contact(*pair)],[0.,0.,0.],[0.,0.,0.])['kind']=='forbidden_contact'
    assert witness.report(elapsed_physics_steps=1,records_written=1)['status']=='VIOLATION'


@pytest.mark.parametrize('mode',['missing_step','nonfinite_force','unknown_contact'])
def test_unknown_evidence_fails_closed(mode):
    witness=WipeSafety(RULES)
    with pytest.raises(ValueError):
        witness.observe(.002 if mode=='missing_step' else 0.,.004 if mode=='missing_step' else .002,
            [contact('tool','other')] if mode=='unknown_contact' else [],
            [float('nan'),0.,0.] if mode=='nonfinite_force' else [0.,0.,0.],[0.,0.,0.])
    assert witness.report(elapsed_physics_steps=1,records_written=0)['status']=='UNKNOWN'


def test_coverage_count_mismatch_is_unknown():
    witness=WipeSafety(RULES);witness.observe(0.,.002,[],[0.,0.,0.],[0.,0.,0.])
    assert witness.report(elapsed_physics_steps=2,records_written=1)['status']=='UNKNOWN'


def test_geometry_only_native_wiping_shapes_are_tool():
    assert classify_geometry('gripper0_wiping_corner1',['gripper0_eef','robot0_right_hand'])=='tool'
    assert classify_geometry('gripper0_mount',['gripper0_wiping_gripper','robot0_right_hand'])=='robot_link'
    assert classify_geometry('robot0_link7_collision',['robot0_link7'])=='robot_link'
    assert classify_geometry('table_collision',['table'])=='table'


def test_native_success_termination_exempts_hold_without_fabrication():
    result={'status':'MEASURED','program_ok':True,'native_task_success':True,
        'safety':{'status':'PASS','coverage_complete':True},'settle_steps':0,'settle_success_steps':0,
        'terminal_native_success':True,'native_success_terminated':True,
        'terminal_hold_status':'NATIVE_SUCCESS_TERMINATED_EXEMPT'}
    decision=decide(result,active=None,binding_ok=True,expected_settle_steps=20)
    assert decision['accepted'] and decision['terminal_hold_status']=='NATIVE_SUCCESS_TERMINATED_EXEMPT'
    result['native_success_terminated']=False
    assert not decide(result,active=None,binding_ok=True,expected_settle_steps=20)['accepted']
    result['safety']['status']='UNKNOWN'
    assert decide(result,active={'version':1},binding_ok=True,expected_settle_steps=20)['retained_parent']


def test_batch_prepare_binds_parent_and_program(tmp_path):
    root=tmp_path/'batch';BATCH.init(root,[716],versions=2)
    source=tmp_path/'empty.py';source.write_text('pass\n')
    folder=prepare(root,1,1,source,'infrastructure smoke')
    assert json.loads((folder/'proposal.json').read_text())['gate_base'] is None
    preflight(root,1,1)
    (folder/'program.py').write_text('a=1\n')
    with pytest.raises(ValueError,match='differs'):preflight(root,1,1)
    BATCH.summarize(root)
    assert BATCH.read(root/'summary.json')['requested_rollouts']==2
