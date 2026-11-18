"""Only evidence bound to the actual candidate and fixed initial state promotes."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from scripts.runners import run_aspire_wipe_batch as batch
from scripts.gates.wipe_gate_state import prepare,commit,load_chain,digest


def measured(root,version,status='PASS'):
    folder=root/'round_01'/f'version_{version:02d}'
    out=folder/'execution';out.mkdir()
    (out/'safety_physics.jsonl.gz').write_bytes(b'test witness bound by content hash')
    batch.save(root/'round_01'/'initial_reference.json',{'initial_state_id':'fixed'})
    batch.save(out/'result.json',{'status':'MEASURED','round':1,'version':version,'seed':716,
        'initial_state_id':'fixed','program_ok':True,'native_task_success':True,
        'program_sha256':digest(folder/'program.py'),'protocol_sha256':digest(root/'protocol.json'),
        'terminal_native_success':True,'native_success_terminated':True,'settle_steps':0,'settle_success_steps':0,
        'terminal_hold_status':'NATIVE_SUCCESS_TERMINATED_EXEMPT',
        'safety':{'status':status,'coverage_complete':status=='PASS','rules_sha256':digest(root/'safety_rules.json'),
                  'witness_sha256':digest(out/'safety_physics.jsonl.gz')}})


def test_promote_then_unknown_retains_parent_and_next_proposal_uses_parent(tmp_path):
    root=tmp_path/'batch';batch.init(root,[716],versions=3)
    source=tmp_path/'program.py';source.write_text('pass\n')
    prepare(root,1,1,source,'test');measured(root,1)
    assert commit(root,1,1)['accepted']
    prepare(root,1,2,source,'test');measured(root,2,'UNKNOWN')
    decision=commit(root,1,2)
    assert not decision['accepted'] and decision['retained_parent']
    assert decision['parent_after']['version']==1
    folder=prepare(root,1,3,source,'test')
    assert json.loads((folder/'proposal.json').read_text())['gate_base']['version']==1
    assert load_chain(root,1)['active']['version']==1


def test_corrupted_witness_never_promotes(tmp_path):
    root=tmp_path/'batch';batch.init(root,[716],versions=1)
    source=tmp_path/'program.py';source.write_text('pass\n')
    prepare(root,1,1,source,'test');measured(root,1)
    (root/'round_01/version_01/execution/safety_physics.jsonl.gz').write_bytes(b'changed')
    decision=commit(root,1,1)
    assert decision['reason']=='binding_or_integrity_error' and decision['parent_after'] is None
