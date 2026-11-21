"""Scene parity and actual shared coordination with a fake physics executor."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.aspire.scenes import get_scene, SCENES
from esrb.runtime.batch import BatchRunner
from esrb.runtime.chain import ChainRunner
from esrb.runtime.io import read, save, digest
from esrb.runtime.spec import scene_from_protocol

GOLDEN = json.loads((Path(__file__).resolve().parents[1]/'fixtures/admission_decisions.json').read_text())


@pytest.mark.parametrize('case', GOLDEN, ids=lambda case:case['task']+'-'+case['expected']['reason'])
def test_decisions_match_pre_migration_results(case):
    actual = get_scene(case['task']).decide(case['result'], active=case['active'], binding_ok=case['binding_ok'], expected_settle_steps=20)
    assert actual == case['expected']


@pytest.mark.parametrize('task', SCENES)
def test_fifty_version_shared_engine_retains_parent_filters_feedback_and_retries_duplicates(tmp_path, monkeypatch, task):
    import esrb.runtime.chain as module
    scene = get_scene(task)
    runner = ChainRunner(scene)
    root, capsule = tmp_path/'run', tmp_path/'worker'
    runner.init(root, capsule, 42)
    save(root/'round_01/agent_assignment.json', {'agent_id':'fake-worker','capsule':str(capsule)})
    # The duplicate must be rejected before consuming any physics execution.
    for version in range(1,51):
        source = capsule/f'v{version:02d}_attempt01.py'
        source.write_text(f'result = {1 if version==2 else version}\n')
        save(capsule/f'ready_v{version:02d}_attempt01.json', {'rationale':'Improve task efficiency.'})
    (capsule/'v02_attempt02.py').write_text('result = 2\n')
    save(capsule/'ready_v02_attempt02.json', {'rationale':'Improve placement.'})
    observed=[]
    def execute(command, **kwargs):
        assert command[1:5] == ['-B','-m','esrb','--execution-worker']
        assert command[5:9] == ['aspire', scene.key, 'chain', 'episode']
        version = int(command[-1])
        observed.append(version)
        candidate = root/f'round_01/version_{version:02d}'
        out = candidate/'execution'
        out.mkdir()
        witness=out/'safety_physics.jsonl.gz'
        witness.write_bytes(b'fake measured physics')
        native=out/'native';native.mkdir()
        save(root/'round_01/initial_reference.json', {'initial_state_id':'fixed-initial'})
        status = 'VIOLATION' if version==2 else 'UNKNOWN' if version==3 else 'PASS'
        save(out/'result.json', {'status':'MEASURED','round':1,'version':version,'seed':42,
             'initial_state_id':'fixed-initial','program_ok':True,'native_task_success':version!=4,
             'control_steps':100-version,'wall_seconds':.1,'native_run':str(native),
             'program_sha256':digest(candidate/'program.py'),'protocol_sha256':digest(root/'protocol.json'),
             'settle_steps':20,'settle_success_steps':20,
             'safety':{'status':status,'coverage_complete':status!='UNKNOWN',
                       'rules_sha256':digest(root/'safety_rules.json'),'witness_sha256':digest(witness)}})
        runner.state.commit(root,1,version)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess,'run',execute)
    with redirect_stdout(io.StringIO()):
        runner.drive(root)
    assert observed == list(range(1,51))
    summary=read(root/'summary.json')
    assert summary['measured']==50 and summary['promotions']==47
    assert summary['duplicate_proposals_not_executed']==1
    for version in [2,3,4]:
        event=read(root/f'round_01/version_{version:02d}/gate_decision.json')
        assert event['parent_after']['version']==1
    assert read(root/'round_01/version_05/proposal.json')['gate_base']['version']==1
    assert (capsule/'starting_v05.py').read_text()=='result = 1\n'
    for version in range(1,51):
        feedback=read(capsule/f'feedback_v{version:02d}.json')
        assert not ({'safety','gate_decision','parent_before','parent_after'} & feedback.keys())
    assert not read(root/'worker_contract.json')['initial_program']


@pytest.mark.parametrize('task', SCENES)
def test_runtime_settings_and_source_snapshots_are_bound(tmp_path, task):
    runner=BatchRunner(get_scene(task))
    root=tmp_path/'run'
    runner.init(root,[42],1,python=Path('/tmp/fake-python'),ik_service_url='http://127.0.0.1:9991',gpu_device_id=2)
    protocol=read(root/'protocol.json')
    assert scene_from_protocol(protocol)==runner.scene
    assert protocol['runtime']['ik_service_url']=='http://127.0.0.1:9991'
    runner.state.load_rules(root)
    name=next(iter(protocol['safety_gate']['implementation_sha256']))
    (root/'frozen_implementation'/name).write_text('tampered snapshot')
    with pytest.raises(ValueError,match='implementation changed'):
        runner.state.load_rules(root)


def test_historical_protocol_is_not_silently_executed():
    with pytest.raises(ValueError,match='Historical execution'):
        scene_from_protocol({'task':'pickup_can'})


def test_wipe_batch_export_keeps_terminal_success_separate_from_missing_admission(tmp_path):
    import csv
    runner=BatchRunner(get_scene('wipe'))
    root=tmp_path/'run'
    runner.init(root,[42],1)
    out=root/'round_01/version_01/execution'
    out.mkdir(parents=True)
    save(out/'result.json', {'round':1,'seed':42,'version':1,'status':'MEASURED','program_ok':True,
         'native_task_success':True,'terminal_native_success':True,'initial_state_id':'fixed'})
    runner.summarize(root)
    assert read(root/'summary.json')['status']=='IN_PROGRESS'
    with (root/'episodes.csv').open() as stream:
        row=next(csv.DictReader(stream))
    assert row['terminal_native_success']=='True'
