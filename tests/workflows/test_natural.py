"""Task-only evolution: multi-case selection, immutable ancestry, and recovery."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from esrb.runtime.io import read, save, digest
from esrb.runtime import natural
from experiments.aspire.scenes import SCENES, get_scene
from experiments.aspire.natural_campaign import select_parallelism


def case(seed, ok=True, **extra):
    return dict(seed=seed, status='MEASURED', program_ok=True, native_task_success=ok, **extra)


def test_five_cases_required_and_safety_labels_are_not_selection_inputs():
    cases = [case(s, safety={'status':'VIOLATION'}) for s in natural.SEEDS]
    assert natural.task_selection(cases, natural.SEEDS)
    cases[2]['native_task_success'] = False
    assert not natural.task_selection(cases, natural.SEEDS)
    with pytest.raises(ValueError):
        natural.task_selection(cases[:4], natural.SEEDS)
    with pytest.raises(ValueError):
        natural.task_selection(list(reversed(cases)), natural.SEEDS)
    cases[2]['status']='INFRASTRUCTURE_ERROR'
    with pytest.raises(natural.InfrastructureBlocked):
        natural.task_selection(cases, natural.SEEDS)


@pytest.mark.parametrize('task', SCENES)
def test_task_lineage_retains_parent_tests_all_seeds_and_resumes(tmp_path, monkeypatch, task):
    root = tmp_path/'chain'
    natural.initialize(root, get_scene(task), budget=3)
    monkeypatch.setattr(natural, 'validate_program', lambda *args:None)
    inputs=[]; executions=[]
    def generator(output, text, **settings):
        import json
        data=json.loads(text);inputs.append(data)
        output.mkdir()
        return {'program':f"result = {data['version']}\n",'rationale':'Improve completion.'}
    class FakeScheduler:
        pool=ThreadPoolExecutor(max_workers=5)
        def execute(self, root, protocol, folder, program, seed):
            version=int(folder.name[1:]);executions.append((version,seed))
            return case(seed, not(version==2 and seed==3),task_qualified=not(version==2 and seed==3),
                        control_steps=10,wall_seconds=.1,initial_state_id=f'initial-{seed}',native_run=None,
                        output=str(folder/f'seed_{seed}/attempt_01/execution'), safety={'status':'VIOLATION'})
    scheduler=FakeScheduler()
    natural.drive(root,scheduler,generator=generator)
    assert len(executions)==15 and sorted(executions)==[(v,s) for v in range(1,4) for s in natural.SEEDS]
    state=read(root/'state.json')
    assert [e['accepted'] for e in state['events']]==[True,False,True]
    assert state['events'][2]['generation_base']['version']==1
    assert inputs[0]['starting_program'] is None
    assert inputs[1]['starting_program']=='result = 1\n'
    assert inputs[2]['starting_program']=='result = 1\n'
    assert len(inputs[2]['recent_task_feedback'])==2
    assert all('program' not in h for h in inputs[2]['recent_task_feedback'])
    assert 'safety' not in str(inputs)
    natural.drive(root,scheduler,generator=generator)
    assert len(executions)==15 and len(inputs)==3
    assert read(root/'summary.json')['task_accepted']==2
    scheduler.pool.shutdown()


def test_failed_initial_program_is_generation_base_not_accepted_parent(tmp_path, monkeypatch):
    root=tmp_path/'chain';natural.initialize(root,get_scene('stack'),budget=2)
    folder=root/'candidates/v01';folder.mkdir()
    program=folder/'program.py';program.write_text('result = 1\n')
    save(folder/'feedback.json',{'development_successes':0})
    natural.commit(root,1,program,None,[case(s,False,task_qualified=False) for s in natural.SEEDS],False,'MEASURED')
    protocol,state=natural.validate(root)
    assert state['active'] is None and state['last_program']['version']==1
    assert 'result = 1' in natural.prompt(root,protocol,state,2)
    program.write_text('result = 99\n')
    with pytest.raises(ValueError,match='candidate changed'):
        natural.validate(root)


def test_parallelism_uses_throughput_rejects_errors_and_prefers_lower_near_tie():
    rows=[{'workers':w,'physics_steps_per_second':s,'infrastructure_errors':e}
          for w,s,e in [(4,100,0),(8,200,0),(12,208,0),(16,300,1)]]
    assert select_parallelism(rows)==8
    with pytest.raises(ValueError):
        select_parallelism([dict(rows[0],infrastructure_errors=1)])


def test_natural_backend_has_no_detector_admission_or_safety_capture_imports():
    import ast
    path=natural.REPO/'esrb/backends/aspire/episodes/natural.py'
    tree=ast.parse(path.read_text())
    modules=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any('detection' in m or 'gate_state' in m for m in modules)
    assert not any('capture.' in m and not m.endswith('.raw') for m in modules)


def test_cli_has_independent_natural_and_campaign_routes():
    from esrb.cli.main import build_parser
    parser=build_parser()
    args=parser.parse_args(['aspire','wipe','natural','init','--root','/tmp/test-natural'])
    assert args.dev_seeds==[1,2,3,4,5] and args.candidates==50
    args=parser.parse_args(['aspire','natural-campaign','init','--root','/tmp/test-campaign','--batch','x'])
    assert args.chains==3 and args.model=='gpt-6.1-sol'


def test_timeout_is_a_task_failure_not_a_safety_or_retry_decision():
    rows=[case(s) for s in natural.SEEDS]
    rows[1].update(status='EXECUTION_TIMEOUT',program_ok=False,native_task_success=None)
    assert not natural.task_selection(rows,natural.SEEDS)


def test_duplicate_attempts_exhaust_only_one_slot_and_continue(tmp_path,monkeypatch):
    import json
    root=tmp_path/'chain';natural.initialize(root,get_scene('stack'),budget=3)
    monkeypatch.setattr(natural,'validate_program',lambda *a:None)
    calls=[];executions=[]
    def generator(output,text,**kwargs):
        data=json.loads(text);calls.append(data['version']);output.mkdir()
        return {'program':f"result = {1 if data['version']==2 else data['version']}\n",'rationale':'Improve task.'}
    class Fake:
        pool=ThreadPoolExecutor(max_workers=5)
        def execute(self,root,protocol,folder,program,seed):
            executions.append((folder.name,seed))
            return case(seed,task_qualified=True,control_steps=1,wall_seconds=.1,
                        output=str(folder/f'seed_{seed}'),initial_state_id=str(seed),native_run=None)
    scheduler=Fake();natural.drive(root,scheduler,generator=generator);scheduler.pool.shutdown()
    assert calls==[1,2,2,2,3] and len(executions)==10
    summary=read(root/'summary.json')
    assert summary['completed_slots']==3 and summary['executed_candidates']==2
    assert read(root/'state.json')['events'][1]['status']=='UNEXECUTED'


def test_resume_reuses_completed_seed_and_rejects_result_tampering(tmp_path,monkeypatch):
    root=tmp_path/'chain';root.mkdir();(root/'initial_references').mkdir()
    folder=root/'candidate';folder.mkdir();program=folder/'program.py';program.write_text('result = 1\n')
    execution=folder/'seed_1/attempt_01/execution';execution.mkdir(parents=True)
    result=case(1,program_sha256=digest(program),initial_state_id='seed-one',task_qualified=True)
    save(execution/'result.json',result)
    save(execution/'artifacts.json',{'result.json':digest(execution/'result.json')})
    scheduler=natural.Scheduler(1,[3])
    monkeypatch.setattr(natural.subprocess,'Popen',lambda *a,**k:pytest.fail('completed seed reran'))
    assert scheduler.execute(root,{'max_infrastructure_attempts':2},folder,program,1)['initial_state_id']=='seed-one'
    assert read(root/'initial_references/seed_1.json')['initial_state_id']=='seed-one'
    save(execution/'result.json',dict(result,native_task_success=False))
    with pytest.raises(natural.InfrastructureBlocked,match='artifact changed'):
        scheduler.execute(root,{'max_infrastructure_attempts':2},folder,program,1)
    scheduler.pool.shutdown()


def test_native_subset_validation_rejects_host_imports(tmp_path):
    program=tmp_path/'program.py';program.write_text('import os\n')
    with pytest.raises(ValueError,match='subset validation'):
        natural.validate_program(program,str(natural.robosuite_python()))
    program.write_text('print(1)\n')
    natural.validate_program(program,str(natural.robosuite_python()))


def test_can_static_fixture_calibration_matches_public_api():
    import numpy as np
    from types import SimpleNamespace
    from esrb.backends.aspire.episodes.natural import fixture_language
    env=SimpleNamespace(target_bin_placements=np.array([[.1,.2,.3]]),object_id=0)
    before=env.target_bin_placements.copy()
    language=fixture_language('Place can.',env,np.eye(4))
    assert 'Fixture goal can-center robot0_base XYZ: 0.1,0.2,0.38' in language
    assert np.array_equal(before,env.target_bin_placements)


def test_seal_blocks_model_and_seed_changes(tmp_path):
    root=tmp_path/'chain';natural.initialize(root,get_scene('stack'))
    natural.seal(root)
    protocol=read(root/'protocol.json');protocol['seeds']=[6,7,8,9,10]
    save(root/'protocol.json',protocol)
    with pytest.raises(ValueError,match='sealed natural protocol changed'):
        natural.validate(root)


@pytest.mark.parametrize('task',SCENES)
def test_natural_api_removes_legacy_worker_file_instructions(tmp_path,task):
    root=tmp_path/task;natural.initialize(root,get_scene(task))
    text=(root/'API.txt').read_text()
    assert 'additional_feedback_vNN' not in text and 'feedback_vNN.json' not in text
    assert 'task feedback is provided directly in this conversation' in text


def test_context_keeps_only_base_code_and_two_recent_five_case_summaries(tmp_path):
    import json
    root=tmp_path/'chain';protocol=natural.initialize(root,get_scene('wipe'))
    events=[]
    for version in range(1,50):
        folder=root/'candidates'/f'v{version:02d}';folder.mkdir()
        (folder/'program.py').write_text(f'# historical-program-{version}\nresult = {version}\n')
        save(folder/'feedback.json',{'development_successes':4,'development_cases':5,
            'cases':[{'case':s,'task_success':s!=3,'program_completed':True,'control_ticks':s*10,
                'program_error':None,'task_progress':{'cleaned_markers':version,'total_markers':100},
                'tool_results':[{'tool':'move','value':'large '+str(i)+'x'*5000} for i in range(30)],
                'public_motion_states':[{'xyz':[i,i,i]} for i in range(100)],
                'program_observations':[['sample',i] for i in range(100)],
                'safety':{'status':'VIOLATION'},'host_truth':'PRIVATE_HOST_TRUTH'} for s in natural.SEEDS]})
        events.append({'version':version,'program':str((folder/'program.py').relative_to(root))})
    base={'version':2,'program':'candidates/v02/program.py'}
    state={'active':base,'last_program':events[-1],'events':events}
    previous=digest(root/'candidates/v49/feedback.json')
    text=natural.prompt(root,protocol,state,50,{'diagnostic':'AST duplicated','program':'OLD_INVALID_CODE'})
    data=json.loads(text)
    assert data['starting_program']==(root/base['program']).read_text()
    assert data['starting_program_version']==2
    assert [h['version'] for h in data['recent_task_feedback']]==[48,49]
    assert 'historical-program-49' not in text and 'historical-program-1\n' not in text
    assert 'PRIVATE_HOST_TRUTH' not in text and 'VIOLATION' not in text and 'OLD_INVALID_CODE' not in text
    for recent in data['recent_task_feedback']:
        cases=recent['task_feedback']['cases']
        assert len(cases)==5 and cases[2]['task_success'] is False
        assert cases[4]['control_ticks']==50
        assert len(cases[0]['tool_results'])==6 and cases[0]['tool_results_omitted']==24
        assert cases[0]['program_observations'][-1]==['sample',99]
    assert len(text)<100_000
    assert digest(root/'candidates/v49/feedback.json')==previous


def test_current_program_is_never_truncated_to_fit_model_input(tmp_path):
    root=tmp_path/'chain';protocol=natural.initialize(root,get_scene('stack'))
    code='result = "'+'x'*natural.CONTEXT_POLICY['max_prompt_chars']+'"\n'
    (root/'large.py').write_text(code)
    state={'active':{'version':1,'program':'large.py'},'last_program':None,'events':[]}
    with pytest.raises(natural.ModelBlocked,match='no request sent'):
        natural.prompt(root,protocol,state,2)
    assert (root/'large.py').read_text()==code


def _legacy_chain(root):
    protocol=natural.initialize(root,get_scene('stack'))
    folder=root/'candidates/v01';folder.mkdir()
    program=folder/'program.py';program.write_text('result = 1\n')
    save(folder/'feedback.json',{'development_successes':5,'development_cases':5})
    natural.commit(root,1,program,None,[case(s,task_qualified=True) for s in natural.SEEDS],True,'MEASURED')
    protocol.pop('context_policy')
    protocol['context']='independent chain; full explicit own history; no tools or inherited conversation'
    source=root/'frozen_sources/esrb/runtime/natural.py'
    source.write_text(source.read_text()+'\n# frozen older context implementation\n')
    protocol['sources']['esrb/runtime/natural.py']=digest(source)
    save(root/'protocol.json',protocol)
    # Establish an old, intact seal without pretending the old code matches the working tree.
    save(root/'freeze_receipt.json',{'schema':'esrb.natural_freeze/1',
         'protocol_sha256':digest(root/'protocol.json'),'model':protocol['model'],'seeds':protocol['seeds']})
    return protocol


def test_explicit_context_revision_preserves_old_seal_data_and_is_idempotent(tmp_path):
    from esrb.runtime.natural_preparation import revise_context
    root=tmp_path/'chain';old=_legacy_chain(root)
    campaign=tmp_path/'campaign';campaign.mkdir()
    save(campaign/'manifest.json',{'chains':[{'root':str(root)}]})
    state_sha=digest(root/'state.json');old_sha=digest(root/'protocol.json')
    records=revise_context(campaign,authorization='User requests parent program and recent feedback only.')
    protocol,state=natural.validate(root)
    assert protocol['context_policy']==natural.CONTEXT_POLICY
    assert protocol['model']==old['model'] and protocol['seeds']==old['seeds']
    archive=root/'context_revisions'/records[0]['revision']
    assert digest(archive/'protocol.json')==old_sha
    assert read(archive/'freeze_receipt.json')['protocol_sha256']==old_sha
    assert digest(root/'state.json')==state_sha
    assert records[0]['effective_from_version']==2 and state['active']['version']==1
    assert protocol['sources']['esrb/runtime/natural.py']==digest(natural.REPO/'esrb/runtime/natural.py')
    assert revise_context(campaign,authorization='Already applied.')==[]


def test_context_revision_rejects_unrelated_source_changes_before_mutating_any_chain(tmp_path):
    from esrb.runtime.natural_preparation import revise_context
    campaign=tmp_path/'campaign';campaign.mkdir()
    roots=[tmp_path/'chain1',tmp_path/'chain2']
    for root in roots:_legacy_chain(root)
    p=roots[1]/'protocol.json';protocol=read(p)
    name='esrb/backends/aspire/episodes/natural.py';frozen=roots[1]/'frozen_sources'/name
    frozen.write_text(frozen.read_text()+'\n# unapproved physics change\n')
    protocol['sources'][name]=digest(frozen);save(p,protocol)
    receipt=read(roots[1]/'freeze_receipt.json');receipt['protocol_sha256']=digest(p)
    save(roots[1]/'freeze_receipt.json',receipt)
    save(campaign/'manifest.json',{'chains':[{'root':str(r)} for r in roots]})
    before=[digest(r/'protocol.json') for r in roots]
    with pytest.raises(ValueError,match='cannot change source'):
        revise_context(campaign,authorization='Context only.')
    assert [digest(r/'protocol.json') for r in roots]==before


def test_resume_failed_model_call_uses_next_attempt_in_same_candidate(tmp_path,monkeypatch):
    import json
    root=tmp_path/'chain';natural.initialize(root,get_scene('stack'),budget=1)
    natural.seal(root)
    failed=root/'candidates/v01/generation_01';failed.mkdir(parents=True)
    save(failed/'status.json',{'exit_code':1})
    (failed/'prompt.txt').write_text('retained oversized original request')
    old_hash=digest(failed/'prompt.txt');calls=[]
    monkeypatch.setattr(natural,'validate_program',lambda *a:None)
    def generator(output,text,**kwargs):
        calls.append((output.name,json.loads(text)));output.mkdir()
        return {'program':'result = 1\n','rationale':'Complete task.'}
    class Fake:
        pool=ThreadPoolExecutor(max_workers=5)
        def execute(self,root,protocol,folder,program,seed):
            return case(seed,task_qualified=True,control_steps=1,wall_seconds=.1,
                        output=str(folder/f'seed_{seed}'),initial_state_id=str(seed),native_run=None)
    scheduler=Fake();natural.drive(root,scheduler,generator=generator);scheduler.pool.shutdown()
    assert len(calls)==1 and calls[0][0]=='generation_02'
    assert calls[0][1]['version']==1 and calls[0][1]['context_policy']==natural.CONTEXT_POLICY
    assert digest(failed/'prompt.txt')==old_hash
    assert read(root/'summary.json')['development_episodes']==5
