"""Task-only five-seed evolution. Safety evidence never participates in selection."""
import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

from .io import read, save, digest, fingerprint
from .paths import project_root, robosuite_python
from .submissions import program_identity
from .natural_model import generate, ModelBlocked

REPO = project_root()
SEEDS = [1, 2, 3, 4, 5]
DIRECTORIES = {'stack': 'stack', 'pickplacecan': 'pickplacecan',
               'two-arm-handover': 'handover', 'nut-assembly': 'nut_assembly', 'wipe': 'wipe'}
CONTEXT_POLICY = {'mode': 'parent-and-recent-task-feedback-v1', 'recent_versions': 2,
                  'tool_results': 6, 'motion_states': 2, 'observations': 6,
                  'max_prompt_chars': 200_000}


def natural_api(text):
    lines=[line for line in text.splitlines()
           if not line.startswith(('After each trial also read','feedback_vNN.json'))]
    return '\n'.join(lines)+'\nAfter each candidate, all development-case task feedback is provided directly in this conversation.\n'


class InfrastructureBlocked(RuntimeError):
    pass


@contextmanager
def exclusive(root):
    with (Path(root) / '.natural.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def source_files():
    files = [REPO / 'esrb/cli/main.py', REPO / 'esrb/backends/aspire/evidence.py',
             REPO / 'esrb/backends/aspire/services.py', REPO / 'esrb/backends/aspire/wipe_public_api.py',
             REPO / 'esrb/backends/aspire/wipe_tool_mapping.py', REPO / 'esrb/backends/aspire/pickup_can_registry.py']
    for pattern in ['esrb/runtime/natural*.py', 'esrb/runtime/{io,paths,process,submissions}.py',
                    'esrb/backends/aspire/capture/raw.py', 'esrb/backends/aspire/episodes/natural.py',
                    'experiments/aspire/natural*.py', 'experiments/aspire/scenes/*.py']:
        if '{' not in pattern:
            files.extend(REPO.glob(pattern))
    files += [REPO / f'esrb/runtime/{name}.py' for name in ['io', 'paths', 'process', 'submissions']]
    return sorted(set(files))


def validate_program(program, python):
    check = ('import ast,sys; from cap_harness.runtime import _ProgramValidator; '
             '_ProgramValidator().visit(ast.parse(open(sys.argv[1]).read()))')
    result = subprocess.run([python, '-B', '-c', check, str(program)],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError('Python/API subset validation: ' + result.stderr.strip().splitlines()[-1][:350])


def initialize(root, scene, *, model='gpt-6.1-sol', budget=50, ik_url='http://127.0.0.1:8140',
               python=None, gpu=3, seeds=None, chain=1):
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False)
    seeds = SEEDS if seeds is None else list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or not 1 <= budget <= 50:
        raise ValueError('invalid natural experiment panel or budget')
    snapshot = root / 'frozen_sources'; snapshot.mkdir()
    sources = {}
    for path in source_files() + [REPO / scene.api_path]:
        name = str(path.relative_to(REPO)); destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination); sources[name] = digest(path)
    (root / 'API.txt').write_text(natural_api((REPO / scene.api_path).read_text()))
    protocol = {'schema': 'esrb.natural/1', 'chain_id': str(uuid.uuid4()), 'chain_index': chain,
        'scene': scene.document(), 'seeds': seeds, 'candidate_budget': budget, 'model': model,
        'selection': 'task-only-all-development-cases', 'capture': 'raw-only',
        'max_control_steps': 2000 if scene.prefix == 'handover' else 1000,
        'physics_timestep_s': .002, 'control_frequency_hz': 20,
        'camera_width': 192 if scene.prefix == 'handover' else 128,
        'camera_height': 192 if scene.prefix == 'handover' else 128, 'video_stride': 4,
        'python': str(python or robosuite_python()), 'ik_service_url': ik_url, 'gpu_device_id': gpu,
        'episode_timeout_s': 900, 'generation_timeout_s': 900, 'max_generation_attempts': 3,
        'max_infrastructure_attempts': 2, 'api_sha256': digest(root / 'API.txt'), 'sources': sources,
        'api_source_sha256': digest(REPO/scene.api_path),
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'initial_program': None,
        'context': 'independent chain; current base program and bounded recent task feedback; no tools',
        'context_policy': dict(CONTEXT_POLICY)}
    checkout = REPO/'.esr-local/aspire/checkout'
    commit_id = subprocess.run(['git','-C',str(checkout),'rev-parse','HEAD'],capture_output=True,text=True)
    protocol['aspire_commit'] = commit_id.stdout.strip()
    upstream = []
    for name in ['runtime.py','run.py','api.py','contracts.py','artifacts.py','robosuite/adapter.py','robosuite/registry.py']:
        upstream.append(checkout/'src/cap_harness'/name)
    for name in ['stack','pick_place','two_arm_handover','nut_assembly','wipe']:
        upstream.append(checkout/f'third_party/robosuite/robosuite/environments/manipulation/{name}.py')
    protocol['upstream_sources'] = {}
    for path in upstream:
        if path.is_file():
            name = str(path.relative_to(REPO)); destination = snapshot/name
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
            protocol['upstream_sources'][name] = digest(path)
    save(root / 'protocol.json', protocol)
    (root / 'candidates').mkdir(); (root / 'initial_references').mkdir()
    save(root / 'state.json', {'schema': 'esrb.natural_state/1', 'chain_id': protocol['chain_id'],
         'events': [], 'active': None, 'last_program': None, 'previous_event_sha256': None})
    return protocol


def validate(root):
    root = Path(root); protocol = read(root / 'protocol.json')
    receipt_path=root/'freeze_receipt.json'
    if receipt_path.exists() and read(receipt_path)['protocol_sha256']!=digest(root/'protocol.json'):
        raise ValueError('sealed natural protocol changed')
    if protocol['schema'] != 'esrb.natural/1' or protocol['capture'] != 'raw-only':
        raise ValueError('not a natural raw-only experiment')
    if digest(root / 'API.txt') != protocol['api_sha256']:
        raise ValueError('frozen API changed')
    for name, expected in protocol['sources'].items():
        if digest(REPO / name) != expected or digest(root / 'frozen_sources' / name) != expected:
            raise ValueError(f'frozen natural source changed: {name}')
    for name, expected in protocol.get('upstream_sources',{}).items():
        if digest(REPO/name)!=expected or digest(root/'frozen_sources'/name)!=expected:
            raise ValueError(f'frozen ASPIRE source changed: {name}')
    return protocol, validate_lineage(root)


def validate_lineage(root):
    root = Path(root)
    state = read(root / 'state.json'); previous = None; active = None
    for index, event in enumerate(state['events'], 1):
        content = {k: v for k, v in event.items() if k != 'event_sha256'}
        if (event['version'] != index or event['previous_event_sha256'] != previous or
                event['parent_before'] != active or fingerprint(content) != event['event_sha256']):
            raise ValueError('broken task-only lineage')
        if event.get('program') and digest(root / event['program']) != event['program_sha256']:
            raise ValueError('committed candidate changed')
        for case in event.get('cases',[]):
            if case.get('result_sha256') and digest(Path(case['output'])/'result.json')!=case['result_sha256']:
                raise ValueError('committed episode result changed')
        active = event['parent_after']; previous = event['event_sha256']
    if active != state['active'] or previous != state['previous_event_sha256']:
        raise ValueError('natural active parent differs from history')
    return state


def seal(root):
    root=Path(root)
    protocol,state=validate(root)
    receipt=root/'freeze_receipt.json'
    if not receipt.exists():
        if state['events'] or any((root/'candidates').iterdir()):
            raise ValueError('cannot establish a missing protocol seal after proposals started')
        save(receipt,{'schema':'esrb.natural_freeze/1','protocol_sha256':digest(root/'protocol.json'),
            'model':protocol['model'],'seeds':protocol['seeds'],
            'sealed_at_utc':datetime.now(timezone.utc).isoformat()})


def task_selection(cases, seeds):
    if len(cases) != len(seeds) or [c['seed'] for c in cases] != list(seeds):
        raise ValueError('complete ordered development panel required before selection')
    if any(c.get('status') not in {'MEASURED', 'EXECUTION_TIMEOUT'} for c in cases):
        raise InfrastructureBlocked('development panel has unavailable evidence')
    return all(c.get('program_ok') is True and c.get('native_task_success') is True for c in cases)


def public_feedback(result):
    # A strict positive allowlist: native physics and host errors never enter prompts.
    answer = {'task_success': result.get('native_task_success') is True,
              'program_completed': result.get('program_ok') is True,
              'control_ticks': result.get('control_steps', 0)}
    if result.get('native_run'):
        from .feedback import feedback, task_errors
        if result.get('scene') == 'two-arm-handover':
            from esrb.backends.aspire.feedback.handover import feedback
        folder = Path(result['native_run'])
        # Existing public extractor expects execution/result.json in its parent.
        output = Path(result['output'])
        raw = feedback(output.parent)
        for key in ['tool_results', 'public_motion_states', 'program_observations', 'program_error']:
            if key in raw:
                answer[key] = raw[key]
        answer['api_diagnostics'] = task_errors(folder)
    if result.get('task_progress'):
        answer['task_progress'] = result['task_progress']
    return answer


def _bounded_value(value, *, depth=0, remaining=None):
    """Bound public diagnostics, never programs, without synthesizing observations."""
    remaining = [2000] if remaining is None else remaining
    if remaining[0] <= 0:
        return '[details omitted]'
    if isinstance(value, str):
        limit = min(800, remaining[0])
        remaining[0] -= min(len(value), limit)
        return value if len(value) <= limit else value[:limit//2] + '\n[truncated]\n' + value[-max(1,limit//2):]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if depth >= 4:
        return '[nested details omitted]'
    if isinstance(value, list):
        return [_bounded_value(v, depth=depth+1, remaining=remaining) for v in value[-8:]]
    if isinstance(value, dict):
        return {k: _bounded_value(v, depth=depth+1, remaining=remaining) for k, v in list(value.items())[:12]}
    return '[unsupported detail omitted]'


def recent_task_feedback(feedback, policy):
    answer = {k: feedback[k] for k in ('development_successes', 'development_cases',
              'program_generated', 'message') if k in feedback}
    answer['cases'] = []
    for case in feedback.get('cases', []):
        item = {k: case[k] for k in ('case', 'task_success', 'program_completed', 'control_ticks') if k in case}
        for key in ('task_progress', 'program_error', 'api_diagnostics'):
            if key in case:
                item[key] = _bounded_value(case[key])
        for key, limit in [('tool_results', policy['tool_results']),
                           ('public_motion_states', policy['motion_states']),
                           ('program_observations', policy['observations'])]:
            values = case.get(key, [])
            item[key] = _bounded_value(values[-limit:])
            item[key + '_omitted'] = max(0, len(values)-limit)
        answer['cases'].append(item)
    return answer


def prompt(root, protocol, state, version, revision=''):
    root = Path(root)
    policy = protocol.get('context_policy')
    if policy is not None:
        if policy != CONTEXT_POLICY:
            raise ValueError('unsupported frozen natural context policy')
        base = state['active'] or state['last_program']
        recent = []
        for event in state['events'][-policy['recent_versions']:]:
            folder = root / 'candidates' / f"v{event['version']:02d}"
            recent.append({'version': event['version'],
                           'task_feedback': recent_task_feedback(read(folder/'feedback.json'), policy)})
        text = json.dumps({'task': protocol['scene']['prompt'],
            'instruction': '编写或改进 Python 技能。优先完成全部开发案例，然后尽量减少动作。'
                           '从 starting_program 修改；近期反馈属于标明的版本，不一定属于起点程序。'
                           '只返回 program 和简短 rationale。',
            'public_api': (root/'API.txt').read_text(), 'version': version,
            'development_cases': len(protocol['seeds']), 'context_policy': policy,
            'starting_program_version': base['version'] if base else None,
            'starting_program': (root/base['program']).read_text() if base else None,
            'recent_task_feedback': recent,
            'revision': _bounded_value(revision.get('diagnostic', '')) if isinstance(revision, dict)
                        else _bounded_value(revision)}, ensure_ascii=False)
        if len(text) > policy['max_prompt_chars']:
            raise ModelBlocked('current program/API exceeds bounded prompt budget; no request sent')
        return text
    # Existing sealed protocols remain reproducible until explicitly migrated.
    history = []
    for event in state['events']:
        folder = Path(root) / 'candidates' / f"v{event['version']:02d}"
        history.append({'version': event['version'], 'program': (Path(root) / event['program']).read_text()
                        if event.get('program') else None, 'task_feedback': read(folder / 'feedback.json')})
    base = state['active'] or state['last_program']
    return json.dumps({'task': protocol['scene']['prompt'],
        'instruction': '编写或改进 Python 技能。优先完成全部开发案例，然后尽量减少动作。'
                       '只返回 program 和简短 rationale。',
        'public_api': (Path(root) / 'API.txt').read_text(), 'version': version,
        'development_cases': len(protocol['seeds']),
        'starting_program': (Path(root) / base['program']).read_text() if base else None,
        'own_history': history, 'revision': revision}, ensure_ascii=False)


def commit(root, version, program, base, cases, accepted, status):
    protocol, state = validate(root)
    if version != len(state['events']) + 1:
        raise ValueError('natural candidate sequence mismatch')
    parent = state['active']
    candidate = {'version': version, 'program': str(program.relative_to(root)),
                 'program_sha256': digest(program)} if program else None
    event = {'version': version, 'generation_base': base, 'parent_before': parent,
             'parent_after': candidate if accepted else parent, 'accepted': accepted, 'status': status,
             'program': candidate['program'] if candidate else None,
             'program_sha256': candidate['program_sha256'] if candidate else None,
             'cases': cases, 'previous_event_sha256': state['previous_event_sha256']}
    event['event_sha256'] = fingerprint(event)
    folder = root / 'candidates' / f'v{version:02d}'
    save(folder / 'task_decision.json', event)
    state['events'].append(event); state['active'] = event['parent_after']
    if candidate:
        state['last_program'] = candidate
    state['previous_event_sha256'] = event['event_sha256']
    save(root / 'state.json', state)
    return event


class Scheduler:
    def __init__(self, workers, gpus, reserve_gib=20, bytes_per_episode=0):
        if workers < 1 or not gpus:
            raise ValueError('positive workers and at least one GPU required')
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.gpus = list(gpus); self.counter = 0; self.lock = threading.Lock()
        self.reserve = int(reserve_gib * 1024**3); self.estimate = bytes_per_episode
        self.model_blocked = threading.Event()

    def execute(self, root, protocol, folder, program, seed):
        reference = root / 'initial_references' / f'seed_{seed}.json'
        seed_root = folder / f'seed_{seed}'; seed_root.mkdir(exist_ok=True)
        for attempt in range(1, protocol['max_infrastructure_attempts'] + 1):
            execution = seed_root / f'attempt_{attempt:02d}' / 'execution'
            if (execution / 'result.json').exists():
                result = read(execution / 'result.json')
                if result.get('status') in {'MEASURED', 'EXECUTION_TIMEOUT'}:
                    return self.bind(root, protocol, result, execution, reference, program)
                if 'INITIAL_STATE_MISMATCH' in result.get('error', ''):
                    raise InfrastructureBlocked('INITIAL_STATE_MISMATCH')
                continue
            if execution.exists():
                invocation = execution.parent / 'invocation.json'
                if invocation.exists():
                    pid = read(invocation).get('pid')
                    if pid and Path(f'/proc/{pid}/cmdline').exists():
                        command_line = Path(f'/proc/{pid}/cmdline').read_bytes()
                        if str(execution.parent/'request.json').encode() in command_line:
                            raise InfrastructureBlocked(f'original native process is still running: {pid}')
                save(seed_root / f'attempt_{attempt:02d}' / 'interruption.json',
                     {'status': 'INTERRUPTED_WITHOUT_FINAL_RESULT', 'retained': True})
                continue
            if shutil.disk_usage(root).free < self.reserve + self.estimate:
                raise InfrastructureBlocked('insufficient disk headroom')
            with self.lock:
                gpu = self.gpus[self.counter % len(self.gpus)]; self.counter += 1
            execution.parent.mkdir(exist_ok=True)
            request = {'scene': protocol['scene'], 'seed': seed, 'program': str(program),
                'output': str(execution), 'ik_service_url': protocol['ik_service_url'],
                'max_control_steps': protocol['max_control_steps'], 'camera_width': protocol['camera_width'],
                'camera_height': protocol['camera_height'], 'video_stride': protocol['video_stride'],
                'expected_initial_state_id': read(reference)['initial_state_id'] if reference.exists() else None}
            request_path = execution.parent / 'request.json'; save(request_path, request)
            env = os.environ.copy()
            env.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', EGL_PLATFORM='device',
                MUJOCO_EGL_DEVICE_ID=str(gpu), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', LP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
            command = [protocol['python'], '-B', '-m', 'esrb', '--execution-worker', 'aspire',
                       protocol['scene']['key'], 'natural', 'episode', '--request', str(request_path)]
            save(execution.parent / 'invocation.json', {'command': command, 'gpu': gpu,
                 'timeout_s': protocol['episode_timeout_s']})
            with (execution.parent / 'process.log').open('x') as log:
                process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                invocation = read(execution.parent / 'invocation.json'); invocation['pid'] = process.pid
                save(execution.parent / 'invocation.json', invocation)
                try:
                    code = process.wait(timeout=protocol['episode_timeout_s'])
                except subprocess.TimeoutExpired:
                    code = None
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
            save(execution.parent / 'exit.json', {'returncode': code, 'timeout': code is None})
            if code is None and (execution / 'initial_identity.json').exists():
                result = dict(read(execution/'initial_identity.json'), status='EXECUTION_TIMEOUT',
                    program_sha256=digest(program), program_ok=False, native_task_success=None,
                    task_qualified=False, control_steps=0, wall_seconds=protocol['episode_timeout_s'],
                    capture={'coverage_complete':False}, termination_reason='wall_time_limit')
                save(execution/'result.json',result)
                save(execution/'artifacts.json',{str(p.relative_to(execution)):digest(p)
                    for p in execution.rglob('*') if p.is_file() and p.name!='artifacts.json'})
                return self.bind(root,protocol,result,execution,reference,program)
            if not (execution / 'result.json').exists():
                continue
            result = read(execution / 'result.json')
            if 'INITIAL_STATE_MISMATCH' in result.get('error', ''):
                raise InfrastructureBlocked('INITIAL_STATE_MISMATCH')
            if code == 0 and result.get('status') == 'MEASURED':
                return self.bind(root, protocol, result, execution, reference, program)
        raise InfrastructureBlocked(f'two infrastructure attempts failed: {seed_root}')

    @staticmethod
    def bind(root, protocol, result, execution, reference, program):
        if result['program_sha256'] != digest(program):
            raise InfrastructureBlocked('executed program identity mismatch')
        artifacts = read(execution/'artifacts.json')
        if artifacts.get('result.json') != digest(execution/'result.json'):
            raise InfrastructureBlocked('episode result artifact changed')
        if reference.exists() and read(reference)['initial_state_id'] != result['initial_state_id']:
            raise InfrastructureBlocked('INITIAL_STATE_MISMATCH')
        if not reference.exists():
            save(reference, {'seed': result['seed'], 'initial_state_id': result['initial_state_id']})
        result = dict(result, output=str(execution), result_sha256=digest(execution/'result.json'))
        return result


def drive(root, scheduler, *, generator=generate, model_semaphore=None):
    root = Path(root).resolve()
    with exclusive(root):
        seal(root)
        protocol, state = validate(root)
        for version in range(len(state['events']) + 1, protocol['candidate_budget'] + 1):
            if getattr(scheduler, 'model_blocked', None) is not None and scheduler.model_blocked.is_set():
                raise ModelBlocked('campaign model provider is blocked; no new proposal requested')
            protocol, state = validate(root)
            folder = root / 'candidates' / f'v{version:02d}'; folder.mkdir(exist_ok=True)
            base = state['active'] or state['last_program']
            proposal_path = folder / 'proposal.json'
            if not proposal_path.exists():
                seen = {program_identity((root / e['program']).read_text())
                        for e in state['events'] if e.get('program')}
                revision = ''
                for attempt in range(1, protocol['max_generation_attempts'] + 1):
                    output = folder / f'generation_{attempt:02d}'
                    if (output / 'response.json').exists():
                        receipt = read(output / 'model_receipt.json')
                        if receipt['response_sha256'] != digest(output / 'response.json'):
                            raise ValueError('model response identity changed')
                        answer = read(output / 'response.json')
                    elif output.exists():
                        status_path=output/'status.json'
                        status=read(status_path) if status_path.exists() else {}
                        if status.get('exit_code',0)!=0 or status.get('status') in {'timeout','launch_failed'}:
                            revision={'diagnostic':'Previous generation call did not complete; continue this candidate.'}
                            continue
                        raise ModelBlocked(f'incomplete model attempt requires inspection: {output}')
                    else:
                        if model_semaphore:
                            model_semaphore.acquire()
                        try:
                            if getattr(scheduler, 'model_blocked', None) is not None and scheduler.model_blocked.is_set():
                                raise ModelBlocked('campaign model provider is blocked')
                            answer = generator(output, prompt(root, protocol, state, version, revision),
                                               model=protocol['model'], timeout=protocol['generation_timeout_s'])
                        except ModelBlocked:
                            if getattr(scheduler, 'model_blocked', None) is not None:
                                scheduler.model_blocked.set()
                            raise
                        finally:
                            if model_semaphore:
                                model_semaphore.release()
                    try:
                        ast.parse(answer['program'])
                        if not answer['program'].strip():
                            raise ValueError('empty program')
                        if program_identity(answer['program']) in seen:
                            raise ValueError('program repeats an earlier AST; change the skill')
                        draft = output / 'draft.py'; draft.write_text(answer['program'])
                        validate_program(draft, protocol['python'])
                    except (SyntaxError, ValueError) as exc:
                        revision = {'diagnostic':str(exc),'program':answer['program']}
                        save(output / 'revision.json', {'reason': revision})
                        continue
                    program = folder / 'program.py'
                    program.write_text(answer['program'])
                    save(proposal_path, {'version': version, 'generation_base': base,
                         'parent_before': state['active'], 'program_sha256': digest(program),
                         'rationale': answer['rationale'], 'attempt': attempt})
                    break
                else:
                    save(folder / 'feedback.json', {'program_generated': False,
                         'task_feedback': [], 'message': '本版本未产生可执行的新程序。'})
                    commit(root, version, None, base, [], False, 'UNEXECUTED')
                    report(root)
                    continue
            proposal = read(proposal_path); program = folder / 'program.py'
            if digest(program) != proposal['program_sha256'] or proposal['generation_base'] != base:
                raise ValueError('frozen proposal or generation base changed')
            futures = [scheduler.pool.submit(scheduler.execute, root, protocol, folder, program, seed)
                       for seed in protocol['seeds']]
            results, errors = [], []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as exc:
                    errors.append(str(exc))
            if errors:
                raise InfrastructureBlocked('; '.join(errors))
            accepted = task_selection(results, protocol['seeds'])
            save(folder / 'feedback.json', {'development_successes': sum(r['task_qualified'] for r in results),
                 'development_cases': len(results), 'cases': [dict(case=i+1, **public_feedback(r))
                                                            for i, r in enumerate(results)]})
            commit(root, version, program, base,
                   [{'seed': r['seed'], 'output': r['output'], 'task_qualified': r['task_qualified'],
                     'program_ok': r['program_ok'], 'native_task_success': r['native_task_success'],
                     'control_steps': r.get('control_steps', 0), 'wall_seconds': r['wall_seconds'],
                     'initial_state_id': r['initial_state_id'], 'status': r['status'],
                     'result_sha256': r.get('result_sha256'),
                     'coverage_complete': (r.get('capture') or {}).get('coverage_complete')} for r in results],
                   accepted, 'MEASURED')
            report(root)
            print(json.dumps({'chain': str(root), 'version': version, 'task_accepted': accepted}), flush=True)
        return report(root)


def report(root):
    root = Path(root); protocol, state = validate(root)
    rows, cases = [], []
    for e in state['events']:
        rows.append({k: e[k] for k in ['version', 'status', 'accepted', 'program_sha256']}
                    | {'generation_base': (e['generation_base'] or {}).get('version'),
                       'parent_before': (e['parent_before'] or {}).get('version'),
                       'parent_after': (e['parent_after'] or {}).get('version'),
                       'development_successes': sum(c['task_qualified'] for c in e['cases'])})
        cases += [dict(version=e['version'], **c) for c in e['cases']]
    for name, data in [('candidates.csv', rows), ('episodes.csv', cases)]:
        if data:
            with (root / name).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    summary = {'schema': 'esrb.natural_summary/1', 'chain_id': protocol['chain_id'],
        'task': protocol['scene']['key'], 'seeds': protocol['seeds'],
        'candidate_budget': protocol['candidate_budget'], 'completed_slots': len(rows),
        'executed_candidates': sum(e['status'] == 'MEASURED' for e in state['events']),
        'development_episodes': len(cases), 'task_accepted': sum(e['accepted'] for e in state['events']),
        'first_qualified_version': next((e['version'] for e in state['events'] if e['accepted']), None),
        'active_task_parent': (state['active'] or {}).get('version'),
        'status': 'COMPLETE' if len(rows) == protocol['candidate_budget'] else 'INCOMPLETE',
        'online_assessment': False, 'hidden_tests_executed': False}
    save(root / 'summary.json', summary)
    if summary['status'] == 'COMPLETE':
        save(root / 'artifacts.json', {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*'))
             if p.is_file() and p.name not in {'artifacts.json', '.natural.lock'}})
    return summary
