"""Resume interrupted Stack V16, then continue the original frozen coordinator logic.

No completed policy is replayed. The original gate/episode/protocol remain frozen.
"""

# Allow direct execution from this subdirectory as well as package imports.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.runners.run_stack_long_chain import *
import shutil


def archive_incomplete_execution(root, version):
    candidate = root / f'round_01/version_{version:02d}'
    execution = candidate / 'execution'
    if (candidate / 'gate_decision.json').exists() or (execution / 'result.json').exists():
        raise ValueError('refuse to archive completed or measured execution')
    target = root / 'interrupted_attempts' / f'version_{version:02d}_launch01'
    target.mkdir(parents=True, exist_ok=False)
    if execution.exists():
        execution.rename(target / 'execution')
    if (candidate / 'execution.log').exists():
        (candidate / 'execution.log').rename(target / 'execution.log')
    save(target / 'interruption.json', {'version': version, 'status': 'INTERRUPTED_NO_RESULT',
         'reason': 'Interactive turn interruption terminated host background processes.',
         'result_present': False, 'gate_decision_present': False,
         'note': 'No recorded physics rows; do not infer that no unflushed computation occurred.'})
    return target


def resume_prepared_v16(root):
    state.load_rules(root)
    chain = state.load_chain(root, 1)
    if len(chain['events']) != 15:
        raise ValueError('expected exactly 15 completed decisions')
    folder = root / 'round_01'
    candidate = folder / 'version_16'
    generation = folder / 'generation_16'
    assignment = read(folder / 'agent_assignment.json')
    capsule = Path(assignment['capsule'])
    source = capsule / 'v16_attempt01.py'
    original = source.read_bytes()
    assert sha(source) == sha(candidate / 'program.py') == read(candidate / 'proposal.json')['source_sha256']
    assert sha(generation / 'agent_input.json') == read(candidate / 'generation.json')['agent_input_sha256']
    archive_incomplete_execution(root, 16)
    env = os.environ.copy()
    env.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', EGL_PLATFORM='device',
               MUJOCO_EGL_DEVICE_ID=str(read(root / 'protocol.json')['gpu_device_id']),
               LP_NUM_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    with (candidate / 'execution.log').open('w') as stream:
        process = subprocess.run([str(PYTHON), '-B', str(REPO / 'scripts/runners/run_stack_long_chain.py'),
                                  'episode', '--root', str(root), '--version', '16'],
                                 cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode or not (candidate / 'gate_decision.json').exists():
        raise RuntimeError('resumed V16 failed; preserve all records')
    assert source.read_bytes() == original
    public = feedback(candidate)
    execution = read(candidate / 'execution/result.json')
    public['api_diagnostics'] = task_errors(Path(execution['native_run']))
    save(generation / 'task_feedback.json', public)
    save(capsule / 'additional_feedback_v16.json', public['api_diagnostics'])
    active = state.load_chain(root, 1)['active']
    base = root / active['program_path'] if active else candidate / 'program.py'
    (capsule / 'starting_v17.py').write_bytes(base.read_bytes())
    save(generation / 'host_provenance.json', {'active_version': (active or {}).get('version'),
                                             'next_starting_sha256': sha(base)})
    summary = summarize(root)
    save(capsule / 'feedback_v16.json', public)
    print(json.dumps(summary), flush=True)


def drive_remaining(root):
    folder = root / 'round_01'
    wait_file(folder / 'agent_assignment.json')
    assignment = read(folder / 'agent_assignment.json')
    capsule = Path(assignment['capsule'])
    contract = read(root / 'worker_contract.json')
    seen = {program_identity((p / 'program.py').read_text()): int(p.name.split('_')[1])
            for p in folder.glob('version_*') if (p / 'gate_decision.json').exists()}
    for version in range(17, VERSIONS + 1):
        for attempt in range(1, 101):
            stem = f'v{version:02d}_attempt{attempt:02d}'
            ready, source = capsule / f'ready_{stem}.json', capsule / f'{stem}.py'
            wait_file(ready)
            wait_file(source)
            original = source.read_bytes()
            if sha(capsule / 'API.txt') != contract['api_sha256']:
                raise ValueError('worker API changed')
            submission = folder / 'submissions' / stem
            submission.mkdir(parents=True, exist_ok=False)
            (submission / 'program.py').write_bytes(original)
            identity = program_identity(original.decode())
            save(submission / 'submission.json', {'version': version, 'attempt': attempt,
                 'source_sha256': sha(source), 'ast_sha256': identity, 'ready': read(ready)})
            if identity not in seen:
                break
            revision = {'version': version, 'attempt': attempt,
                        'message': '程序与此前已测程序相同（忽略注释及说明文字）。请实际修改技能，以下一 attempt 提交本版本；本次未运行。'}
            save(submission / 'revision.json', revision)
            save(capsule / f'revision_{stem}.json', revision)
        else:
            raise ValueError('excessive duplicate proposals')
        state.load_rules(root)
        generation = folder / f'generation_{version:02d}'
        generation.mkdir()
        initial = {'task_prompt': TASK, 'api': (capsule / 'API.txt').read_text(), 'iteration': version,
                   'starting_program': (capsule / f'starting_v{version:02d}.py').read_text() if version > 1 else None,
                   'previous_task_feedback': read(capsule / f'feedback_v{version-1:02d}.json') if version > 1 else None}
        save(generation / 'agent_input.json', initial)
        (generation / 'agent_output.py').write_bytes(original)
        save(generation / 'agent_output.json', read(submission / 'submission.json'))
        with state.round_lock(folder):
            chain = state.load_chain(root, 1)
            if version != len(chain['events']) + 1:
                raise ValueError('version sequence mismatch')
            active = chain['active']
            candidate = folder / f'version_{version:02d}'
            candidate.mkdir()
            (candidate / 'program.py').write_bytes(original)
            (candidate / 'candidate_prompt.txt').write_text(TASK + '\n' + initial['api'])
            save(candidate / 'task_feedback.json', initial['previous_task_feedback'])
            save(candidate / 'proposal.json', {'round': 1, 'version': version,
                 'gate_base': {'version': active['version'], 'program_sha256': active['program_sha256']} if active else None,
                 'source_sha256': sha(source), 'kind': 'candidate', 'rationale': read(ready).get('rationale', ''),
                 'candidate_prompt_sha256': sha(candidate / 'candidate_prompt.txt'),
                 'task_feedback_sha256': sha(candidate / 'task_feedback.json'),
                 'parent_program': active['program_path'] if active else None})
            save(candidate / 'execution_settings.json', {'renderer': 'egl', 'gpu_device_id': read(root / 'protocol.json')['gpu_device_id'],
                 'camera_sampling_hz': 5, 'fresh_explicit_observation': True})
        save(candidate / 'generation.json', {'agent_id': assignment['agent_id'], 'fork_turns': 'none',
             'initial_program_provided': False, 'attempt': attempt, 'api_sha256': contract['api_sha256'],
             'agent_input_sha256': sha(generation / 'agent_input.json'), 'controlled_injection': False})
        env = os.environ.copy()
        env.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', EGL_PLATFORM='device', MUJOCO_EGL_DEVICE_ID=str(read(root / 'protocol.json')['gpu_device_id']),
                   LP_NUM_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        with (candidate / 'execution.log').open('w') as stream:
            result = subprocess.run([str(PYTHON), '-B', str(REPO / 'scripts/runners/run_stack_long_chain.py'), 'episode', '--root', str(root),
                                     '--version', str(version)], cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode or not (candidate / 'gate_decision.json').exists():
            raise RuntimeError(f'episode {version} failed; inspect execution.log')
        if source.read_bytes() != original:
            raise ValueError('committed submission changed')
        seen[identity] = version
        public = feedback(candidate)
        execution = read(candidate / 'execution/result.json')
        public['api_diagnostics'] = task_errors(Path(execution['native_run']))
        save(generation / 'task_feedback.json', public)
        save(capsule / f'additional_feedback_v{version:02d}.json', public['api_diagnostics'])
        if version < VERSIONS:
            active = state.load_chain(root, 1)['active']
            base = root / active['program_path'] if active else candidate / 'program.py'
            (capsule / f'starting_v{version+1:02d}.py').write_bytes(base.read_bytes())
            save(generation / 'host_provenance.json', {'active_version': (active or {}).get('version'), 'next_starting_sha256': sha(base)})
        summary = summarize(root)
        save(capsule / f'feedback_v{version:02d}.json', public)
        print(json.dumps(summary), flush=True)
    save(folder / 'fresh_agent_complete.json', {'status': 'COMPLETE', 'measured': VERSIONS,
         'agent_id': assignment['agent_id'], 'fork_turns': 'none', 'finished_at_utc': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    root = parser.parse_args().root.resolve()
    expected = read(root / 'resume_coordinator_manifest.json')['source_sha256']
    assert sha(Path(__file__)) == expected
    resume_prepared_v16(root)
    drive_remaining(root)
