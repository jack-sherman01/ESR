"""Bridge external, initially empty workers to immutable short-batch candidates."""
import subprocess
from .batch import BatchRunner, child_environment, episode_command
from .feedback import task_errors
from .io import read, save, digest
from .paths import project_root
from .submissions import wait_file


def drive_round(scene, root, number):
    runner = BatchRunner(scene)
    state = runner.state
    protocol = read(root / 'protocol.json')
    folder = root / f'round_{number:02d}'
    wait_file(folder / 'agent_assignment.json')
    assignment = read(folder / 'agent_assignment.json')
    from pathlib import Path
    capsule = Path(assignment['capsule'])
    api_hash = digest(capsule / 'API.txt')
    budget = protocol['versions_per_round']
    for version in range(1, budget + 1):
        source = capsule / f'v{version:02d}.py'
        ready = capsule / f'ready_v{version:02d}.json'
        wait_file(ready)
        wait_file(source)
        original = source.read_bytes()
        if digest(capsule / 'API.txt') != api_hash:
            raise ValueError('API input changed during round')
        generation = folder / f'generation_{version:02d}'
        generation.mkdir(exist_ok=False)
        initial = {'task_prompt': assignment['task_prompt'], 'api': (capsule / 'API.txt').read_text(), 'iteration': version,
                   'starting_program': (capsule / f'starting_v{version:02d}.py').read_text() if version > 1 else None,
                   'previous_task_feedback': read(capsule / f'feedback_v{version-1:02d}.json') if version > 1 else None}
        scene.validate_prompt(__import__('json').dumps(initial))
        save(generation / 'agent_input.json', initial)
        (generation / 'agent_output.py').write_bytes(original)
        save(generation / 'agent_output.json', {'rationale': read(ready).get('rationale', ''), 'source_sha256': digest(source)})
        state.load_rules(root)
        with state.round_lock(folder):
            chain = state.load_chain(root, number)
            if version != len(chain['events']) + 1:
                raise ValueError('version ordering mismatch')
            active = chain['active']
            candidate = folder / f'version_{version:02d}'
            candidate.mkdir(exist_ok=False)
            (candidate / 'program.py').write_bytes(original)
            (candidate / 'candidate_prompt.txt').write_text(initial['task_prompt'] + '\n' + initial['api'])
            save(candidate / 'task_feedback.json', initial['previous_task_feedback'])
            save(candidate / 'proposal.json', {'round': number, 'version': version,
                 'gate_base': {'version': active['version'], 'program_sha256': active['program_sha256']} if active else None,
                 'source_sha256': digest(source), 'rationale': read(ready).get('rationale', ''), 'kind': 'candidate',
                 'candidate_prompt_sha256': digest(candidate / 'candidate_prompt.txt'),
                 'task_feedback_sha256': digest(candidate / 'task_feedback.json'),
                 'parent_program': active['program_path'] if active else None})
            settings = {'renderer': 'egl', 'gpu_device_id': protocol['runtime']['gpu_device_id'] if protocol['runtime']['gpu_device_id'] is not None else (number-1) % 4,
                        'camera_sampling_hz': 5, 'fresh_explicit_observation': True}
            save(candidate / 'execution_settings.json', settings)
        save(candidate / 'generation.json', {'agent_id': assignment['agent_id'], 'fork_turns': 'none',
             'initial_program_provided': False, 'api_sha256': api_hash, 'agent_input_sha256': digest(generation / 'agent_input.json'),
             'agent_input_path': str((generation / 'agent_input.json').relative_to(root)), 'controlled_injection': False})
        with (candidate / 'execution.log').open('w') as log:
            result = subprocess.run(episode_command(protocol, scene, root, number, version), cwd=project_root(),
                                    env=child_environment(settings), stdout=log, stderr=subprocess.STDOUT)
        if result.returncode or not (candidate / 'gate_decision.json').exists():
            raise RuntimeError(f'episode process failed: {number}/{version}; inspect execution.log')
        if source.read_bytes() != original:
            raise ValueError('submitted program changed during execution')
        public = scene.feedback(candidate)
        native = read(candidate / 'execution/result.json').get('native_run')
        if native:
            public['api_diagnostics'] = task_errors(Path(native))
        save(generation / 'task_feedback.json', public)
        if version < budget:
            active = state.load_chain(root, number)['active']
            base = root / active['program_path'] if active else candidate / 'program.py'
            (capsule / f'starting_v{version+1:02d}.py').write_bytes(base.read_bytes())
            save(generation / 'host_provenance.json', {'active_version': (active or {}).get('version'), 'next_starting_sha256': digest(base)})
        save(capsule / f'feedback_v{version:02d}.json', public)
    save(folder / 'fresh_agent_complete.json', {'status': 'COMPLETE', 'measured': budget, 'agent_id': assignment['agent_id'], 'fork_turns': 'none'})
    return number
