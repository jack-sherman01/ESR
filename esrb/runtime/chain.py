"""Single implementation of immutable, feedback-driven fifty-version chains."""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import subprocess

from .batch import BatchRunner
from .feedback import task_errors
from .gate_state import GateState
from .io import read, save, digest as sha
from .paths import project_root
from .submissions import program_identity, wait_file

REPO = project_root()
VERSIONS = 50

class ChainRunner:
    def __init__(self, scene):
        self.scene=scene
        self.runner=BatchRunner(scene)
        self.state=GateState(scene)

    def init(self, root, capsule, seed=None, **settings):
        seed=seed if seed is not None else secrets.randbelow(2**31-1)+1
        if capsule.exists():
            raise ValueError('capsule must be a new directory')
        settings.setdefault('ik_service_url',f'http://127.0.0.1:{self.scene.chain_port}')
        settings.setdefault('gpu_device_id',self.scene.chain_gpu)
        self.runner.init(root,[seed],versions=5,**settings)
        protocol=read(root/'protocol.json')
        protocol.update(schema=f'esrb.{self.scene.prefix}_long_chain/1',versions_per_round=VERSIONS,
                        generator='one fresh worker; fork_turns=none; no initial skill or prior conversation',
                        candidate_prompt=self.scene.prompt,
                        ik_service_url=protocol['runtime']['ik_service_url'],gpu_device_id=protocol['runtime']['gpu_device_id'],
                        duplicate_policy='Before physics, reject matching AST after comments/docstrings are removed; retry same version with an immutable new attempt; do not consume execution budget.')
        save(root/'protocol.json',protocol)
        capsule.mkdir(parents=True,exist_ok=False)
        (capsule/'API.txt').write_bytes((REPO/self.scene.api_path).read_bytes())
        save(root/'worker_contract.json',{'capsule':str(capsule),'fork_turns':'none','initial_program':None,
             'task_prompt':self.scene.prompt,'api_sha256':sha(capsule/'API.txt'),
             'feedback':'same-trial task/public motion/Python and API errors only',
             'submission':'vNN_attemptMM.py then ready_vNN_attemptMM.json; immutable after ready',
             'retry':'revision_vNN_attemptMM.json means submit a different program at next attempt, same version',
             'versions':VERSIONS})
        print(json.dumps({'root':str(root),'seed':seed,'versions':VERSIONS,'capsule':str(capsule)}),flush=True)

    def summarize(self, root):

        state, TASK, VERSIONS = self.state, self.scene.prompt, 50
        feedback, summarize = self.scene.feedback, self.summarize
        PYTHON = read(root / 'protocol.json')['runtime']['python']
        protocol = read(root / 'protocol.json')
        rows = []
        for folder in sorted((root / 'round_01').glob('version_*')):
            if not (folder / 'gate_decision.json').exists():
                continue
            decision = read(folder / 'gate_decision.json')
            result = read(folder / 'execution/result.json')
            rows.append({'version': result['version'], 'seed': result['seed'],
                         'program_ok': result['program_ok'], 'native_task_success': result['native_task_success'],
                         'safety_status': decision['safety_status'], 'decision': decision['decision'],
                         'reason': decision['reason'], 'parent_before': (decision['parent_before'] or {}).get('version'),
                         'parent_after': (decision['parent_after'] or {}).get('version'),
                         'observed_safety_regression': decision['observed_safety_regression'],
                         'control_steps': result['control_steps'], 'wall_seconds': result['wall_seconds'],
                         'initial_state_id': result['initial_state_id'], 'source_sha256': sha(folder / 'program.py'),
                         'ast_sha256': program_identity((folder / 'program.py').read_text()),
                         'coverage_complete': result['safety']['coverage_complete']})
        if rows:
            with (root / 'episodes.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        summary = {'target': VERSIONS, 'measured': len(rows), 'seed': protocol['seeds'][0],
                   'native_successes': sum(r['native_task_success'] for r in rows),
                   'promotions': sum(r['decision'] == 'PROMOTE' for r in rows),
                   'safety_rejections': sum(r['safety_status'] == 'VIOLATION' for r in rows),
                   'unknown': sum(r['safety_status'] == 'UNKNOWN' for r in rows),
                   'observed_safety_regressions': sum(r['observed_safety_regression'] for r in rows),
                   'active_parent': rows[-1]['parent_after'] if rows else None,
                   'unique_source_programs': len({r['source_sha256'] for r in rows}),
                   'unique_ast_programs': len({r['ast_sha256'] for r in rows}),
                   'actual_initial_identities': len({r['initial_state_id'] for r in rows}),
                   'all_sampling_complete': all(r['coverage_complete'] for r in rows),
                   'duplicate_proposals_not_executed': len(list((root / 'round_01/submissions').glob('*/revision.json'))),
                   'independent_retest': False,
                   'status': 'COMPLETE' if len(rows) == VERSIONS else 'RUNNING'}
        save(root / 'summary.json', summary)
        return summary

    def drive(self, root):

        state, TASK, VERSIONS = self.state, self.scene.prompt, 50
        feedback, summarize = self.scene.feedback, self.summarize
        PYTHON = read(root / 'protocol.json')['runtime']['python']
        folder = root / 'round_01'
        wait_file(folder / 'agent_assignment.json')
        assignment = read(folder / 'agent_assignment.json')
        capsule = Path(assignment['capsule'])
        contract = read(root / 'worker_contract.json')
        seen = {}
        for version in range(1, VERSIONS + 1):
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
                result = subprocess.run([str(PYTHON), '-B', '-m', 'esrb', '--execution-worker', 'aspire', self.scene.key, 'chain', 'episode', '--root', str(root), '--version', str(version)], cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
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
