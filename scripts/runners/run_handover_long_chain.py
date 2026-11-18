"""One fresh worker, one seed and fifty immutable TwoArmHandover candidates."""
from __future__ import annotations

# Allow direct execution from this subdirectory as well as package imports.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import ast
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
from scripts.gates import handover_gate_state as state
from scripts.runners import run_aspire_handover_batch as runner
from scripts.runners.run_fresh_agent_trials import wait_file
from scripts.feedback.handover_task_feedback import feedback
from scripts.feedback.watch_fresh_agent_api_errors_v2 import task_errors

PYTHON = REPO / '.esr-local/aspire/checkout/.venv-robosuite/bin/python'
TASK = '用双臂 Panda 把锤子从 primary 交给 secondary，尽量可靠且少用动作。'
VERSIONS = 50


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def program_identity(source):
    """Ignore comments and docstrings, so cosmetic text cannot satisfy novelty."""
    class RemoveDocstrings(ast.NodeTransformer):
        def visit(self, node):
            node = super().visit(node)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                    node.body = node.body[1:]
            return node
    try:
        tree = RemoveDocstrings().visit(ast.parse(source))
        canonical = ast.dump(tree, include_attributes=False)
    except SyntaxError:
        canonical = source.strip()
    return hashlib.sha256(canonical.encode()).hexdigest()


def init(root, capsule, seed=None):
    seed = seed if seed is not None else secrets.randbelow(2**31 - 1) + 1
    with contextlib.redirect_stdout(io.StringIO()):
        runner.init(root, [seed], versions=5)
    # Construction is complete only after these long-chain fields are frozen.
    protocol = read(root / 'protocol.json')
    protocol.update(schema='esrb.handover_long_chain/1', versions_per_round=VERSIONS,
                    generator='one fresh worker; fork_turns=none; no initial skill or prior conversation',
                    candidate_prompt=TASK, ik_service_url='http://127.0.0.1:8130', gpu_device_id=2,
                    duplicate_policy='Before physics, reject matching AST after comments/docstrings are removed; retry same version with an immutable new attempt; do not consume execution budget.')
    extra_sources = ['scripts/runners/run_handover_long_chain.py', 'scripts/runners/run_fresh_agent_trials.py',
                     'scripts/feedback/watch_fresh_agent_api_errors.py', 'scripts/feedback/watch_fresh_agent_api_errors_v2.py',
                     'integrations/aspire/handover_skill_api.txt', 'integrations/aspire/aspire_pyroki_service.py', 'scripts/runners/run_long_chain_job.py']
    protocol['safety_gate']['implementation_sha256'].update({name: sha(REPO / name) for name in extra_sources})
    save(root / 'protocol.json', protocol)
    capsule.mkdir(parents=True, exist_ok=False)
    api = REPO / 'integrations/aspire/handover_skill_api.txt'
    (capsule / 'API.txt').write_bytes(api.read_bytes())
    save(root / 'worker_contract.json', {'capsule': str(capsule), 'fork_turns': 'none',
         'initial_program': None, 'task_prompt': TASK, 'api_sha256': sha(capsule / 'API.txt'),
         'feedback': 'same-trial task/public motion/Python and API errors only',
         'submission': 'vNN_attemptMM.py then ready_vNN_attemptMM.json; immutable after ready',
         'retry': 'revision_vNN_attemptMM.json means submit a different program at next attempt, same version',
         'versions': VERSIONS})
    print(json.dumps({'root': str(root), 'seed': seed, 'versions': VERSIONS, 'capsule': str(capsule)}), flush=True)


def summarize(root):
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


def drive(root):
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
            result = subprocess.run([str(PYTHON), '-B', str(Path(__file__).resolve()), 'episode', '--root', str(root),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init', 'episode', 'drive', 'summarize'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--capsule', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--version', type=int, choices=range(1, VERSIONS + 1))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'init':
        if args.capsule is None:
            parser.error('init requires --capsule')
        init(root, args.capsule.resolve(), args.seed)
    elif args.action == 'episode':
        if args.version is None:
            parser.error('episode requires --version')
        sys.exit(runner.episode(root, 1, args.version))
    elif args.action == 'drive':
        drive(root)
    else:
        print(json.dumps(summarize(root), indent=2))
