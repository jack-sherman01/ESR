"""Fifteen chains, one shared simulator queue, independent explicit model histories."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import threading
import time

from esrb.runtime.io import read, save, digest
from esrb.runtime.paths import project_root

TASKS = ['stack', 'pickplacecan', 'two-arm-handover', 'nut-assembly', 'wipe']


def init(root, batch, *, chains=3, candidates=50, model='gpt-6.1-sol', ik_urls=None):
    from esrb.runtime.natural import initialize, DIRECTORIES
    from .scenes import get_scene
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False)
    ik_urls = ik_urls or ['http://127.0.0.1:8140', 'http://127.0.0.1:8141']
    entries = []
    for task in TASKS:
        for index in range(1, chains+1):
            path = project_root() / '.esr-local' / DIRECTORIES[task] / f'natural-seeds1to5-{batch}' / f'chain_{index:02d}'
            initialize(path, get_scene(task), model=model, budget=candidates,
                       ik_url=ik_urls[len(entries) % len(ik_urls)], chain=index)
            entries.append({'task': task, 'chain': index, 'root': str(path)})
    manifest = {'schema': 'esrb.natural_campaign/1', 'batch': batch, 'chains': entries,
                'seeds': [1,2,3,4,5], 'candidate_budget': len(entries)*candidates,
                'development_episode_budget': len(entries)*candidates*5,
                'created_at_utc': datetime.now(timezone.utc).isoformat()}
    save(root / 'manifest.json', manifest)
    return manifest


def select_parallelism(rows):
    valid = [r for r in rows if r['infrastructure_errors'] == 0 and r['physics_steps_per_second'] > 0]
    if not valid:
        raise ValueError('no concurrency profile passed infrastructure checks')
    best = max(r['physics_steps_per_second'] for r in valid)
    return min(r['workers'] for r in valid if r['physics_steps_per_second'] >= .95*best)


def status(root):
    manifest = read(Path(root) / 'manifest.json'); rows = []
    for item in manifest['chains']:
        path = Path(item['root']) / 'summary.json'
        rows.append(dict(item, **(read(path) if path.exists() else {'status': 'NOT_STARTED', 'completed_slots': 0,
                              'development_episodes': 0, 'task_accepted': 0})))
    return {'batch': manifest['batch'], 'chains': rows,
            'completed_slots': sum(r['completed_slots'] for r in rows),
            'development_episodes': sum(r['development_episodes'] for r in rows),
            'task_accepted': sum(r['task_accepted'] for r in rows),
            'complete': all(r['status'] == 'COMPLETE' for r in rows)}


def run(root, workers=None, gpus=(3,2), model_workers=3):
    from esrb.runtime.natural import Scheduler, drive, exclusive
    root = Path(root).resolve()
    with exclusive(root):
        preflight = read(root / 'preflight.json')
        if preflight.get('status') != 'PASS':
            raise ValueError('campaign preflight must pass before formal model generation')
        calibration = read(root / 'calibration.json')
        selected = select_parallelism(calibration['profiles'])
        if workers is not None and workers != selected:
            raise ValueError('workers must match the measured concurrency selection')
        estimate = preflight['storage']['bytes_per_episode']
        per_scene = preflight['storage']['per_scene_bytes_per_episode']
        remaining_bytes = 0
        for item in read(root/'manifest.json')['chains']:
            protocol = read(Path(item['root'])/'protocol.json')
            state = read(Path(item['root'])/'state.json')
            remaining_bytes += (protocol['candidate_budget']-len(state['events']))*5*per_scene[item['task']]
        if shutil.disk_usage(root).free < 20*1024**3 + remaining_bytes:
            raise ValueError('insufficient projected storage for remaining development episodes')
        scheduler = Scheduler(selected, gpus, bytes_per_episode=estimate)
        model_semaphore = threading.Semaphore(model_workers)
        manifest = read(root / 'manifest.json'); failures = []
        save(root / 'execution_config.json', {'workers': selected, 'gpus': list(gpus), 'model_workers': model_workers,
             'calibration_sha256': digest(root / 'calibration.json'), 'preflight_sha256': digest(root / 'preflight.json')})
        with ThreadPoolExecutor(max_workers=len(manifest['chains'])) as pool:
            jobs = {pool.submit(drive, Path(item['root']), scheduler, model_semaphore=model_semaphore): item
                    for item in manifest['chains']}
            for job in as_completed(jobs):
                item = jobs[job]
                try:
                    job.result()
                except Exception as exc:
                    failures.append({'chain': item, 'type': type(exc).__name__, 'error': str(exc)})
                    save(Path(item['root']) / 'blocked.json', failures[-1])
                save(root / 'status.json', status(root))
        scheduler.pool.shutdown(wait=True)
        save(root / 'completion.json', {'status': 'BLOCKED' if failures else 'COMPLETE',
                                      'failures': failures, 'summary': status(root)})
        return 1 if failures else 0


def main(args):
    if args.action == 'init':
        value = init(args.root, args.batch, chains=args.chains, candidates=args.candidates,
                     model=args.model, ik_urls=args.ik_urls)
    elif args.action in {'drive', 'resume'}:
        return run(args.root, args.workers, args.gpus, args.model_workers)
    elif args.action in {'preflight', 'calibrate'}:
        from .natural_preflight import execute
        return execute(args.root.resolve(), args.gpus, args.action)
    else:
        value = status(args.root)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


def add_parser(tasks):
    actions = tasks.add_parser('natural-campaign', help='Prepare and run a task-only five-scene campaign').add_subparsers(dest='action', required=True)
    for action in ['init', 'preflight', 'calibrate', 'drive', 'resume', 'status']:
        leaf = actions.add_parser(action)
        leaf.set_defaults(_module='experiments.aspire.natural_campaign', _route=['aspire', 'natural-campaign', action], _leaf=leaf)
        leaf.add_argument('--root', type=Path, required=True)
        if action == 'init':
            leaf.add_argument('--batch', required=True)
            leaf.add_argument('--chains', type=int, default=3)
            leaf.add_argument('--candidates', type=int, default=50)
            leaf.add_argument('--model', default='gpt-6.1-sol')
            leaf.add_argument('--ik-urls', nargs='+')
        elif action in {'preflight', 'calibrate', 'drive', 'resume'}:
            leaf.add_argument('--gpus', nargs='+', type=int, default=[3,2])
            if action in {'drive', 'resume'}:
                leaf.add_argument('--workers', type=int)
                leaf.add_argument('--model-workers', type=int, default=3)
