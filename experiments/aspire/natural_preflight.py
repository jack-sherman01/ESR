"""Separate engineering accounting and measured concurrency calibration."""
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time
import threading
import statistics
import urllib.request

from esrb.runtime.io import read, save, digest
from esrb.runtime.natural import Scheduler, initialize, DIRECTORIES, validate
from esrb.runtime.paths import project_root
from .scenes import get_scene
from .natural_campaign import TASKS, select_parallelism


def engineering_protocol(scene, url):
    root = project_root()
    return {'scene': scene.document(), 'ik_service_url': url,
            'max_control_steps': 2000 if scene.prefix == 'handover' else 1000,
            'camera_width': 192 if scene.prefix == 'handover' else 128,
            'camera_height': 192 if scene.prefix == 'handover' else 128,
            'video_stride': 4, 'python': str(root / '.esr-local/aspire/checkout/.venv-robosuite/bin/python'),
            'episode_timeout_s': 900, 'max_infrastructure_attempts': 1}


def held_joints(task):
    grip = 'None' if task == 'wipe' else 'state.gripper_positions[arm]'
    return ("state = get_robot_state()\nfor tick in range(3):\n    arms = {}\n"
            "    for arm in state.joint_positions:\n        arms[arm] = ArmCommand(mode='joint_position', "
            f"target=state.joint_positions[arm], gripper_position={grip})\n"
            "    step(RobotAction(arms=arms))\n")


def existing_skill(task):
    base = project_root() / '.esr-local' / DIRECTORIES[task]
    choices = sorted(base.glob('one-round-versions50-esr-fresh-*/round_01/version_50/program.py'))
    if not choices:
        raise ValueError(f'no existing engineering skill available for {task}')
    return choices[-1]


def one_case(base, scheduler, task, protocol, seed, source, tag):
    folder = base / task / tag; folder.mkdir(parents=True, exist_ok=True)
    references = folder / 'initial_references'; references.mkdir(exist_ok=True)
    candidate = folder / 'candidate'; candidate.mkdir(exist_ok=True)
    program = candidate / 'program.py'
    if program.exists() and program.read_text() != source:
        raise ValueError('engineering source changed')
    if not program.exists():
        program.write_text(source)
    return scheduler.execute(folder, protocol, candidate, program, seed)


def execute(root, gpus, action):
    manifest = read(root / 'manifest.json'); engineering = root / 'engineering'; engineering.mkdir(exist_ok=True)
    urls = sorted({read(Path(e['root']) / 'protocol.json')['ik_service_url'] for e in manifest['chains']})
    for url in urls:
        with urllib.request.urlopen(url + '/healthz', timeout=5) as response:
            if response.status != 200:
                raise ValueError(f'IK service unavailable: {url}')
    for item in manifest['chains']:
        validate(Path(item['root']))
    protocols = {task: engineering_protocol(get_scene(task), urls[i % len(urls)]) for i, task in enumerate(TASKS)}
    structural_path = engineering / 'structural.json'
    if not structural_path.exists():
        scheduler = Scheduler(4, gpus)
        def structural(task):
            records = []
            # Reuse the first five real three-tick checks; they count toward the 100-episode cap.
            first = project_root()/'.esr-local/_checks/natural-dev5/first-native-smoke'/task/'v01/seed_1001/attempt_01/execution'
            if (first/'result.json').exists():
                records.append(dict(read(first/'result.json'), output=str(first)))
            for index, seed in enumerate(([1001,1002] if records else [1001,1001,1002])):
                records.append(one_case(engineering / 'structural', scheduler, task, protocols[task], seed,
                                        held_joints(task), f'reset_{index+1}'))
            if records[0]['initial_state_id'] != records[1]['initial_state_id']:
                raise ValueError(f'non-reproducible initial state: {task}')
            if records[0]['initial_state_id'] == records[2]['initial_state_id']:
                raise ValueError(f'initial state does not vary with seed: {task}')
            return {'task': task, 'records': records, 'same_seed_match': True, 'different_seed_changes': True}
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                rows = list(pool.map(structural, TASKS))
            save(structural_path, {'status': 'PASS', 'episodes': 15, 'tasks': rows})
        finally:
            scheduler.pool.shutdown(wait=True)
    warm_path = engineering / 'warmup.json'
    if not warm_path.exists():
        scheduler = Scheduler(4, gpus)
        try:
            jobs = [scheduler.pool.submit(one_case, engineering / 'warmup', scheduler, task,
                         protocols[task], 9001, existing_skill(task).read_text(), 'existing_skill') for task in TASKS]
            rows = [job.result() for job in jobs]
            save(warm_path, {'episodes': 5, 'records': rows})
        finally:
            scheduler.pool.shutdown(wait=True)
    profiles = []
    aborted_path = engineering/'aborted_api_calibration.json'
    spent = read(aborted_path)['started_episodes'] if aborted_path.exists() else 0
    cases_per_profile = min(20,(100-20-spent)//4)
    if cases_per_profile<16:
        raise ValueError('engineering budget cannot cover all four concurrency levels at full occupancy')
    case_plan = [(task,repetition) for repetition in range(1,5) for task in TASKS][:cases_per_profile]
    calibration_root = engineering/('calibration_after_api_fix' if spent else 'calibration')
    for workers in [4,8,12,16]:
        profile_path = calibration_root/f'parallel_{workers}.json'
        profile_path.parent.mkdir(exist_ok=True)
        if profile_path.exists():
            profiles.append(read(profile_path)); continue
        scheduler = Scheduler(workers, gpus); start = time.monotonic(); results = []; errors = []
        stop_sampling=threading.Event();gpu_samples=[]
        def sample_gpus():
            while not stop_sampling.is_set():
                try:
                    sample=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                    gpu_samples.append({'elapsed_s':time.monotonic()-start,'csv':sample.stdout})
                except (OSError,subprocess.TimeoutExpired) as exc:
                    gpu_samples.append({'elapsed_s':time.monotonic()-start,'error':str(exc)})
                stop_sampling.wait(2)
        sampling=threading.Thread(target=sample_gpus,daemon=True);sampling.start()
        jobs = []
        for task,repetition in case_plan:
            skill=existing_skill(task)
            seed=read(skill.parents[2]/'protocol.json')['seeds'][0]
            jobs.append(scheduler.pool.submit(one_case,calibration_root/f'parallel_{workers}',scheduler,
                task,protocols[task],seed,skill.read_text(),f'repetition_{repetition}'))
        for job in jobs:
            try:
                result=job.result()
                if result.get('status')!='MEASURED':
                    raise ValueError(f"calibration execution unavailable: {result.get('status')}")
                results.append(result)
            except Exception as exc:
                errors.append(str(exc))
        scheduler.pool.shutdown(wait=True)
        stop_sampling.set();sampling.join(timeout=6)
        elapsed = time.monotonic()-start; steps = sum(r['capture']['steps'] for r in results)
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                              '--format=csv'], capture_output=True, text=True)
        profile = {'workers': workers, 'episodes': cases_per_profile, 'completed': len(results), 'errors': errors,
             'infrastructure_errors': len(errors), 'wall_seconds': elapsed,
             'physics_steps': steps, 'physics_steps_per_second': steps/elapsed,
             'episode_wall_seconds': [r['wall_seconds'] for r in results],
             'episode_p95_wall_seconds':statistics.quantiles([r['wall_seconds'] for r in results],n=20)[18] if len(results)>1 else None,
             'gpu_snapshot': gpu.stdout,'gpu_samples':gpu_samples, 'results': results}
        save(profile_path, profile); profiles.append(profile)
        print(json.dumps({k:v for k,v in profile.items() if k not in {'results','gpu_snapshot','gpu_samples'}}), flush=True)
    selected = select_parallelism(profiles)
    save(root / 'calibration.json', {'profiles': profiles, 'selected_workers': selected,
                                    'engineering_episode_budget': 100,'excluded_prior_episodes':spent})
    all_results = read(warm_path)['records'] + [r for p in profiles for r in p['results']]
    ratios = []
    per_scene = {}
    for result in all_results:
        path = Path(result['output']); size = sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
        steps = result['capture']['steps']
        if steps:
            request = read(path.parent / 'request.json'); task = request['scene']['key']
            value = size/steps
            per_scene[task] = max(per_scene.get(task, 0), value)
            ratios.append(value)
    # Bound formal episode storage by full control horizon plus a conservative overhead.
    estimates = {task: int(1.3*per_scene[task]*(50000 if task=='two-arm-handover' else 25000)) for task in TASKS}
    projected = sum(estimates[e['task']]*read(Path(e['root'])/'protocol.json')['candidate_budget']*5
                    for e in manifest['chains'])
    free = shutil.disk_usage(root).free
    model_probe = project_root()/'.esr-local/_checks/natural-dev5/model-probe'
    receipt = read(model_probe/'model_receipt.json')
    model = read(Path(manifest['chains'][0]['root'])/'protocol.json')['model']
    if receipt['requested_model'] != model:
        raise ValueError('model availability check used a different model')
    passed = free >= projected + 20*1024**3
    report = {'status': 'PASS' if passed else 'STORAGE_BLOCKED',
        'structural_sha256': digest(structural_path), 'calibration_sha256': digest(root/'calibration.json'),
        'model_probe': str(model_probe), 'model_receipt_sha256': digest(model_probe/'model_receipt.json'),
        'engineering_episodes': 20+spent+4*cases_per_profile, 'storage': {'bytes_per_episode': max(estimates.values()),
            'per_scene_bytes_per_episode': estimates, 'projected_bytes': projected,
            'available_bytes': free, 'reserve_bytes': 20*1024**3}}
    save(root/'preflight.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if passed else 1
