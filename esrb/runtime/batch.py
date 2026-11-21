"""Shared batch construction, parallel dispatch and result bookkeeping."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib import import_module
import json
import os
from pathlib import Path
import secrets
import subprocess
import time

from .evidence import freeze_sources
from .gate_state import GateState
from .io import read, save, digest as sha
from .paths import project_root, robosuite_python

REPO = project_root()

def episode_command(protocol, scene, root, number, version):
    return [protocol['runtime']['python'], '-B', '-m', 'esrb', '--execution-worker', 'aspire', scene.key, 'batch', 'episode',
            '--root', str(root), '--round', str(number), '--version', str(version)]

def child_environment(settings):
    env = os.environ.copy()
    renderer = settings.get('renderer', 'egl')
    env.update(MUJOCO_GL=renderer, PYOPENGL_PLATFORM=renderer, LP_NUM_THREADS='1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    if renderer == 'egl':
        env.update(EGL_PLATFORM='device', MUJOCO_EGL_DEVICE_ID=str(settings.get('gpu_device_id', 0)))
    return env

class BatchRunner:
    def __init__(self, scene):
        self.scene = scene
        self.state = GateState(scene)

    def init(self, root, seeds=None, versions=5, *, python=None, ik_service_url=None, gpu_device_id=None):
        if versions not in range(1, 6):
            raise ValueError('versions per round must be between 1 and 5')
        supplied = seeds is not None
        seeds = list(seeds) if supplied else []
        if supplied and (not seeds or len(seeds)>10 or len(set(seeds))!=len(seeds)
                         or any(type(seed) is not int or not 0<=seed<2**31 for seed in seeds)):
            raise ValueError('expected 1-10 distinct nonnegative 31-bit seeds')
        while not supplied and len(seeds)<10:
            seed = secrets.randbelow(2**31-1)+1
            if seed not in seeds:
                seeds.append(seed)
        root.mkdir(parents=True, exist_ok=False)
        rules = read(REPO / self.scene.rules_path)
        self.scene.validate_rules(rules)
        save(root / 'safety_rules.json', rules)
        protocol = dict(self.scene.protocol_defaults)
        protocol.update(schema=self.scene.protocol_defaults['schema'], task=self.scene.task, environment=self.scene.environment,
                        rounds=len(seeds), versions_per_round=versions, stop_on_success=False, seeds=seeds,
                        seed_selection='explicit frozen panel' if supplied else '10 distinct SystemRandom draws; fixed within round',
                        execution_layout='unified-cli/1', scene=self.scene.document(),
                        runtime={'python':str(Path(python).resolve()) if python else str(robosuite_python()),
                                 'ik_service_url':ik_service_url or self.scene.protocol_defaults.get('ik_service_url', 'http://127.0.0.1:8116'), 'gpu_device_id':gpu_device_id},
                        created_at_utc=datetime.now(timezone.utc).isoformat(),
                        safety_gate={'enabled':True, 'rules_sha256':sha(root/'safety_rules.json'),
                                     'implementation_sha256':freeze_sources(root,self.scene),
                                     'sampling':'every 0.002-second physics step',
                                     'parent_update':'accept only; reject/unknown retain parent'})
        protocol['ik_service_url'] = protocol['runtime']['ik_service_url']
        save(root/'protocol.json', protocol)
        for number, seed in enumerate(seeds, 1):
            folder=root/f'round_{number:02d}'
            folder.mkdir()
            save(folder/'round.json', {'round':number,'seed':seed})
        print(json.dumps({'root':str(root),'task':self.scene.task,'rounds':len(seeds),'versions_per_round':versions,'seeds':seeds}))

    def episode(self, root, number, version):
        with self.state.round_lock(root/f'round_{number:02d}'):
            rules, _chain=self.state.preflight(root,number,version)
            backend=import_module(f'esrb.backends.aspire.episodes.{self.scene.prefix}')
            code=backend.execute(root,number,version,rules)
            decision=self.state.commit(root,number,version)
            print(json.dumps({'gate_decision':decision['decision'],'reason':decision['reason'],
                              'active_version':(decision['parent_after'] or {}).get('version')}), flush=True)
            return code

    def wave(self, root, version, workers, rounds):
        start=time.monotonic()
        protocol=read(root/'protocol.json')
        load_rules, load_chain = self.state.load_rules, self.state.load_chain
        load_rules(root)  # Historical no-gate batches are not extended in place.
        def launch(number):
            folder=root/f'round_{number:02d}'/f'version_{version:02d}'
            if (folder/'execution').exists():
                return {'round':number,'status':'EXISTING_EXECUTION_PRESERVED'}
            if not (folder/'program.py').exists():
                return {'round':number,'status':'MISSING_PROGRAM'}
            env=os.environ.copy()
            settings_path=folder/'execution_settings.json'
            settings=read(settings_path) if settings_path.exists() else {}
            renderer=settings.get('renderer','osmesa')
            env.update(MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',LP_NUM_THREADS='1',OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            env.update(MUJOCO_GL=renderer,PYOPENGL_PLATFORM=renderer)
            if renderer=='egl':
                env.update(EGL_PLATFORM='device',MUJOCO_EGL_DEVICE_ID=str(settings.get('gpu_device_id',0)))
            with (folder/'execution.log').open('w') as log:
                proc=subprocess.run(episode_command(protocol, self.scene, root, number, version),cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            result=read(folder/'execution/result.json') if (folder/'execution/result.json').exists() else {'status':'PROCESS_ERROR'}
            return {'round':number,'returncode':proc.returncode,'status':result['status'],
                    'native_task_success':result.get('native_task_success'),'control_steps':result.get('control_steps'),
                    'wall_seconds':result.get('wall_seconds')}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(launch,n) for n in rounds]):
                print(json.dumps(future.result()),flush=True)
        suffix='' if rounds==list(range(1,11)) else '.rounds_'+'-'.join(map(str,rounds))
        save(root/f'wave_{version:02d}{suffix}.json',{'version':version,'workers':workers,'rounds':rounds,'wall_seconds':time.monotonic()-start,
                                           'protocol_sha256':sha(root/'protocol.json')})
        self.summarize(root)

    def summarize(self, root):
        import csv
        protocol=read(root/'protocol.json')
        load_rules, load_chain = self.state.load_rules, self.state.load_chain
        rows=[]
        for number,seed in enumerate(protocol['seeds'],1):
            for version in range(1,protocol['versions_per_round']+1):
                folder=root/f'round_{number:02d}'/f'version_{version:02d}'
                result=folder/'execution/result.json'
                if result.exists():
                    row=read(result)
                    row['proposal']=read(folder/'proposal.json') if (folder/'proposal.json').exists() else None
                    rows.append(row)
        counts=[{'round':n,'seed':seed,'measured_versions':sum(r['round']==n and r['status']=='MEASURED' for r in rows),
                 'distinct_actual_initial_states':len({r['initial_state_id'] for r in rows if r['round']==n and r['status']=='MEASURED'}),
                 'successful_versions':[r['version'] for r in rows if r['round']==n and r.get('native_task_success')]} for n,seed in enumerate(protocol['seeds'],1)]
        gated=bool((protocol.get('safety_gate') or {}).get('enabled'))
        decisions=[read(p) for p in sorted(root.glob('round_*/version_*/gate_decision.json'))] if gated else []
        for row in counts:
            if gated:
                chain=load_chain(root,row['round'])
                row['active_parent_version']=(chain['active'] or {}).get('version')
        expected=len(protocol['seeds'])*protocol['versions_per_round']
        summary={'schema':f'esrb.{self.scene.prefix}_summary/2' if gated else f'esrb.{self.scene.prefix}_summary/1','requested_rollouts':expected,
                 'measured_rollouts':sum(r['status']=='MEASURED' for r in rows),'native_success_rollouts':sum(bool(r.get('native_task_success')) for r in rows),
                 'infrastructure_errors':sum(r['status']!='MEASURED' for r in rows),
                 'independent_retests':0,'rounds':counts,'executions':rows,
                 'status':'COMPLETE' if len(rows)==expected and all(r['status']=='MEASURED' for r in rows)
                     and (not gated or len(decisions)==len(rows)) else 'IN_PROGRESS'}
        if gated:
            summary.update(gate_decisions=decisions,promotions=sum(d['accepted'] for d in decisions),
                           safety_rejections=sum(d['reason']=='observed_safety_violation' for d in decisions),
                           observed_safety_regressions=sum(d['observed_safety_regression'] for d in decisions))
        save(root/'summary.json',summary)
        fields=['round','seed','version','status','program_ok','native_task_success']
        fields += ['terminal_native_success'] if self.scene.prefix == 'wipe' else ['grasp_observed','lift_observed']
        fields += ['control_steps','wall_seconds','termination_reason']
        with (root/'episodes.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
        print(json.dumps({k:v for k,v in summary.items() if k not in {'executions','rounds'}},indent=2),flush=True)
