"""Twelve-scene controlled measurement acceptance; unified CLI only.

Each phase is append-only. Code/config changes require a new revision directory;
failed episodes are retained and never silently retried or relabelled.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import shutil

from esrb.measurement.spec import SCENES,NATIVE,JAM,cases,primary,protocol,case_status
from esrb.runtime.paths import project_root


def read(p):return json.loads(Path(p).read_text())


def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(p)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@contextmanager
def execution_lock(root):
    """A report may read progress; two coordinators may not mutate one run."""
    import fcntl
    with (Path(root)/'.coordinator.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('another coordinator is executing this frozen run') from None
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def source_files():
    repo=project_root()
    files=[p for directory in ('esrb','experiments','configs','integrations','examples')
                   for p in (repo/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for directory in ('.esr-local/aspire/checkout/src/cap_harness',
                      '.esr-local/aspire/checkout/third_party/robosuite/robosuite'):
        files.extend(p for p in (repo/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                     and p.suffix not in ('.pyc','.log'))
    return sorted(files+[repo/'pyproject.toml'])


def source_hashes():
    repo=project_root()
    return {str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files()}


class Study:
    def __init__(self,args):
        self.args=args;self.root=args.out_dir.resolve();self.repo=project_root()
        self.workers=args.workers
        if not 1<=args.workers<=8:raise ValueError('workers must be between 1 and 8')
        import math
        if not math.isfinite(args.episode_timeout) or args.episode_timeout<=0:raise ValueError('episode timeout must be finite and positive')
        if args.scenes and len(set(args.scenes))!=len(args.scenes):raise ValueError('scenes must be distinct')
        if self.root.exists():
            self.protocol=read(self.root/'protocol.json')
            frozen=read(self.root/'source_hashes.json');source=Path(self.protocol.get('source_root',self.repo))
            if args.phase!='report' and any(not (source/p).is_file() or hashlib.sha256((source/p).read_bytes()).hexdigest()!=value for p,value in frozen.items()):
                raise ValueError('frozen sources changed; create a NEW revision directory and retain this run')
            if args.scenes and list(args.scenes)!=self.protocol['scenes']:
                raise ValueError('scene selection differs from the frozen protocol')
        else:
            if args.phase not in ('all','calibrate'):raise ValueError('new runs must start with calibrate or all')
            self.root.mkdir(parents=True)
            self.protocol=protocol(args.scenes or SCENES,args.workers)
            self.protocol['episode_timeout_s']=args.episode_timeout
            self.protocol['source_sha256']=digest(source_hashes())
            self.protocol['source_root']=str(self.root/'source')
            self.protocol['publication_root']=str(self.repo)
            self.protocol['interpreters']={
                # Resolving the executable symlink escapes pyvenv.cfg and loses
                # the environment's installed dependencies. Keep the venv path.
                'selfbuilt':str((args.selfbuilt_python or self.repo/'.venv/bin/python').absolute()),
                'aspire':str((args.aspire_python or self.repo/'.esr-local/aspire/checkout/.venv-robosuite/bin/python').absolute())}
            self.protocol['prechange_archive']=Path('/tmp/esrb-measurement-archive-path').read_text().strip() if Path('/tmp/esrb-measurement-archive-path').exists() else None
            self.protocol['created_utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
            self.protocol['case_budgets']={'calibration':3,'main':30,'diagnostic':24,'replay':2,
                'additional_per_scene':{s:12*(len(cases(s))-2) for s in self.protocol['scenes']}}
            save(self.root/'source_hashes.json',source_hashes())
            with tarfile.open(self.root/'sources.tar.gz','w:gz',compresslevel=1,dereference=True) as archive:
                for p in source_files():archive.add(p,arcname=str(p.relative_to(self.repo)),recursive=False)
            with tarfile.open(self.root/'sources.tar.gz','r:gz') as archive:
                archive.extractall(self.root/'source',**({'filter':'data'} if hasattr(tarfile,'data_filter') else {}))
            save(self.root/'protocol.json',self.protocol)
        self.scenes=self.protocol['scenes']
        self.args.episode_timeout=self.protocol.get('episode_timeout_s',args.episode_timeout)

    def interpreter(self,scene):
        return self.protocol['interpreters']['aspire' if scene in NATIVE else 'selfbuilt']

    def subprocess(self,request,scene,log):
        python=self.interpreter(scene)
        if not Path(python).is_file():return dict(status='BLOCKED',error='missing interpreter: '+python)
        env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                                        NUMEXPR_NUM_THREADS='1',MUJOCO_GL='osmesa',PYTHONDONTWRITEBYTECODE='1')
        env['PYTHONPATH']=str(self.repo)+os.pathsep+env.get('PYTHONPATH','')
        env['PYTHONPATH']=os.pathsep.join([str(self.repo),
            str(self.repo/'.esr-local/aspire/checkout/src'),
            str(self.repo/'.esr-local/aspire/checkout/third_party/robosuite'),env.get('PYTHONPATH','')])
        command=[python,'-B','-c','import sys; from esrb.measurement.episodes import worker; worker(sys.argv[1])',str(request)]
        with Path(log).open('wb') as stream:
            try:
                proc=subprocess.run(command,cwd=self.repo,env=env,stdout=stream,stderr=subprocess.STDOUT,
                                    timeout=self.args.episode_timeout)
                return dict(returncode=proc.returncode,status='FINISHED' if proc.returncode==0 else 'BLOCKED')
            except subprocess.TimeoutExpired:
                return dict(status='BLOCKED',error='infrastructure timeout; partial raw evidence retained')

    def episode(self,phase,scene,case,seed,tag='',program=None,reference=True):
        key=f'{case}-seed{seed:04d}'+('-'+tag if tag else '')
        folder=self.root/phase/scene/key
        result_path=folder/'result.json'
        if result_path.exists():return read(result_path)
        if shutil.disk_usage(self.root).free<5*1024**3:
            result=dict(status='BLOCKED',scene=scene,case=case,seed=seed,task_success=False,
                        error='less than 5 GiB free; no rollout started',file=str(folder.relative_to(self.root)))
            save(result_path,result);return result
        if folder.exists():
            # Never erase/retry a crashed worker in place.
            result=dict(status='BLOCKED',scene=scene,case=case,seed=seed,
                        error='unfinished prior attempt retained',file=str(folder.relative_to(self.root)))
            save(result_path,result)
            return result
        request_dir=self.root/'requests'/phase/scene;request_dir.mkdir(parents=True,exist_ok=True)
        limits_path=self.root/'calibration'/scene/'limits.json'
        program=program or dict(scene=scene,case=case,source_sha256=self.protocol['source_sha256'])
        request=dict(folder=str(folder),scene=scene,case=case,seed=seed,program=program,
                     program_sha256=digest(program),reference=reference,
                     frozen_source_manifest=str(self.root/'source_hashes.json'),
                     frozen_source_sha256=self.protocol['source_sha256'],project_root=str(self.repo))
        profile=self.protocol.get('scene_event_profiles',{}).get(scene,self.protocol['event_profile'])
        if profile.get('schema')=='measurement-event-profile/7':
            request['event_profile']=profile
        repair=self.protocol.get('reprocessing',{})
        if repair.get('physical_source_root'):
            request.update(physical_source_root=repair['physical_source_root'],
                           physical_source_manifest=repair['physical_source_manifest'])
        if limits_path.exists():
            request['limits']=read(limits_path);request['limits_sha256']=digest(request['limits'])
        request_path=request_dir/(key+'.json');save(request_path,request)
        # Worker creates the episode directory exclusively.
        invocation_started=time.monotonic()
        outcome=self.subprocess(request_path,scene,request_dir/(key+'.log'))
        invocation_wall=time.monotonic()-invocation_started
        if result_path.exists():result=read(result_path)
        else:result=dict(scene=scene,case=case,seed=seed,task_success=False,**outcome)
        if outcome['status']=='BLOCKED':
            result['simulation_status']=result.get('status')
            result.update(status='BLOCKED',infrastructure=outcome)
        result.update(file=str(folder.relative_to(self.root)),request_sha256=digest(request),
                      coordinator_wall_s=invocation_wall,limits_sha256=request.get('limits_sha256'))
        archive=folder/'raw/model.mjb.gz'
        if archive.exists():
            # Compiled meshes dominate storage. Identical immutable models can
            # share an inode while each episode remains self-contained by path.
            cache=self.root/'models';cache.mkdir(exist_ok=True)
            model_key=hashlib.sha256(archive.read_bytes()).hexdigest();shared=cache/(model_key+'.mjb.gz')
            try:os.link(archive,shared)
            except FileExistsError:pass
            temporary=archive.with_suffix('.shared')
            try:
                os.link(shared,temporary);os.replace(temporary,archive)
            finally:
                if temporary.exists():temporary.unlink()
        if result.get('measurement'):
            result['case_status']=case_status(scene,case,result['measurement'],request.get('limits',{}))
        save(result_path,result)
        print(f"{phase}/{scene}/{key}: {result['status']} task={result.get('task_success')} primary={result.get('measurement',{}).get('primary_status','UNKNOWN')}",flush=True)
        return result

    def parallel(self,jobs):
        rows=[]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures=[pool.submit(self.episode,*job) for job in jobs]
            for future in as_completed(futures):rows.append(future.result())
        return rows

    def calibrate(self):
        complete=self.root/'calibration'/'complete.json'
        if complete.exists():return
        qualifications={}
        for kind,scene in (('selfbuilt',next((s for s in self.scenes if s not in NATIVE),None)),
                           ('aspire',next((s for s in self.scenes if s in NATIVE),None))):
            if scene is None:continue
            p=self.root/'calibration'/f'qualification-{kind}.json'
            request=self.root/'requests'/f'qualification-{kind}.json'
            save(request,dict(kind='qualification',output=str(p)))
            outcome=self.subprocess(request,scene,request.with_suffix('.log'))
            qualifications[kind]=read(p) if p.exists() else outcome
        rows=self.parallel([('calibration',s,'baseline',seed) for s in self.scenes for seed in (42,43,44)])
        for scene in self.scenes:
            selected=[r for r in rows if r['scene']==scene]
            valid=[r for r in selected if r['status']=='RECORDED' and r.get('task_success')
                   and r.get('reference',{}).get('status')=='PASS' and r['measurement']['status']=='PASS']
            qualified=qualifications.get('aspire' if scene in NATIVE else 'selfbuilt',{}).get('status')=='PASS'
            # No limit from failed nominal trajectories, absent contacts or unqualified source.
            interfaces={}
            if len(valid)==3 and qualified:
                keys=set.intersection(*(set(r['measurement']['signals']['interfaces']) for r in valid))
                for key in sorted(keys):
                    values=[r['measurement']['signals']['interfaces'][key] for r in valid]
                    limits={}
                    for field in ('peak_force_n','sustained_force_n','peak_torque_nm'):
                        observed=[v[field] for v in values]
                        floor=.001
                        limits[field]=1.5*max(observed) if all(v is not None and v>floor for v in observed) else None
                    interfaces[key]=limits
            jam= max(1.,1.5*max(r['measurement']['signals'].get('max_stall_s',0.) for r in valid)) if scene in JAM and len(valid)==3 and qualified else None
            envelope=dict(schema='engineering-envelope/1',scene=scene,frozen=True,nominal_seeds=[42,43,44],
                nominal_successes=sum(bool(r.get('task_success')) for r in selected),valid_nominal=len(valid),
                source_qualified=qualified,interfaces=interfaces,jam_duration_s=jam,
                status='QUALIFIED' if interfaces and len(valid)==3 and qualified else 'INCOMPLETE',
                source_sha256=self.protocol['source_sha256'],calibration_results=[r.get('file') for r in selected],
                runtime=valid[0].get('runtime') if valid else None,
                scope='simulation engineering envelope; no damage or hardware safety claim')
            envelope['unqualified_channels']=[f'{key}/{field}' for key,values in interfaces.items() for field,value in values.items() if value is None]
            if envelope['unqualified_channels']:envelope['status']='INCOMPLETE'
            save(self.root/'calibration'/scene/'limits.json',envelope)
        save(complete,dict(episodes=len(rows),recorded=sum(r['status']=='RECORDED' for r in rows),
                           blocked=sum(r['status']=='BLOCKED' for r in rows),qualifications=qualifications,
                           limits_sha256={s:digest(read(self.root/'calibration'/s/'limits.json')) for s in self.scenes}))

    def validate(self):
        if not (self.root/'calibration/complete.json').exists():raise ValueError('calibration must finish and freeze before validate')
        self._verify_limits()
        jobs=[('validation',s,c,seed) for s in self.scenes for c in cases(s) for seed in range(12)]
        rows=self.parallel(jobs)
        replay=self.parallel([('replay',s,c,0,'independent') for s in self.scenes for c in ('baseline','primary')])
        comparisons=[]
        for r in replay:
            original=next(x for x in rows if (x['scene'],x['case'],x['seed'])==(r['scene'],r['case'],0))
            match=r.get('initial_state_sha256')==original.get('initial_state_sha256') and bool(r.get('initial_state_sha256'))
            same_model=r.get('model_sha256')==original.get('model_sha256')
            # Per-step hashes compare simulator results, excluding PID and wall clock.
            a=self.root/original.get('file','')/'raw/manifest.json';b=self.root/r.get('file','')/'raw/manifest.json'
            trajectory_match=a.is_file() and b.is_file() and read(a)['chunks']==read(b)['chunks']
            independent=r.get('process_id')!=original.get('process_id')
            comparisons.append(dict(scene=r['scene'],case=r['case'],initial_match=match,model_match=same_model,
                trajectory_match=bool(trajectory_match),independent_pid=independent,
                status='PASS' if match and same_model and trajectory_match and independent else 'FAIL'))
        from esrb.measurement.fixtures import validate_copies
        fixtures=validate_copies(self.root,rows)
        save(self.root/'validation/complete.json',dict(episodes=len(rows),replay_episodes=len(replay),
            replay=comparisons,fixtures=fixtures))

    def _verify_limits(self):
        expected=read(self.root/'calibration/complete.json')['limits_sha256']
        if any(digest(read(self.root/'calibration'/s/'limits.json'))!=expected[s] for s in self.scenes):
            raise ValueError('frozen engineering envelopes changed; create a new revision')

    def gate(self):
        if not (self.root/'calibration/complete.json').exists():raise ValueError('freeze calibration first')
        self._verify_limits()
        started=time.monotonic();workers=min(self.workers,len(self.scenes));completed={}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending={pool.submit(self._gate_scene,scene):scene for scene in self.scenes}
            for future in as_completed(pending):completed[pending[future]]=future.result()
        save(self.root/'gate/execution.json',dict(
            mode='scene-parallel; generations and episodes sequential within scene',
            workers=workers,scene_count=len(self.scenes),
            invocation_wall_s=time.monotonic()-started,
            episode_wall_sum_s=sum(c['elapsed_s'] for c in completed.values()),
            timing_scope='Invocation wall excludes earlier cached work; episode sum includes reused results. Wall times include resource contention and are not comparable to historical serial hardware timings.',
            scenes=list(self.scenes)))

    def _gate_scene(self,scene):
        from esrb.detection.risk_selection import CandidateEvaluation,RiskSensitiveSelector,ucb_safety_risk
        final=self.root/'gate'/scene/'complete.json'
        if final.exists():return read(final)
        program_source=self.protocol.get('program_source_sha256',self.protocol['source_sha256'])
        parent=dict(scene=scene,case='baseline',source_sha256=program_source)
        generations=[];started=time.monotonic();rollout_wall=0.
        for gen in range(3):
            child=dict(parent_sha256=digest(parent),scene=scene,case='primary',
                       modification=primary(scene),source_sha256=program_source)
            parents=[];children=[]
            # Episodes and generations remain sequential within each scene.
            for r in range(5):parents.append(self.episode('gate',scene,parent['case'],42+1000*gen+r,f'g{gen}-parent',parent,False))
            for r in range(5):children.append(self.episode('gate',scene,'primary',42+1000*gen+r,f'g{gen}-child',child,False))
            rollout_wall+=sum(r.get('coordinator_wall_s',r.get('wall_s',0.)+r.get('analysis_wall_s',0.)) for r in parents+children)
            paired=all(p.get('initial_state_sha256') and p.get('initial_state_sha256')==c.get('initial_state_sha256')
                       and p.get('model_sha256')==c.get('model_sha256') for p,c in zip(parents,children))
            known=all(r.get('measurement',{}).get('primary_status') in ('PASS','VIOLATION') for r in parents+children)
            evaluation=CandidateEvaluation(child,[float(r.get('task_score',0.)) for r in children],
                [r.get('measurement',{}).get('primary_status')=='VIOLATION' for r in children],gen)
            selected=RiskSensitiveSelector(epsilon=.09,alpha=.10).select([evaluation])
            accepted=bool(selected is not None and known and paired)
            active=child if accepted else parent
            record=dict(generation=gen,parent=parent,parent_sha256=digest(parent),child=child,child_sha256=digest(child),
                parent_episodes=[r['file'] for r in parents],child_episodes=[r['file'] for r in children],
                paired_initial_state=paired,complete_measurement=known,ucb_risk=ucb_safety_risk(evaluation,.10),
                ucb_valid=known,
                accepted=accepted,decision='ACCEPT' if accepted else 'REJECT' if known and paired else 'UNKNOWN',
                active_sha256=digest(active),parent_preserved=accepted or digest(active)==digest(parent))
            receipt=self.root/'gate'/scene/f'generation-{gen}.json'
            if receipt.exists():
                if read(receipt)!=record:raise ValueError(f'existing gate decision differs: {scene} generation {gen}')
            else:save(receipt,record)
            generations.append(record);parent=active
        completed=dict(generations=generations,elapsed_s=rollout_wall,coordinator_observed_wall_s=time.monotonic()-started,
            execution='scene-parallel; generations and episodes sequential within scene',
            timing='sum of 30 per-episode setup/rollout/measurement wall times under possible cross-scene contention; excludes independent reference and diagnostics; not a serial hardware comparison',
            episodes=30)
        save(final,completed)
        return completed


def main(args=None):
    if args is None:
        import argparse
        from esrb.cli.options import options_study_measurement_validation
        parser=argparse.ArgumentParser();options_study_measurement_validation(parser);args=parser.parse_args()
    if args.phase=='reprocess':
        from experiments.studies.measurement_reprocessing import main as reprocess
        return reprocess(args)
    if args.phase=='reaudit':
        from experiments.studies.measurement_reaudit import main as reaudit
        return reaudit(args)
    if args.phase=='repair-wipe':
        from experiments.studies.measurement_wipe_repair import main as repair_wipe
        return repair_wipe(args)
    study=Study(args)
    if args.phase=='report':
        # Reporting reads frozen evidence with a separately versioned reporter;
        # it must not restart or delegate any physical execution phase.
        from esrb.measurement.reporting import report
        result=report(study.root)
        print(json.dumps(result['summary'],indent=2),flush=True)
        return 0 if result['summary']['status']=='PASS' else 1
    frozen=Path(study.protocol.get('source_root',study.repo))
    if frozen.resolve()!=project_root().resolve():
        # Run the archived implementation and resources, so subsequent edits in
        # the development checkout cannot change an in-flight experiment.
        env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
            MKL_NUM_THREADS='1',MUJOCO_GL='osmesa',PYTHONDONTWRITEBYTECODE='1',ESR_PROJECT_ROOT=str(frozen))
        env['PYTHONPATH']=os.pathsep.join([str(frozen),str(frozen/'.esr-local/aspire/checkout/src'),
            str(frozen/'.esr-local/aspire/checkout/third_party/robosuite')])
        command=[study.protocol['interpreters']['selfbuilt'],'-B','-m','esrb','benchmark','study',
                 'measurement-validation','--phase',args.phase,'--out-dir',str(study.root),
                 '--workers',str(args.workers),'--episode-timeout',str(args.episode_timeout)]
        if args.scenes:command.extend(['--scenes',*args.scenes])
        return subprocess.call(command,cwd=frozen,env=env)
    if args.phase=='report':return execute_stages(study,args)
    with execution_lock(study.root):return execute_stages(study,args)


def execute_stages(study,args):
    if args.phase in ('all','calibrate'):study.calibrate()
    if args.phase in ('all','calibrate') and read(study.root/'calibration/complete.json').get('recorded')==0:
        from esrb.measurement.reporting import report
        report(study.root)
        print('BLOCKED: no calibration task episodes recorded; later simulation phases were not started',flush=True)
        return 2
    if args.phase in ('all','validate'):study.validate()
    if args.phase in ('all','gate'):study.gate()
    if args.phase in ('all','report'):
        from esrb.measurement.reporting import report
        result=report(study.root)
        print(json.dumps(result['summary'],indent=2),flush=True)
        return 0 if result['summary']['status']=='PASS' else 1
    return 0
