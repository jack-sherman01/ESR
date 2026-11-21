"""Selective, append-only repair of a completed controlled measurement run.

This study-specific revision reuses intact evidence, reruns infrastructure
failures, and preserves actual violations and structurally invalid recordings.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reuse_tree(source, target):
    """Share immutable bytes; writers must always replace paths atomically."""
    def link(a,b):
        try:os.link(a,b)
        except OSError:shutil.copy2(a,b)
    shutil.copytree(source,target,copy_function=link)


def repair_kind(row):
    if row.get('status')=='BLOCKED':
        error=row.get('error','')
        if row.get('case')=='legal_hold' and 'not JSON serializable' in error:return 'rerun_serialization'
        if error=='less than 5 GiB free; no rollout started':return 'rerun_not_started'
        raise ValueError('Unrecognized blocked episode requires investigation: '+error)
    if row.get('reference',{}).get('status')=='UNKNOWN':return 'reference_only'
    return 'reuse'


def reusable_hold_repair(cache,origin,relative):
    """A cached repair must reproduce every original raw chunk exactly."""
    from experiments.studies.measurement_validation import read,digest
    folder=Path(cache)/relative;original=Path(origin)/relative
    if not (folder/'result.json').exists():return False
    row=read(folder/'result.json')
    if row.get('status')!='RECORDED' or not row.get('file') or row.get('reference',{}).get('status')!='PASS':return False
    if row.get('case')!='legal_hold':return False
    try:
        a=read(original/'raw/manifest.json');b=read(folder/'raw/manifest.json')
        if not a['complete'] or not b['complete']:return False
        for key in ('initial_state_sha256','model_sha256','physics_steps','chunks'):
            if a[key]!=b[key]:raise ValueError('cached physical trajectory differs: '+str(relative))
        if sha(folder/'raw/manifest.json')!=row['raw_manifest_sha256']:raise ValueError('cached raw manifest changed')
        if any(sha(folder/'raw'/item['file'])!=item['sha256'] for item in b['chunks']):raise ValueError('cached raw chunk changed')
        request=read(Path(cache)/'requests'/relative.with_suffix('.json'))
        if digest(request)!=row['request_sha256'] or digest(request['program'])!=row['program_sha256']:raise ValueError('cached request binding changed')
        if digest(read(Path(origin)/'calibration'/row['scene']/'limits.json'))!=row['limits_sha256']:raise ValueError('cached engineering limits differ')
        return True
    except FileNotFoundError:return False


def prepare(args):
    from experiments.studies.measurement_validation import read,save,digest
    from esrb.runtime.paths import project_root
    if not args.from_run:raise ValueError('reprocess requires --from-run')
    origin=args.from_run.resolve();root=args.out_dir.resolve();repo=project_root()
    if root.exists():raise ValueError('reprocessing requires a NEW output directory')
    if not 1<=args.workers<=8:raise ValueError('workers must be between 1 and 8')
    old=read(origin/'protocol.json')
    if args.scenes and list(args.scenes)!=old['scenes']:raise ValueError('repair preserves the original scene panel')
    if not (origin/'validation/complete.json').exists() or any(not (origin/'gate'/s/'complete.json').exists() for s in old['scenes']):
        raise ValueError('the source run must have completed all execution stages')
    # The planned five-trial gate cannot accept any update. Reusing old program
    # identities is safe only after checking that no accepted descendant exists.
    if any(g['accepted'] for s in old['scenes'] for g in read(origin/'gate'/s/'complete.json')['generations']):
        raise ValueError('selective repair cannot reuse an accepted descendant chain')
    root.mkdir(parents=True)
    original_files=[origin/'protocol.json',origin/'source_hashes.json']
    original_files += [p for directory in ('calibration','validation','gate','replay','requests')
                      for p in (origin/directory).rglob('*.json')]
    save(root/'origin_receipts.json',{str(p.relative_to(origin)):sha(p) for p in original_files})
    frozen=Path(old['source_root']);original_hashes=read(origin/'source_hashes.json')
    if any(not (frozen/p).is_file() or sha(frozen/p)!=value for p,value in original_hashes.items()):
        raise ValueError('original physical capsule changed')
    reuse_tree(frozen,root/'source')
    patches={name:(repo/name).read_bytes() for name in (
        'experiments/studies/measurement_validation.py','experiments/studies/measurement_reprocessing.py',
        'esrb/measurement/reporting.py','esrb/measurement/reference.py','esrb/measurement/physical_sources.py','esrb/cli/options.py')}
    # Preserve all original task/controller code. Only convert this logged field
    # to a Python bool; no control action, geometry, stage or threshold changes.
    name='esrb/measurement/episodes.py';before=(frozen/name).read_text()
    old_line="holding=case=='legal_hold' and lifted and transport and hold_ticks<60"
    new_line="holding=bool(case=='legal_hold' and lifted and transport and hold_ticks<60)"
    if old_line not in before:raise ValueError('unexpected original serialization implementation')
    repaired=before.replace(old_line,new_line)
    repaired=repaired.replace('    result=run(request)\n',
        "    if request.get('physical_source_root'):\n"
        "        from .physical_sources import bind\n"
        "        bind(request['project_root'],request['physical_source_root'],request['physical_source_manifest'])\n"
        "    result=run(request)\n")
    repaired=repaired.replace("        root=Path(request['project_root'])\n",
        "        if request.get('physical_source_root'):\n"
        "            from .physical_sources import verify\n"
        "            if not verify(request['physical_source_root'],request['physical_source_manifest']):return False\n"
        "        root=Path(request['project_root'])\n")
    patches[name]=repaired.encode()
    for name,data in patches.items():
        path=root/'source'/name;path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.repair-tmp');temporary.write_bytes(data);temporary.replace(path)
    hashes=dict(original_hashes)
    for name in patches:hashes[name]=sha(root/'source'/name)
    save(root/'source_hashes.json',hashes)
    with tarfile.open(root/'sources.tar.gz','w:gz',compresslevel=1) as archive:
        for name in sorted(hashes):archive.add(root/'source'/name,arcname=name,recursive=False)
    protocol=dict(old,source_root=str(root/'source'),source_sha256=digest(hashes),
        program_source_sha256=old.get('program_source_sha256',old['source_sha256']),
        workers=args.workers,publication_root=str(repo),main_execution='scene-parallel; generations and episodes sequential within scene',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reprocessing=dict(origin=str(origin),origin_source_sha256=old['source_sha256'],
            physical_source_root=old['source_root'],physical_source_manifest=str(origin/'source_hashes.json'),
            patched_files=sorted(patches),calibration='Frozen original limits reused byte-for-byte; no recalibration',
            evidence='Clean episodes reused by immutable file links. New physical attempts have new capsule receipts; program identifiers retain original logical definitions.',
            timing='Mixed original and repaired episode timings; not a speed ranking'))
    save(root/'protocol.json',protocol)
    for directory in ('calibration','replay','requests','models','fixtures'):
        if (origin/directory).exists():reuse_tree(origin/directory,root/directory)
    jobs=[];references=[];counts={};cached=[]
    cache=getattr(args,'reuse_from',None)
    if cache:
        cache=cache.resolve();cached_protocol=read(cache/'protocol.json')
        if cached_protocol.get('reprocessing',{}).get('origin')!=str(origin) or cached_protocol['event_profile']!=old['event_profile']:
            raise ValueError('cached repair belongs to a different original run or profile')
    for phase in ('validation','gate'):
        for path in sorted((origin/phase).glob('*/*/result.json')):
            row=read(path);kind=repair_kind(row)
            relative=path.parent.relative_to(origin)
            if kind=='rerun_serialization' and cache and reusable_hold_repair(cache,origin,relative):
                (root/relative).parent.mkdir(parents=True,exist_ok=True)
                reuse_tree(cache/relative,root/relative)
                target=root/'requests'/relative.with_suffix('.json');temporary=target.with_suffix('.reuse-tmp')
                shutil.copy2(cache/'requests'/relative.with_suffix('.json'),temporary);temporary.replace(target)
                cached.append(dict(file=str(relative),cache=str(cache),result_sha256=sha(cache/relative/'result.json')))
                kind='reused_completed_repair'
            counts[kind]=counts.get(kind,0)+1
            if kind in ('reuse','reference_only'):
                (root/relative).parent.mkdir(parents=True,exist_ok=True)
                reuse_tree(path.parent,root/relative)
                if kind=='reference_only':references.append(str(relative))
            elif phase=='validation' and kind!='reused_completed_repair':
                jobs.append(dict(phase=phase,scene=row['scene'],case=row['case'],seed=row['seed'],
                    program=read(origin/'requests'/relative.with_suffix('.json'))['program']))
    # Validation completion contains only replay/corruption checks on unchanged
    # baseline/primary trajectories; the repaired legal holds have no such check.
    save(root/'validation/complete.json',read(origin/'validation/complete.json'))
    save(root/'repair_plan.json',dict(origin=str(origin),counts=counts,validation_jobs=jobs,
        reference_jobs=references,reused_repairs=cached,
        physical_attempts_expected=counts.get('rerun_serialization',0)+counts.get('rerun_not_started',0)))
    return root


def reference_worker(root,relative):
    from experiments.studies.measurement_validation import read,save
    from esrb.measurement.reference import check
    folder=Path(root)/relative;row=read(folder/'result.json')
    old=row['reference'];started=time.monotonic()
    row['reference']=check(folder,row['measurement'],limits=read(Path(root)/'calibration'/row['scene']/'limits.json'))
    row['reference_wall_s']=time.monotonic()-started
    row['reprocessing']=dict(kind='reference_only',previous_reference=old,
        raw_sha256_unchanged=True,physical_rollouts_started=0)
    save(folder/'result.json',row)


def execute(root):
    from argparse import Namespace
    from experiments.studies.measurement_validation import Study,read,save,execution_lock
    from esrb.measurement.reporting import report
    root=Path(root);protocol=read(root/'protocol.json');plan=read(root/'repair_plan.json')
    args=Namespace(out_dir=root,phase='gate',workers=protocol['workers'],scenes=None,
        episode_timeout=protocol['episode_timeout_s'],selfbuilt_python=None,aspire_python=None)
    study=Study(args);status=root/'reprocessing-status.json'
    with execution_lock(root):
        try:
            save(status,dict(status='RUNNING',phase='repair',pid=os.getpid(),expected_physical_attempts=plan['physical_attempts_expected']))
            def repair_reference(relative):
                scene=relative.split('/')[1]
                command=[study.interpreter(scene),'-B','-c',
                    'import sys;from experiments.studies.measurement_reprocessing import reference_worker;reference_worker(sys.argv[1],sys.argv[2])',str(root),relative]
                log=root/'requests'/('reference-reprocess-'+relative.replace('/','--')+'.log')
                with log.open('wb') as stream:
                    subprocess.run(command,cwd=study.repo,env=os.environ.copy(),stdout=stream,stderr=subprocess.STDOUT,
                        check=True,timeout=protocol['episode_timeout_s'])
            with ThreadPoolExecutor(max_workers=study.workers) as pool:
                futures=[pool.submit(study.episode,job['phase'],job['scene'],job['case'],job['seed'],'',job['program'],True)
                         for job in plan['validation_jobs']]
                futures += [pool.submit(repair_reference,relative) for relative in plan['reference_jobs']]
                for future in as_completed(futures):future.result()
            save(status,dict(status='RUNNING',phase='gate',expected_physical_attempts=plan['physical_attempts_expected']))
            study.gate()
            receipts=read(root/'origin_receipts.json');origin=Path(plan['origin'])
            changed=[name for name,value in receipts.items() if sha(origin/name)!=value]
            changed += [item['cache']+'/'+item['file'] for item in plan.get('reused_repairs',[])
                        if sha(Path(item['cache'])/item['file']/'result.json')!=item['result_sha256']]
            if changed:raise ValueError('original evidence changed: '+str(changed[:3]))
            inventory=[]
            for phase in ('validation','gate'):
                for path in (root/phase).glob('*/*/result.json'):
                    relative=str(path.relative_to(root));old=read(origin/relative);new=read(path)
                    kind=repair_kind(old)
                    if kind!='reuse':inventory.append(dict(file=relative,kind=kind,previous_status=old['status'],
                        current_status=new['status'],primary=new.get('measurement',{}).get('primary_status'),
                        reference=new.get('reference',{}).get('status')))
            save(root/'reprocessing-audit.json',dict(original_evidence_unchanged=True,calibration_unchanged=True,
                physical_attempts_expected=plan['physical_attempts_expected'],repaired=inventory,
                note='Invalid time-reset trajectories remain UNKNOWN; real violations and engineering-limit gaps are preserved.'))
            result=report(root)
            save(status,dict(status='COMPLETE',scientific_status=result['summary']['status'],
                original_evidence_unchanged=True,physical_attempts=plan['physical_attempts_expected']))
            return 0
        except BaseException as error:
            save(status,dict(status='FAILED',error=f'{type(error).__name__}: {error}'))
            raise


def main(args):
    root=prepare(args);source=root/'source'
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1',MUJOCO_GL='osmesa',PYTHONDONTWRITEBYTECODE='1',ESR_PROJECT_ROOT=str(source),
        NUMBA_CACHE_DIR=str(root/'runtime_cache/numba'),
        PYTHONPATH=os.pathsep.join([str(source),str(source/'.esr-local/aspire/checkout/src'),
            str(source/'.esr-local/aspire/checkout/third_party/robosuite')]))
    protocol=json.loads((root/'protocol.json').read_text())
    command=[protocol['interpreters']['selfbuilt'],'-B','-c',
        'import sys;from experiments.studies.measurement_reprocessing import execute;execute(sys.argv[1])',str(root)]
    return subprocess.call(command,cwd=source,env=env)
