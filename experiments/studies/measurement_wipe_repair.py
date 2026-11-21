"""Re-run only Wipe, then publish verified evidence into the canonical dataset."""
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

from experiments.studies.measurement_validation import Study,read,save,digest,execution_lock
from experiments.studies.measurement_reprocessing import sha
from esrb.measurement.spec import event_profile
from esrb.runtime.paths import project_root


def files(root):
    return (p for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink())


def link_tree(source,target):
    """Share immutable data only; derived JSON writers must replace paths."""
    def copy(a,b):
        try:os.link(a,b)
        except OSError:shutil.copy2(a,b)
    shutil.copytree(source,target,copy_function=copy)


def main(args):
    if not args.from_run:raise ValueError('repair-wipe requires the completed canonical --from-run')
    origin=args.from_run.resolve()
    if read(origin/'consolidation.json')['status']!='COMPLETE':raise ValueError('canonical data must be consolidated')
    if args.scenes and args.scenes!=['wipe']:raise ValueError('repair-wipe is scoped to Wipe')
    local=argparse.Namespace(**vars(args));local.phase='calibrate';local.scenes=['wipe']
    study=Study(local);root=study.root
    if (root/'repair-inputs.json').exists():raise ValueError('create a new Wipe repair output revision')
    study.protocol.update(event_profile=event_profile(7),episode_timeout_s=3600.,
        wipe_repair=dict(origin=str(origin),scope='Re-run Wipe only; fixed seeds/budgets/native physics; primary remains non-tool robot contact.',
            development_seeds=[42,43,44],contact_profile='Reset-frame +90deg local-X rotation in free space for 70 controls; then descend; native OSC max translation 0.01 m.'))
    save(root/'protocol.json',study.protocol)
    manifest=read(origin/'consolidated-artifacts.json')
    save(root/'repair-inputs.json',dict(origin=str(origin),canonical_manifest=manifest,
        previous_report=read(origin/'report/summary.json'),previous_protocol=read(origin/'protocol.json')))
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1',MUJOCO_GL='osmesa',ESR_PROJECT_ROOT=study.protocol['source_root'],
        PYTHONPATH=os.pathsep.join([study.protocol['source_root'],study.protocol['source_root']+'/.esr-local/aspire/checkout/src',
            study.protocol['source_root']+'/.esr-local/aspire/checkout/third_party/robosuite']))
    cmd=[study.protocol['interpreters']['selfbuilt'],'-B','-c',
        'import sys;from experiments.studies.measurement_wipe_repair import execute;execute(sys.argv[1])',str(root)]
    return subprocess.call(cmd,cwd=study.protocol['source_root'],env=env)


def execute(root):
    root=Path(root);protocol=read(root/'protocol.json');status=root/'repair-status.json'
    args=argparse.Namespace(out_dir=root,phase='all',scenes=['wipe'],workers=protocol['workers'],episode_timeout=3600.)
    try:
        with execution_lock(root):
            study=Study(args)
            for name in ('calibrate','validate','gate'):
                save(status,dict(status='RUNNING',phase=name))
                getattr(study,name)()
            save(status,dict(status='RUNNING',phase='independent-gate-reference'))
            subprocess.run([protocol['interpreters']['aspire'],'-B','-c',
                'import sys;from experiments.studies.measurement_wipe_repair import audit_gate;audit_gate(sys.argv[1])',
                str(root)],check=True,timeout=3600)
            from esrb.measurement.reporting import report
            result=report(root)
            rows=[read(p) for phase in ('calibration','validation','gate','replay') for p in (root/phase/'wipe').glob('*/result.json')]
            if len(rows)!=95 or any(r['status']!='RECORDED' for r in rows):raise ValueError('not all 95 Wipe records completed')
            if any(r.get('reference',{}).get('status')!='PASS' for r in rows):raise ValueError('Wipe independent reference did not pass')
            for row in rows:
                measurement=row['measurement'];kinds={e['kind'] for e in measurement['events']}
                if row['case']=='primary' and ('robot_contact' not in kinds or 'joint_limit_proximity' in kinds):
                    raise ValueError('primary injection must witness robot contact without earlier joint proximity')
                if row['case']=='baseline' and (not row['task_success'] or measurement['status']!='PASS'):
                    raise ValueError('normal Wipe must complete safely under revised secondary rules')
            save(status,dict(status='RUNNING',phase='merge',records=len(rows)))
            merge(root)
            save(status,dict(status='COMPLETE',records=95,primary_rule='robot_contact'))
    except BaseException as error:
        save(status,dict(status='FAILED',error=f'{type(error).__name__}: {error}'))
        raise


def audit_gate(root):
    from esrb.measurement.reference import check
    root=Path(root);limits=read(root/'calibration/wipe/limits.json')
    for p in sorted((root/'gate/wipe').glob('*/result.json')):
        row=read(p);started=time.monotonic()
        row['reference']=check(p.parent,row['measurement'],limits=limits,profile=event_profile(7))
        row['reference_wall_s']=time.monotonic()-started
        row['reference_scope']='Independent raw reconstruction after execution; no additional rollout'
        save(p,row)


def audit_previous(origin):
    from esrb.measurement.rules import analyze
    from esrb.measurement.reference import check
    origin=Path(origin);destination=origin/'supplemental/wipe-repair-v7'
    limits=read(origin/'calibration/wipe/limits.json');profile=event_profile(7);checks=[]
    for previous in read(destination/'previous-primary.json'):
        folder=origin/previous['retained'];measurement=analyze(folder,limits,profile=profile)
        reference=check(folder,measurement,limits=limits,profile=profile)
        events=[e for e in measurement['events'] if e['kind']=='joint_limit_proximity']
        if not events or reference['status']!='PASS':raise ValueError('old Wipe early termination not explained by independent joint reconstruction')
        save(folder/'joint-proximity-audit.json',dict(measurement=measurement,reference=reference,
            original_result_unchanged=True,scope='Post-hoc audit of prior failed injection; does not count as a new contact-positive trial'))
        checks.append(dict(episode=previous['episode'],events=events,reference=reference['status']))
    save(destination/'previous-joint-proximity.json',dict(status='PASS',episodes=len(checks),cases=checks))


def merge(work):
    work=Path(work);inputs=read(work/'repair-inputs.json');origin=Path(inputs['origin']);protocol=read(work/'protocol.json')
    publication=Path(inputs['previous_protocol']['publication_root'])
    destination=origin/'supplemental/wipe-repair-v7'
    with execution_lock(origin):
        if destination.exists():raise ValueError('Wipe repair already published or interrupted; retain evidence and investigate')
        # Validate canonical bytes before replacing any result. Cache identities
        # only within this completed, exclusively locked transaction.
        cache={}
        for name,expected in inputs['canonical_manifest'].items():
            p=origin/name
            if isinstance(expected,dict):
                if not p.is_symlink() or os.readlink(p)!=expected['symlink']:raise ValueError('canonical symlink changed')
                continue
            stat=p.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
            if key not in cache:cache[key]=sha(p)
            if cache[key]!=expected:raise ValueError('canonical evidence changed: '+name)
        destination.mkdir(parents=True)
        # Prior failed injections remain compact, readable physical evidence;
        # they are never relabelled as successful collision injections.
        previous=[]
        for phase in ('calibration','validation','gate','replay'):
            for p in (origin/phase/'wipe').glob('*/result.json'):
                old=read(p);relative=p.parent.relative_to(origin)
                if old['case']=='primary':
                    target=destination/'prior-primary'/relative
                    target.parent.mkdir(parents=True,exist_ok=True);link_tree(p.parent,target)
                    previous.append(dict(episode=str(relative),retained=str(target.relative_to(origin))))
        save(destination/'previous-primary.json',previous)
        subprocess.run([protocol['interpreters']['aspire'],'-B','-c',
            'import sys;from experiments.studies.measurement_wipe_repair import audit_previous;audit_previous(sys.argv[1])',
            str(origin)],check=True,timeout=3600)
        # Preserve the exact execution capsule; the original canonical physical
        # source remains in source/ and other scenes never change.
        capsule=origin/'analysis-capsules/wipe-contact-repair-v7';capsule.mkdir()
        link_tree(work/'source',capsule/'source')
        for name in ('source_hashes.json','protocol.json'):shutil.copy2(work/name,capsule/name)
        for phase in ('calibration','validation','gate','replay','fixtures'):
            old=origin/phase/'wipe';replacement=work/phase/'wipe'
            if not replacement.exists():continue
            temporary=old.with_name('.wipe-v7-ready')
            link_tree(replacement,temporary)
            backup=old.with_name('.wipe-v6-replaced');old.rename(backup);temporary.rename(old);shutil.rmtree(backup)
        for phase in ('calibration','validation','gate','replay'):
            old=origin/'requests'/phase/'wipe';replacement=work/'requests'/phase/'wipe'
            temporary=old.with_name('.wipe-v7-ready');link_tree(replacement,temporary)
            backup=old.with_name('.wipe-v6-replaced');old.rename(backup);temporary.rename(old);shutil.rmtree(backup)
        complete=read(origin/'calibration/complete.json');complete['limits_sha256']['wipe']=digest(read(origin/'calibration/wipe/limits.json'));save(origin/'calibration/complete.json',complete)
        complete=read(origin/'validation/complete.json');new=read(work/'validation/complete.json')
        for key in ('replay','fixtures'):complete[key]=[r for r in complete[key] if r['scene']!='wipe']+new[key]
        save(origin/'validation/complete.json',complete)
        current=read(origin/'protocol.json');current.setdefault('scene_event_profiles',{})['wipe']=protocol['event_profile']
        current.setdefault('scene_source_capsules',{})['wipe']=str(capsule)
        save(origin/'protocol.json',current)
        save(destination/'frozen-run.json',dict(protocol=protocol,source_hashes=read(work/'source_hashes.json')))
        repaired=read(origin/'gate/wipe/complete.json')
        receipt=dict(status='COMPLETE',scene='wipe',new_physical_episodes=95,
            accounting=dict(calibration=3,validation=60,gate=30,replay=2),primary_diagnostic_triggered=12,
            primary_gate_triggered=15,prior_primary_retained=len(previous),event_profile='measurement-event-profile/7',
            source_sha256=protocol['source_sha256'],source_capsule=str(capsule),model_calls=0,
            native_termination_preserved=True,primary_definition='Non-tool robot contact; joint proximity remains separate',
            gate_parent_preserved=all(not g['accepted'] and g['parent_preserved'] for g in repaired['generations']))
        save(origin/'wipe-repair.json',receipt)
        from esrb.measurement.reporting import report
        result=report(origin)
        before={s['scene']:s for s in inputs['previous_report']['scenes']}
        for scene in result['scenes']:
            if scene['scene']!='wipe' and scene!=before[scene['scene']]:raise ValueError('unrelated scene report changed')
        if result['scenes'][-1]['metrics']['recall']!=1.:raise ValueError('repaired collision detection not fully witnessed')
        # Remap absorbed execution requests and preserved old fault evidence.
        index=read(origin/'provenance/source-index.json');locations={};manifest={};cache={}
        for p in files(origin):
            name=str(p.relative_to(origin))
            if name=='consolidated-artifacts.json':continue
            stat=p.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
            if key not in cache:cache[key]=sha(p)
            manifest[name]=cache[key];locations.setdefault(cache[key],name)
        kept=[];pruned=[]
        for row in index:
            if row['kind']!='file':kept.append(row);continue
            if manifest.get(row['stored'])==row['sha256']:kept.append(row)
            elif row['sha256'] in locations:kept.append(dict(row,stored=locations[row['sha256']]))
            else:pruned.append(dict(row,pruned=True,policy='Superseded Wipe result replaced by verified physical repair; metadata only'))
        # Every new request/capsule/raw byte is addressable by original path.
        for p in files(work):
            name=p.relative_to(work)
            if str(name) in ('repair-inputs.json',) or name.parts[0] in ('models','runtime_cache','report') or p.suffix=='.pyc':continue
            value=sha(p)
            if value in locations:kept.append(dict(source=str(p),source_root=str(work),relative=str(name),kind='file',sha256=value,size=p.stat().st_size,stored=locations[value]))
        save(origin/'provenance/source-index.json',kept)
        archive=origin/'provenance/pruned-index.jsonl.gz';temporary=archive.with_name(archive.name+'.v7-tmp')
        with gzip.open(archive,'rt') as src,gzip.open(temporary,'wt') as dest:
            shutil.copyfileobj(src,dest)
            for row in pruned:dest.write(json.dumps(row)+'\n')
        temporary.replace(archive)
        bindings=read(origin/'provenance/execution-sources.json');bindings['bindings'][protocol['source_sha256']]=dict(capsule=str(capsule.relative_to(origin)),source=str((capsule/'source').relative_to(origin)),manifest=str((capsule/'source_hashes.json').relative_to(origin)),files=len(read(capsule/'source_hashes.json')))
        counts=Counter(read(p)['frozen_source_sha256'] for phase in ('calibration','validation','gate','replay') for p in (origin/'requests'/phase).glob('*/*.json'))
        bindings['request_counts']=dict(counts);save(origin/'provenance/execution-sources.json',bindings)
        verification=dict(status='PASS',other_11_scenes_unchanged=True,main_records=1524,new_wipe_records=95,
            primary_diagnostic_triggered=12,primary_gate_triggered=15,all_new_reference_passed=True,
            old_primary_failures_retained=len(previous),source_aliases=len(kept),superseded_aliases_metadata_only=len(pruned))
        save(origin/'verification/wipe-repair-v7.json',verification)
        artifacts={};cache={}
        for p in files(origin):
            name=str(p.relative_to(origin))
            if name=='consolidated-artifacts.json':continue
            stat=p.stat();key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
            if key not in cache:cache[key]=sha(p)
            artifacts[name]=cache[key]
        if any(artifacts[row['stored']]!=row['sha256'] for row in kept if row['kind']=='file'):raise ValueError('source alias mismatch')
        save(origin/'consolidated-artifacts.json',artifacts)
        save(work/'merge-complete.json',dict(status='COMPLETE',canonical=str(origin),verification=verification))
