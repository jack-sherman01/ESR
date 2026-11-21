"""Versioned offline rule audit; no task/controller execution or model calls."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse
import collections
import os
import subprocess
import time

from experiments.studies.measurement_validation import Study,read,save,digest,execution_lock
from experiments.studies.measurement_reprocessing import reuse_tree,sha


def prepare(args):
    if not args.from_run:raise ValueError('reaudit requires --from-run')
    origin=args.from_run.resolve();root=args.out_dir.resolve()
    if root.exists():raise ValueError('reaudit requires a NEW output revision')
    previous=read(origin/'protocol.json')
    if args.scenes and list(args.scenes)!=previous['scenes']:
        raise ValueError('the complete original panel is re-audited to avoid case selection')
    if any(not (origin/'gate'/scene/'complete.json').exists() for scene in previous['scenes']):
        raise ValueError('finish original run before offline re-audit')
    if any(g['accepted'] for scene in previous['scenes']
           for g in read(origin/'gate'/scene/'complete.json')['generations']):
        raise ValueError('offline re-audit cannot reuse an accepted descendant chain')
    local=argparse.Namespace(**vars(args));local.phase='calibrate';local.scenes=previous['scenes']
    study=Study(local);protocol=study.protocol
    from esrb.runtime.paths import project_root
    archives=sorted((project_root()/'.esr-local/_archives').glob('measurement-rules-v6-*'))
    protocol.update(interpreters=previous['interpreters'],program_source_sha256=previous.get('program_source_sha256',previous['source_sha256']),
        episode_timeout_s=previous['episode_timeout_s'],main_execution='offline rule re-audit; no physical rollouts or online interventions',
        offline_reaudit=dict(origin=str(origin),previous_event_profile=previous['event_profile'],
            analysis_source_sha256=protocol['source_sha256'],original_physical_source_sha256=previous['source_sha256'],
            physical_rollouts_started=0,model_calls=0,limits='unchanged original engineering envelope; no recalibration',
            prechange_rule_archive=str(archives[-1].resolve()) if archives else None,
            interpretation='Post-hoc rule revision after inspecting failure modes; not a held-out measurement qualification or a new online update experiment.',
            timing='Legacy RDC uses original physical experiment timings; offline processing time is separate.'))
    save(root/'protocol.json',protocol)
    receipts={}
    for directory in ('calibration','validation','gate','replay','requests'):
        for p in (origin/directory).rglob('*.json'):receipts[str(p.relative_to(origin))]=sha(p)
    receipts['protocol.json']=sha(origin/'protocol.json');receipts['source_hashes.json']=sha(origin/'source_hashes.json')
    save(root/'origin_receipts.json',receipts)
    for directory in ('calibration','validation','replay','requests','fixtures'):
        if (origin/directory).exists():reuse_tree(origin/directory,root/directory)
    for p in (origin/'gate').glob('*/*/result.json'):
        relative=p.parent.relative_to(origin);(root/relative).parent.mkdir(parents=True,exist_ok=True)
        reuse_tree(p.parent,root/relative)
    jobs=[str(p.parent.relative_to(root)) for phase in ('gate','validation','replay','calibration')
          for p in sorted((root/phase).glob('*/*/result.json'))]
    save(root/'reaudit_plan.json',dict(origin=str(origin),jobs=jobs,episode_count=len(jobs),
        physical_rollouts_started=0,model_calls=0))
    return root


def worker(root,relative):
    from esrb.measurement.rules import analyze
    from esrb.measurement.reference import check
    from esrb.measurement.spec import case_status
    root=Path(root);folder=root/relative;protocol=read(root/'protocol.json')
    profile=protocol['event_profile'];row=read(folder/'result.json')
    original=Path(protocol['offline_reaudit']['origin'])/relative/'result.json'
    old=read(original);limits_path=root/'calibration'/row['scene']/'limits.json'
    import sys,mujoco,numpy
    runtime=dict(python=sys.version,mujoco=mujoco.__version__,numpy=numpy.__version__)
    if old.get('runtime') and old['runtime']!=runtime:raise ValueError('runtime differs from original recorded episode')
    limits=read(limits_path) if limits_path.exists() else None
    started=time.monotonic();measurement=analyze(folder,limits,profile=profile)
    analyze_s=time.monotonic()-started
    started=time.monotonic();reference=check(folder,measurement,limits=limits,profile=profile)
    reference_s=time.monotonic()-started
    row.update(measurement=measurement,reference=reference,
        case_status=case_status(row['scene'],row['case'],measurement,limits or {}),
        offline_reaudit=dict(origin=str(original),origin_result_sha256=sha(original),
            previous_primary_status=old.get('measurement',{}).get('primary_status'),
            previous_missing=old.get('measurement',{}).get('missing',[]),event_profile_schema=profile['schema'],
            analysis_wall_s=analyze_s,reference_wall_s=reference_s,physical_rollouts_started=0,runtime=runtime))
    # Keep original execution timings and binding receipts; only derived fields
    # are replaced atomically, so linked original results cannot be changed.
    save(folder/'result.json',row)


def fixture_worker(root,scene):
    """Use the scene's original MuJoCo environment, including for corrupt copies."""
    from esrb.measurement.rules import analyze
    from esrb.measurement.limits import exceeds
    root=Path(root)
    protocol=read(root/'protocol.json');profile=protocol['event_profile']
    completion=read(root/'validation/complete.json');fixtures=[]
    for old in (r for r in completion['fixtures'] if r['scene']==scene):
        row=dict(old,event_profile_schema=profile['schema'],offline_reaudit=True)
        if old['case']=='threshold_boundary':
            row['actual']=exceeds(old['value'],old['limit'])
            row['status']='PASS' if row['actual']==old['expected'] else 'FAIL'
        else:
            folder=root/old['file'];measurement=analyze(folder,profile=profile)
            row['actual']=measurement['primary_status']
            row['status']='PASS' if row['actual']==old['expected'] else 'FAIL'
            row['unknown_reasons']=measurement.get('unknown_reasons',[])
            save(folder/'verification.json',row)
        fixtures.append(row)
    save(root/'fixtures'/scene/'reaudit-complete.json',fixtures)


def refresh_validation_witnesses(root):
    """Recheck faults with their original interpreter; raw copies stay immutable."""
    root=Path(root);protocol=read(root/'protocol.json')
    completion=read(root/'validation/complete.json');fixtures=[]
    for scene in protocol['scenes']:
        python=protocol['interpreters']['aspire' if scene in ('stack','pickplacecan','two-arm-handover','nut-assembly','wipe') else 'selfbuilt']
        subprocess.run([python,'-B','-c',
            'import sys;from experiments.studies.measurement_reaudit import fixture_worker;fixture_worker(sys.argv[1],sys.argv[2])',
            str(root),scene],env=os.environ.copy(),check=True,timeout=protocol['episode_timeout_s'])
        fixtures.extend(read(root/'fixtures'/scene/'reaudit-complete.json'))
    # Replay compares immutable physical chunks and original independent PIDs;
    # no new replay is claimed. Their event analyses have been revised separately.
    save(root/'validation/complete.json',dict(completion,fixtures=fixtures,
        offline_reaudit=True,replay_scope='Original cross-process physical replay receipts retained; no new rollouts.'))


def recompute_gates(root,origin):
    from esrb.detection.risk_selection import CandidateEvaluation,RiskSensitiveSelector,ucb_safety_risk
    protocol=read(root/'protocol.json')
    for scene in protocol['scenes']:
        original=read(origin/'gate'/scene/'complete.json');generations=[]
        for old in original['generations']:
            parents=[read(root/p/'result.json') for p in old['parent_episodes']]
            children=[read(root/p/'result.json') for p in old['child_episodes']]
            paired=all(p.get('initial_state_sha256') and p['initial_state_sha256']==c.get('initial_state_sha256')
                and p['model_sha256']==c.get('model_sha256') for p,c in zip(parents,children))
            known=all(r['measurement']['primary_status'] in ('PASS','VIOLATION') for r in parents+children)
            candidate=CandidateEvaluation(old['child'],[r['task_score'] for r in children],
                [r['measurement']['primary_status']=='VIOLATION' for r in children],old['generation'])
            accepted=known and paired and RiskSensitiveSelector(.09,.10).select([candidate]) is not None
            if accepted:raise ValueError('offline re-audit cannot invent an unexecuted descendant chain')
            g=dict(old,paired_initial_state=paired,complete_measurement=known,ucb_valid=known,
                ucb_risk=ucb_safety_risk(candidate,.10),accepted=False,decision='REJECT' if known and paired else 'UNKNOWN',
                active_sha256=old['parent_sha256'],parent_preserved=True,
                offline_reaudit=True,original_decision=old['decision'])
            generations.append(g);save(root/'gate'/scene/f"generation-{g['generation']}.json",g)
        save(root/'gate'/scene/'complete.json',dict(original,generations=generations,offline_reaudit=True,
            timing='Original physical execution times reused; offline re-audit time is separately recorded.'))
    save(root/'gate/execution.json',dict(mode='offline recomputation only',physical_rollouts_started=0,
        warning='Decisions are recomputed evidence, not online interventions; original active-policy lineage preserved.'))


def execute(root):
    from esrb.measurement.reporting import report
    root=Path(root);protocol=read(root/'protocol.json');origin=Path(protocol['offline_reaudit']['origin'])
    plan=read(root/'reaudit_plan.json');status=root/'reaudit-status.json';started=time.monotonic()
    with execution_lock(root):
        try:
            done=0;reference_counts=collections.Counter();changes=collections.Counter()
            save(status,dict(status='RUNNING',completed=0,total=len(plan['jobs']),physical_rollouts_started=0,pid=os.getpid()))
            def run(relative):
                scene=relative.split('/')[1]
                python=protocol['interpreters']['aspire' if scene in ('stack','pickplacecan','two-arm-handover','nut-assembly','wipe') else 'selfbuilt']
                request_log=root/'reaudit_logs'/(relative.replace('/','--')+'.log');request_log.parent.mkdir(exist_ok=True)
                command=[python,'-B','-c',
                    'import sys;from experiments.studies.measurement_reaudit import worker;worker(sys.argv[1],sys.argv[2])',str(root),relative]
                with request_log.open('wb') as stream:
                    subprocess.run(command,cwd=protocol['source_root'],env=os.environ.copy(),stdout=stream,
                        stderr=subprocess.STDOUT,check=True,timeout=protocol['episode_timeout_s'])
                return relative,read(root/relative/'result.json')
            with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
                futures=[pool.submit(run,relative) for relative in plan['jobs']]
                for future in as_completed(futures):
                    relative,row=future.result();done+=1
                    reference_counts[row['reference']['status']]+=1
                    changes[(row['offline_reaudit']['previous_primary_status'],row['measurement']['primary_status'])]+=1
                    save(status,dict(status='RUNNING',completed=done,total=len(plan['jobs']),
                        reference=dict(reference_counts),physical_rollouts_started=0))
                    print(relative,row['measurement']['primary_status'],row['reference']['status'],f'{done}/{len(plan["jobs"])}',flush=True)
            refresh_validation_witnesses(root)
            recompute_gates(root,origin)
            changed=[name for name,value in read(root/'origin_receipts.json').items() if sha(origin/name)!=value]
            if changed:raise ValueError('original evidence changed: '+str(changed[:3]))
            source=Path(protocol['source_root'])
            if any(sha(source/name)!=value for name,value in read(root/'source_hashes.json').items()):raise ValueError('frozen analysis source changed')
            audit=dict(origin=str(origin),original_evidence_unchanged=True,engineering_limits_unchanged=True,
                physical_rollouts_started=0,model_calls=0,episodes=done,elapsed_s=time.monotonic()-started,
                primary_status_transitions=[dict(previous=a,current=b,count=n) for (a,b),n in changes.items()],
                reference=dict(reference_counts),event_profile_schema=protocol['event_profile']['schema'])
            save(root/'rule-revision-audit.json',audit)
            result=report(root)
            save(status,dict(status='COMPLETE',completed=done,total=len(plan['jobs']),
                scientific_status=result['summary']['status'],physical_rollouts_started=0,original_evidence_unchanged=True))
            return 0
        except BaseException as error:
            save(status,dict(status='FAILED',completed=done,total=len(plan['jobs']),error=f'{type(error).__name__}: {error}'))
            raise


def main(args):
    root=prepare(args);protocol=read(root/'protocol.json');source=Path(protocol['source_root'])
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        MUJOCO_GL='osmesa',NUMBA_CACHE_DIR=str(root/'runtime_cache/numba'),PYTHONDONTWRITEBYTECODE='1',
        ESR_PROJECT_ROOT=str(source),PYTHONPATH=os.pathsep.join([str(source),
            str(source/'.esr-local/aspire/checkout/src'),str(source/'.esr-local/aspire/checkout/third_party/robosuite')]))
    command=[protocol['interpreters']['selfbuilt'],'-B','-c',
        'import sys;from experiments.studies.measurement_reaudit import execute;execute(sys.argv[1])',str(root)]
    return subprocess.call(command,cwd=source,env=env)
