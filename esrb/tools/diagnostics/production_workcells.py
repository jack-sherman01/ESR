"""Engineering acceptance for the three articulated Panda production cells."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import tarfile
import time

NAMES=('long_horizon_assembly','moving_transport','moving_assembly')


def task_and_policy(name,candidate=False):
    from esrb.tasks.long_horizon.assembly import LongHorizonAssemblyTask
    from esrb.agent.assembly_policy import AssemblyPolicy
    from esrb.update_operators.assembly_orientation import AssemblyOrientationOperator
    from esrb.tasks.extensions.registry import EXTENSIONS,ExtensionOperator
    if name=='long_horizon_assembly':
        task,policy=LongHorizonAssemblyTask(),AssemblyPolicy()
        if candidate: policy=AssemblyOrientationOperator().apply(policy,severity=.5).candidate_policy
    else:
        cls,policy_cls=EXTENSIONS[name]; task,policy=cls(),policy_cls()
        if candidate: policy=ExtensionOperator(name).apply(policy,severity=.5).candidate_policy
    return task,policy


def reset_worker(name,seed):
    task,_=task_and_policy(name)
    try:
        task.reset(seed)
        return {'pid':os.getpid(),'scenario':name,'initial_state_digest':task._log['initial_state_digest']}
    finally: task.close()


def episode_worker(root,name,seed,candidate):
    from esrb.oracles.multiconstraint import scenario_oracle
    from esrb.utils.hashing import hash_artifact
    task,policy=task_and_policy(name,candidate)
    started=time.monotonic()
    try:
        obs=task.reset(seed)
        while True:
            obs,_,done,trace=task.step(policy.act(obs))
            if done: break
        elapsed=time.monotonic()-started
        verdict=scenario_oracle(name).evaluate(trace,0)
        success=task.evaluate(trace).success
        record={'scope':'controlled_engineering_no_model_calls','scenario':name,'seed':seed,
                'candidate':candidate,'policy':policy.state_dict(),'trace':trace,'task_success':success,
                'oracle':asdict(verdict),'simulation_wall_s':elapsed,'total_wall_s':time.monotonic()-started}
        path=Path(root)/name/f"seed{seed:02d}-{'candidate' if candidate else 'baseline'}.json.gz"
        path.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(path,'wb',compresslevel=1) as stream:
            stream.write(json.dumps(record,separators=(',',':'),allow_nan=False).encode())
        return {'scenario':name,'seed':seed,'candidate':candidate,'task_success':success,
                'oracle_status':verdict.status,'violations':verdict.violated_constraints,
                'missing_evidence':verdict.missing_required,'failure_reason':trace['failure_reason'],
                'steps':trace['control_step_count'],'simulation_time_s':trace['times_s'][-1],
                'simulation_wall_s':elapsed,'total_wall_s':time.monotonic()-started,
                'initial_state_digest':trace['initial_state_digest'],'model_digest':trace['model_digest'],
                'file':str(path),'file_sha256':hash_artifact(path)}
    finally: task.close()


def freeze_sources(root):
    from esrb.utils.hashing import hash_artifact
    repo=Path(__file__).resolve().parents[3]
    files=[]
    for directory in ('esrb','experiments','configs'):
        files += [p for p in (repo/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files.append(repo/'pyproject.toml')
    manifest={str(p.relative_to(repo)):hash_artifact(p) for p in files}
    (root/'source_hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open(root/'sources.tar.gz','w:gz',compresslevel=1) as archive:
        for path in files: archive.add(path,arcname=str(path.relative_to(repo)),recursive=False)
    return hash_artifact(root/'source_hashes.json')


def main(args=None):
    if args is None:
        import argparse
        from esrb.cli.options import options_tools_diagnose_production_workcells
        parser=argparse.ArgumentParser(); options_tools_diagnose_production_workcells(parser); args=parser.parse_args()
    # Small IK systems are much faster with one BLAS thread per independent worker.
    os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
    root=args.out_dir.resolve()
    if root.exists(): raise ValueError('acceptance output must be a new directory; frozen runs are never resumed')
    if args.workers<1 or not args.seeds or min(args.seeds)<0 or len(set(args.seeds))!=len(args.seeds):
        raise ValueError('positive workers and distinct nonnegative seeds are required')
    root.mkdir(parents=True)
    protocol={'schema':'panda-production-engineering/1','scenarios':args.scenarios,'seeds':args.seeds,
              'candidate_seed':args.seeds[0],'workers':args.workers,'physics_dt_s':.002,'control_dt_s':.02,
              'episode_limit_s':120,'minimum_success_fraction':.9,'model_calls':0,'hardware_actions':0,
              'source_manifest_sha256':freeze_sources(root)}
    import platform
    from importlib.metadata import version
    protocol['runtime']={'python':platform.python_version(),'mujoco':version('mujoco'),
                         'numpy':version('numpy'),'scipy':version('scipy')}
    protocol['models']={}
    for name in args.scenarios:
        task,_=task_and_policy(name)
        protocol['models'][name]=task.model_digest
        task.close()
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    context=multiprocessing.get_context('spawn')
    resets=[]
    # A fresh pool per witness guarantees independent processes on Python 3.10+.
    for name in args.scenarios:
        for _ in range(2):
            with ProcessPoolExecutor(max_workers=1,mp_context=context) as pool:
                resets.append(pool.submit(reset_worker,name,args.seeds[0]).result())
    rows=[]
    jobs=[(name,seed,False) for name in args.scenarios for seed in args.seeds]
    jobs += [(name,args.seeds[0],True) for name in args.scenarios]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=context) as pool:
        pending=[pool.submit(episode_worker,str(root),*job) for job in jobs]
        for future in as_completed(pending):
            row=future.result(); rows.append(row)
            print(f"{len(rows)}/{len(jobs)} {row['scenario']} seed={row['seed']} candidate={row['candidate']} task={row['task_success']} ESR={row['oracle_status']}",flush=True)
            (root/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
    summaries=[]
    for name in args.scenarios:
        group=[r for r in rows if r['scenario']==name]
        baselines=[r for r in group if not r['candidate']]
        witnesses=[r for r in resets if r['scenario']==name]
        reset_ok=len({r['pid'] for r in witnesses})==2 and witnesses[0]['initial_state_digest']==witnesses[1]['initial_state_digest']
        success=sum(r['task_success'] for r in baselines)
        model_ok=all(r['model_digest']==protocol['models'][name] for r in group)
        ok=success>=.9*len(args.seeds) and reset_ok and model_ok and not any(r['missing_evidence'] for r in group)
        ok=ok and all(r['oracle_status']=='PASS' for r in baselines if r['task_success'])
        ok=ok and any(not r['task_success'] for r in group if r['candidate'])
        summaries.append({'scenario':name,'baseline_successes':success,'baseline_episodes':len(baselines),
                          'independent_reset_match':reset_ok,'frozen_model_match':model_ok,'status':'PASS' if ok else 'FAIL'})
    summary={'status':'PASS' if all(r['status']=='PASS' for r in summaries) else 'FAIL',
             'scope':'controlled_engineering_acceptance','scenarios':summaries,'episodes':rows,'reset_witnesses':resets,
             'model_calls':0,'hardware_actions':0}
    (root/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary['scenarios'],indent=2),flush=True)
    if args.render and summary['status']=='PASS':
        from esrb.tools.render.focused_workcells import render_acceptance
        render_acceptance(summary,args.assets_dir.resolve())
    if summary['status']!='PASS': raise RuntimeError('engineering acceptance failed; retain all records and inspect verification.json')
