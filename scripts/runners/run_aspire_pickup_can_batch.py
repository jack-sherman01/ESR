"""Parallel fixed-within-round PickPlaceCan skill experiments, without independent retests.

Policies execute in upstream ASPIRE's restricted public API. New batches use a
physics-step safety witness and an atomic single-execution policy admission gate.
"""
from __future__ import annotations

# Allow direct execution from this subdirectory as well as package imports.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / '.esr-local/aspire/checkout/.venv-robosuite/bin/python'
from scripts.gates.pickup_can_gate_rules import classify_geometry, validate_rules
from scripts.gates.pickup_can_gate_state import (commit as commit_gate, load_chain, load_rules,
                              prepare, preflight, round_lock)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def init(root, seeds=None, versions=5):
    if versions not in range(1,6):
        raise ValueError('versions per round must be between 1 and 5')
    root.mkdir(parents=True, exist_ok=False)
    supplied = seeds is not None
    seeds = list(seeds) if supplied else []
    if supplied and (not seeds or len(seeds)>10 or len(set(seeds))!=len(seeds)
                     or any(type(s) is not int or not 0<=s<2**31 for s in seeds)):
        raise ValueError('expected 1-10 distinct nonnegative 31-bit seeds')
    while not supplied and len(seeds) < 10:
        seed = secrets.randbelow(2**31 - 1) + 1
        if seed not in seeds:
            seeds.append(seed)
    rules = read(REPO/'configs/experiment/pickup_can_safety_gate.json')
    validate_rules(rules)
    save(root/'safety_rules.json',rules)
    implementation_paths = ['scripts/runners/run_aspire_pickup_can_batch.py','scripts/gates/pickup_can_gate_state.py',
                            'scripts/gates/pickup_can_safety_capture.py','scripts/gates/pickup_can_gate_rules.py','esrb/detection/pickup_can_gate.py','scripts/gates/stack_gate_rules.py','esrb/detection/stack_gate.py','scripts/feedback/pickup_can_task_feedback.py','integrations/aspire/pickup_can_registry.py']
    implementation_paths.append('scripts/gates/simulation_capture.py')
    protocol = {'schema': 'esrb.pickup_can_batch/1', 'task': 'pickup_can', 'environment': 'PickPlaceCan',
                'rounds': len(seeds), 'versions_per_round': versions, 'stop_on_success': False,
                'seed_selection': 'explicit frozen panel' if supplied else '10 distinct SystemRandom draws; fixed within round',
                'seeds': seeds, 'max_control_steps': 1000, 'control_frequency_hz': 20,
                'camera_height': 128, 'camera_width': 128, 'video_stride': 4,
                'renderer': 'EGL with per-round GPU assignment', 'controller': 'JOINT_POSITION',
                'physics_timestep_s': .002, 'independent_retest': False, 'offline_physics_audit': False,
                'telemetry': '20 Hz post-control state, object poses, sampled contacts; not all physics contacts',
                'safety_gate': {'enabled':True,'rules_sha256':sha(root/'safety_rules.json'),
                                'implementation_sha256':{name:sha(REPO/name) for name in implementation_paths},
                                'sampling':'every 0.002-second physics step','parent_update':'accept only; reject/unknown retain parent'},
                'generator': 'current interactive assistant; archived task-only prompt and feedback; session context is not blinded',
                'candidate_prompt': 'PickPlaceCan task and public motion/vision feedback only; no admission instructions',
                'perception': 'public calibrated RGB-D color/geometry', 'ik': 'local CPU PyRoki',
                'planner': 'official joint interpolation', 'goal_observation': 'static destination fixture center supplied in public task language; no actual object state exposed', 'created_at_utc': datetime.now(timezone.utc).isoformat()}
    save(root / 'protocol.json', protocol)
    for number, seed in enumerate(seeds, 1):
        folder = root / f'round_{number:02d}'
        folder.mkdir()
        save(folder / 'round.json', {'round': number, 'seed': seed})
    print(json.dumps(protocol, indent=2))


def episode(root, number, version):
    with round_lock(root/f'round_{number:02d}'):
        rules, _chain = preflight(root,number,version)
        code = _episode(root,number,version,rules)
        decision = commit_gate(root,number,version)
        print(json.dumps({'gate_decision':decision['decision'],'reason':decision['reason'],
                          'active_version':(decision['parent_after'] or {}).get('version')}),flush=True)
        return code


def _episode(root, number, version, rules):
    import numpy as np
    import mujoco
    from scripts.gates.simulation_capture import native_state, identity
    from cap_harness.robosuite.adapter import RobosuiteAdapter
    from cap_harness.artifacts import RunRecorder
    from cap_harness.runtime import ProgramExecutor
    from cap_harness.contracts import ArmCommand, RobotAction, StepResult
    from cap_harness.errors import ApiError, ErrorCode
    from scripts.gates.pickup_can_safety_capture import PickupCanSafetyCapture
    run = importlib.import_module('cap_harness.run')
    adapter_module = importlib.import_module('cap_harness.robosuite.adapter')
    protocol = read(root / 'protocol.json')
    round_folder = root / f'round_{number:02d}'
    folder = round_folder / f'version_{version:02d}'
    out = folder / 'execution'
    out.mkdir(exist_ok=False)
    program = folder / 'program.py'
    seed = protocol['seeds'][number - 1]
    settings_path = folder / 'execution_settings.json'
    settings = read(settings_path) if settings_path.exists() else {'camera_sampling_hz':20, 'fresh_explicit_observation':False}
    save(out/'execution_settings.json', settings)
    (out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    for name in protocol['safety_gate']['implementation_sha256']:
        destination = out/'gate_sources'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((REPO/name).read_bytes())
    start = time.monotonic()
    holder = {}
    holder['settle_success_steps'] = holder['settle_steps'] = 0
    stream = (out / 'telemetry.jsonl').open('w')

    class Recorder(RunRecorder):
        def __init__(self, **kwargs):
            kwargs['video_frame_stride'] = protocol['video_stride']
            super().__init__(**kwargs)

    class Adapter(RobosuiteAdapter):
        def _after_native_reset(self, env):
            super()._after_native_reset(env)
            self.target = env.obj_body_id['Can']
            self.initial = native_state(env.sim)
            xml = env.sim.model.get_xml()
            (out / 'model.xml').write_text(xml)
            np.save(out / 'initial_state.npy', self.initial)
            self.initial_id = identity({'model': sha(out/'model.xml'), 'state': self.initial.tolist()})
            reference = round_folder / 'initial_reference.json'
            if reference.exists():
                if read(reference)['initial_state_id'] != self.initial_id:
                    raise ValueError('INITIAL_STATE_MISMATCH within fixed-seed round')
            elif version == 1:
                save(reference, {'seed': seed, 'initial_state_id': self.initial_id,
                                  'model_sha256': sha(out/'model.xml')})
            else:
                raise ValueError('MISSING_ROUND_INITIAL_REFERENCE')
            self.rows, self.lifted, self.grasped = [], False, False
            self.table_z = float(env.bin1_pos[2])+.02
            self.groups = []
            self.camera_observables = [name for name in env._observables if name.endswith(('_image','_depth'))]
            for name in self.camera_observables:
                env.modify_observable(name, 'sampling_rate', settings['camera_sampling_hz'])
            self.in_native_step = False
            for i in range(env.sim.model.ngeom):
                name = env.sim.model.geom_id2name(i) or ''
                body = int(env.sim.model.geom_bodyid[i])
                lineage = []
                while body:
                    lineage.append(env.sim.model.body_id2name(body) or '')
                    body = int(env.sim.model.body_parentid[body])
                group = classify_geometry(name,lineage)
                self.groups.append(group)
            save(out/'geometry_groups.json', [{'id':i,'name':env.sim.model.geom_id2name(i),'group':g} for i,g in enumerate(self.groups)])
            holder['adapter'] = self
            self.safety = PickupCanSafetyCapture(env,self.groups,rules,out)

        def _episode_language(self, metadata):
            # Static fixture goal, not object state: expose it explicitly as task calibration.
            goal = self.native_env.target_bin_placements[self.native_env.object_id].copy()
            goal[2] += .08
            base_goal = self._primary_from_world() @ np.array([*goal,1.])
            return metadata.language + " Fixture goal can-center robot0_base XYZ: " + ",".join(str(float(x)) for x in base_goal[:3])

        def get_observation(self):
            if (settings['fresh_explicit_observation'] and hasattr(self,'camera_observables')
                    and not self.in_native_step):
                self._raw_observation = self.native_env._get_observations(force_update=True)
            return super().get_observation()

        def native_step(self, action):
            if self.safety.stopped:
                return StepResult(ok=False,error=ApiError(ErrorCode.SAFETY_INTERLOCK,'ESRB safety gate stopped this episode'))
            self.in_native_step = True
            try:
                result = super().native_step(action)
            finally:
                self.in_native_step = False
            if result.ok:
                env, sim = self.native_env, self.native_env.sim
                contacts = []
                for i in range(sim.data.ncon):
                    c = sim.data.contact[i]
                    g1, g2 = int(c.geom1), int(c.geom2)
                    groups = {self.groups[g1], self.groups[g2]}
                    if groups & {'can','gripper'}:
                        wrench = np.zeros(6)
                        mujoco.mj_contactForce(sim.model._model, sim.data._data, i, wrench)
                        contacts.append({'geom1':g1,'geom2':g2,'groups':[self.groups[g1],self.groups[g2]],
                                         'position_m':c.pos.tolist(),'normal_force_n':float(wrench[0])})
                can = sim.data.body_xpos[self.target].copy()
                grasp = bool(env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.objects[env.object_id]))
                self.grasped |= grasp
                self.lifted |= bool(grasp and can[2]-.06 > self.table_z+.04)
                row = {'step':self._elapsed_steps,'time_s':float(sim.data.time),
                       'can_xyz_m':can.tolist(),'can_quat_wxyz':sim.data.body_xquat[self.target].tolist(),
                       'can_grasped':grasp,'native_success':bool(env._check_success()),
                       'jaw_open_fraction':float(self.get_robot_state().gripper_positions['primary']),
                       'contacts_sampled':contacts}
                self.rows.append(row)
                stream.write(json.dumps(row,allow_nan=False)+'\n')
            return result

        def close(self):
            if 'summary' not in holder and hasattr(self,'rows'):
                holder['summary'] = {'initial_state_id':self.initial_id, 'control_steps':len(self.rows),
                                     'grasp_observed':self.grasped,'lift_observed':self.lifted,
                                     'native_success_at_any_sample':any(r['native_success'] for r in self.rows),
                                     'last_telemetry':self.rows[-1] if self.rows else None,
                                     'physics_timestep_s':float(self.native_env.sim.model.opt.timestep)}
                if hasattr(self,'safety'):
                    holder['summary']['safety'] = self.safety.finish()
                np.save(out/'final_state.npy',native_state(self.native_env.sim))
            super().close()

    class Executor(ProgramExecutor):
        def execute_program(self, code):
            result = super().execute_program(code)
            adapter = holder['adapter']
            # A terminal hold in this same execution is not an independent retest.
            for _ in range(rules['settle_control_steps']):
                if adapter.safety.stopped or adapter._elapsed_steps >= protocol['max_control_steps']:
                    break
                state = adapter.get_robot_state()
                step = adapter.step(RobotAction(arms={'primary':ArmCommand(mode='joint_position',
                    target=state.joint_positions['primary'],gripper_position=adapter._gripper_targets['primary'],embodiment='robosuite')}))
                if not step.ok:
                    break
                holder['settle_steps'] += 1
                holder['settle_success_steps'] += int(adapter.check_success())
            return result

    from integrations.aspire.pickup_can_registry import install_registry
    install_registry()
    adapter_module.RobosuiteAdapter = Adapter
    run.RunRecorder = Recorder
    run.ProgramExecutor = Executor
    exit_code = 0
    try:
        outcome = run.run_program(benchmark='robosuite',suite='pickup_can',task_id=0,seed=seed,
                                  program_path=program.resolve(),output_root=out/'native',max_steps=protocol['max_control_steps'],
                                  camera_width=128,camera_height=128,capture_videos=True)
        summary = holder.get('summary',{})
        summary.update(status='MEASURED',round=number,version=version,seed=seed,
                       program_ok=outcome.program_ok,native_task_success=outcome.task_success,
                       termination_reason=outcome.termination_reason,native_run=str(outcome.run_dir),
                       wall_seconds=time.monotonic()-start,program_sha256=sha(program),
                       protocol_sha256=sha(root/'protocol.json'), independent_retest=False,
                       execution_settings=settings,
                       settle_steps=holder['settle_steps'],settle_success_steps=holder['settle_success_steps'],
                       safety_assessment='physics-step PickPlaceCan gate; bounded frozen rule scope')
        summary.setdefault('safety',{'status':'UNKNOWN','coverage_complete':False,'errors':['missing physics witness']})
        summary['safety'].update(rules_sha256=protocol['safety_gate']['rules_sha256'],
            witness_sha256=sha(out/'safety_physics.jsonl.gz') if (out/'safety_physics.jsonl.gz').exists() else None)
        save(out/'result.json',summary)
        print(json.dumps({k:v for k,v in summary.items() if k!='last_telemetry'},indent=2),flush=True)
    except Exception:
        save(out/'result.json',{'status':'INFRASTRUCTURE_ERROR','round':number,'version':version,'seed':seed,
                               'wall_seconds':time.monotonic()-start,'error':traceback.format_exc()})
        traceback.print_exc()
        exit_code = 1
    finally:
        stream.close()
        save(out/'artifacts.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifacts.json'})
    return exit_code


def wave(root, version, workers, rounds):
    start=time.monotonic()
    protocol=read(root/'protocol.json')
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
            proc=subprocess.run([str(PYTHON),'-B',str(Path(__file__).resolve()),'episode','--root',str(root),
                                 '--version',str(version),'--round',str(number)],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
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
    summarize(root)


def summarize(root):
    import csv
    protocol=read(root/'protocol.json')
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
    summary={'schema':'esrb.pickup_can_summary/1' if gated else 'esrb.stack_summary/1','requested_rollouts':expected,
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
    fields=['round','seed','version','status','program_ok','native_task_success','grasp_observed','lift_observed','control_steps','wall_seconds','termination_reason']
    with (root/'episodes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:v for k,v in summary.items() if k not in {'executions','rounds'}},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['init','prepare','episode','wave','summarize'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--version',type=int,choices=range(1,6))
    parser.add_argument('--round',type=int,choices=range(1,11))
    parser.add_argument('--rounds',type=int,nargs='+')
    parser.add_argument('--seeds',type=int,nargs='+')
    parser.add_argument('--versions-per-round',type=int,choices=range(1,6),default=5)
    parser.add_argument('--program',type=Path)
    parser.add_argument('--generation-base',type=Path)
    parser.add_argument('--rationale',default='feedback-driven candidate')
    parser.add_argument('--kind',choices=['candidate','controlled_acceptance'],default='candidate')
    parser.add_argument('--workers',type=int,default=10)
    args=parser.parse_args(); root=args.root.resolve()
    if args.action=='init':init(root,args.seeds,args.versions_per_round)
    elif args.action=='prepare':
        if args.program is None or args.round is None or args.version is None:
            parser.error('prepare requires --program, --round and --version')
        print(prepare(root,args.round,args.version,args.program,args.rationale,args.kind,generation_base=args.generation_base))
    elif args.action=='episode':sys.exit(episode(root,args.round,args.version))
    elif args.action=='wave':wave(root,args.version,args.workers,args.rounds or list(range(1,len(read(root/'protocol.json')['seeds'])+1)))
    else:summarize(root)
