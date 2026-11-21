"""Native Stack execution; called only in the simulator interpreter."""
import importlib
import json
import time
import traceback
from pathlib import Path
from esrb.runtime.io import read, save, digest as sha
from esrb.runtime.paths import project_root
from esrb.backends.aspire.services import configure_native_runtime
from esrb.detection.stack_gate import validate_rules
from esrb.detection.stack_gate import classify_geometry

REPO = project_root()

def execute(root, number, version, rules):
    import numpy as np
    import mujoco
    from esrb.backends.aspire.evidence import native_state, identity
    from cap_harness.robosuite.adapter import RobosuiteAdapter
    from cap_harness.artifacts import RunRecorder
    from cap_harness.runtime import ProgramExecutor
    from cap_harness.contracts import ArmCommand, RobotAction, StepResult
    from cap_harness.errors import ApiError, ErrorCode
    from esrb.backends.aspire.capture.stack import StackSafetyCapture
    run = importlib.import_module('cap_harness.run')
    adapter_module = importlib.import_module('cap_harness.robosuite.adapter')
    protocol = read(root / 'protocol.json')
    configure_native_runtime(run, protocol)
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
            self.target, self.base = env.cubeA_body_id, env.cubeB_body_id
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
            self.table_z = float(env.sim.data.get_site_xpos('table_top')[2])
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
            self.safety = StackSafetyCapture(env,self.groups,rules,out)

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
                    if groups & {'red_cube','green_cube','gripper'}:
                        wrench = np.zeros(6)
                        mujoco.mj_contactForce(sim.model._model, sim.data._data, i, wrench)
                        contacts.append({'geom1':g1,'geom2':g2,'groups':[self.groups[g1],self.groups[g2]],
                                         'position_m':c.pos.tolist(),'normal_force_n':float(wrench[0])})
                red = sim.data.body_xpos[self.target].copy()
                green = sim.data.body_xpos[self.base].copy()
                grasp = bool(env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.cubeA))
                self.grasped |= grasp
                self.lifted |= bool(grasp and red[2] > self.table_z+.06)
                row = {'step':self._elapsed_steps,'time_s':float(sim.data.time),
                       'red_xyz_m':red.tolist(),'green_xyz_m':green.tolist(),
                       'red_quat_wxyz':sim.data.body_xquat[self.target].tolist(),
                       'green_quat_wxyz':sim.data.body_xquat[self.base].tolist(),
                       'red_grasped':grasp,'native_success':bool(env._check_success()),
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

    adapter_module.RobosuiteAdapter = Adapter
    run.RunRecorder = Recorder
    run.ProgramExecutor = Executor
    exit_code = 0
    try:
        outcome = run.run_program(benchmark='robosuite',suite='cube_stack',task_id=0,seed=seed,
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
                       safety_assessment='physics-step Stack gate; bounded frozen rule scope')
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
