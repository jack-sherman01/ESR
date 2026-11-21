"""One isolated native episode. No ESR detector or admission module is imported."""
import json
from pathlib import Path
import time
import traceback
import importlib.metadata
import sys
import xml.etree.ElementTree as ET
import hashlib

from esrb.runtime.io import read, save, digest


def fixture_language(metadata_language, env, primary_from_world):
    """Public static destination calibration; never the measured can pose."""
    import numpy as np
    goal = env.target_bin_placements[env.object_id].copy()
    goal[2] += .08
    base_goal = primary_from_world @ np.array([*goal,1.])
    return metadata_language + ' Fixture goal can-center robot0_base XYZ: ' + ','.join(str(float(x)) for x in base_goal[:3])


def execute(request_path):
    import numpy as np
    import mujoco
    from cap_harness import run
    from cap_harness.robosuite import adapter as adapter_module
    from cap_harness.robosuite.adapter import RobosuiteAdapter
    from cap_harness.artifacts import RunRecorder
    from esrb.backends.aspire.capture.raw import RawPhysicsCapture
    from esrb.backends.aspire.evidence import native_state, identity
    from esrb.backends.aspire.services import configure_native_runtime

    request = read(request_path)
    out = Path(request['output']); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); holder = {}; telemetry = (out / 'telemetry.jsonl').open('x')
    source_identities = {'episode_source_sha256': digest(Path(__file__)),
                         'capture_source_sha256': digest(Path(__file__).parents[1]/'capture/raw.py')}
    if request['scene']['prefix'] == 'pickup_can':
        from esrb.backends.aspire.pickup_can_registry import install_registry
        install_registry()
    configure_native_runtime(run, {'runtime': {'ik_service_url': request['ik_service_url']}})
    BaseApi = run.CapApi
    if request['scene']['prefix'] == 'wipe':
        from esrb.backends.aspire.wipe_public_api import WipeCapApi
        BaseApi = WipeCapApi
    run.CapApi = BaseApi

    class Recorder(RunRecorder):
        def __init__(self, **kwargs):
            kwargs['video_frame_stride'] = request.get('video_stride', 4)
            super().__init__(**kwargs)

    class Adapter(RobosuiteAdapter):
        def _episode_language(self, metadata):
            if request['scene']['prefix'] == 'pickup_can':
                return fixture_language(metadata.language,self.native_env,self._primary_from_world())
            return super()._episode_language(metadata)

        def _after_native_reset(self, env):
            super()._after_native_reset(env)
            holder['adapter'] = self
            if abs(float(env.sim.model.opt.timestep) - .002) > 1e-12:
                raise ValueError('physics timestep differs from frozen specification')
            if float(env.control_freq) != 20:
                raise ValueError('control frequency differs from frozen specification')
            if request['scene']['prefix'] == 'wipe':
                self.hand_from_grip = (np.linalg.inv(self._body_transform(env.sim, 'robot0_right_hand'))
                                       @ self._site_transform(env.sim, self._grip_site(env.robots[0])))
                save(out / 'tool_mapping.json', {'hand_from_grip': self.hand_from_grip.tolist(),
                     'native_action_dimension': env.robots[0].action_dim})
                if env.robots[0].action_dim != 7:
                    raise ValueError('Wipe fixed-tool action dimension changed')
            (out / 'model.xml').write_text(env.sim.model.get_xml())
            resources = []
            for element in ET.fromstring((out/'model.xml').read_text()).iter():
                filename = element.get('file')
                if filename:
                    path = Path(filename)
                    resources.append({'file': filename, 'resolved': path.is_file(),
                                      'sha256': digest(path) if path.is_file() else None})
            save(out/'model_resources.json', resources)
            cameras = []
            for i in range(env.sim.model.ncam):
                cameras.append({'name': env.sim.model.camera_id2name(i), 'fovy_deg': float(env.sim.model.cam_fovy[i]),
                    'position_world_m': env.sim.data.cam_xpos[i].tolist(),
                    'rotation_opengl_camera_to_world': env.sim.data.cam_xmat[i].reshape(3,3).tolist()})
            save(out/'camera_calibration.json', {'width': request['camera_width'], 'height': request['camera_height'],
                 'initial_cameras': cameras, 'dynamic_calibration': 'native public observation trace camera_pose/intrinsics',
                 'pixel_convention': 'ASPIRE vertically flipped RGB and metric depth'})
            initial = native_state(env.sim); np.save(out / 'initial_state.npy', initial)
            initial_id = identity({'model': digest(out / 'model.xml'), 'state': initial.tolist()})
            holder.update(initial_state_id=initial_id, native_success_at_any_sample=False, control_steps=0)
            save(out / 'initial_identity.json', {'seed': request['seed'], 'initial_state_id': initial_id,
                 'model_sha256': digest(out / 'model.xml')})
            if request.get('expected_initial_state_id') not in (None, initial_id):
                raise ValueError('INITIAL_STATE_MISMATCH')
            self.camera_observables = [n for n in env._observables if n.endswith(('_image', '_depth'))]
            for name in self.camera_observables:
                env.modify_observable(name, 'sampling_rate', 5)
            self.in_native_step = False
            self.raw = RawPhysicsCapture(env, out / 'physics')
            self.rgbd_root = out/'rgbd'; self.rgbd_root.mkdir()
            self.rgbd_payloads = self.rgbd_root/'payloads'; self.rgbd_payloads.mkdir()
            self.rgbd_stream = (self.rgbd_root/'observations.jsonl').open('x')
            self.rgbd_count = 0

        def get_observation(self):
            if hasattr(self, 'camera_observables') and not self.in_native_step:
                self._raw_observation = self.native_env._get_observations(force_update=True)
            observation = super().get_observation()
            if hasattr(self,'rgbd_stream'):
                import imageio.v2 as imageio
                cameras = {}
                for name,camera in observation.cameras.items():
                    rgb = np.ascontiguousarray(camera.rgb); depth = np.ascontiguousarray(camera.depth_m)
                    key = hashlib.sha256(rgb.tobytes()+depth.dtype.str.encode()+str(depth.shape).encode()+depth.tobytes()).hexdigest()
                    rgb_file=self.rgbd_payloads/f'{key}.png';depth_file=self.rgbd_payloads/f'{key}.npz'
                    if not rgb_file.exists():
                        imageio.imwrite(rgb_file,rgb,compress_level=1)
                        np.savez_compressed(depth_file,depth_m=depth)
                    pose=camera.camera_pose
                    cameras[name]={'rgb':f'payloads/{key}.png','depth':f'payloads/{key}.npz',
                        'intrinsics':camera.intrinsics.tolist(),'frame':camera.frame,
                        'camera_pose':{'position':pose.position.tolist(),
                            'quaternion_wxyz':pose.quaternion_wxyz.tolist(),'frame':pose.frame},
                        'timestamp_s':camera.timestamp_s}
                self.rgbd_stream.write(json.dumps({'index':self.rgbd_count,'timestamp_s':observation.timestamp_s,
                    'explicit_fresh_request':not self.in_native_step,'cameras':cameras},allow_nan=False)+'\n')
                self.rgbd_count+=1
            return observation

        def ik_request(self, target_pose, state, arm):
            if request['scene']['prefix'] == 'wipe':
                from esrb.backends.aspire.wipe_tool_mapping import provider_input_transform
                from cap_harness.geometry import matrix_to_pose
                return matrix_to_pose(provider_input_transform(target_pose.as_matrix(), self.hand_from_grip),
                                      frame=self.BASE_FRAME), state
            return super().ik_request(target_pose, state, arm)

        def native_step(self, action):
            self.in_native_step = True
            try:
                result = super().native_step(action)
            finally:
                self.in_native_step = False
            if result.ok:
                success = bool(self.native_env._check_success())
                holder['native_success_at_any_sample'] |= success
                holder['control_steps'] += 1
                telemetry.write(json.dumps({'control_index': holder['control_steps'],
                    'time_s': float(self.native_env.sim.data.time), 'native_task_success': success,
                    'native_terminated': bool(self._last_terminated),
                    'native_truncated': bool(self._last_truncated)}, allow_nan=False) + '\n')
            return result

        def close(self):
            if self.native_env is not None and hasattr(self, 'raw'):
                holder['terminal_native_success'] = bool(self.native_env._check_success())
                holder['native_terminated'] = bool(self._last_terminated)
                holder['native_truncated'] = bool(self._last_truncated)
                holder['capture'] = self.raw.finish()
                if not self.rgbd_stream.closed:
                    self.rgbd_stream.close()
                holder['rgbd_observations'] = self.rgbd_count
                np.save(out / 'final_state.npy', native_state(self.native_env.sim))
                if request['scene']['prefix'] == 'wipe':
                    holder['task_progress'] = {'cleaned_markers': len(self.native_env.wiped_markers),
                                               'total_markers': self.native_env.num_markers}
            super().close()

    run.RunRecorder = Recorder
    adapter_module.RobosuiteAdapter = Adapter
    result = {'status': 'INFRASTRUCTURE_ERROR', 'seed': request['seed'],
              'scene': request['scene']['key'],
              'program_sha256': digest(Path(request['program'])), 'request_sha256': digest(request_path)}
    code = 1
    try:
        outcome = run.run_program(benchmark='robosuite', suite=request['scene']['task'], task_id=0,
            seed=request['seed'], program_path=Path(request['program']), output_root=out / 'native',
            max_steps=request['max_control_steps'], camera_width=request['camera_width'],
            camera_height=request['camera_height'], capture_videos=True)
        capture = holder.get('capture', {})
        terminal = holder.get('terminal_native_success') is True
        result.update({k: v for k, v in holder.items() if k != 'adapter'})
        result.update(status='MEASURED' if capture.get('coverage_complete') else 'EVIDENCE_INCOMPLETE',
                      program_ok=bool(outcome.program_ok), native_task_success=terminal,
                      task_qualified=bool(outcome.program_ok and terminal),
                      termination_reason=outcome.termination_reason, native_run=str(outcome.run_dir))
        for path in Path(outcome.run_dir).glob('trace/calls/*/output.json'):
            error = read(path).get('error')
            if isinstance(error, dict) and error.get('code') == 'provider_unavailable':
                result.update(status='INFRASTRUCTURE_ERROR', error='provider_unavailable')
                break
        # A valid zero-motion program can be measured; no physical records are fabricated.
        code = 0 if result['status'] == 'MEASURED' else 1
    except Exception:
        result['error'] = traceback.format_exc()
        traceback.print_exc()
        adapter = holder.get('adapter')
        if adapter is not None:
            try:
                adapter.close()
            except Exception:
                result['cleanup_error'] = traceback.format_exc()
    finally:
        telemetry.close()
        result['wall_seconds'] = time.monotonic() - started
        result['online_assessment'] = False
        packages = {}
        for name in ['mujoco', 'numpy', 'robosuite', 'cap-harness']:
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        save(out/'environment.json', {'python': sys.version, 'packages': packages,
             **source_identities})
        save(out / 'result.json', result)
        save(out / 'artifacts.json', {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*'))
                                     if p.is_file() and p.name != 'artifacts.json'})
    return code
