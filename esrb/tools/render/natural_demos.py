"""Render archived natural-campaign successes at native 480p.

Replays recorded control actions, checks every control boundary against the
lossless physical trace, and never calls a model, IK service or generated skill.
Run with the campaign's robosuite interpreter and MUJOCO_GL=egl (or osmesa).
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from esrb.runtime.io import digest, read, save

SCENES = {
    'stack': ('Stack', 'stack', 1),
    'pickplacecan': ('PickPlaceCan', 'pickplacecan', 1),
    'two-arm-handover': ('TwoArmHandover', 'handover', 1),
    'nut-assembly': ('NutAssemblySquare', 'nut_assembly', 1),
    'wipe': ('Wipe', 'wipe', 2),
}


def selection(campaign, task, seed=1):
    title, _, index = SCENES[task]
    entry = next(c for c in read(campaign/'manifest.json')['chains']
                 if c['task'] == task and c['chain'] == index)
    root = Path(entry['root'])
    state = read(root/'state.json')
    event = next(e for e in reversed(state['events']) if e['accepted'])
    case = next(c for c in event['cases'] if c['seed'] == seed)
    execution = Path(case['output'])
    result = read(execution/'result.json')
    if (not result['task_qualified'] or not result['program_ok'] or
            not result['native_task_success'] or digest(execution/'result.json') != case['result_sha256']):
        raise ValueError('selected source is not an intact task-qualified episode')
    if digest(root/event['program']) != event['program_sha256']:
        raise ValueError('selected source program changed')
    return title, root, event, execution, result


def recorded_boundaries(execution, control_steps):
    """Read exact start states at each 20Hz control boundary; no resampling."""
    positions, velocities, times = [], [], []
    offset = 0
    physics = execution/'physics'
    artifacts = read(execution/'artifacts.json')
    sources = []
    for chunk in read(physics/'summary.json')['chunks']:
        path = physics/chunk['file']
        sha = digest(path)
        if sha != artifacts[str(path.relative_to(execution))]:
            raise ValueError(f'physical source changed: {path}')
        sources.append({'file': str(path), 'sha256': sha})
        with np.load(path, allow_pickle=False) as data:
            rows = np.arange((-offset) % 25, len(data['time']), 25)
            positions.extend(data['qpos_start'][rows])
            velocities.extend(data['qvel_start'][rows])
            times.extend(data['time'][rows, 0])
            offset += len(data['time'])
    if offset != control_steps*25 or len(positions) != control_steps:
        raise ValueError('physical/control step count mismatch')
    return np.asarray(positions), np.asarray(velocities), np.asarray(times), sources


def decode_action(record):
    from cap_harness.contracts import ArmCommand, RobotAction
    return RobotAction(arms={name: ArmCommand(mode=c['mode'],
        target=np.asarray(c['target']['$array_inline'], dtype=float),
        gripper_position=c.get('gripper_position'), embodiment=c['embodiment'])
        for name, c in record['arms'].items()})


def encode_gif(frames, destination):
    """125 frames at 25fps: five seconds including the last native task state."""
    height, width = frames[0].shape[:2]
    args = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-threads', '1',
            '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
            '-framerate', '25', '-i', '-', '-filter_complex_threads', '1',
            '-filter_complex', '[0:v]split[a][b];[a]palettegen=stats_mode=diff[p];'
                               '[b][p]paletteuse=dither=sierra2_4a',
            '-loop', '0', str(destination)]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=errors)
        try:
            for frame in frames:
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
            process.stdin.close()
            if process.wait() != 0:
                errors.seek(0)
                raise RuntimeError(errors.read().decode())
        finally:
            if process.poll() is None:
                process.kill(); process.wait()


def render(campaign, out, task, *, preview=False):
    import mujoco
    from PIL import Image
    from cap_harness.robosuite.adapter import RobosuiteAdapter
    from esrb.backends.aspire.evidence import native_state
    if task == 'pickplacecan':
        from esrb.backends.aspire.pickup_can_registry import install_registry
        install_registry()
    from cap_harness.robosuite import registry
    title, root, event, execution, result = selection(campaign, task)
    protocol = read(root/'protocol.json')
    width, height = 640, 480
    out.mkdir(parents=True, exist_ok=True)
    stem = task.replace('-', '_')+'_success_480p'
    destination = out/(stem+'.gif')
    if destination.exists() and not preview:
        raise FileExistsError(destination)
    adapter = RobosuiteAdapter(registry=registry.RobosuiteTaskRegistry(), camera_width=protocol['camera_width'],
        camera_height=protocol['camera_height'], horizon=protocol['max_control_steps'])
    initial_observation = adapter.reset(protocol['scene']['task'], result['seed'])
    env = adapter.native_env
    model, data = env.sim.model._model, env.sim.data._data
    renderer = None
    try:
        actual_xml_sha = hashlib.sha256(env.sim.model.get_xml().encode()).hexdigest()
        if actual_xml_sha != digest(execution/'model.xml'):
            raise ValueError('render replay initial model does not match source')
        expected_initial = np.load(execution/'initial_state.npy', allow_pickle=False)
        if not np.array_equal(native_state(env.sim), expected_initial):
            raise ValueError('render replay initial integration state does not match source')
        # Only rendering settings change, after exact original-model identity check.
        model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
        model.vis.global_.offheight = max(height, model.vis.global_.offheight)
        renderer = mujoco.Renderer(model, height=height, width=width)
        options = mujoco.MjvOption()
        # Match robosuite's visible geometry groups (hide collision-only duplicates).
        options.geomgroup[0] = 0
        options.geomgroup[1] = 1

        def frame():
            renderer.update_scene(data, camera='agentview', scene_option=options)
            return renderer.render().copy()

        if preview:
            final = np.load(execution/'final_state.npy', allow_pickle=False)
            mujoco.mj_setState(model, data, final, mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(model, data)
            Image.fromarray(frame()).save(out/(stem+'_preview.png'))
            return {'task': task, 'preview': True}
        qpos, qvel, times, sources = recorded_boundaries(execution, result['control_steps'])
        steps_path = Path(result['native_run'])/'episode/steps.jsonl'
        records = [json.loads(line) for line in steps_path.read_text().splitlines() if line.strip()]
        if len(records) != result['control_steps'] or not all(r['ok'] for r in records):
            raise ValueError('native action record is incomplete')
        # Same native controller/physics, no costly per-tick camera observation rendering.
        for name in env._observables:
            if name.endswith(('_image', '_depth')):
                env.modify_observable(name, 'enabled', False)
        # Native API needs the initial camera data only to package its step result.
        adapter.get_observation = lambda: initial_observation
        indices = np.rint(np.linspace(0, len(records), 125)).astype(int)
        counts = {int(i): int(np.count_nonzero(indices == i)) for i in np.unique(indices)}
        frames = [frame() for _ in range(counts.get(0, 0))]
        max_position_error = max_velocity_error = 0.
        for index, record in enumerate(records):
            pos_error = float(np.max(np.abs(data.qpos-qpos[index])))
            vel_error = float(np.max(np.abs(data.qvel-qvel[index])))
            max_position_error = max(pos_error, max_position_error)
            max_velocity_error = max(vel_error, max_velocity_error)
            if pos_error > 1e-9 or vel_error > 1e-8 or abs(data.time-times[index]) > 1e-9:
                raise ValueError(f'replay drift before control {index}: qpos={pos_error}, qvel={vel_error}')
            replay = adapter.native_step(decode_action(record['action']))
            if not replay.ok:
                raise ValueError(f'failed native replay step {index}: {replay.error}')
            if index+1 in counts:
                rendered = frame()
                frames.extend([rendered]*counts[index+1])
        expected_final = np.load(execution/'final_state.npy', allow_pickle=False)
        actual_final = native_state(env.sim)
        final_error = float(np.max(np.abs(actual_final-expected_final)))
        if final_error > 1e-8 or not env._check_success() or len(frames) != 125:
            raise ValueError(f'final task/state replay mismatch: {final_error}')
        encode_gif(frames, destination)
        Image.fromarray(frames[-1]).save(out/(stem+'.png'))
        with Image.open(destination) as gif:
            duration = sum((gif.seek(i), gif.info.get('duration', 0))[1] for i in range(gif.n_frames))/1000
            gif_frames = gif.n_frames
            if gif.size != (width, height) or abs(duration-5) > .01:
                raise ValueError('encoded GIF dimensions or duration mismatch')
        metadata = {'task': title, 'campaign': str(campaign.resolve()),
            'chain': protocol['chain_index'], 'version': event['version'], 'seed': result['seed'],
            'program_ok': result['program_ok'], 'native_task_success': result['native_task_success'],
            'all_five_development_cases_successful': event['accepted'], 'safety_audited': False,
            'source_execution': str(execution), 'source_result_sha256': digest(execution/'result.json'),
            'source_program': str(root/event['program']), 'source_program_sha256': event['program_sha256'],
            'model_sha256': actual_xml_sha, 'initial_state_id': result['initial_state_id'],
            'source_initial_state_sha256': digest(execution/'initial_state.npy'),
            'source_final_state_sha256': digest(execution/'final_state.npy'),
            'source_actions': str(steps_path), 'source_actions_sha256': digest(steps_path),
            'source_physics': sources, 'control_steps': len(records),
            'physics_timestep_s': .002, 'control_frequency_hz': 20,
            'source_duration_s': len(records)/20, 'output_duration_s': duration,
            'playback_speed': (len(records)/20)/duration, 'size_px': [width,height],
            'output_frames': gif_frames, 'camera': 'agentview',
            'max_control_boundary_qpos_error': max_position_error,
            'max_control_boundary_qvel_error': max_velocity_error,
            'max_final_integration_state_error': final_error,
            'replayed_terminal_task_success': bool(env._check_success()),
            'render_only_overrides': ['framebuffer dimensions', 'disabled camera observation updates',
                'temporal sampling', 'GIF palette'],
            'method': 'Replay archived native actions without model/API planning; verify every '
                      'control-boundary state and the full final integration state against raw records.',
            'renderer_source_sha256': digest(Path(__file__)),
            'output': destination.name, 'gif_sha256': digest(destination)}
        if task == 'wipe':
            metadata['replayed_cleaned_markers'] = len(env.wiped_markers)
            metadata['total_markers'] = env.num_markers
        save(out/(stem+'.source.json'), metadata)
        return {k: metadata[k] for k in ('task','chain','version','seed','output','output_duration_s',
                                        'max_final_integration_state_error')}
    finally:
        if renderer is not None:
            renderer.close()
        adapter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--task', choices=list(SCENES), required=True)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    print(json.dumps(render(args.campaign, args.out_dir, args.task, preview=args.preview)), flush=True)


if __name__ == '__main__':
    main()
