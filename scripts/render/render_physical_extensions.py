"""Replay audited engineering examples as README GIFs (MuJoCo + Pillow).

Only rendering is added: the recorded policy, physical reset and entire trace
must match the saved evidence exactly. No agent service is called.

MUJOCO_GL=egl python scripts/render/render_physical_extensions.py \
  --acceptance private/runs/physical_extensions_work_20261116/verified/acceptance.json \
  --out-dir assets
"""

# Allow direct execution from this subdirectory as well as package imports.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import gc
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from esrb.agent_api.profiles import BY_SCENARIO
from esrb.oracles.multiconstraint import scenario_oracle
from esrb.tasks.extensions.registry import EXTENSIONS
from esrb.tasks.extensions.settings import configured_task
from esrb.utils.hashing import hash_artifact, hash_dict

WIDTH, HEIGHT, STRIDE, PLAYBACK_SPEED = 480, 360, 5, 2.0
TITLES = {
    'moving_transport': 'Moving Transport', 'moving_assembly': 'Moving Assembly',
    'constrained_push': 'Constrained Push', 'force_alignment': 'Force Alignment',
    'compliant_assembly': 'Compliant Assembly',
}


def font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def camera_for(name):
    """Free viewing camera: no model geometry, dynamics or state is modified."""
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    planar = name in {'constrained_push', 'force_alignment'}
    camera.lookat[:] = [0.015, 0., 0.015 if planar else 0.12]
    camera.distance = 0.40 if planar else 0.56
    camera.azimuth = 100
    camera.elevation = -65 if planar else -25
    return camera


def measurement_label(name, trace):
    if name.startswith('moving_'):
        return f"surface gap {1000 * trace['surface_distances'][-1]:.1f} mm"
    if name in {'constrained_push', 'force_alignment'}:
        return f"pusher {trace['pusher_forces'][-1]:.1f} N"
    if trace['stages'][-1] in {'insert', 'verify'}:
        return f"depth {1000 * trace['insertion_depth_m'][-1]:.1f} mm"
    return f"load {np.linalg.norm(trace['contact_forces'][-1]):.1f} N"


def caption(frame, title, detail, unsafe):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 34), fill='#141820')
    draw.text((10, 7), title, font=font(17), fill='#ffb17a' if unsafe else '#8fe4bb')
    draw.rectangle((0, HEIGHT-32, WIDTH, HEIGHT), fill='#141820')
    draw.text((10, HEIGHT-25), detail, font=font(14), fill='#eef1f6')
    return image.quantize(colors=128, dither=Image.Dither.NONE)


def first_bad_step(name, trace, protocol):
    """Add a frame for the first observed scalar/contact breach, even off stride."""
    limits = {rule['metric']: rule['limit'] for rule in protocol['rules'] if rule['limit'] is not None}
    checks = []
    for metric, channel, lower in (
        ('min_surface_distance_m', 'surface_distances', True),
        ('proxy_collision', 'proxy_collisions', False),
        ('peak_pusher_force_n', 'pusher_forces', False),
        ('peak_wall_force_n', 'wall_forces', False),
        ('forbidden_collision', 'forbidden_collisions', False),
        ('boundary_excess_m', 'boundary_excess', False),
        ('sustained_force_n', 'sustained_forces', False),
        ('jamming_duration_s', 'jamming_durations', False),
    ):
        if metric in limits and channel in trace:
            values = np.asarray(trace[channel])
            indices = np.flatnonzero(values < limits[metric] if lower else values > limits[metric])
            if len(indices):
                checks.append(int(indices[0])+1)
    for metric, channel in (('peak_force_n', 'contact_forces'), ('peak_torque_nm', 'contact_torques')):
        if metric in limits:
            indices = np.flatnonzero(np.linalg.norm(trace[channel], axis=1) > limits[metric])
            if len(indices):
                checks.append(int(indices[0])+1)
    checks.extend(event['step'] for event in trace.get('drop_events', []))
    return min(checks) if checks else None


def render_example(name, report, source, wrapper, role, out):
    raw, trace = wrapper['raw_result'], wrapper['raw_result']['episode_data']
    receipt = report['generations'][0]['proposal_receipt']
    components = receipt['parent_components' if role == 'success' else 'candidate_components']
    profile = BY_SCENARIO[name]
    if raw['policy_digest'] != profile.policy_digest(components):
        raise ValueError('selected rollout does not belong to the recorded policy')
    task_type = configured_task(EXTENSIONS[name][0], trace.get('task_settings'))
    task, policy = task_type(), profile.build(components['policy'])
    limits = {rule['metric']: rule['limit'] for rule in report['measurement_protocol']['rules'] if rule['limit'] is not None}
    oracle = scenario_oracle(name, limits=limits)
    unsafe = role != 'success'
    first_bad = first_bad_step(name, trace, report['measurement_protocol'])
    capture = {1, len(trace['commands'])}
    capture.update(range(STRIDE, len(trace['commands'])+1, STRIDE))
    capture.update(index+1 for index in range(1, len(trace['stages'])) if trace['stages'][index] != trace['stages'][index-1])
    if first_bad is not None:
        capture.update(i for i in (first_bad-1, first_bad, first_bad+1) if 1 <= i <= len(trace['commands']))
    frames, sample_steps, times = [], [], []
    renderer = None
    try:
        obs = task.reset(raw['case']['seed'], disturbance=raw['case']['disturbance'])
        renderer = mujoco.Renderer(task._model, height=HEIGHT, width=WIDTH)
        camera = camera_for(name)
        done = False
        for step in range(1, len(trace['commands'])+1):
            obs, _, done, replay = task.step(policy.act(obs))
            if replay['commands'][-1] != trace['commands'][step-1]:
                raise ValueError(f'{name}: policy command mismatch at {step}')
            if step in capture:
                renderer.update_scene(task._data, camera=camera)
                title = TITLES[name] + (' | Unsafe candidate' if unsafe else ' | Success')
                detail = f"{replay['stages'][-1].upper()} | {replay['times_s'][-1]:.2f}s | {measurement_label(name, replay)}"
                if done:
                    detail = ('TASK FAILED' if not task.evaluate(replay).success else 'TASK COMPLETE') + (' | UNSAFE' if unsafe else ' | SAFE')
                frames.append(caption(renderer.render(), title, detail, unsafe))
                sample_steps.append(step)
                times.append(replay['times_s'][-1])
            if done:
                if step != len(trace['commands']):
                    raise ValueError('episode terminated before the recorded final step')
                break
        if not done:
            raise ValueError('recorded command stream did not terminate the episode')
        replay['control_step_count'] = len(replay['commands'])
        if hash_dict(replay) != hash_dict(trace):
            raise ValueError(f'{name}: rendering changed the recorded physical trace')
        assessment = oracle.evaluate(replay, raw['oracle']['generation'])
        completed = bool(task.evaluate(replay).success)
        if hash_dict(raw) != wrapper['evidence_id'] or assessment.safe != raw['oracle']['safe']:
            raise ValueError('source evidence or safety outcome mismatch')
        if unsafe and assessment.status != 'VIOLATION' or not unsafe and not (assessment.safe and completed):
            raise ValueError('selected example does not satisfy the displayed label')
        filename = name + ('_failure.gif' if unsafe else '.gif')
        durations = [max(20, round(1000*(b-a)/PLAYBACK_SPEED/10)*10) for a, b in zip(times, times[1:])] + [1200]
        frames[0].save(out/filename, save_all=True, append_images=frames[1:], duration=durations,
                       loop=0, optimize=True, disposal=2)
        return {'file': filename, 'gif_digest': hash_artifact(out/filename),
            'scenario': name, 'role': role, 'source_report': str(source.resolve()),
            'source_evidence_id': wrapper['evidence_id'], 'case': raw['case'],
            'policy': components['policy'], 'policy_digest': raw['policy_digest'],
            'policy_version': receipt['base_version' if role == 'success' else 'candidate_version'],
            'trace_digest': hash_dict(replay), 'exact_replay': True,
            'viewing_camera': {'lookat_m': camera.lookat.tolist(), 'distance_m': camera.distance,
                               'azimuth_deg': camera.azimuth, 'elevation_deg': camera.elevation},
            'task_success': completed, 'oracle_status': assessment.status,
            'violations': assessment.violated_constraints,
            'frames': len(frames), 'sampled_steps': sample_steps, 'sample_times_s': times,
            'durations_ms': durations, 'first_observed_bad_step': first_bad,
            'simulation_steps': len(replay['commands']), 'simulation_time_s': replay['times_s'][-1]}
    finally:
        if renderer is not None:
            renderer.close()
        task.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acceptance', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, default=Path('assets'))
    args = parser.parse_args()
    acceptance = json.loads(args.acceptance.read_text())
    if acceptance['status'] != 'PASS' or {row['scenario'] for row in acceptance['scenarios']} != set(EXTENSIONS):
        parser.error('a complete audited engineering panel is required')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    demos = []
    for source in acceptance['scenarios']:
        name, path = source['scenario'], Path(source['source_report'])
        if hash_artifact(path) != source['report_file_digest']:
            raise ValueError('source report changed since its audit')
        report = json.loads(path.read_text())
        row = report['generations'][0]
        parent = next(item for item in row['baseline_measurements']
                      if item['raw_result']['task_success'] and item['raw_result']['oracle']['status'] == 'PASS')
        candidates = [item for item in row['gate_plan']['executed'] if item['raw_result']['oracle']['status'] == 'VIOLATION']
        child = next((item for item in candidates if not item['raw_result']['task_success']), candidates[0])
        for role, wrapper in (('success', parent), ('unsafe', child)):
            demo = render_example(name, report, path, wrapper, role, args.out_dir)
            demos.append(demo)
            print(f"{demo['file']}: exact replay, task={demo['task_success']}, safety={demo['oracle_status']}, frames={demo['frames']}", flush=True)
        del report, row, parent, candidates, child, wrapper
        gc.collect()
    metadata = {'scope': 'selected_controlled_engineering_examples_not_success_rates',
        'acceptance_digest': hash_artifact(args.acceptance), 'size_px': [WIDTH, HEIGHT],
        'capture_stride': STRIDE, 'playback_speed': PLAYBACK_SPEED, 'final_frame_pause_ms': 1200,
        'min_frame_duration_ms': 20, 'duration_quantization_ms': 10,
        'camera_only_override': True, 'model_calls': 0, 'hardware_actions': 0, 'demos': demos}
    (args.out_dir/'physical_extensions_animation.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
