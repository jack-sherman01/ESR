"""Render controlled examples or replay saved engineering trajectories.

With --acceptance, physical reset and model must match saved evidence; a saved
task-success panel can omit an independent safety assessment. With
--reference-preview, recorded inputs are reused for a
current controlled rollout. No agent service is called.

MUJOCO_GL=egl python -m esrb tools render physical-extensions \
  --acceptance NEW_VERIFIED_PANEL/acceptance.json \
  --out-dir assets
"""


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
        return f"belt {trace['belt_speed_ms']:.3f} m/s"
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
        ('robot_collision', 'robot_collisions', False),
        ('part_drop', 'part_drops', False),
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
    checks.extend(event.get('step',round(event.get('time_s',0.)/.02)) for event in trace.get('drop_events', []))
    return min(checks) if checks else None


def render_example(name, report, source, wrapper, role, out, duration=5.):
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
    stride=max(STRIDE,int(np.ceil(len(trace['commands'])/80))) if name.startswith('moving_') else STRIDE
    capture.update(range(stride, len(trace['commands'])+1, stride))
    capture.update(index+1 for index in range(1, len(trace['stages'])) if trace['stages'][index] != trace['stages'][index-1])
    if first_bad is not None:
        capture.update(i for i in (first_bad-1, first_bad, first_bad+1) if 1 <= i <= len(trace['commands']))
    frames, sample_steps, times = [], [], []
    renderer = None
    try:
        obs = task.reset(raw['case']['seed'], disturbance=raw['case']['disturbance'])
        moving = name.startswith('moving_')
        if not moving:
            renderer = mujoco.Renderer(task._model, height=HEIGHT, width=WIDTH)
        camera = camera_for(name)
        done = False
        for step in range(1, len(trace['commands'])+1):
            obs, _, done, replay = task.step(policy.act(obs))
            if replay['commands'][-1] != trace['commands'][step-1]:
                raise ValueError(f'{name}: policy command mismatch at {step}')
            if step in capture:
                title = TITLES[name] + (' | Unsafe candidate' if unsafe else ' | Success')
                detail = f"{replay['stages'][-1].upper()} | {replay['times_s'][-1]:.2f}s | {measurement_label(name, replay)}"
                if done:
                    detail = ('TASK FAILED' if not task.evaluate(replay).success else 'TASK COMPLETE') + (' | UNSAFE' if unsafe else ' | SAFE')
                if moving:
                    from esrb.tools.render.focused_workcells import annotate
                    frames.append(annotate(task.render_frame(WIDTH, HEIGHT), name, replay['stages'][-1],
                                           unsafe, bool(replay['robot_collisions'][-1]),
                                           done and task.evaluate(replay).success,
                                           replay.get('failure_reason') if done else None))
                else:
                    renderer.update_scene(task._data, camera=camera)
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
        if (unsafe and (assessment.status=='UNKNOWN' or completed and assessment.safe)
                or not unsafe and not (assessment.safe and completed)):
            raise ValueError('selected example does not satisfy the displayed label')
        filename = name + ('_failure.gif' if unsafe else '.gif')
        durations = [max(20, round(1000*(b-a)/PLAYBACK_SPEED/10)*10) for a, b in zip(times, times[1:])] + [1200]
        if moving:
            from esrb.tools.render.tool_workcell import gif_frames
            gif_frames(frames, out/filename, duration)
            with Image.open(out/filename) as image:
                from PIL import ImageSequence
                durations = [frame.info.get('duration', 0) for frame in ImageSequence.Iterator(image)]
        else:
            frames[0].save(out/filename, save_all=True, append_images=frames[1:], duration=durations,
                           loop=0, optimize=True, disposal=2)
        view = {'lookat_m': camera.lookat.tolist(), 'distance_m': camera.distance,
                'azimuth_deg': camera.azimuth, 'elevation_deg': camera.elevation}
        if moving:
            cid = task._model.camera('overview').id
            view = {'position_m': task._model.cam_pos[cid].tolist(),
                    'quaternion_wxyz': task._model.cam_quat[cid].tolist(),
                    'fovy_deg': float(task._model.cam_fovy[cid])}
        return {'file': filename, 'gif_digest': hash_artifact(out/filename),
            'scenario': name, 'role': role, 'source_report': str(source.resolve()),
            'source_evidence_id': wrapper['evidence_id'], 'case': raw['case'],
            'policy': components['policy'], 'policy_digest': raw['policy_digest'],
            'policy_version': receipt['base_version' if role == 'success' else 'candidate_version'],
            'trace_digest': hash_dict(replay), 'exact_replay': True,
            'viewing_camera': view, 'size_px': [WIDTH, HEIGHT], 'inset': False,
            'source_kind': 'audited_engineering_replay', 'capture_stride': STRIDE,
            'camera_only_override': not moving, 'playback_duration_s': sum(durations)/1000,
            'task_success': completed, 'oracle_status': assessment.status,
            'violations': assessment.violated_constraints,
            'frames': len(frames), 'sampled_steps': sample_steps, 'sample_times_s': times,
            'durations_ms': durations, 'first_observed_bad_step': first_bad,
            'simulation_steps': len(replay['commands']), 'simulation_time_s': replay['times_s'][-1]}
    finally:
        if renderer is not None:
            renderer.close()
        task.close()


def main(args=None):
    global WIDTH, HEIGHT
    from esrb.cli.options import options_tools_render_physical_extensions
    parser = argparse.ArgumentParser()
    options_tools_render_physical_extensions(parser)
    args = parser.parse_args() if args is None else args
    if args.width < 320 or args.height < 240 or not np.isfinite(args.duration) or args.duration < 1:
        parser.error('render size must be at least 320x240 and duration at least 1 second')
    WIDTH, HEIGHT = args.width, args.height
    if getattr(args,'success_only',False):
        if args.acceptance is not None or args.reference_preview is not None:
            parser.error('--success-only uses a fresh seed; do not combine with a saved panel')
        from esrb.tools.render.focused_workcells import render_successful,merge_panel
        args.out_dir.mkdir(parents=True,exist_ok=True)
        demos=[]
        for name in args.scenarios or ['constrained_push','force_alignment','compliant_assembly']:
            demos.extend(render_successful(name,args.out_dir,args.seed,WIDTH,HEIGHT,args.duration))
        merge_panel(args.out_dir,demos)
        return
    if args.acceptance is None:
        from esrb.tools.render.focused_workcells import render_controlled, merge_panel
        selected = args.scenarios or ['moving_transport', 'moving_assembly']
        if any(name not in {'long_horizon_assembly','moving_transport','moving_assembly'} for name in selected):
            parser.error('--acceptance is required for the contact-extension evidence previews')
        args.out_dir.mkdir(parents=True, exist_ok=True)
        references = json.loads(args.reference_preview.read_text())['demos'] if args.reference_preview else None
        if references is not None and any(not any(row['scenario'] == name and (row['role'] != 'success') == unsafe
                                                  for row in references) for name in selected for unsafe in (False, True)):
            parser.error('reference preview must contain baseline and candidate inputs for each selected scene')
        demos = []
        for name in selected:
            demos.extend(render_controlled(name, args.out_dir, args.seed, WIDTH, HEIGHT, args.duration, references))
        merge_panel(args.out_dir, demos)
        return
    acceptance = json.loads(args.acceptance.read_text())
    if acceptance.get('scope') in {'controlled_engineering_acceptance','measured_successful_engineering_trajectories'}:
        if acceptance.get('status')!='PASS': parser.error('a passed engineering acceptance is required')
        names={row['scenario'] for row in acceptance['scenarios']}
        if args.scenarios:
            if set(args.scenarios)-names: parser.error('selected scene is absent from this acceptance')
            acceptance['episodes']=[row for row in acceptance['episodes'] if row['scenario'] in args.scenarios]
        from esrb.tools.render.focused_workcells import render_acceptance
        render_acceptance(acceptance,args.out_dir,WIDTH,HEIGHT,args.duration)
        return
    if acceptance['status'] != 'PASS' or {row['scenario'] for row in acceptance['scenarios']} != set(EXTENSIONS):
        parser.error('a complete audited engineering panel is required')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    demos = []
    for source in acceptance['scenarios']:
        name, path = source['scenario'], Path(source['source_report'])
        if args.scenarios and name not in args.scenarios:
            continue
        if hash_artifact(path) != source['report_file_digest']:
            raise ValueError('source report changed since its audit')
        report = json.loads(path.read_text())
        row = report['generations'][0]
        parent = next(item for item in row['baseline_measurements']
                      if item['raw_result']['task_success'] and item['raw_result']['oracle']['status'] == 'PASS')
        candidates = [item for item in row['gate_plan']['executed'] if not item['raw_result']['task_success']
                      or item['raw_result']['oracle']['status'] == 'VIOLATION']
        if not candidates:
            raise ValueError(f'{name}: no measured candidate failure or violation in the selected generation')
        child = next((item for item in candidates if not item['raw_result']['task_success']), candidates[0])
        for role, wrapper in (('success', parent), ('failure', child)):
            demo = render_example(name, report, path, wrapper, role, args.out_dir, args.duration)
            demo['acceptance_digest'] = hash_artifact(args.acceptance)
            demos.append(demo)
            print(f"{demo['file']}: exact replay, task={demo['task_success']}, safety={demo['oracle_status']}, frames={demo['frames']}", flush=True)
        del report, row, parent, candidates, child, wrapper
        gc.collect()
    from esrb.tools.render.focused_workcells import merge_panel
    merge_panel(args.out_dir, demos)
