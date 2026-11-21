"""Matched-time Panda previews, rendered from separate visualization-only data.

The two physical rollouts run before any display overrides. Camera, texture and
lighting changes therefore cannot affect the measured success/drop trajectories.
"""
from contextlib import ExitStack
import json

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from esrb.agent.tool_handling_policy import ToolHandlingPolicy
from esrb.oracles.multiconstraint import scenario_oracle
from esrb.oracles.panda import validate_panda_measurements
from esrb.tasks.tool_handling import ToolHandlingTask
from esrb.update_operators.tool_grip import ToolGripRelaxationOperator
from esrb.utils.hashing import hash_dict

STRIDE = 4
GREEN, ORANGE, INK = '#92e9c1', '#ffb174', '#17212d'


def font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', max(9, round(size)))
    except OSError:
        return ImageFont.load_default()


def capture_pair(seed):
    """Measure the baseline to completion, then the child to that same horizon."""
    demos = []
    horizon = ToolHandlingTask.MAX_STEPS
    for relaxed in (False, True):
        task, policy = ToolHandlingTask(), ToolHandlingPolicy()
        if relaxed:
            policy = ToolGripRelaxationOperator().apply(policy, seed, .5).candidate_policy
        states = []
        try:
            observation = task.reset(seed)
            for step in range(horizon):
                observation, _, done, trace = task.step(policy.act(observation))
                states.append({
                    'step': step+1, 'time_s': float(task._data.time),
                    'qpos': task._data.qpos.tolist(), 'qvel': task._data.qvel.tolist(),
                    'stage': trace['stages'][-1], 'dropped': bool(trace['drop_events']),
                    'success': bool(trace['success']),
                    'tool_position': trace['tool_positions'][-1],
                    'tip_position': trace['tool_tip_positions'][-1],
                    'surface_force_n': trace['surface_forces'][-1],
                    'grip_normal_forces_n': trace['grip_normal_forces'][-1],
                })
                if done:
                    break
            validate_panda_measurements(trace)
            verdict = scenario_oracle('tool_handling').evaluate(trace, 0)
            success = bool(task.evaluate(trace).success)
            if (success != (not relaxed) or verdict.safe != (not relaxed)
                    or bool(trace['drop_events']) != relaxed or verdict.status == 'UNKNOWN'):
                raise RuntimeError('Measured Panda preview did not meet the expected demo conditions')
            demos.append({'states': states, 'trace': trace, 'info': {
                'seed': seed, 'transport_closure_m': policy.transport_closure,
                'policy': policy.state_dict(), 'task_success': success,
                'safe': bool(verdict.safe), 'status': verdict.status,
                'signals': verdict.signal_values, 'drop_events': trace['drop_events'],
                'stage_events': trace['stage_events'], 'model_digest': trace['model_digest'],
                'model_resource_digest': trace['model_resource_digest'],
                'model_resources': trace['model_resources'], 'trace_digest': hash_dict(trace),
                'simulation_time_s': float(task._data.time), 'simulation_steps': len(states),
                'episode_done': bool(done),
                'end_reason': 'task_completed' if done else 'matched_baseline_horizon',
            }})
            horizon = len(states)
        finally:
            task.close()
    if len(demos[0]['states']) != len(demos[1]['states']):
        raise ValueError('Preview rollouts must share the same number of physical steps')
    if hash_dict(demos[0]['trace']['reset_readback']) != hash_dict(demos[1]['trace']['reset_readback']):
        raise ValueError('Preview rollouts must start from the same measured initial condition')
    if not np.allclose([s['time_s'] for s in demos[0]['states']],
                       [s['time_s'] for s in demos[1]['states']], rtol=0, atol=1e-10):
        raise ValueError('Preview rollouts must share the same simulation timestamps')
    return {'seed': seed, 'demos': demos}


def camera_pose(model, name, position, target, fovy=None):
    camera = model.camera(name).id
    position, target = np.asarray(position), np.asarray(target)
    backward = position-target
    backward /= np.linalg.norm(backward)
    right = np.cross([0., 0., 1.], backward)
    right /= np.linalg.norm(right)
    rotation = np.column_stack((right, np.cross(backward, right), backward))
    model.cam_pos[camera] = position
    mujoco.mju_mat2Quat(model.cam_quat[camera], rotation.ravel())
    if fovy is not None:
        model.cam_fovy[camera] = fovy


def presentation_style(model):
    """Apply the task's shared robosuite-inspired arena appearance."""
    from esrb.tasks.tool_handling.appearance import apply_appearance
    apply_appearance(model)
    camera_pose(model, 'overview', [1.05, -1.10, .85], [.10, -.16, .28])


def detail_pose(reference):
    """Both trajectories use the baseline's camera pose at every matched time."""
    if reference['stage'] in {'orient', 'operate', 'done'}:
        return [ .24, -.24, .19], [.10, 0., .075], 30., 'CONTACT DETAIL'
    center = np.asarray(reference['tool_position'])
    # Include the lifted grip and the table underneath it, so a falling tool
    # remains visible in the same camera rather than disappearing below frame.
    return (center + [.18, -.34, .20]).tolist(), [float(center[0]), float(center[1]), .105], 40., 'GRIP DETAIL'


def project(model, data, camera, point, width, height):
    cam = model.camera(camera).id
    local = data.cam_xmat[cam].reshape(3, 3).T @ (np.asarray(point)-data.cam_xpos[cam])
    if local[2] >= 0:
        return None
    focal = .5*height / np.tan(np.deg2rad(model.cam_fovy[cam])*.5)
    return (width*.5 + local[0]/-local[2]*focal,
            height*.5 - local[1]/-local[2]*focal)


def slot_guide(image, model, data):
    """Screen-space guide beside the physical slot; never a simulated contact."""
    draw = ImageDraw.Draw(image)
    axis = np.array([np.cos(.7), np.sin(.7), 0.])
    center = np.array([.10, 0., .052])
    for sign in (-1, 1):
        a, b = [project(model, data, 'detail', center + sign*r*axis, *image.size)
                for r in (.012, .022)]
        if a is not None and b is not None:
            draw.line([a, b], fill='#007f90', width=max(2, image.width//180))
            draw.ellipse((b[0]-2, b[1]-2, b[0]+2, b[1]+2), fill='#007f90')
    draw.text((10, image.height-23), 'SLOT AXIS',
              font=font(image.width*.032), fill='#b8f6ff', stroke_width=1, stroke_fill=INK)


def compose(overview, detail, state, relaxed, detail_label, dropped_tool_pixel=None):
    image = Image.fromarray(overview)
    width, height = image.size
    scale = width/960
    draw = ImageDraw.Draw(image)
    margin, bar = round(16*scale), round(40*scale)
    draw.rectangle((0, 0, width, bar), fill=INK)
    draw.text((margin, round(8*scale)), 'Tool alignment & contact',
              font=font(23*scale), fill='#eef3f8')
    phase = {'grasp': 'Grasp', 'lift': 'Lift', 'transport': 'Transport',
             'orient': 'Align', 'operate': 'Align + contact', 'done': 'Contact verified'}[state['stage']]
    if state['success']:
        phase = 'Contact verified'
    if state['dropped']:
        phase = 'Tool dropped'
    accent = ORANGE if relaxed else GREEN
    label = ('Reduced grip  |  ' if relaxed else '') + phase
    draw.text((margin, bar+round(12*scale)), label, font=font(21*scale),
              fill=accent, stroke_width=1, stroke_fill=INK)
    x, y = width-margin-detail.width, bar+round(12*scale)
    draw.rectangle((x-2, y-2, x+detail.width+1, y+detail.height+round(27*scale)), fill=INK)
    image.paste(detail, (x, y))
    draw.text((x+round(8*scale), y+detail.height+round(5*scale)), detail_label,
              font=font(14*scale), fill='#d8e3ec')
    if state['dropped'] and dropped_tool_pixel is not None:
        px, py = dropped_tool_pixel
        radius = 15*scale
        draw.ellipse((px-radius, py-radius, px+radius, py+radius), outline=ORANGE, width=max(1, round(2*scale)))
        end = (px-65*scale, py+45*scale)
        draw.line([(px-radius, py+radius), end], fill=ORANGE, width=max(1, round(2*scale)))
        draw.text((end[0]-60*scale, end[1]+3*scale), 'Dropped tool', font=font(17*scale),
                  fill=ORANGE, stroke_width=1, stroke_fill=INK)
    return image


def gif_frames(frames, path, duration):
    # One palette for the entire timeline preserves colours between frames.
    samples = frames[::max(1, len(frames)//12)]
    atlas = Image.new('RGB', (320*len(samples), 240))
    for index, frame in enumerate(samples):
        atlas.paste(frame.resize((320, 240)), (320*index, 0))
    palette = atlas.quantize(colors=256)
    indexed = [im.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG) for im in frames]
    ticks = round(duration*100)
    hold = min(50, ticks//5)
    boundaries = np.rint(np.linspace(0, ticks-hold, len(indexed))).astype(int)
    durations = [int(t)*10 for t in np.diff(boundaries)] + [hold*10]
    if min(durations) < 10:
        raise ValueError('GIF duration is too short for the captured frames')
    indexed[0].save(path, save_all=True, append_images=indexed[1:],
                    duration=durations, loop=0, optimize=False, disposal=2)
    return sum(durations)/1000


def comparison_image(parent, child, state, seed, panel_width=640):
    height = round(parent.height*panel_width/parent.width)
    scale = panel_width/640
    header, footer = round(38*scale), round(34*scale)
    image = Image.new('RGB', (panel_width*2, height+header+footer), INK)
    draw = ImageDraw.Draw(image)
    draw.text((12*scale, 9*scale),
              f'Grip comparison  |  seed {seed}  |  same simulation time: {state["time_s"]:.2f} s',
              font=font(19*scale), fill='#edf2f7')
    for index, (panel, label, color) in enumerate([
            (parent, 'BASELINE  /  retained grip', GREEN),
            (child, 'REDUCED GRIP  /  slip and drop', ORANGE)]):
        image.paste(panel.resize((panel_width, height), Image.Resampling.LANCZOS), (index*panel_width, header))
        draw.text((index*panel_width+12*scale, header+height+7*scale), label,
                  font=font(17*scale), fill=color)
    return image


def comparison_analysis(comparison, capture, step):
    """Keep measured time, step-peak forces and relative translation off the demo."""
    panel_width = comparison.width//2
    image = Image.new('RGB', (comparison.width, comparison.height+150), INK)
    image.paste(comparison, (0, 0))
    draw, metrics = ImageDraw.Draw(image), []
    for column, demo in enumerate(capture['demos']):
        state, trace = demo['states'][step-1], demo['trace']
        reference = int(np.flatnonzero(trace['retention_active'])[0])
        positions = np.asarray(trace['relative_positions'])
        displacement_mm = 1000*float(np.linalg.norm(positions[step-1]-positions[reference]))
        metrics.append({
            'step': step, 'time_s': state['time_s'], 'grip_normal_forces_n': state['grip_normal_forces_n'],
            'surface_force_n': state['surface_force_n'], 'relative_translation_mm': displacement_mm,
            'translation_reference_step': reference+1, 'dropped': state['dropped'],
        })
        rows = [f"t = {state['time_s']:.3f} s   /   control step {step}",
                f"Grip normal (step peaks): {state['grip_normal_forces_n'][0]:.2f} / {state['grip_normal_forces_n'][1]:.2f} N",
                f"Surface contact (step peak): {state['surface_force_n']:.2f} N",
                f"Relative translation: {displacement_mm:.2f} mm  /  drop: {state['dropped']}"]
        for row, text in enumerate(rows):
            draw.text((column*panel_width+14, comparison.height+12+row*32),
                      text, font=font(18), fill='#d4dfe9')
    return image, metrics


def write_metadata(metadata, out):
    # GIF encoding may combine identical consecutive images. Preserve the
    # shared source sampling timeline and distinguish it from encoded frames.
    for demo in metadata['demos']:
        with Image.open(out/demo['file']) as image:
            demo['encoded_frames'] = image.n_frames
    with Image.open(out/metadata['comparison']['file']) as image:
        metadata['comparison']['encoded_frames'] = image.n_frames
    metadata['frame_count_semantics'] = 'frames counts rendered samples; identical consecutive GIF images may be combined.'
    (out/'tool_handling_preview.json').write_text(json.dumps(metadata, indent=2)+'\n')


def render_views(capture, width, height, samples):
    """Render paired states with identical cameras; also used for still previews."""
    from esrb.tasks.tool_handling.appearance import decorate_scene
    task = ToolHandlingTask()
    frames, camera_records = [[], []], []
    try:
        model = task._model
        presentation_style(model)
        model.vis.global_.offwidth = width
        model.vis.global_.offheight = height
        detail_width, detail_height = round(width*.42), round(height*.42)
        data = mujoco.MjData(model)  # Visualization data, never passed to task.step.
        options = mujoco.MjvOption()
        options.geomgroup[3] = 0
        options.geomgroup[5] = 0
        with ExitStack() as stack:
            overview = stack.enter_context(mujoco.Renderer(model, height=height, width=width))
            detail_renderer = stack.enter_context(mujoco.Renderer(model, height=detail_height, width=detail_width))
            for index in samples:
                reference = capture['demos'][0]['states'][index]
                position, target, fovy, label = detail_pose(reference)
                camera_pose(model, 'detail', position, target, fovy)
                camera_records.append({'step': reference['step'], 'position': position,
                                       'target': target, 'fovy_deg': fovy, 'label': label})
                for relaxed in (False, True):
                    state = capture['demos'][int(relaxed)]['states'][index]
                    data.qpos[:], data.qvel[:], data.time = state['qpos'], state['qvel'], state['time_s']
                    mujoco.mj_forward(model, data)
                    overview.update_scene(data, camera='overview', scene_option=options)
                    decorate_scene(overview.scene)
                    main_view = overview.render().copy()
                    detail_renderer.update_scene(data, camera='detail', scene_option=options)
                    decorate_scene(detail_renderer.scene)
                    detail = Image.fromarray(detail_renderer.render())
                    if label == 'CONTACT DETAIL':
                        slot_guide(detail, model, data)
                    locator = project(model, data, 'overview', data.xpos[model.body('tool').id], width, height)
                    frames[int(relaxed)].append(compose(main_view, detail, state, relaxed, label, locator))
    finally:
        task.close()
    return frames, camera_records


def render_pair(capture, out, width=960, height=720, duration=5.):
    """Replay recorded poses for display, without stepping or changing the plant."""
    samples = list(range(0, len(capture['demos'][0]['states']), STRIDE))
    final = len(capture['demos'][0]['states'])-1
    if samples[-1] != final:
        samples.append(final)
    seed = capture['seed']
    frames, camera_records = render_views(capture, width, height, samples)
    playback = []
    for index, name in enumerate(('tool_handling.gif', 'tool_handling_drop.gif')):
        playback.append(gif_frames(frames[index], out/name, duration))
    comparisons = [comparison_image(a, b, capture['demos'][0]['states'][s], seed)
                   for a, b, s in zip(*frames, samples)]
    gif_frames(comparisons, out/'tool_handling_comparison.gif', duration)
    frames[0][-1].save(out/'tool_handling_workcell.png')
    # Three numbered steps belong only to the successful trajectory.
    panel_width = 640
    panel_height = round(height*panel_width/width)
    stages = Image.new('RGB', (panel_width*3, panel_height+80), INK)
    draw = ImageDraw.Draw(stages)
    draw.text((14, 10), 'SUCCESS  /  Tool alignment & contact', font=font(23), fill=GREEN)
    selected = []
    for column, (phase, label) in enumerate([
            ('grasp', '1. GRASP'), ('transport', '2. TRANSPORT'), ('operate', '3. ALIGN + CONTACT')]):
        choices = [i for i, s in enumerate(samples) if capture['demos'][0]['states'][s]['stage'] == phase]
        chosen = choices[-1] if phase != 'transport' else choices[len(choices)//2]
        selected.append({'label': label, 'step': samples[chosen]+1,
                         'time_s': capture['demos'][0]['states'][samples[chosen]]['time_s']})
        stages.paste(frames[0][chosen].resize((panel_width, panel_height), Image.Resampling.LANCZOS),
                     (column*panel_width, 42))
        draw.text((column*panel_width+14, panel_height+52), label, font=font(19), fill='#e8eef6')
    stages.save(out/'tool_handling_stages.png')
    drop_time = capture['demos'][1]['info']['drop_events'][0]['time_s']
    # Show the fall shortly after first grip loss, at one actual shared timestamp.
    comparison_index = min(range(len(samples)), key=lambda i:
                           abs(capture['demos'][0]['states'][samples[i]]['time_s']-(drop_time+.40)))
    comparisons[comparison_index].save(out/'tool_handling_comparison.png')
    analysis, analysis_metrics = comparison_analysis(comparisons[comparison_index], capture, samples[comparison_index]+1)
    analysis.save(out/'tool_handling_comparison_analysis.png')
    demos = []
    for index, demo in enumerate(capture['demos']):
        demos.append({**demo['info'], 'file': ('tool_handling_drop.gif' if index else 'tool_handling.gif'),
                      'frames': len(samples), 'size_px': [width, height],
                      'playback_duration_s': playback[index],
                      'sample_times_s': [demo['states'][s]['time_s'] for s in samples],
                      'render_states_digest': hash_dict([demo['states'][s] for s in samples])})
    metadata = {
        'scope': 'controlled_engineering_visualization', 'task_label': 'Tool alignment & contact',
        'task_semantics': 'Aligned tool tip and sustained surface contact; no tightening mechanism.',
        'render_only_overrides': ['overview camera', 'detail camera', 'robosuite-inspired display-only arena', 'ceramic texture', 'directional lights', 'hidden non-task decorations', '2D slot-axis guide', '2D dropped-tool locator'],
        'physics_overrides': [], 'synchronized_comparison': True,
        'capture_stride': STRIDE, 'final_frame_pause_ms': min(50, round(duration*100)//5)*10,
        'analysis': {'file': 'tool_handling_comparison_analysis.png', 'metrics': analysis_metrics,
                     'force_reduction': 'Maximum of the 10 physics substeps in each control step.',
                     'translation_reference': 'Tool position in the gripper frame at the first retention-active step.'},
        'comparison': {'file': 'tool_handling_comparison.gif', 'still': 'tool_handling_comparison.png',
                       'step': samples[comparison_index]+1, 'simulation_time_s': capture['demos'][0]['states'][samples[comparison_index]]['time_s'],
                       'playback_duration_s': playback[0],
                       'camera_rule': 'Both trajectories use the baseline camera at each identical timestamp.',
                       'detail_cameras': camera_records},
        'success_panels': selected, 'demos': demos,
    }
    from esrb.tasks.tool_handling import appearance
    from esrb.utils.hashing import hash_artifact
    metadata['visual_style'] = appearance.STYLE
    metadata['visual_resources'] = {str(path.relative_to(appearance.ASSETS.parent.parent)): hash_artifact(path)
                                    for path in [appearance.ASSETS.parent.parent/'appearance.py', *sorted(appearance.ASSETS.iterdir())]}
    metadata['arena_provenance'] = json.loads((appearance.ASSETS/'provenance.json').read_text())
    write_metadata(metadata, out)
    return metadata
