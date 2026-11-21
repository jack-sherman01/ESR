"""Measured single-view previews of assembly and moving-workspace workcells."""
import json
import gzip
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from esrb.oracles.multiconstraint import scenario_oracle
STYLE = 'robosuite-inspired-panda-production/1'
from esrb.tools.render.tool_workcell import font, gif_frames
from esrb.utils.hashing import hash_artifact, hash_dict

TITLES = {'long_horizon_assembly': 'Long-horizon Assembly',
          'moving_transport': 'Moving Transport', 'moving_assembly': 'Moving Assembly',
          'constrained_push':'Constrained Push','force_alignment':'Force Alignment',
          'compliant_assembly':'Compliant Assembly'}


def make_demo(name, seed, unsafe=False):
    if name == 'long_horizon_assembly':
        from esrb.tasks.long_horizon.assembly import LongHorizonAssemblyTask
        from esrb.agent.assembly_policy import AssemblyPolicy
        from esrb.update_operators.assembly_orientation import AssemblyOrientationOperator
        task, policy = LongHorizonAssemblyTask(), AssemblyPolicy()
        if unsafe:
            policy = AssemblyOrientationOperator().apply(policy, seed, .5).candidate_policy
        point = None
    else:
        from esrb.tasks.extensions.registry import EXTENSIONS, ExtensionOperator
        task_type, policy_type = EXTENSIONS[name]
        task, policy = task_type(), policy_type()
        if unsafe:
            policy = ExtensionOperator(name).apply(policy, seed, .5).candidate_policy
        point = None
    return task, policy, point


def capture_demo(name, seed, unsafe=False, *, disturbance=None, policy_state=None):
    """Run real dynamics once before changing any display settings."""
    task, policy, point = make_demo(name, seed, unsafe)
    if policy_state is not None:
        if set(policy_state) != set(policy.state_dict()):
            raise ValueError(f'{name}: reference policy parameters do not match the task')
        for key, value in policy_state.items():
            setattr(policy, key, float(value))
    if disturbance is not None:
        point = disturbance
    states = []
    try:
        observation = task.reset(seed, disturbance=point)
        for step in range(task.MAX_STEPS):
            observation, _, done, trace = task.step(policy.act(observation))
            states.append({'step': step+1, 'time_s': float(task._data.time),
                           'qpos': task._data.qpos.tolist(), 'qvel': task._data.qvel.tolist(),
                           'mocap_pos': task._data.mocap_pos.tolist(),
                           'mocap_quat': task._data.mocap_quat.tolist()})
            if done:
                break
        if not done:
            raise RuntimeError(f'{name}: demo did not reach an episode termination')
        verdict = scenario_oracle(name).evaluate(trace, 0)
        completed = bool(task.evaluate(trace).success)
        if unsafe:
            if verdict.status == 'UNKNOWN' or completed and verdict.safe:
                raise RuntimeError(f'{name}: the candidate has no verified failure or safety violation')
        elif not (completed and verdict.safe):
            raise RuntimeError(f'{name}: the baseline was not a completed, safe trajectory')
        return {'name': name, 'seed': seed, 'unsafe': unsafe, 'states': states, 'trace': trace,
                'policy': policy.state_dict(), 'task_success': completed,
                'oracle_status': verdict.status, 'violations': verdict.violated_constraints,
                'trace_digest': hash_dict(trace)}
    finally:
        task.close()


def annotate(frame, name, stage, unsafe, contact=False, completed=False, failure_reason=None):
    """A compact title and measured phase; no inset, metrics bar or full-arm view."""
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    width = image.width
    scale = width/960
    draw.rectangle((0, 0, width, round(36*scale)), fill='#17212d')
    draw.text((round(14*scale), round(7*scale)), TITLES[name], font=font(22*scale), fill='#edf3f8')
    phases = {'grasp': 'Grasp', 'lift': 'Lift', 'transport': 'Transport',
              'orient': 'Align', 'preposition': 'Align over socket', 'insert': 'Insert',
              'verify': 'Release + verify', 'operate': 'Align + contact', 'done': 'Complete'}
    label = 'Robot contact' if contact else phases.get(stage, stage.replace(':', ' / ').replace('_',' ').title())
    if completed:
        label = {'constrained_push':'Pallet docked','force_alignment':'Alignment verified',
                 'compliant_assembly':'Keyed shaft seated'}.get(name,
                 'Assembly verified' if 'assembly' in name else 'Three-part kit verified')
    if failure_reason:
        label = 'Task failed: '+failure_reason.replace('_',' ').replace(':',' / ')
    if unsafe:
        label = 'Controlled candidate  |  '+label
    draw.text((round(14*scale), round(47*scale)), label, font=font(20*scale),
              fill='#ffb174' if unsafe else '#92e9c1', stroke_width=1, stroke_fill='#17212d')
    return image


def sample_indices(demo):
    trace, count = demo['trace'], len(demo['states'])
    stride = max(4, int(np.ceil(count/80)))
    indices = set(range(0, count, stride)) | {count-1}
    indices.update(i for i in range(1, count) if trace['stages'][i] != trace['stages'][i-1])
    contacts = np.flatnonzero(trace.get('robot_collisions', []))
    if len(contacts):
        first = int(contacts[0])
        indices.update(i for i in range(first-3, first+4) if 0 <= i < count)
    return sorted(indices)


def focused_frame(task,name,width,height):
    """View the new fixtures from the table side opposite the robot base."""
    if name not in {'constrained_push','force_alignment','compliant_assembly'}:
        return task.render_frame(width,height)
    if task._renderer is None:task._renderer=mujoco.Renderer(task._model,width=width,height=height)
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:]=[.01,.03,.075];camera.azimuth=235;camera.elevation=-38
    camera.distance=.85 if name=='constrained_push' else .78
    option=mujoco.MjvOption();option.geomgroup[3]=0
    task._renderer.update_scene(task._data,camera=camera,scene_option=option)
    from esrb.tasks.tool_handling.appearance import decorate_scene
    decorate_scene(task._renderer.scene)
    task._last_view={'type':'free','lookat_m':camera.lookat.tolist(),'distance_m':float(camera.distance),
                     'azimuth_deg':float(camera.azimuth),'elevation_deg':float(camera.elevation)}
    return task._renderer.render().copy()


def render_demo(demo, out, width=960, height=720, duration=5.):
    """Display measured states in a separate task instance without plant stepping."""
    task, _, _ = make_demo(demo['name'], demo['seed'], demo['unsafe'])
    trace, indices, frames, snapshots = demo['trace'], sample_indices(demo), [], {}
    try:
        task.reset(demo['seed'], disturbance=trace['reset_readback'])
        task._model.vis.global_.offwidth = width
        task._model.vis.global_.offheight = height
        for index in indices:
            state = demo['states'][index]
            task._data.qpos[:], task._data.qvel[:], task._data.time = state['qpos'], state['qvel'], state['time_s']
            task._data.mocap_pos[:] = np.asarray(state['mocap_pos']).reshape(-1, 3)
            task._data.mocap_quat[:] = np.asarray(state['mocap_quat']).reshape(-1, 4)
            mujoco.mj_forward(task._model, task._data)
            contact = bool(trace.get('robot_collisions', [False]*len(demo['states']))[index])
            image = annotate(focused_frame(task,demo['name'],width,height), demo['name'], trace['stages'][index],
                             demo['unsafe'], contact, demo['task_success'] and index == len(demo['states'])-1,
                             trace.get('failure_reason') if index==len(demo['states'])-1 else None)
            frames.append(image)
            snapshots[trace['stages'][index]] = image
            if contact and 'robot_contact' not in snapshots:
                snapshots['robot_contact'] = image
        stem = demo['name']+('_failure' if demo['unsafe'] else '')
        playback = gif_frames(frames, out/(stem+'.gif'), duration)
        frames[-1].save(out/(stem+'.png'))
        if 'robot_contact' in snapshots:
            snapshots['robot_contact'].save(out/(demo['name']+'_contact.png'))
        camera = task._model.camera('overview').id
        with Image.open(out/(stem+'.gif')) as image:
            encoded = image.n_frames
        return {
            'file': stem+'.gif', 'gif_digest': hash_artifact(out/(stem+'.gif')),
            'scenario': demo['name'], 'role': 'failure' if demo['unsafe'] else 'success',
            'source_kind': demo.get('source_kind','current_controlled_engineering_rollout'), 'seed': demo['seed'],
            'policy': demo['policy'], 'disturbance': trace['reset_readback'],
            'task_success': demo['task_success'],
            'safe': (demo['oracle_status'] == 'PASS') if demo['oracle_status'] in {'PASS','VIOLATION'} else None,
            'oracle_status': demo['oracle_status'], 'violations': demo['violations'],
            'failure_reason': trace.get('failure_reason'), 'stage_events': trace['stage_events'],
            'trace_digest': demo['trace_digest'], 'model_digest': trace['model_digest'],
            'size_px': [width, height], 'frames': len(frames), 'encoded_frames': encoded,
            'sampled_steps': [i+1 for i in indices], 'sample_times_s': [demo['states'][i]['time_s'] for i in indices],
            'playback_duration_s': playback, 'simulation_steps': len(demo['states']),
            'simulation_time_s': demo['states'][-1]['time_s'],
            'visual_style': 'robosuite-inspired-panda-contact/2' if trace.get('model','').startswith('panda-contact/') else STYLE,
            'playback_speed': demo['states'][-1]['time_s']/playback,
            'raw_source': demo.get('raw_source'), 'raw_source_sha256': demo.get('raw_source_sha256'),
            'inset': False, 'physics_overrides': [],
            'camera_only_override': trace.get('model','').startswith('panda-contact/'),
            'render_only_overrides': ['camera', 'caption', 'temporal sampling', 'GIF palette'],
            'viewing_camera': getattr(task,'_last_view',{'position_m': task._model.cam_pos[camera].tolist(),
                               'quaternion_wxyz': task._model.cam_quat[camera].tolist(),
                               'fovy_deg': float(task._model.cam_fovy[camera])}),
            'visual_resources': {str(path.relative_to(Path(__file__).resolve().parents[3])):hash_artifact(path) for path in
                (Path(__file__),Path(__file__).resolve().parents[2]/('tasks/contact_workcells/workcell.py' if demo['name'] in {'constrained_push','force_alignment','compliant_assembly'} else 'tasks/production/workcell.py'),
                 Path(__file__).resolve().parents[2]/('tasks/contact_workcells/geometry.py' if demo['name'] in {'constrained_push','force_alignment','compliant_assembly'} else 'tasks/production/geometry.py'))},
        }
    finally:
        task.close()


def render_controlled(name, out, seed=0, width=960, height=720, duration=5., references=None):
    demos = []
    for unsafe in (False, True):
        reference = next((row for row in (references or []) if row['scenario'] == name
                          and (row['role'] != 'success') == unsafe), None)
        case = reference.get('case', {}) if reference else {}
        initial = case.get('disturbance', reference.get('disturbance')) if reference else None
        actual_seed = case.get('seed', reference.get('seed', seed)) if reference else seed
        capture = capture_demo(name, actual_seed, unsafe, disturbance=initial,
                               policy_state=reference['policy'] if reference else None)
        from datetime import datetime
        archive=Path(__file__).resolve().parents[3]/'.esr-local/_checks/production-renders'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        archive.mkdir(parents=True)
        raw=archive/(name+'.json.gz')
        with gzip.open(raw,'wb',compresslevel=1) as stream:
            stream.write(json.dumps(capture,separators=(',',':')).encode())
        capture['raw_source']=str(raw); capture['raw_source_sha256']=hash_artifact(raw)
        metadata = render_demo(capture, out, width, height, duration)
        if reference:
            metadata['reference_input_digest'] = hash_dict({
                'seed': actual_seed, 'disturbance': initial, 'policy': reference['policy']})
        demos.append(metadata)
        print(f"{metadata['file']}: task={metadata['task_success']}, {metadata['oracle_status']}, {duration:g}s", flush=True)
    result = {'scope': 'controlled_engineering_visualization', 'visual_style': STYLE,
              'inset': False, 'size_px': [width, height], 'demos': demos}
    (out/(name+'_animation.json')).write_text(json.dumps(result, indent=2)+'\n')
    (out/(name+'_preview.json')).write_text(json.dumps({
        'scope': 'measured_final_state_preview', 'scenario': name, 'inset': False,
        'file': name+'.png', 'trace_digest': demos[0]['trace_digest'],
        'seed': seed, 'step': demos[0]['simulation_steps'], 'visual_style': STYLE,
    }, indent=2)+'\n')
    return demos


def render_acceptance(summary,out,width=960,height=720,duration=5.):
    """Render the exact accepted engineering records, without further rollouts."""
    out.mkdir(parents=True,exist_ok=True)
    all_demos=[]
    for name in TITLES:
        entries=[r for r in summary['episodes'] if r['scenario']==name]
        if not entries: continue
        candidate=next((r for r in entries if r['candidate']),None)
        baseline=next(r for r in entries if not r['candidate'] and r['task_success']
                      and (candidate is None or r['seed']==candidate['seed']))
        demos=[]
        for entry in (baseline,candidate):
            if entry is None: continue
            raw=Path(entry['file'])
            if hash_artifact(raw)!=entry['file_sha256']: raise ValueError('source trajectory changed')
            with gzip.open(raw,'rt') as stream: record=json.load(stream)
            trace=record['trace']
            seed=record.get('seed',entry['seed'])
            oracle=record.get('oracle',{'status':entry.get('oracle_status','NOT_RUN'),'violated_constraints':[]})
            model,_,_=make_demo(name,seed)
            try:
                if trace['model_digest']!=model.model_digest:
                    raise ValueError('render model does not match frozen trajectory')
            finally: model.close()
            states=[{'qpos':r['qpos'],'qvel':r['qvel'],'time_s':r['time_s'],'mocap_pos':[], 'mocap_quat':[]}
                    for r in trace['physics_samples'][9::10]]
            capture={'name':name,'seed':seed,'unsafe':entry['candidate'],'states':states,'trace':trace,
                     'source_kind':summary.get('scope','frozen_controlled_engineering_trajectory'),
                     'policy':record['policy'],'task_success':record.get('task_success',entry['task_success']),'oracle_status':oracle['status'],
                     'violations':oracle['violated_constraints'],'trace_digest':hash_dict(trace),
                     'raw_source':str(raw),'raw_source_sha256':entry['file_sha256']}
            demos.append(render_demo(capture,out,width,height,duration))
            if not entry['candidate']:
                task,_,_=make_demo(name,seed)
                try:
                    task.reset(seed,disturbance=trace['reset_readback'])
                    task.render_frame(width,height)
                    camera=mujoco.MjvCamera(); camera.type=mujoco.mjtCamera.mjCAMERA_FREE
                    camera.lookat[:]=[-.05,-.10,.25]; camera.distance=1.7; camera.azimuth=135; camera.elevation=-30
                    option=mujoco.MjvOption(); option.geomgroup[3]=0
                    task._renderer.update_scene(task._data,camera=camera,scene_option=option)
                    Image.fromarray(task._renderer.render()).save(out/(name+'_workcell.png'))
                finally: task.close()
            del record,trace,capture,states
        (out/(name+'_animation.json')).write_text(json.dumps({'scope':'frozen_controlled_engineering_records','inset':False,'demos':demos},indent=2)+'\n')
        (out/(name+'_preview.json')).write_text(json.dumps({'file':name+'.png','seed':baseline['seed'],
            'raw_source':baseline['file'],'raw_source_sha256':baseline['file_sha256'],'inset':False},indent=2)+'\n')
        all_demos.extend(demos)
    merge_panel(out,[r for r in all_demos if r['scenario']!='long_horizon_assembly'])
    return all_demos


def merge_panel(out, demos):
    """Retain untouched contact previews while updating only selected tasks."""
    path = out/'physical_extensions_animation.json'
    old = json.loads(path.read_text()) if path.exists() else {}
    replaced = {row['scenario'] for row in demos}
    retained = []
    for row in old.get('demos', []):
        if row['scenario'] in replaced:
            continue
        row = dict(row)
        for key in ('size_px', 'capture_stride', 'playback_speed', 'final_frame_pause_ms', 'camera_only_override',
                    'acceptance_digest', 'min_frame_duration_ms', 'duration_quantization_ms'):
            if key in old:
                row.setdefault(key, old[key])
        retained.append(row)
    path.write_text(json.dumps({
        'scope': 'selected_controlled_engineering_examples_not_success_rates',
        'settings_scope': 'per_demo', 'model_calls': 0, 'hardware_actions': 0,
        'demos': retained+demos,
    }, indent=2)+'\n')


def render_successful(name,out,seed=0,width=640,height=480,duration=5.):
    """One physical episode, then render its measured states without more stepping."""
    from datetime import datetime
    import gc
    from esrb.tools.diagnostics.production_workcells import freeze_sources
    repo=Path(__file__).resolve().parents[3]
    evidence=repo/'.esr-local/_checks/panda-contact-renders'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    evidence.mkdir(parents=True)
    freeze_sources(evidence)
    task,policy,_=make_demo(name,seed)
    states=[]
    try:
        observation=task.reset(seed)
        for _ in range(task.MAX_STEPS):
            observation,_,done,trace=task.step(policy.act(observation))
            states.append({'qpos':task._data.qpos.tolist(),'qvel':task._data.qvel.tolist(),
                           'time_s':float(task._data.time),'mocap_pos':task._data.mocap_pos.tolist(),
                           'mocap_quat':task._data.mocap_quat.tolist()})
            if done:break
        success=bool(task.evaluate(trace).success)
        record={'scope':'measured_successful_engineering_trajectories','scenario':name,'seed':seed,
                'policy':policy.state_dict(),'task_success':success,'trace':trace,
                'oracle':{'status':'NOT_RUN','violated_constraints':[]}}
        raw=evidence/(name+'.json.gz')
        enabled=gc.isenabled();gc.disable()
        try:
            with gzip.open(raw,'wt',compresslevel=1) as stream:json.dump(record,stream,separators=(',',':'))
            digest=hash_dict(trace)
        finally:
            if enabled:gc.enable()
        if not success:raise RuntimeError(f'{name}: {trace.get("failure_reason")}; raw trajectory retained at {raw}')
        print(f'{name}: task completed in {states[-1]["time_s"]:.2f}s; rendering saved states',flush=True)
    finally:task.close()
    capture={'name':name,'seed':seed,'unsafe':False,'states':states,'trace':trace,'policy':record['policy'],
             'task_success':True,'oracle_status':'NOT_RUN','violations':[],'trace_digest':digest,
             'source_kind':'measured_successful_engineering_trajectories','raw_source':str(raw),'raw_source_sha256':hash_artifact(raw)}
    metadata=render_demo(capture,out,width,height,duration)
    task,_,_=make_demo(name,seed)
    try:
        task.reset(seed,disturbance=trace['reset_readback'])
        Image.fromarray(task.render_frame(width,height,overview=True)).save(out/(name+'_workcell.png'))
    finally:task.close()
    (out/(name+'_animation.json')).write_text(json.dumps({'scope':'measured_successful_engineering_trajectories',
        'inset':False,'demos':[metadata]},indent=2)+'\n')
    (out/(name+'_preview.json')).write_text(json.dumps({'file':name+'.png','seed':seed,'raw_source':str(raw),
        'raw_source_sha256':metadata['raw_source_sha256'],'size_px':[width,height],'inset':False},indent=2)+'\n')
    print(f'{name}: {width}x{height}, {duration:g}s GIF saved',flush=True)
    return [metadata]
