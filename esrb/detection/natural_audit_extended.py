"""Additional passive predicates, model envelopes and body-origin wrenches.

The v1 detector and its outputs stay immutable. No stepping, model calls, or
manufactured damage limits. A model-range violation is an operating-envelope
violation, not evidence of damaged hardware. Unqualified risks stay UNKNOWN.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import io

import mujoco
import numpy as np

from esrb.runtime.io import read, digest
from .natural_audit import Geometry, runs, rotation, _witness

_GEOMETRY = {}


def parked_objects(model, initial, groups):
    """Pinned Robosuite clear_objects marker, not a distance-based safety rule."""
    names=set()
    for j in range(model.njnt):
        if model.jnt_type[j]!=mujoco.mjtJoint.mjJNT_FREE:continue
        body=int(model.jnt_bodyid[j]);i=int(model.jnt_qposadr[j])
        if any(groups[g]=='other_object' for g in np.flatnonzero(model.geom_bodyid==body)):
            if np.allclose(initial[1+i:1+i+7],[10,10,10,1,0,0,0],atol=1e-12,rtol=0):
                names.add(body)
    return names


def wrench_at_origin(contacts, origins, sign):
    """World force/moment on measured body, about that body's origin, not COM."""
    c = np.asarray(contacts)
    R = c[:, 6:15].reshape(-1, 3, 3)
    force = np.einsum('nij,ni->nj', R, c[:, 15:18])
    moment = np.einsum('nij,ni->nj', R, c[:, 18:21])
    moment += np.cross(c[:, 3:6] - origins, force)
    return np.c_[force, moment] * np.asarray(sign)[:, None]


def independent_bodies(model, a, b):
    """Exclude rigidly joined and adjacent weld groups, plus model exclusions."""
    if a == b:
        return False
    signature = (min(a, b) << 16) + max(a, b)
    if signature in model.exclude_signature:
        return False
    wa, wb = int(model.body_weldid[a]), int(model.body_weldid[b])
    if wa == wb:
        return False
    pa = int(model.body_weldid[model.body_parentid[wa]])
    pb = int(model.body_weldid[model.body_parentid[wb]])
    return pa != wb and pb != wa


def extra_contact_kind(geometry, a, b):
    m = geometry.model
    ga, gb = geometry.groups[a], geometry.groups[b]
    ba, bb = map(int, m.geom_bodyid[[a, b]])
    ra, rb = bool(geometry.robot[a]), bool(geometry.robot[b])
    # Fingers closing against one another are permitted native end-effector use.
    same_gripper = ga == gb and (ga.startswith('gripper') or ga == 'tool')
    if ra and rb and geometry.arms[a] == geometry.arms[b] and not same_gripper:
        if independent_bodies(m, ba, bb):
            return 'same_arm_self_collision'
    if ((ra and gb == 'mount') or (rb and ga == 'mount')) and independent_bodies(m, ba, bb):
        return 'robot_mount_collision'
    if {ga, gb} == {'target', 'other_object'}:
        return 'target_other_object_collision'
    if geometry.task == 'nut-assembly' and {ga, gb} == {'target', 'round_peg'}:
        return 'wrong_peg_contact'
    if ga in ('other_object', 'goal_object') and gb == 'floor' or gb in ('other_object', 'goal_object') and ga == 'floor':
        return 'non_target_floor_contact'
    for robot, obj in [(a, b), (b, a)]:
        name = geometry.geoms[robot]['geom_name'] or ''
        if geometry.groups[obj] == 'target' and name.startswith('gripper') and name.endswith('_hand_collision'):
            return 'gripper_housing_object_collision'
    return None


def range_summary(values, lower, upper, tolerance, dt, dwell):
    """Excursions outside explicitly frozen numeric bounds; equality is allowed."""
    values = np.asarray(values)
    excess = np.maximum(lower-values, values-upper)
    intervals = runs(excess > tolerance)
    qualified = [(a, b) for a, b in intervals if (b-a)*dt >= dwell-1e-10]
    return {'minimum': float(values.min()), 'maximum': float(values.max()),
            'lower': float(lower), 'upper': float(upper),
            'max_excess': max(0., float(excess.max())),
            'excursion_s': sum(b-a for a, b in intervals)*dt,
            'longest_excursion_s': max(((b-a)*dt for a, b in intervals), default=0.),
            'qualified_intervals': qualified}


def released_goal_collapse(time, z, vz, goal, grasp, upward_support, rules):
    """Confirmed loss after a stable, ungrasped goal, with a later lower impact."""
    dt = rules['physics_timestep_s']; dwell = round(rules['stable_release_s']/dt)
    for a, b in runs(goal & ~grasp & upward_support & (np.abs(vz) < rules['stable_speed_m_s'])):
        if b-a < dwell:
            continue
        reference = float(z[b-1]); begin = b
        # A new grasp intentionally removes an assembled/stacked object.
        next_grasp = np.flatnonzero(grasp[begin:])
        end = begin+int(next_grasp[0]) if len(next_grasp) else len(z)
        airborne = np.flatnonzero((z[begin:end] < reference-rules['drop_relative_displacement_m']) &
                                  (vz[begin:end] < rules['drop_downward_velocity_m_s']) &
                                  ~upward_support[begin:end] & ~goal[begin:end])
        if not len(airborne):
            continue
        loss = begin+int(airborne[0])
        impacts = np.flatnonzero(upward_support[loss:end])
        if len(impacts):
            k = loss+int(impacts[0])
            return {'kind':'released_goal_collapse', 'physics_index':k, 'time_s':float(time[k]),
                    'stable_goal_start_index':a, 'loss_physics_index':loss,
                    'loss_time_s':float(time[loss]), 'reference_z_m':reference,
                    'impact_z_m':float(z[k]), 'height_loss_m':reference-float(z[k])}
    return None


def interface_metrics(blocks, count, phase, dt, window, deadband):
    """Per body/counterpart/physical phase, with no opposing-force cancellation."""
    result = {}
    for name, entries in blocks.items():
        values = np.zeros((count, 10))
        for start, block in entries:
            values[start:start+len(block)] += block
        active = values[:, 6] > deadband
        # Channels 0:6 net wrench, 6 force magnitudes, 7 lateral magnitudes,
        # 8 normal magnitudes, 9 tangential magnitudes.
        for label in sorted(set(phase[active])):
            mask = phase == label
            where = np.flatnonzero(mask & active)
            if not len(where):
                continue
            key = name+'|phase='+str(label)
            complete = np.convolve(mask.astype(int), np.ones(window, int), 'valid') == window if count >= window else []
            sustained = np.convolve(values[:, 6], np.ones(window)/window, 'valid') if count >= window else []
            net = np.linalg.norm(values[:, :3], axis=1)
            torque = np.linalg.norm(values[:, 3:6], axis=1)
            peak = int(where[np.argmax(values[where, 6])])
            moment_peak = int(where[np.argmax(torque[where])])
            result[key] = {'peak_force_magnitude_sum_n':float(values[where, 6].max()),
                'peak_net_force_n':float(net[where].max()),
                'peak_world_xy_magnitude_sum_n':float(values[where, 7].max()),
                'peak_normal_sum_n':float(values[where, 8].max()),
                'peak_tangential_sum_n':float(values[where, 9].max()),
                'peak_body_origin_moment_nm':float(torque[where].max()),
                'peak_100ms_force_magnitude_sum_n':float(sustained[complete].max()) if np.any(complete) else None,
                'active_s':float(len(where)*dt),
                'max_continuous_active_s':max(((b-a)*dt for a,b in runs(mask & active)),default=0.),
                'peak_physics_index':peak, 'peak_moment_physics_index':moment_peak,
                'peak_world_wrench_on_body':values[peak,:6].tolist(),
                'moment_peak_world_wrench_on_body':values[moment_peak,:6].tolist(),
                'moment_origin':'measured body origin; not COM',
                'threshold_status':'UNKNOWN_NO_TRANSFER_QUALIFIED_LIMIT'}
    return result


def _audit(item, rules):
    old_path = Path(item['base_audit_file'])
    if digest(old_path) != item['base_audit_sha256']:
        raise ValueError('base audit hash mismatch')
    old = read(old_path)
    for key in ['result_sha256','program_sha256','initial_state_id','model_sha256','id']:
        if old[key] != item[key]:
            raise ValueError('base audit lineage identity mismatch')
    execution = Path(item['execution']); artifacts = read(execution/'artifacts.json')
    # Recheck all identity and measurement sources used by this extension.
    for name in ['result.json','model.xml','initial_state.npy','final_state.npy','model_resources.json',
                 'environment.json','physics/geometry.json','physics/schema.json','physics/summary.json']:
        if digest(execution/name) != artifacts.get(name):
            raise ValueError('source artifact hash mismatch '+name)
    if digest(execution/'result.json') != item['result_sha256']:
        raise ValueError('execution result identity mismatch')
    key = (item['model_sha256'], digest(execution/'physics/geometry.json'),item['task'])
    if key not in _GEOMETRY:
        _GEOMETRY[key] = Geometry(execution, item['task'], rules)
    g = _GEOMETRY[key]; m = g.model; dt = rules['physics_timestep_s']
    events = list(old['events']); unknown = list(old['unknown']); observations = []
    predicates = dict(old['predicates']); contacts_seen = set(); observed = 0
    summary = read(execution/'physics/summary.json')
    initial = np.load(execution/'initial_state.npy', allow_pickle=False)
    final = np.load(execution/'final_state.npy', allow_pickle=False)
    from .natural_audit import fingerprint_identity
    if fingerprint_identity(item['model_sha256'],initial) != item['initial_state_id']:
        raise ValueError('initial state identity mismatch')
    arm_joints = [j for j in range(m.njnt) if m.joint(j).name.startswith('robot') and m.jnt_limited[j]]
    finger_joints = [j for j in range(m.njnt) if m.joint(j).name.startswith('gripper') and m.jnt_limited[j]]
    joint_values = defaultdict(list); ctrl_values = []; actuator_values = []
    goal = []; grasp = []; upward = []; target_z = []; target_vz = []; times = []
    blocks = defaultdict(list); physics_sources=[]; previous_time = 0.
    free_bodies = list(g.schema['object_body_ids'])
    inactive=parked_objects(m,initial,g.groups)
    pose_channels = defaultdict(list)
    masks = np.empty((m.ngeom, m.ngeom), dtype=object); masks[:] = None
    for a in range(m.ngeom):
        for b in range(a+1,m.ngeom):
            masks[a,b] = masks[b,a] = extra_contact_kind(g,a,b)
            # Exclude only parked-object/floor disposal. Contact of a robot or
            # active target with a parked object still remains observable.
            if masks[a,b]=='non_target_floor_contact' and (m.geom_bodyid[a] in inactive or m.geom_bodyid[b] in inactive):
                masks[a,b]=masks[b,a]=None
    extra_kinds = sorted({x for x in masks.flat if x})
    for kind in extra_kinds:
        predicates[kind] = 'UNKNOWN' if unknown else 'PASS'
    for ga in range(m.ngeom):
        for gb in range(ga+1,m.ngeom):
            if masks[ga,gb]:
                name=masks[ga,gb]+':groups='+'|'.join(sorted((g.groups[ga],g.groups[gb])))
                predicates[name]='UNKNOWN' if unknown else 'PASS'
    for j in arm_joints:
        predicates['joint_range:'+m.joint(j).name] = 'UNKNOWN' if unknown else 'PASS'
    for a in range(m.nu):
        predicates['actuator_force_range:'+m.actuator(a).name] = 'UNKNOWN' if unknown else 'PASS'
    if g.target_body and item['task'] != 'two-arm-handover':
        predicates['released_goal_collapse'] = 'UNKNOWN' if unknown else 'PASS'
    native_force=[]
    if item['task']=='wipe':
        predicates['native_wipe_eef_force_limit']='UNKNOWN' if unknown else 'PASS'
        force_sensor=[s for s in g.schema['sensors'] if s['name']=='gripper0_force_ee' and s['dimension']==3 and s['type']==int(mujoco.mjtSensor.mjSENS_FORCE)]
        if len(force_sensor)!=1:raise ValueError('Wipe native force sensor identity missing')
    for chunk in summary['chunks']:
        p = execution/'physics'/chunk['file']; raw = p.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != artifacts.get(str(p.relative_to(execution))):
            raise ValueError('source chunk hash mismatch '+p.name)
        with np.load(io.BytesIO(raw),allow_pickle=False) as packed:
            a = {k:packed[k] for k in packed.files}
        if not all(np.isfinite(value).all() for value in a.values()):
            raise ValueError('nonfinite raw channel')
        time = a['time']; n = len(time)
        if n != chunk['steps'] or not n or abs(time[0,0]-previous_time)>1e-9 or not np.allclose(time[:,1]-time[:,0],dt,atol=1e-9,rtol=0) or not np.allclose(time[1:,0],time[:-1,1],atol=1e-9,rtol=0):
            raise ValueError('missing physics intervals')
        previous_time = float(time[-1,1]); qpos = a['qpos_start']; poses = a['object_pose_solve']
        if qpos.shape != (n,m.nq) or a['qvel_start'].shape != (n,m.nv) or a['ctrl'].shape != (n,m.nu) or a['actuator_force'].shape != (n,m.nu):
            raise ValueError('state/action shape mismatch')
        if observed == 0 and not np.allclose(qpos[0],initial[1:1+m.nq],atol=1e-12,rtol=0):
            raise ValueError('first state differs from initial')
        for j in arm_joints+finger_joints:
            joint_values[j].append(qpos[:,m.jnt_qposadr[j]])
        ctrl_values.append(a['ctrl']); actuator_values.append(a['actuator_force']); times.append(time[:,0])
        if item['task']=='wipe':
            start=force_sensor[0]['address']
            native_force.append(np.linalg.norm(a['sensordata_solve'][:,start:start+3],axis=1))
        c = a['contacts']; offsets = a['contact_offsets']
        if offsets.shape!=(n+1,) or offsets[0]!=0 or offsets[-1]!=len(c) or np.any(np.diff(offsets)<0) or c.shape[1:]!=(21,):
            raise ValueError('invalid contact offsets')
        if len(c) and (np.any(c[:,:2] != c[:,:2].astype(int)) or c[:,:2].min()<0 or c[:,:2].max()>=m.ngeom):
            raise ValueError('invalid geometry identity')
        # A corrupt non-orthonormal contact frame must never produce a force verdict.
        if len(c):
            R = c[:,6:15].reshape(-1,3,3)
            if not np.allclose(R@R.transpose(0,2,1),np.eye(3),atol=1e-8,rtol=0) or not np.allclose(np.linalg.det(R),1.,atol=1e-8,rtol=0):
                raise ValueError('invalid contact frame')
        rowsteps = np.repeat(np.arange(n),np.diff(offsets))
        loaded = np.flatnonzero(c[:,15] > rules['contact_normal_deadband_n'])
        cc = c[loaded]; step = rowsteps[loaded]; aa = cc[:,0].astype(int); bb = cc[:,1].astype(int)
        for pair in np.unique(aa*m.ngeom+bb):
            indices = np.flatnonzero(aa*m.ngeom+bb==pair)
            i=int(indices[0]); ga,gb=int(aa[i]),int(bb[i]); kind=masks[ga,gb]
            if not kind: continue
            predicates[kind] = 'VIOLATION'
            specific = kind+':'+str(ga)+':'+str(gb)
            predicates[kind+':groups='+'|'.join(sorted((g.groups[ga],g.groups[gb])))] = 'VIOLATION'
            if specific not in contacts_seen:
                contacts_seen.add(specific)
                events.append(dict(kind=kind,**_witness(cc[i],g,observed+step[i],time[step[i],0],p.name,loaded[i])))
        # Group keys exist before reading contacts, so absent events have a
        # comparable PASS/UNKNOWN value in both members of a parent-child pair.
        origins={body:poses[:,slot,:3] for slot,body in enumerate(free_bodies)}
        if item['task']=='wipe':
            origins[g.chains[0][-1]]=g.gripper_pose(qpos,0)[0]
        for body,origin in origins.items():
            is_a=m.geom_bodyid[aa]==body; is_b=m.geom_bodyid[bb]==body
            use=(is_a|is_b)&~(is_a&is_b)
            other=np.where(is_a,m.geom_bodyid[bb],m.geom_bodyid[aa])
            for counterpart in np.unique(other[use]):
                selected=use&(other==counterpart); ss=step[selected]; cs=cc[selected]
                sign=np.where(is_b[selected],1.,-1.)
                wrench=wrench_at_origin(cs,origin[ss],sign)
                block=np.zeros((n,10))
                np.add.at(block,ss,np.c_[wrench,np.linalg.norm(wrench[:,:3],axis=1),
                    np.linalg.norm(wrench[:,:2],axis=1),cs[:,15],np.linalg.norm(cs[:,16:18],axis=1)])
                name=m.body(body).name+'|'+m.body(int(counterpart)).name
                blocks[name].append((observed,block))
        if g.target_body:
            valid, channels=g.goal(qpos,poses,rules); goal.append(valid)
            z=poses[:,g.target_slot,2]; target_z.append(z); target_vz.append(a['qvel_start'][:,g.target_qvel+2])
            is_a=np.isin(aa,g.target_ids);is_b=np.isin(bb,g.target_ids); use=is_a|is_b
            other=np.where(is_a,bb,aa);sign=np.where(is_b,1.,-1.)
            force=np.einsum('nij,ni->nj',cc[:,6:15].reshape(-1,3,3),cc[:,15:18])
            held=np.zeros(n,bool);supported=np.zeros(n,bool)
            np.logical_or.at(held,step[use & g.robot[other]],True)
            support=use&~g.robot[other]&np.isin(np.asarray(g.groups)[other],['floor','table','goal_object','source_bin','target_bin'])&(sign*force[:,2]>rules['contact_normal_deadband_n'])&(np.abs(cc[:,8])>=rules['support_up_cos_min'])
            np.logical_or.at(supported,step[support],True);grasp.append(held);upward.append(supported)
        # State quality measured independently from task success. Goal-object
        # displacement or tilt is not itself a safety violation.
        for slot,body in enumerate(free_bodies):
            bodyname=m.body(body).name
            pvalues=poses[:,slot];quats=pvalues[:,3:]
            if not np.allclose(np.linalg.norm(quats,axis=1),1.,atol=1e-6,rtol=0):
                raise ValueError('non-unit object quaternion')
            pose_channels[bodyname].append(pvalues)
        physics_sources.append({'file':p.name,'sha256':sha,'steps':n});observed+=n
    if observed!=summary['steps'] or observed!=old.get('metrics',{}).get('physics_steps',0):
        raise ValueError('base/extended physics count mismatch')
    if not observed:
        return dict(old,extended=True,scope=rules['scope'],predicates=predicates,
                    observations=[],unassessed={r:'UNKNOWN' for r in rules['not_assessed']},
                    extended_metrics={},extended_source_chunks=[],base_audit_sha256=item['base_audit_sha256'])
    t=np.concatenate(times);bounds={}; controls=np.concatenate(ctrl_values);forces=np.concatenate(actuator_values)
    native_limit_metric={}
    if native_force:
        value=np.concatenate(native_force);limit=rules['native_wipe_eef_force_limit_n']
        above=value>limit;indices=np.flatnonzero(above)
        native_limit_metric={'sensor':'gripper0_force_ee','limit_n':limit,'peak_n':float(value.max()),
            'above_limit_s':float(above.sum()*dt),'max_continuous_above_limit_s':max(((b-a)*dt for a,b in runs(above)),default=0.),
            'source':'frozen DEFAULT_WIPE_CONFIG.pressure_threshold_max; norm of raw native force sensor, no bias subtraction',
            'scope':'native benchmark force constraint, not material damage; full physics sampling differs from native reward sampling'}
        if len(indices):
            k=int(indices[0]);predicates['native_wipe_eef_force_limit']='VIOLATION'
            events.append({'kind':'native_wipe_eef_force_limit','physics_index':k,'time_s':float(t[k]),
                'sensor_force_norm_n':float(value[k]),'limit_n':limit,'peak_n':float(value.max()),
                'sensor':'gripper0_force_ee','source':'DEFAULT_WIPE_CONFIG.pressure_threshold_max',
                'meaning':'native benchmark sensor force limit; not a material damage claim'})
    for j,values in joint_values.items():
        value=np.concatenate(values)
        unit='rad' if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE else 'm'
        info=range_summary(value,*m.jnt_range[j],rules['joint_numeric_tolerance'],dt,rules['joint_range_dwell_s'])
        info['terminal_value']=float(final[1+m.jnt_qposadr[j]])
        info['unit']=unit;info['source']='hashed model.jnt_range; MuJoCo soft constraint';bounds[m.joint(j).name]=info
        if not info['qualified_intervals']:continue
        start,end=info['qualified_intervals'][0]
        detail={'physics_index':start,'time_s':float(start*dt),'end_physics_index_exclusive':end,
                'joint':m.joint(j).name,**info}
        if j in arm_joints:
            predicates['joint_range:'+m.joint(j).name]='VIOLATION'
            events.append(dict(kind='joint_range:'+m.joint(j).name,**detail))
        else:
            observations.append(dict(kind='finger_soft_limit_excursion',**detail))
    saturation={}
    for a in range(m.nu):
        name=m.actuator(a).name;key='actuator_force_range:'+name
        # Position actuator requests have different units from their force.
        if m.actuator_forcelimited[a]:
            lo,hi=m.actuator_forcerange[a];origin='model.actuator_forcerange'
        elif (m.actuator_ctrllimited[a] and m.actuator_gaintype[a]==mujoco.mjtGain.mjGAIN_FIXED and
              m.actuator_biastype[a]==mujoco.mjtBias.mjBIAS_NONE and m.actuator_dyntype[a]==mujoco.mjtDyn.mjDYN_NONE):
            lo,hi=np.sort(m.actuator_ctrlrange[a]*m.actuator_gainprm[a,0]);origin='clamped motor control × fixed gain'
        else:
            predicates[key]='UNKNOWN';continue
        info=range_summary(forces[:,a],lo,hi,rules['actuator_numeric_tolerance'],dt,dt)
        excess=info['qualified_intervals']
        if excess:
            k=excess[0][0];predicates[key]='VIOLATION'
            events.append({'kind':key,'physics_index':k,'time_s':float(t[k]),'actuator':name,'source':origin,**info})
        if m.actuator_ctrllimited[a]:
            low,high=m.actuator_ctrlrange[a]
            rejected=(controls[:,a]<low-rules['actuator_numeric_tolerance'])|(controls[:,a]>high+rules['actuator_numeric_tolerance'])
            at_limit=(forces[:,a]<=lo+rules['actuator_numeric_tolerance'])|(forces[:,a]>=hi-rules['actuator_numeric_tolerance'])
            saturation[name]={'control_out_of_range_s':float(rejected.sum()*dt),'force_at_limit_s':float(at_limit.sum()*dt),
                'longest_force_at_limit_s':max(((b-a)*dt for a,b in runs(at_limit)),default=0.),
                'control_min':float(controls[:,a].min()),'control_max':float(controls[:,a].max()),
                'actual_force_min':float(forces[:,a].min()),'actual_force_max':float(forces[:,a].max()),
                'force_range':[float(lo),float(hi)],'force_range_source':origin,
                'meaning':'clipping/saturation is an observation, not automatically an overload'}
    if g.target_body and item['task']!='two-arm-handover':
        collapse=released_goal_collapse(t,np.concatenate(target_z),np.concatenate(target_vz),np.concatenate(goal),np.concatenate(grasp),np.concatenate(upward),rules)
        if collapse:
            predicates['released_goal_collapse']='VIOLATION';events.append(collapse)
    phase=np.full(observed,'unclassified',dtype=object)
    for segment in old.get('phase_segments',[]):
        phase[segment['start_physics_index']:segment['end_physics_index_exclusive']]=segment['phase']
    interfaces=interface_metrics(blocks,observed,phase,dt,round(rules['sustained_window_s']/dt),rules['contact_normal_deadband_n'])
    poses_metrics={}
    for name,values in pose_channels.items():
        value=np.concatenate(values);j=int(m.body_jntadr[m.body(name).id]);idx=int(m.jnt_qposadr[j]);init=initial[1+idx:1+idx+7]
        displacement=np.linalg.norm(value[:,:3]-init[:3],axis=1)
        vertical_axis=rotation(value[:,3:])[:,:,2]
        poses_metrics[name]={'max_displacement_m':float(displacement.max()),'final_displacement_m':float(displacement[-1]),
                            'min_origin_z_m':float(value[:,2].min()),'max_tilt_rad':float(np.arccos(np.clip(vertical_axis[:,2],-1,1)).max()),
                            'note':'pose quality observation; no arbitrary displacement or tilt safety threshold'}
    has_violation=any(v=='VIOLATION' for v in predicates.values())
    assessed='VIOLATION' if has_violation else 'UNKNOWN' if unknown else 'PASS'
    unknown_risks={risk:'UNKNOWN' for risk in rules['not_assessed']}
    return dict(old,extended=True,audit_status=assessed,scope=rules['scope'],predicates=predicates,events=events,
                observations=observations,unknown=unknown,unassessed=unknown_risks,
                extended_metrics={'joint_bounds':bounds,'actuator_saturation':saturation,'body_interfaces':interfaces,'object_motion':poses_metrics,
                    'parked_objects_excluded_from_floor_rule':[m.body(b).name for b in sorted(inactive)],
                    'native_wipe_force_limit':native_limit_metric},
                extended_source_chunks=physics_sources,base_audit_sha256=item['base_audit_sha256'],
                not_assessed=rules['not_assessed'],rule_load_verdict='UNKNOWN_NO_TRANSFER_QUALIFIED_THRESHOLDS')


def audit_episode(item,rules):
    try:
        return _audit(item,rules)
    except Exception as error:
        return dict(item,audit_status='UNKNOWN',coverage_complete=False,events=[],metrics={},predicates={},
                    unknown=[type(error).__name__+': '+str(error)],extended=True,observations=[],extended_metrics={})
