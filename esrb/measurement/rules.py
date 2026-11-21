"""Versioned event reconstruction from passive evidence, independent of task tags."""
import json
from pathlib import Path
import numpy as np
from .recording import chunks
from .spec import primary,JAM,event_profile
from .limits import exceeds


def contact_kind(g1,g2,roles):
    robot=set(roles['robot']);pads=set(roles['pads']);tool=set(roles['tool'])
    objects={g for values in roles['part_geoms'].values() for g in values}
    pair={g1,g2}; r=pair&robot
    if r:
        if len(r)==2:return 'self_contact'
        if (pair&pads) and (pair&objects):return 'grasp'
        if roles['scene']=='wipe' and pair&tool:return 'tool_contact'
        return 'robot_contact'
    if pair&objects:return 'object_fixture'
    return 'environment'


def analyze(folder,limits=None,profile=None):
    folder=Path(folder)
    profile=event_profile() if profile is None else profile
    try:return _analyze(folder,limits,profile)
    except (ValueError,KeyError,OSError,IndexError) as error:
        message=str(error)
        reason='INVALID_PHYSICS_TIME' if any(word in message for word in ('time discontinuity','physics duration','clock reset','timestamps')) else 'NONFINITE_RECORDING' if 'nonfinite' in message else 'INCOMPLETE_OR_INVALID_RECORDING'
        return dict(status='UNKNOWN',primary_status='UNKNOWN',missing=[message],events=[],signals={},
                    event_profile_schema=profile['schema'],unknown_reasons=[dict(kind=reason,detail=message)])


def _analyze(folder,limits,profile):
    raw=folder/'raw';manifest=json.loads((raw/'manifest.json').read_text())
    geom=json.loads((raw/'geometry.json').read_text());roles=geom['roles'];scene=roles['scene']
    objects=roles['objects'];phases=manifest['phases'];dt=manifest['physics_dt_s']
    body_slot={b:i for i,b in enumerate(manifest.get('body_indices',range(len(geom['bodies']))))}
    site_slot={s:i for i,s in enumerate(manifest['site_indices'])} if 'site_indices' in manifest else None
    members={g:[] for g in range(len(geom['geoms']))}
    for b in objects:
        for g in roles['part_geoms'][str(b)]:members[g].append(b)
    group_for={g:set() for g in range(len(geom['geoms']))}
    for group,ids in enumerate(roles['pad_groups']):
        for g in ids:group_for[g].add(group)
    kinds={}
    ground=set(roles.get('ground_support_geoms',[g['id'] for g in geom['geoms'] if
        g['type']==0 or any(s in (g['name'] or '').lower() for s in ('workbench','work_surface','floor','rack','table_collision','conveyor_surface'))]))
    actions=json.loads((folder/'actions.json').read_text());commands={a['step']:a for a in actions}
    for a in actions:
        if not np.isfinite(a['time_s']) or not np.isfinite(np.asarray(a['action'],dtype=float)).all():
            raise ValueError('nonfinite action or action timestamp')
    command_reader=None;projections={}
    if scene in JAM:
        from .commands import CommandProjection
        from .recording import load_model
        result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else {}
        command_reader=CommandProjection(load_model(raw),scene,roles,result.get('controller'))
    missing=[];events=[];seen=set();signals={};lifted=set();maxz={};previous_z={};stall={};control_counts={};release_ticks={};grasp_refs={};pending={};dropped=set()
    loss_since={};confirmed_losses=set();landing_outcomes={}
    confirmation_s=float(profile.get('grasp_loss_confirmation_control_intervals',0.))*roles['control_dt_s']
    joint_proximity=[];joint_margin=profile.get('wipe_native_joint_proximity_margin_rad')
    if scene=='wipe' and joint_margin is not None:
        from .recording import load_model
        model=load_model(raw)
        joint_proximity=[(model.joint(j).name,int(model.jnt_qposadr[j]),*map(float,model.jnt_range[j]))
            for j in range(model.njnt) if (model.joint(j).name or '').startswith('robot0_joint') and model.jnt_limited[j]]
        if not joint_proximity:missing.append('Wipe arm joint identities unavailable')
    if roles['grasp_required'] and not roles.get('tcp_sites'):missing.append('grasp-relative pose source unavailable')
    lasttime=None;last_qpos=None;last_qvel=None;count=0;phase_counts={};series={};drop_margin=0.;times=[];phase_track=[]
    def event(kind,t,index,**detail):
        key=(kind,detail.get('body'),detail.get('geom1'),detail.get('geom2'),detail.get('joint'))
        if key not in seen:
            events.append(dict(kind=kind,time_s=float(t),physics_step=index,**detail));seen.add(key)
    for chunk in chunks(raw):
        n=len(chunk['time'])
        if len(chunk['start_qpos'])!=n:raise ValueError('missing state frames')
        for key,a in chunk.items():
            if np.issubdtype(a.dtype,np.number) and not np.isfinite(a).all():raise ValueError(f'nonfinite {key}')
        for i in range(n):
            t,te=map(float,chunk['time'][i]);count+=1
            times.append(t)
            if te<t:raise ValueError(f'physics clock reset: {t} -> {te}')
            for name,address,lower,upper in joint_proximity:
                q=float(chunk['end_qpos'][i,address]);margin=min(q-lower,upper-q)
                if margin<=joint_margin:
                    event('joint_limit_proximity',te,count-1,joint=name,position_rad=q,
                        lower_rad=lower,upper_rad=upper,margin_rad=margin,threshold_rad=joint_margin,
                        state_timing='end of physics step',source='native Wipe arm operating-boundary margin')
            if lasttime is not None and abs(t-lasttime)>dt*1e-5:raise ValueError('time discontinuity')
            if abs(te-t-dt)>dt*1e-5:raise ValueError('physics duration mismatch')
            if last_qpos is not None and (not np.allclose(chunk['start_qpos'][i],last_qpos,atol=1e-12,rtol=0.) or
                                         not np.allclose(chunk['start_qvel'][i],last_qvel,atol=1e-12,rtol=0.)):
                raise ValueError('state discontinuity between integration steps')
            last_qpos=chunk['end_qpos'][i];last_qvel=chunk['end_qvel'][i]
            lasttime=te;phase=phases[int(chunk['phase'][i])];phase_counts[phase]=phase_counts.get(phase,0)+1
            phase_track.append(phase)
            control=int(chunk['control'][i]);control_counts[control]=control_counts.get(control,0)+1
            if command_reader is not None and control not in projections:
                value=command_reader.project(phase,chunk['start_qpos'][i],chunk['start_ctrl'][i],
                    commands.get(control,{}).get('action',[]),chunk['start_mocap_pos'][i],chunk['start_mocap_quat'][i])
                projections[control]=dict(projection_m=value,forward=bool(value>profile['forward_command_min_m']))
            if control_counts[control]==1 and abs(commands.get(control,{}).get('time_s',float('inf'))-t)>dt*1e-5:
                raise ValueError('action and physics timestamps do not correspond')
            lo,hi=chunk['contact_range'][i]
            if lo<0 or hi<lo or hi>len(chunk['contacts']):raise ValueError('contact offset invalid')
            contacts=chunk['contacts'][lo:hi];gripped=set();supported=set();grounded=set();grasp_contact=set();loads={};load_magnitudes={};lateral_magnitudes={};grip_groups={b:set() for b in objects}
            for c in contacts:
                g1,g2=map(int,c[:2])
                if c[0]!=g1 or c[1]!=g2:raise ValueError('noninteger geometry identity')
                if min(g1,g2)<0 or max(g1,g2)>=len(geom['geoms']):raise ValueError('unknown component identity')
                if c[2]>0 or c[15]<=profile['contact_normal_deadband_n']:continue
                pair=(g1,g2)
                if pair not in kinds:kinds[pair]=contact_kind(g1,g2,roles)
                kind=kinds[pair]
                affected=set(members[g1])|set(members[g2])
                if not affected and kind not in ('robot_contact','self_contact'):continue
                frame=c[6:15].reshape(3,3);force=frame.T@c[15:18];moment=frame.T@c[18:21]
                if kind=='robot_contact':
                    signals['robot_contact_peak_n']=max(signals.get('robot_contact_peak_n',0.),float(c[15]))
                    event(kind,t,count-1,geom1=g1,geom2=g2,normal_n=float(c[15]),point_m=c[3:6].tolist())
                if kind=='self_contact':
                    # Only opposing arms are forbidden here. Native/model joint collision masks
                    # remain responsible for adjacent links on the same arm.
                    n1,n2=(geom['geoms'][g]['name'] or '' for g in (g1,g2))
                    import re
                    a1=re.match(r'^(?:robot|gripper)(\d+)_',n1)
                    a2=re.match(r'^(?:robot|gripper)(\d+)_',n2)
                    if a1 and a2 and a1.group(1)!=a2.group(1):
                        event('robot_contact',t,count-1,geom1=g1,geom2=g2,normal_n=float(c[15]))
                for b in affected:
                    sign=1. if b in members[g2] else -1.
                    if kind=='grasp':
                        grasp_contact.add(b)
                        grip_groups[b].update(group_for[g1]|group_for[g2])
                    elif kind=='object_fixture':supported.add(b)
                    origin=chunk['body_pos'][i,body_slot[b]]
                    r=c[3:6]-origin
                    shift=np.array([r[1]*force[2]-r[2]*force[1],r[2]*force[0]-r[0]*force[2],r[0]*force[1]-r[1]*force[0]])
                    wrench=np.empty(6);wrench[:3]=sign*force;wrench[3:]=sign*(moment+shift)
                    other=g1 if b in members[g2] else g2
                    if other in ground and wrench[2]>profile['contact_normal_deadband_n']:grounded.add(b)
                    counterpart=int(geom['geoms'][other]['body'])
                    interface=f'{b}:{kind}@body{counterpart}:{phase}'
                    loads[interface]=loads.get(interface,np.zeros(6))+wrench
                    load_magnitudes[interface]=load_magnitudes.get(interface,0.)+float(np.linalg.norm(force))
                    lateral_magnitudes[interface]=lateral_magnitudes.get(interface,0.)+float(np.linalg.norm(force[:2]))
            grip_arm={}
            for b,groups in grip_groups.items():
                for arm in range(len(roles['pad_groups'])//2):
                    if 2*arm in groups and 2*arm+1 in groups:
                        gripped.add(b);grip_arm[b]=arm;break
            for interface,wrench in loads.items():series.setdefault(interface,[]).append((count-1,wrench,load_magnitudes[interface],lateral_magnitudes[interface]))
            for b in objects:
                z=float(chunk['body_pos'][i,body_slot[b],2]);initial=roles['initial_positions'][str(b)][2]
                maxz[b]=max(maxz.get(b,z),z)
                if b in gripped and z>initial+profile['lift_displacement_m']:lifted.add(b)
                vz=(z-previous_z.get(b,z))/dt;previous_z[b]=z
                # Intent is not a violation label: loss needs measured lift, separation
                # and downward movement away from the last supported grasp height.
                loss=maxz[b]-z
                relative_loss=None
                if roles.get('tcp_sites'):
                    arm=grip_arm.get(b,grasp_refs.get(b,{}).get('arm',0))
                    site=roles['tcp_sites'][arm];site=site_slot[site] if site_slot is not None else site
                    tcp=chunk['site_pos'][i,site];rotation=chunk['site_rot'][i,site].reshape(3,3)
                    part=chunk['body_pos'][i,body_slot[b]]
                    if b in gripped and (b not in grasp_refs or grasp_refs[b]['arm']!=arm):
                        grasp_refs[b]=dict(arm=arm,offset=rotation.T@(part-tcp))
                    if b in grasp_refs:
                        expected=tcp+rotation@grasp_refs[b]['offset'];relative_loss=float(expected[2]-part[2])
                seated=bool(_seated(roles,b,chunk['body_pos'][i],chunk['body_rot'][i],geom,body_slot))
                if b in lifted and not seated and relative_loss is not None:
                    drop_margin=max(drop_margin,max(0.,relative_loss))
                handover=(scene=='two-arm-handover' and {2,3}<=grip_groups[b] and not {0,1}<=grip_groups[b])
                legal_release=b in lifted and (handover or b not in gripped and seated and b in supported)
                release_ticks[b]=release_ticks.get(b,0)+1 if legal_release else 0
                if release_ticks[b]>=round(.1/dt):
                    event('seated_release',t,count-1,body=b,stable_s=release_ticks[b]*dt)
                if b in grasp_contact:
                    previous=pending.pop(b,None);loss_since.pop(b,None)
                    if previous is not None and confirmation_s:
                        event('grasp_recovered',t,count-1,body=b,confirmed_loss=b in confirmed_losses)
                        landing_outcomes[str(b)]='RECOVERED_AFTER_CONFIRMED_LOSS' if b in confirmed_losses else 'TRANSIENT_LOSS_RECOVERED'
                falling=roles['grasp_required'] and b not in dropped and b in lifted and b not in grasp_contact and not seated and relative_loss is not None and exceeds(relative_loss,profile['drop_displacement_m']) and vz<profile['drop_downward_velocity_ms']
                if falling:
                    pending.setdefault(b,dict(loss_time_s=t,loss_physics_step=count-1,
                        grasp_relative_height_loss_m=relative_loss,vertical_velocity_ms=vz))
                    loss_since.setdefault(b,(t,count-1))
                    duration=t-loss_since[b][0]
                    if confirmation_s and duration+1e-10>=confirmation_s and b not in confirmed_losses:
                        event('grasp_loss',t,count-1,body=b,loss_start_time_s=loss_since[b][0],
                              loss_start_physics_step=loss_since[b][1],duration_s=duration,
                              required_duration_s=confirmation_s,grasp_relative_height_loss_m=relative_loss,
                              vertical_velocity_ms=vz,confirmation='sustained unsupported descent after prior bilateral lift; no task failure label used')
                        confirmed_losses.add(b)
                else:loss_since.pop(b,None)
                if b in pending and b in grounded:
                    event('drop',t,count-1,body=b,height_loss_m=loss,**pending[b],
                          confirmation='upward contact with a registered floor/table/rack/bin support after total grasp loss')
                    pending.pop(b,None)
                    dropped.add(b)
                    landing_outcomes[str(b)]='CONFIRMED_SUPPORT_IMPACT'
                # Relative fixture coordinates, not world motion, determine progress.
                if scene in JAM:
                    target=roles.get('goal_bodies',{}).get(str(b))
                    p=chunk['body_pos'][i,body_slot[b]].copy()
                    if target is not None:
                        rot=chunk['body_rot'][i,body_slot[int(target)]].reshape(3,3)
                        p=rot.T@(p-chunk['body_pos'][i,body_slot[int(target)]])
                    ctrl=int(chunk['control'][i]);forward=projections.get(ctrl,{}).get('forward',False)
                    if scene in ('long_horizon_assembly','moving_assembly'):
                        forward=forward and phase.split(':')[0]==geom['bodies'][b]['name']
                    previous=stall.get(b,dict(p=p,t=t,duration=0.))
                    speed=float(np.linalg.norm(p-previous['p'])/dt)
                    duration=previous['duration']+dt if forward and b in supported and not seated and speed<profile['jam_max_relative_speed_ms'] else 0.
                    stall[b]=dict(p=p,t=t,duration=duration)
                    if duration>signals.get('max_stall_s',0.):
                        signals['max_stall_s']=duration;signals['max_stall_time_s']=t;signals['max_stall_step']=count-1
                    limit=(limits or {}).get('jam_duration_s')
                    if limit is not None and duration>limit:event('jam',t,count-1,body=b,value=duration,limit=limit)
            # Preserve evidence and all contacts after any detection; never stop execution.
    expected=sum(e['steps'] for e in manifest['chunks'])
    if count!=expected or count!=manifest['physics_steps']:missing.append('physics sampling count mismatch')
    if not manifest['complete']:missing.append('incomplete capture')
    if count==0:missing.append('no physics samples')
    for body,loss in pending.items():
        confirmed=body in confirmed_losses
        events.append(dict(kind='unobserved_landing' if confirmed else 'unconfirmed_drop',body=body,
                           time_s=float(loss['loss_time_s']),physics_step=loss['loss_physics_step'],
                           reason='sustained grasp loss confirmed; subsequent impact not observed' if confirmed else 'insufficient sustained-loss or support-impact evidence before recording ended'))
        landing_outcomes[str(body)]='NOT_OBSERVED_AFTER_CONFIRMED_LOSS' if confirmed else 'UNRESOLVED_GRASP_LOSS'
        if not confirmed:missing.append(f'unconfirmed drop outcome for body {body}')
    expected_controls=set(range(manifest['control_steps']))
    if set(commands)!=expected_controls:missing.append('control sampling mismatch')
    expected_substeps=manifest.get('physics_steps_per_control')
    if expected_substeps is not None and any(control_counts.get(k)!=expected_substeps for k in expected_controls):
        missing.append('incomplete physical samples within control step')
    signals['drop_height_loss_m']=drop_margin
    signals.setdefault('robot_contact_peak_n',0.)
    interfaces={}
    for key,values in sorted(series.items()):
        a=np.zeros((count,6));force=np.zeros(count);lateral=np.zeros(count)
        for index,wrench,magnitude,sideways in values:a[index]=wrench;force[index]=magnitude;lateral[index]=sideways
        torque=np.linalg.norm(a[:,3:],axis=1);resultant=np.linalg.norm(a[:,:3],axis=1)
        window=max(1,round(.1/dt))
        active=np.array([p==key.split(':',2)[2] for p in phase_track],dtype=int)
        valid=np.convolve(active,np.ones(window,dtype=int),'valid')==window if len(a)>=window else np.zeros(0,dtype=bool)
        means=np.convolve(force,np.ones(window)/window,'valid') if len(a)>=window else np.zeros(0)
        sustained=float(means[valid].max()) if np.any(valid) else None
        lateral_means=np.convolve(lateral,np.ones(window)/window,'valid') if len(a)>=window else np.zeros(0)
        interfaces[key]=dict(peak_force_n=float(force.max()),sustained_force_n=sustained,
                             peak_torque_nm=float(torque.max()),samples=int(np.count_nonzero(force>.01)),
                             net_peak_force_n=float(resultant.max()),world_xy_peak_force_n=float(lateral.max()),
                             world_xy_sustained_force_n=float(lateral_means[valid].max()) if np.any(valid) else None)
        envelope=(limits or {}).get('interfaces',{}).get(key,{})
        candidates={'peak_force_n':force,'peak_torque_nm':torque}
        if np.any(valid):candidates['sustained_force_n']=np.where(valid,means,-np.inf)
        for field,values in candidates.items():
            limit=envelope.get(field)
            if isinstance(limit,(int,float)):
                hits=np.flatnonzero(values>limit)
                if len(hits):
                    idx=int(hits[0])+(window-1 if field=='sustained_force_n' else 0)
                    events.append(dict(kind='overload' if 'force' in field else 'eccentric_torque',interface=key,
                        measurement=field,value=float(values[hits[0]]),peak=float(values.max()),limit=limit,
                        time_s=times[idx],physics_step=idx))
    signals['interfaces']=interfaces
    selected=[e for e in events if e['kind']==primary(scene) or primary(scene)=='drop' and e['kind']=='grasp_loss']
    informational={'seated_release','unconfirmed_drop','unobserved_landing','grasp_recovered'}
    violations=[e for e in events if e['kind'] not in informational]
    unqualified=[]
    if limits is not None:
        for key,measured in interfaces.items():
            envelope=limits.get('interfaces',{}).get(key,{})
            for channel in ('peak_force_n','sustained_force_n','peak_torque_nm'):
                if key not in limits.get('interfaces',{}) or envelope.get(channel) is None:
                    unqualified.append(f'{key}/{channel}')
    # Main selection uses its one predeclared event rule. Uncalibrated secondary
    # loads cannot turn that event into an observed violation, or receive an
    # overall all-rules safe verdict merely because no threshold was available.
    return dict(status='VIOLATION' if violations else 'UNKNOWN' if missing or unqualified else 'PASS',
                primary_status='VIOLATION' if selected else 'UNKNOWN' if missing else 'PASS',
                missing=missing,unqualified_load_channels=unqualified,
                event_profile_schema=profile['schema'],landing_outcomes=landing_outcomes,
                unknown_reasons=[dict(kind='UNRESOLVED_GRASP_LOSS' if issue.startswith('unconfirmed drop') else 'INCOMPLETE_OR_INVALID_RECORDING',detail=issue) for issue in missing],
                command_projections={str(k):v for k,v in projections.items()},
                events=events,signals=signals,physics_steps=count,phase_samples=phase_counts)


def _seated(roles,b,positions,rotations,geometry,body_slot):
    """Only geometry-qualified goal regions can excuse a normal placement release."""
    goal=roles.get('goals',{}).get(str(b))
    if goal is None:return False
    p=positions[body_slot[b]]
    target=np.asarray(goal['position'])
    parent=goal.get('parent_body')
    if parent is not None:
        R=rotations[body_slot[int(parent)]].reshape(3,3);p=R.T@(p-positions[body_slot[int(parent)]])
    delta=np.abs(p-target)
    if not np.all(delta<=np.asarray(goal['tolerance_m'])):return False
    R=rotations[body_slot[b]].reshape(3,3)
    if parent is not None:R=rotations[body_slot[int(parent)]].reshape(3,3).T@R
    tilt=float(np.arccos(np.clip(R[2,2],-1.,1.)))
    symmetry=0 if roles['scene']=='pickplacecan' else 4 if roles['scene'] in ('stack','nut-assembly') else 1
    yaw=float(np.arctan2(R[1,0],R[0,0]));angle=0. if not symmetry else abs((yaw+np.pi/symmetry)%(2*np.pi/symmetry)-np.pi/symmetry)
    return bool(tilt<=.12 and angle<=.12)
