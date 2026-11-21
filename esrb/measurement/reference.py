"""Independent reconstruction from raw state and model geometry.

This module deliberately imports no production classifier or event rules. The
simulator is shared, so this checks collection/classification, not engine truth.
"""
import json
from pathlib import Path
import numpy as np
import mujoco
from .recording import chunks,load_model


def check(folder,observed,max_samples=32,limits=None,profile=None):
    folder=Path(folder);raw=folder/'raw'
    if profile is None:
        from .spec import event_profile
        schema=observed.get('event_profile_schema','measurement-event-profile/5')
        profile=event_profile(int(schema.rsplit('/',1)[-1]))
    if observed.get('missing') and not observed.get('signals'):
        return dict(status='UNKNOWN',error='invalid raw measurement: '+'; '.join(observed['missing']),
                    scope='Structural recording failure is preserved; no complete-reference claim')
    try:
        manifest=json.loads((raw/'manifest.json').read_text());geometry=json.loads((raw/'geometry.json').read_text())
        model=load_model(raw);data=mujoco.MjData(model)
        role=geometry['roles'];scene=role['scene'];dt=manifest['physics_dt_s']
        body_indices=manifest.get('body_indices',list(range(model.nbody)))
        body_slot={b:i for i,b in enumerate(body_indices)}
        site_indices=manifest.get('site_indices',list(range(model.nsite)))
        site_slot={s:i for i,s in enumerate(site_indices)}
        expected_names={
            'tool_handling':{'tool'},'long_horizon_assembly':{'base','connector','cap'},
            'moving_transport':{'base','connector','cap'},'moving_assembly':{'base','connector','cap'},
            'constrained_push':{'workpiece'},'force_alignment':{'workpiece'},'compliant_assembly':{'workpiece'},
            'stack':{'cubeA_main'},'pickplacecan':{'Can_main'},'two-arm-handover':{'hammer_root'},
            'nut-assembly':{'SquareNut_main'},'wipe':{'gripper0_wiping_gripper'}}
        actual_names={model.body(b).name for b in role['objects']}
        # Small analytic unit fixtures have their own scene identity, whereas
        # registered task models must match their independently named components.
        if not role.get('analytic_fixture',False) and actual_names!=expected_names[scene]:
            raise ValueError(f'task component registration mismatch: {actual_names}')
        if role.get('analytic_fixture',False):
            robot_ids=set(role['robot']);pad_groups=role['pad_groups'];tool_ids=set(role['tool'])
        else:
            names=[model.geom(g).name or '' for g in range(model.ngeom)]
            if scene in ('stack','pickplacecan','two-arm-handover','nut-assembly','wipe'):
                robot_ids={g for g,name in enumerate(names) if name.startswith(('robot0_','robot1_','gripper0_','gripper1_'))}
                pad_groups=[]
                for arm in (0,1):
                    for finger in (1,2):
                        group=[g for g,name in enumerate(names) if name.startswith(f'gripper{arm}_finger{finger}_')]
                        if group:pad_groups.append(group)
            else:
                pad_groups=[[g for g in range(model.ngeom)
                    if model.body(int(model.geom_bodyid[g])).name==side+'_finger'
                    and bool(model.geom_contype[g] or model.geom_conaffinity[g])] for side in ('left','right')]
                robot_ids={g for g,name in enumerate(names) if name.startswith('panda_')}|set(sum(pad_groups,[]))
            tool_ids={g for g,name in enumerate(names) if scene=='wipe' and ('wiping_surface' in name or 'wiping_corner' in name)}
            if robot_ids!=set(role['robot']) or set(sum(pad_groups,[]))!=set(role['pads']) or tool_ids!=set(role['tool']):
                raise ValueError('model-derived robot/pad/tool component registration mismatch')
            if [set(group) for group in pad_groups]!=[set(group) for group in role['pad_groups']]:
                raise ValueError('model-derived opposing finger registration mismatch')
        n=manifest['physics_steps'];stride=max(1,n//max_samples);index=0;checked=0
        maxstate=0.;maxforce=0.;maxpoint=0.;maxfk=0.;events=[];last={};grasped=set();maxheight={}
        witnesses={e['physics_step']+offset for e in observed['events'] if e.get('physics_step') is not None for offset in (-1,0,1)}
        buffers={};times=[];phase_track=[];stalled={b:[] for b in role['objects']};release={b:[] for b in role['objects']}
        relative_positions={b:[] for b in role['objects']}
        anchors={};pending={};dropped=set();descent_start={};loss_confirmed=set()
        duration_required=float(profile.get('grasp_loss_confirmation_control_intervals',0.))*role['control_dt_s']
        tcp_sites=[i for i in range(model.nsite) if model.site(i).name in ('grip_origin','gripper0_grip_site','gripper1_grip_site')]
        actions={a['step']:a for a in json.loads((folder/'actions.json').read_text())}
        command_values={};command_data=None
        jam_scene=scene in ('long_horizon_assembly','moving_assembly','constrained_push','force_alignment','compliant_assembly','nut-assembly')
        if jam_scene and not role.get('analytic_fixture'):
            command_data=mujoco.MjData(model)
            if scene=='nut-assembly':
                from esrb.runtime.paths import project_root
                config=json.loads((project_root()/'.esr-local/aspire/checkout/third_party/robosuite/robosuite/controllers/config/osc_pose.json').read_text())
                translation_scale=(np.asarray(config['output_max'][:3])-np.asarray(config['output_min'][:3]))/2
            else:
                commanded_arm=[int(model.jnt_qposadr[model.joint(f'joint{k}').id]) for k in range(1,8)]
                commanded_site=model.site('grip_origin').id
        # Resolve object and robot identity again from body ancestry / declared model
        # membership. No contact_kind() call and no reuse of its classified contacts.
        object_for_geom={}
        for b in role['objects']:
            for g in range(model.ngeom):
                ancestor=int(model.geom_bodyid[g])
                while ancestor and ancestor!=b:ancestor=int(model.body_parentid[ancestor])
                if ancestor==b:object_for_geom[g]=b
        if (raw/'initial.npz').exists():
            with np.load(raw/'initial.npz',allow_pickle=False) as initial:
                for key in ('qpos','qvel','mocap_pos','mocap_quat'):getattr(data,key)[:]=initial[key]
            mujoco.mj_forward(model,data)
        ground_geoms=set()
        for g in range(model.ngeom):
            if g in robot_ids or g in object_for_geom:continue
            name=(model.geom(g).name or '').lower();size=model.geom_size[g]
            named=any(s in name for s in ('workbench','work_surface','floor','rack','table_collision','conveyor_surface'))
            flat_box=model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX and size[0]>=.05 and size[1]>=.05 and size[2]<=.015
            if data.geom_xmat[g,8]>=.9 and (named or flat_box or model.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE):ground_geoms.add(g)
        robot_bodies=set(int(model.geom_bodyid[g]) for g in robot_ids)
        pad_bodies=set(int(model.geom_bodyid[g]) for g in sum(pad_groups,[]))
        native_margin=profile.get('wipe_native_joint_proximity_margin_rad') if scene=='wipe' else None
        checked_joints=[j for j in range(model.njnt) if (model.joint(j).name or '').startswith('robot0_joint')
            and model.jnt_limited[j]] if native_margin is not None else []
        if native_margin is not None and not checked_joints:raise ValueError('Wipe arm joint identities unavailable')
        for batch in chunks(raw):
            for i,(t,te) in enumerate(batch['time']):
                for joint in checked_joints:
                    lower,upper=model.jnt_range[joint];angle=batch['end_qpos'][i,int(model.jnt_qposadr[joint])]
                    if not lower+native_margin<angle<upper-native_margin:
                        events.append(dict(kind='joint_limit_proximity',time_s=float(te),joint=model.joint(joint).name))
                times.append(float(t));load={};load_norm={};load_lateral={};supported=set()
                phase_track.append(manifest['phases'][int(batch['phase'][i])])
                control=int(batch['control'][i])
                if jam_scene and control not in command_values:
                    phase=phase_track[-1];stage=phase.rsplit(':',1)[-1]
                    direction=None;amount=0.
                    if scene=='constrained_push' and stage=='push':direction=np.array([1.,0.,0.])
                    elif scene=='force_alignment' and stage=='insert':direction=np.array([0.,1.,0.])
                    elif scene=='compliant_assembly' and stage=='insert':direction=np.array([1.,0.,0.])
                    elif scene in ('long_horizon_assembly','moving_assembly','nut-assembly') and stage=='lower':direction=np.array([0.,0.,-1.])
                    if direction is not None and command_data is not None:
                        if scene=='nut-assembly':
                            delta=np.clip(np.asarray(actions[control]['action'][:3]),-1.,1.)*translation_scale
                            amount=float(np.dot(direction,delta))
                        else:
                            command_data.qpos[:]=batch['start_qpos'][i]
                            command_data.mocap_pos[:]=batch['start_mocap_pos'][i]
                            command_data.mocap_quat[:]=batch['start_mocap_quat'][i]
                            mujoco.mj_kinematics(model,command_data)
                            origin=command_data.site_xpos[commanded_site].copy()
                            if scene in ('long_horizon_assembly','moving_assembly'):
                                part_name=phase.split(':')[0]
                                if part_name!='base':reference_body=model.body('base').id
                                elif scene=='moving_assembly':reference_body=model.body('pallet').id
                                else:reference_body=None
                                if reference_body is not None:direction=command_data.xmat[reference_body].reshape(3,3)@direction
                            command_data.qpos[commanded_arm]=batch['start_ctrl'][i,:7]
                            mujoco.mj_kinematics(model,command_data)
                            amount=float(direction@(command_data.site_xpos[commanded_site]-origin))
                    command_values[control]=dict(projection_m=amount,forward=amount>1e-5)
                lo,hi=batch['contact_range'][i];contacts=batch['contacts'][lo:hi]
                held=set();grasp_contact=set();grounded=set();touches={b:set() for b in role['objects']}
                for contact in contacts:
                    if contact[2]>0 or contact[15]<=.01:continue
                    g1,g2=map(int,contact[:2]);a,b=(int(model.geom_bodyid[g]) for g in (g1,g2))
                    obj=object_for_geom.get(g1,object_for_geom.get(g2))
                    robot=(a in robot_bodies) != (b in robot_bodies)
                    hand=(a in pad_bodies or b in pad_bodies) and obj is not None
                    if hand:
                        grasp_contact.add(obj)
                        # Body ancestry readback independently checks finger sides.
                        for group,ids in enumerate(pad_groups):
                            bodies={int(model.geom_bodyid[g]) for g in ids}
                            if a in bodies or b in bodies:touches[obj].add(group)
                    permitted_wipe=scene=='wipe' and (g1 in tool_ids or g2 in tool_ids)
                    if robot and not hand and not permitted_wipe:
                        events.append(dict(kind='robot_contact',time_s=float(t),geom1=g1,geom2=g2))
                    if a in robot_bodies and b in robot_bodies:
                        import re
                        names=[model.geom(g).name or '' for g in (g1,g2)]
                        arms=[re.match(r'^(?:robot|gripper)(\d+)_',name) for name in names]
                        if all(arms) and arms[0].group(1)!=arms[1].group(1):
                            events.append(dict(kind='robot_contact',time_s=float(t),geom1=g1,geom2=g2))
                    for load_obj in {object_for_geom.get(g) for g in (g1,g2)}-{None}:
                        if hand:kind='grasp'
                        elif permitted_wipe:kind='tool_contact'
                        elif robot:kind='robot_contact'
                        elif a in robot_bodies and b in robot_bodies:kind='self_contact'
                        else:kind='object_fixture';supported.add(obj)
                        # Independently shift the spatial wrench from the contact
                        # point to the body origin using a skew-symmetric matrix.
                        basis=contact[6:15].reshape(3,3)
                        f=np.einsum('ij,i->j',basis,contact[15:18])
                        tau=np.einsum('ij,i->j',basis,contact[18:21])
                        r=contact[3:6]-batch['body_pos'][i,body_slot[load_obj]]
                        skew=np.array([[0.,-r[2],r[1]],[r[2],0.,-r[0]],[-r[1],r[0],0.]])
                        direction=1. if object_for_geom.get(g2)==load_obj else -1.
                        support=g1 if object_for_geom.get(g2)==load_obj else g2
                        if support in ground_geoms and direction*f[2]>.01:grounded.add(load_obj)
                        if kind=='object_fixture':supported.add(load_obj)
                        counterpart=int(model.geom_bodyid[support])
                        key=f'{load_obj}:{kind}@body{counterpart}:{manifest["phases"][int(batch["phase"][i])]}'
                        load[key]=load.get(key,np.zeros(6))+direction*np.r_[f,tau+skew@f]
                        load_norm[key]=load_norm.get(key,0.)+float(np.sqrt(np.sum(f*f)))
                        load_lateral[key]=load_lateral.get(key,0.)+float(np.sqrt(np.sum(f[:2]*f[:2])))
                for key,wrench in load.items():buffers.setdefault(key,[]).append((index,wrench,load_norm[key],load_lateral[key]))
                held_arm={}
                for obj,touch in touches.items():
                    for arm in range(len(pad_groups)//2):
                        if {2*arm,2*arm+1}<=touch:held.add(obj);held_arm[obj]=arm;break
                for obj in role['objects']:
                    z=float(batch['body_pos'][i,body_slot[obj],2]);initial=role['initial_positions'][str(obj)][2]
                    maxheight[obj]=max(maxheight.get(obj,z),z)
                    if obj in held and z>initial+.035:grasped.add(obj)
                    vz=(z-last.get(obj,z))/dt;last[obj]=z
                    region=role.get('goals',{}).get(str(obj));seated=False
                    if region:
                        p=batch['body_pos'][i,body_slot[obj]].copy();parent=region.get('parent_body')
                        if parent is not None:
                            R=batch['body_rot'][i,body_slot[int(parent)]].reshape(3,3)
                            p=np.linalg.solve(R,p-batch['body_pos'][i,body_slot[int(parent)]])
                        seated=all(abs(float(x-y))<=tol for x,y,tol in zip(p,region['position'],region['tolerance_m']))
                        R=batch['body_rot'][i,body_slot[obj]].reshape(3,3)
                        if parent is not None:R=np.linalg.solve(batch['body_rot'][i,body_slot[int(parent)]].reshape(3,3),R)
                        n=0 if scene=='pickplacecan' else 4 if scene in ('stack','nut-assembly') else 1
                        yaw=float(np.arctan2(R[1,0],R[0,0]));yaw_error=0. if n==0 else min(abs(yaw-2*np.pi*k/n) for k in range(-n,n+1))
                        seated=seated and R[2,2]>=np.cos(.12) and yaw_error<=.12
                    separation=None
                    if tcp_sites:
                        arm=held_arm.get(obj,anchors.get(obj,{}).get('arm',0));site=site_slot[tcp_sites[arm]]
                        rotation=batch['site_rot'][i,site].reshape(3,3);tcp=batch['site_pos'][i,site]
                        if obj in held and (obj not in anchors or anchors[obj]['arm']!=arm):
                            anchors[obj]=dict(arm=arm,offset=np.linalg.solve(rotation,batch['body_pos'][i,body_slot[obj]]-tcp))
                        if obj in anchors:separation=float((tcp+rotation@anchors[obj]['offset'])[2]-z)
                    if obj in grasp_contact:
                        previous=pending.pop(obj,None);descent_start.pop(obj,None)
                        if previous is not None and duration_required:
                            events.append(dict(kind='grasp_recovered',time_s=float(t),body=obj))
                    falling=role['grasp_required'] and obj not in dropped and obj in grasped and obj not in grasp_contact and not seated and separation is not None and separation>profile['drop_displacement_m'] and vz<profile['drop_downward_velocity_ms']
                    if falling:
                        if obj not in pending:pending[obj]=float(t)
                        if obj not in descent_start:descent_start[obj]=float(t)
                        if duration_required and float(t)-descent_start[obj]+1e-10>=duration_required and obj not in loss_confirmed:
                            events.append(dict(kind='grasp_loss',time_s=float(t),body=obj))
                            loss_confirmed.add(obj)
                    else:descent_start.pop(obj,None)
                    if obj in pending and obj in grounded:
                        events.append(dict(kind='drop',time_s=float(t),body=obj));pending.pop(obj);dropped.add(obj)
                    p=batch['body_pos'][i,body_slot[obj]].copy();parent=role.get('goal_bodies',{}).get(str(obj))
                    if parent is not None:
                        p=np.linalg.solve(batch['body_rot'][i,body_slot[int(parent)]].reshape(3,3),p-batch['body_pos'][i,body_slot[int(parent)]])
                    relative_positions[obj].append(p)
                    forward=command_values.get(int(batch['control'][i]),{}).get('forward',False)
                    if scene in ('long_horizon_assembly','moving_assembly'):
                        forward=forward and phase_track[-1].split(':')[0]==model.body(obj).name
                    stalled[obj].append(bool(forward and obj in supported and not seated))
                    receiver=scene=='two-arm-handover' and {2,3}<=touches[obj] and not {0,1}<=touches[obj]
                    release[obj].append(bool(obj in grasped and (receiver or obj not in held and seated and obj in supported)))
                if index%stride==0 or index==n-1 or index in witnesses:
                    for key in ('qpos','qvel','act','ctrl','qacc_warmstart','qfrc_applied','xfrc_applied','mocap_pos','mocap_quat'):
                        getattr(data,key)[:]=batch['start_'+key][i]
                    data.time=float(t)
                    mujoco.mj_forward(model,data)
                    # Restore the solver warm start AFTER mj_forward before the solve.
                    data.qacc_warmstart[:]=batch['start_qacc_warmstart'][i]
                    mujoco.mj_step(model,data)
                    maxfk=max(maxfk,float(np.max(np.abs(data.xpos[body_indices]-batch['body_pos'][i]))),
                              float(np.max(np.abs(data.xmat[body_indices]-batch['body_rot'][i]))))
                    if site_indices:
                        maxfk=max(maxfk,float(np.max(np.abs(data.site_xpos[site_indices]-batch['site_pos'][i]))),
                                  float(np.max(np.abs(data.site_xmat[site_indices]-batch['site_rot'][i]))))
                    maxstate=max(maxstate,float(np.max(np.abs(data.qpos-batch['end_qpos'][i]))))
                    if data.ncon!=len(contacts):raise ValueError('replayed contact count differs')
                    for j,c in enumerate(data.contact):
                        expected=contacts[j];w=np.zeros(6);mujoco.mj_contactForce(model,data,j,w)
                        if tuple(map(int,expected[:2]))!=(c.geom1,c.geom2):raise ValueError('replayed geometry mismatch')
                        maxforce=max(maxforce,float(np.max(np.abs(w-expected[15:21]))))
                        maxpoint=max(maxpoint,float(np.linalg.norm(c.pos-expected[3:6])))
                    checked+=1
                index+=1
        for obj,time in pending.items():events.append(dict(kind='unobserved_landing' if obj in loss_confirmed else 'unconfirmed_drop',time_s=time,body=obj))
        signal_errors=[]
        reported_commands=observed.get('command_projections',{})
        for control,value in command_values.items():
            reported=reported_commands.get(str(control))
            if reported is None or reported['forward']!=value['forward'] or abs(reported['projection_m']-value['projection_m'])>1e-9:
                signal_errors.append(dict(control=control,error='signed command projection reconstruction differs'))
        for key in set(observed['signals'].get('interfaces',{}))-set(buffers):
            signal_errors.append(dict(interface=key,error='interface missing in independent reconstruction'))
        for key,sparse in buffers.items():
            values=np.zeros((index,6));force=np.zeros(index);lateral=np.zeros(index)
            for at,wrench,magnitude,sideways in sparse:values[at]+=wrench;force[at]+=magnitude;lateral[at]+=sideways
            net_force=np.sqrt(np.sum(values[:,:3]**2,axis=1));torque=np.sqrt(np.sum(values[:,3:]**2,axis=1))
            window=max(1,round(.1/dt));cumulative=np.r_[0.,np.cumsum(force)]
            mean=(cumulative[window:]-cumulative[:-window])/window
            active=np.array([p==key.split(':',2)[2] for p in phase_track],dtype=int)
            counts=np.r_[0,np.cumsum(active)];valid=(counts[window:]-counts[:-window])==window
            measured={'peak_force_n':force,'peak_torque_nm':torque,'net_peak_force_n':net_force,'world_xy_peak_force_n':lateral}
            if np.any(valid):
                measured['sustained_force_n']=np.where(valid,mean,-np.inf)
                side_cumulative=np.r_[0.,np.cumsum(lateral)]
                measured['world_xy_sustained_force_n']=np.where(valid,(side_cumulative[window:]-side_cumulative[:-window])/window,-np.inf)
            expected_signals=observed['signals'].get('interfaces',{}).get(key,{})
            for field,series in measured.items():
                if not len(series):continue
                actual=float(np.max(series));reported=expected_signals.get(field)
                if reported is None:raise ValueError(f'missing reconstructed load interface: {key}/{field}')
                absolute_floor=.001
                if abs(actual-reported)>max(absolute_floor,.05*abs(actual)):
                    signal_errors.append(dict(interface=key,field=field,reference=actual,observed=reported))
                bound=(limits or {}).get('interfaces',{}).get(key,{}).get(field)
                if bound is not None and field in ('peak_force_n','sustained_force_n','peak_torque_nm'):
                    hits=np.flatnonzero(series>bound)
                    if len(hits):
                        at=int(hits[0])+(window-1 if field=='sustained_force_n' else 0)
                        events.append(dict(kind='overload' if 'force' in field else 'eccentric_torque',interface=key,measurement=field,time_s=times[at]))
        def consecutive(mask):
            i=np.arange(len(mask));last_false=np.maximum.accumulate(np.where(~np.asarray(mask,dtype=bool),i,-1))
            return (i-last_false)*dt
        for obj,positions in relative_positions.items():
            if not positions:continue
            p=np.asarray(positions);speed=np.r_[0.,np.linalg.norm(np.diff(p,axis=0),axis=1)/dt]
            lengths=consecutive(np.asarray(stalled[obj])&(speed<.001))
            limit=(limits or {}).get('jam_duration_s')
            if limit is not None:
                hits=np.flatnonzero(lengths>limit+1e-10)
                if len(hits):events.append(dict(kind='jam',time_s=times[int(hits[0])],body=obj))
            lengths=consecutive(release[obj]);hits=np.flatnonzero(lengths>=.1-1e-10)
            if len(hits):events.append(dict(kind='seated_release',time_s=times[int(hits[0])],body=obj))
        comparisons=[]
        def event_order(e):
            binding=e.get('interface',str(e.get('body',sorted([e.get('geom1',-1),e.get('geom2',-1)]))))
            return (e['time_s'],binding)
        event_kinds=('drop','grasp_loss','grasp_recovered','unobserved_landing','unconfirmed_drop','robot_contact','overload','eccentric_torque','jam','seated_release')
        if native_margin is not None:event_kinds+=('joint_limit_proximity',)
        for kind in event_kinds:
            predicted=sorted([e for e in observed['events'] if e['kind']==kind],key=event_order)
            expected=sorted([e for e in events if e['kind']==kind],key=event_order)
            def binding(e):
                if kind in ('drop','grasp_loss','grasp_recovered','unobserved_landing','unconfirmed_drop','jam','seated_release'):return str(e['body'])
                if kind=='robot_contact':return str(sorted((e['geom1'],e['geom2'])))
                if kind=='joint_limit_proximity':return e['joint']
                return e['interface']+'/'+e['measurement']
            def onsets(rows):
                result={}
                for row in rows:result.setdefault(binding(row),float(row['time_s']))
                return result
            found=onsets(predicted);wanted=onsets(expected)
            extra=set(found)-set(wanted);missed=set(wanted)-set(found)
            errors=[abs(found[key]-wanted[key]) for key in set(found)&set(wanted)]
            location=None if not predicted or not expected else abs(predicted[0]['time_s']-expected[0]['time_s'])
            components=None if not predicted and not expected else not extra and not missed
            consistent=not extra and not missed and all(error<=dt+1e-10 for error in errors)
            comparisons.append(dict(rule=kind,detected=bool(predicted),reference=bool(expected),
                false_positive=bool(extra),false_negative=bool(missed),
                first_event_error_s=location,component_match=components,
                max_event_error_s=max(errors,default=None),bindings_checked=len(set(found)|set(wanted)),
                unmatched_detected=sorted(extra),unmatched_reference=sorted(missed),
                status='PASS' if consistent else 'FAIL'))
        status='PASS' if maxstate<1e-7 and maxfk<1e-8 and maxforce<1e-4 and maxpoint<1e-8 and not signal_errors and all(c['status']=='PASS' for c in comparisons) else 'FAIL'
        return dict(status=status,physics_steps=index,replayed_samples=checked,max_state_error=maxstate,
            max_wrench_abs_error=maxforce,max_contact_point_error_m=maxpoint,comparisons=comparisons,load_signal_errors=signal_errors,
            max_fk_error=maxfk,
            reconstructed_commands=len(command_values),
            scope='independent reconstruction using the same simulator, not independent physics truth')
    except Exception as error:
        return dict(status='UNKNOWN',error=f'{type(error).__name__}: {error}')
