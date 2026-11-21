"""Privileged deterministic workcell controller for measurement acceptance.

This is an engineering actor, not an ASPIRE Agent skill. It uses the unchanged
ASPIRE Robosuite task registration, Panda OSC_POSE controller and native
success predicate. Native state/Jacobian access is explicitly recorded.
"""
import json
from pathlib import Path
import numpy as np
import mujoco


class CartesianActor:
    def __init__(self,env,arm=0,*,integral=True,max_translation=.05):
        self.env=env;self.robot=env.robots[arm];self.arm=arm
        self.m=env.sim.model._model;self.d=env.sim.data._data
        self.site=self.robot.eef_site_id
        self.rot=np.array([[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]])
        self.home=self.d.site_xpos[self.site].copy()
        self.home_rotation=self.d.site_xmat[self.site].reshape(3,3).copy()
        self.position_bias=np.zeros(3)
        self.integral=integral;self.max_translation=max_translation

    def action(self,target,close,rotation=None):
        rotation=self.rot if rotation is None else rotation
        from robosuite.utils.transform_utils import mat2quat,quat2axisangle
        current=self.d.site_xmat[self.site].reshape(3,3)
        angular=quat2axisangle(mat2quat(rotation@current.T))
        error=np.asarray(target)-self.d.site_xpos[self.site]
        # Native gravity compensation covers the robot, not the carried object.
        # Bounded integral action removes its steady pose error without changing
        # the model, actuator limits, or the commanded task goal.
        if self.integral:self.position_bias=np.clip(self.position_bias+.01*np.clip(error,-.02,.02),-.04,.04)
        delta=np.r_[np.clip(error+self.position_bias,-self.max_translation,self.max_translation),
                    np.clip(angular,-.15,.15)]
        scale=(self.robot.controller.output_max-self.robot.controller.output_min)/2.
        command=np.clip(delta/scale,-1.,1.)
        return np.r_[command,1. if close else -1.] if self.robot.gripper.dof else command


def execute(request):
    import robosuite as suite
    from robosuite.controllers import load_controller_config
    from cap_harness.robosuite.registry import RobosuiteTaskRegistry
    from esrb.backends.aspire.pickup_can_registry import install_registry
    from esrb.measurement.episodes import roles_for
    from esrb.measurement.recording import Recorder,identity
    scene=request['scene'];case=request['case'];seed=request['seed'];folder=Path(request['folder'])
    if scene=='pickplacecan':metadata=install_registry()
    else:metadata=RobosuiteTaskRegistry().resolve(dict(stack='cube_stack',
        **{'two-arm-handover':'two_arm_handover','nut-assembly':'nut_assembly','wipe':'spill_wipe'})[scene])
    np.random.seed(seed)
    kwargs=dict(robots=list(metadata.robots),controller_configs=load_controller_config(default_controller='OSC_POSE'),
        initialization_noise=None,has_renderer=False,has_offscreen_renderer=False,use_camera_obs=False,
        use_object_obs=True,control_freq=20,horizon=2000 if scene=='two-arm-handover' else 1000,
        ignore_done=False,reward_shaping=True)
    if metadata.env_configuration:kwargs['env_configuration']=metadata.env_configuration
    env=suite.make(metadata.environment,**kwargs)
    try:
        env.reset();m=env.sim.model._model;d=env.sim.data._data
        wipe_contact=scene=='wipe' and case in ('primary','robot_contact')
        actors=[CartesianActor(env,i,integral=scene!='wipe',max_translation=.01 if wipe_contact else .025 if scene=='wipe' else .05)
                for i in range(len(env.robots))]
        roles=roles_for(m,d,scene)
        if scene!='wipe' and not roles['objects']:raise ValueError('task object identity unresolved')
        initial=identity(m,d);actions=[];native_success=False;success_ticks=0;intervened=False;done=False
        phase='approach';ticks=0;hold_ticks=0;receiver_ticks=0;receiver_stage='wait';receiver_parity=None;donor_stable=0;receiver_lift=None;nut_stable=0;nut_exit_rotation=None;anchors={};stages=['approach','descend','close','lift','transfer','lower','release','retreat','verify']
        pick=None;goal=None;offset=np.zeros(3);grasp_rotation=actors[0].rot.copy()
        if scene=='stack':
            b=roles['objects'][0];pick=d.xpos[b].copy()
            goal=d.xpos[m.body('cubeB_main').id].copy()+[0,0,.045]
        elif scene=='pickplacecan':
            b=roles['objects'][0];pick=d.xpos[b].copy()+[0,0,.025]
            idx=next(i for i,o in enumerate(env.objects) if 'Can' in o.name)
            goal=env.target_bin_placements[idx].copy()+[-.025,-.045,.090];offset=np.array([0.,0.,.025])
        elif scene=='nut-assembly':
            b=roles['objects'][0];sid=env.object_site_ids[0]
            pick=d.site_xpos[sid].copy()+d.xmat[b].reshape(3,3)@np.array([.012,0.,.008]);offset=pick-d.xpos[b]
            # Preserve the sampled nut yaw; the square symmetry is checked separately.
            mat=d.xmat[b].reshape(3,3);yaw=np.arctan2(mat[1,0],mat[0,0])
            c,s=np.cos(yaw+np.pi/2),np.sin(yaw+np.pi/2);grasp_rotation=np.array([[c,s,0],[s,-c,0],[0,0,-1.]])
            aligned_yaw=round(yaw/(np.pi/2))*(np.pi/2)
            c,s=np.cos(aligned_yaw),np.sin(aligned_yaw)
            nut_rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
            nut_goal=np.r_[d.xpos[env.peg1_body_id][:2],env.table_offset[2]+.01]
            goal=nut_goal+nut_rotation@np.array([.066,0.,.008])
        elif scene=='two-arm-handover':
            b=roles['objects'][0]
            axis=d.geom_xmat[env.hammer_handle_geom_id].reshape(3,3)[:,2]
            pick=np.asarray(env._handle_xpos).copy()-.35*env.hammer.handle_length*axis+[0,0,.006]
            goal=np.array([0.,.02,1.18]);offset=pick-d.xpos[b]
            axis=d.geom_xmat[env.hammer_handle_geom_id].reshape(3,3)[:,2]
            yaw=np.arctan2(axis[1],axis[0])+np.pi/2;c,s=np.cos(yaw),np.sin(yaw)
            grasp_rotation=np.array([[c,s,0],[s,-c,0],[0,0,-1.]])
            alternate=grasp_rotation@np.diag([-1.,-1.,1.])
            current=d.site_xmat[actors[0].site].reshape(3,3)
            if np.trace(current.T@alternate)>np.trace(current.T@grasp_rotation):grasp_rotation=alternate
        if scene not in ('wipe','two-arm-handover'):
            roles['goals'][str(b)]=dict(position=(nut_goal if scene=='nut-assembly' else goal-offset).tolist(),
                tolerance_m=[.015,.015,.015] if scene!='pickplacecan' else [.04,.04,.05])
        with Recorder(m,d,folder/'raw',roles) as rec:
            def forcing(model,data):
                if not intervened or not roles['objects']:return
                b0=roles['objects'][0]
                if case in ('overload','eccentric_torque'):
                    f=np.array([0.,0.,-20.]);r=np.array([.05,0.,0.])
                    data.xfrc_applied[b0,:]=np.r_[f,np.cross(r,f) if case=='eccentric_torque' else np.zeros(3)]
                elif case=='jam':
                    parts=set(roles['part_geoms'][str(b0)])
                    touch=any((c.geom1 in parts)!=(c.geom2 in parts) and
                              c.geom1 not in roles['pads'] and c.geom2 not in roles['pads'] and c.dist<=0 for c in data.contact)
                    if touch and b0 not in anchors:anchors[b0]=data.xpos[b0].copy()
                    if b0 in anchors:
                        v=data.qvel[model.body_dofadr[b0]:model.body_dofadr[b0]+3]
                        data.xfrc_applied[b0,:3]=np.clip(5000.*(anchors[b0]-data.xpos[b0])-20.*v,-100.,100.)
            rec.forcing=forcing
            for step in range(kwargs['horizon']):
                actor=actors[0];ticks+=1;holding=False
                if scene=='wipe':
                    # Markers are read by this engineering controller, not by an Agent.
                    markers=[marker for marker in env.model.mujoco_arena.markers if marker not in env.wiped_markers]
                    positions=[d.xpos[m.body(marker.root_body).id].copy() for marker in markers]
                    center=min(positions,key=lambda p:float(np.linalg.norm(p[:2]-d.site_xpos[actor.site,:2]))) if positions else np.asarray(env.table_offset)
                    target=center+np.array([0.,0.,-.004])
                    if wipe_contact:
                        # Rotate away from the table before descending. Preserve
                        # the reset yaw instead of winding joint 7 into its limit.
                        target=np.asarray(env.table_offset)+[-.15,0.,.26 if step<70 else -.14]
                        intervened=True
                    if case in ('overload','eccentric_torque') and step>20:intervened=True
                    phase='wipe';close=False
                else:
                    # Robosuite samples some objects before gravity settling.
                    # Track actual object pose while approaching, instead of
                    # assuming the sampled reset pose remains the grasp pose.
                    if phase=='approach' or scene in ('nut-assembly','two-arm-handover') and phase in ('descend','close'):
                        if scene=='nut-assembly':pick=d.site_xpos[env.object_site_ids[0]].copy()+d.xmat[b].reshape(3,3)@np.array([.012,0.,.008])
                        elif scene=='two-arm-handover':
                            axis=d.geom_xmat[env.hammer_handle_geom_id].reshape(3,3)[:,2]
                            pick=np.asarray(env._handle_xpos).copy()-.35*env.hammer.handle_length*axis+[0,0,.006]
                        else:pick=d.xpos[b].copy()+([0,0,.025] if scene=='pickplacecan' else [0,0,0])
                    close=phase in ('close','lift','transfer','lower')
                    if phase=='approach':target=pick+[0,0,.15]
                    elif phase in ('descend','close'):target=pick
                    elif phase=='lift':target=pick+[0,0,.20]
                    elif phase=='transfer':target=goal+[0,0,.10 if scene=='pickplacecan' else .18]
                    elif phase in ('lower','release'):target=goal
                    else:target=goal+[0,0,.18]
                    if scene=='two-arm-handover' and phase in ('transfer','lower','release'):
                        target=goal.copy()
                    if scene=='two-arm-handover' and case!='primary':close=phase not in ('approach','descend') and receiver_ticks<20
                    if phase=='close' and ticks>25:
                        # Keep the measured TCP/object offset for transport.
                        grasp_offset=d.site_xpos[actor.site]-d.xpos[b]
                    if phase in ('transfer','lower','release','retreat','verify') and scene in ('stack','pickplacecan'):
                        target=np.asarray(target)+grasp_offset-(offset if scene=='pickplacecan' else 0.)
                    if case=='primary' and phase=='transfer':intervened=True
                    if case=='primary' and intervened:close=False
                    if case in ('overload','eccentric_torque','jam') and phase=='lower':intervened=True
                    if case=='legal_hold' and phase=='transfer' and hold_ticks<24:
                        target=d.site_xpos[actor.site].copy();hold_ticks+=1;intervened=True;holding=True
                if case=='task_failure':
                    target=actor.home;close=False
                    if step>=25:break
                if case=='robot_contact' and scene!='wipe':
                    target=np.r_[actor.home[:2],float(env.table_offset[2])-.02] if hasattr(env,'table_offset') else np.array([0,0,.75])
                    close=False;intervened=True
                    if step>=300:break
                rotation=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]]) if scene=='nut-assembly' and phase in ('transfer','lower','release','retreat','verify') else grasp_rotation
                if scene=='nut-assembly' and phase in ('transfer','lower','release'):
                    # Servo the measured part pose, preserving its actual grasp
                    # transform. A nominal handle offset alone cannot align the
                    # square opening after a small slip during pickup.
                    desired_part=nut_goal+[0.,0.,.18 if phase=='transfer' else 0.]
                    if phase=='lower':
                        lateral=np.linalg.norm(nut_goal[:2]-d.xpos[b,:2])
                        # Unload a peg-edge contact before centering. Continuing
                        # to press downward can lock a millimetric lateral error
                        # through friction even when the hole orientation fits.
                        desired_part[2]=(float(d.xpos[b,2])+.008 if lateral>.003 else
                                         max(nut_goal[2],float(d.xpos[b,2])-.010))
                        actor.position_bias[2]=0.
                    part_error=desired_part-d.xpos[b]
                    increment=np.clip(part_error,-.025,.025)
                    if phase=='lower':increment=part_error
                    target=d.site_xpos[actor.site]+increment
                    rotation=nut_rotation@d.xmat[b].reshape(3,3).T@d.site_xmat[actor.site].reshape(3,3)
                    nut_exit_rotation=d.site_xmat[actor.site].reshape(3,3).copy()
                if scene=='nut-assembly' and phase in ('retreat','verify') and nut_exit_rotation is not None:
                    rotation=nut_exit_rotation
                if wipe_contact:
                    rotation=actor.home_rotation@np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
                arm_actions=[actor.action(target,close,rotation)]
                if len(actors)>1:
                    receiver=actors[1]
                    donor_arrived=phase=='transfer' and np.linalg.norm(d.site_xpos[actor.site]-goal)<.025
                    donor_stable=donor_stable+1 if donor_arrived else 0
                    if receiver_stage=='wait' and donor_stable>=10:receiver_stage='approach'
                    reception=receiver_stage!='wait'
                    axis=d.geom_xmat[env.hammer_handle_geom_id].reshape(3,3)[:,2]
                    z_axis=np.cross(axis,np.array([0.,0.,1.]))
                    if np.linalg.norm(z_axis)<.1:z_axis=np.array([-1.,0.,0.])
                    z_axis/=np.linalg.norm(z_axis)
                    p=np.asarray(env._handle_xpos).copy()+.35*env.hammer.handle_length*axis if reception else receiver.home
                    if receiver_stage=='approach':
                        p=p-.10*z_axis
                        if np.linalg.norm(d.site_xpos[receiver.site]-p)<.012:receiver_stage='descend'
                    elif receiver_stage=='descend' and np.linalg.norm(d.site_xpos[receiver.site]-p)<.008:
                        receiver_stage='close'
                    if receiver_lift is not None:p=receiver_lift
                    # Approach the exposed handle from its side, leaving the
                    # donor's vertical palm and the hammer head clear.
                    up=np.array([0.,0.,1.]) if abs(z_axis[2])<.9 else np.array([1.,0.,0.])
                    x_axis=up-z_axis*np.dot(up,z_axis);x_axis/=np.linalg.norm(x_axis)
                    receiver_rotation=np.column_stack((x_axis,np.cross(z_axis,x_axis),z_axis))
                    if receiver_parity is None:
                        alt=receiver_rotation@np.diag([-1.,-1.,1.])
                        receiver_parity=np.trace(receiver.home_rotation.T@alt)>np.trace(receiver.home_rotation.T@receiver_rotation)
                    if receiver_parity:receiver_rotation=receiver_rotation@np.diag([-1.,-1.,1.])
                    arm_actions.append(receiver.action(p,receiver_stage=='close',receiver_rotation))
                    receiver_ticks=receiver_ticks+1 if reception and env._get_task_info()[1] else 0
                    if reception and case!='primary' and receiver_ticks>=20:
                        arm_actions[0][-1]=-1.
                        if receiver_lift is None:receiver_lift=d.site_xpos[receiver.site].copy()+[0,0,.04]
                action=np.concatenate(arm_actions)
                rec.control=step;rec.phase=phase
                actions.append(dict(step=step,time_s=float(d.time),phase=phase,action=action.tolist(),
                                    target_position_m=np.asarray(target).tolist(),
                                    object_position_m=d.xpos[roles['objects'][0]].tolist() if roles['objects'] else None,
                                    intervention=intervened,holding=holding,
                                    stage_motion_hint=bool(phase=='lower' and np.linalg.norm(action[:3])>1e-3)))
                _,reward,done,info=env.step(action)
                success=bool(env._check_success());native_success|=success
                success_ticks=success_ticks+1 if success else 0
                if step%100==0:
                    print(f'{scene} {case} seed={seed} step={step} phase={phase} tcp={d.site_xpos[actor.site]} target={target}',flush=True)
                    if scene=='two-arm-handover':
                        print(f'receiver_tcp={d.site_xpos[actors[1].site]} receiver_target={p} grasp={env._get_task_info()[:2]} object={d.xpos[b]}',flush=True)
                if scene=='two-arm-handover' and phase=='close' and ticks==34:
                    print(f'pickup tcp={d.site_xpos[actor.site]} target={target} object={d.xpos[b]} grasp={env._get_task_info()[:2]}',flush=True)
                if done:break
                if scene=='two-arm-handover' and success_ticks>=3:break
                if scene!='wipe':
                    distance=np.linalg.norm(d.site_xpos[actor.site]-target)
                    if scene=='nut-assembly' and phase in ('transfer','lower','release'):
                        distance=np.linalg.norm((nut_goal if phase=='lower' else desired_part)-d.xpos[b])
                    settled=distance<.012
                    if phase in ('approach','descend'):
                        relative=rotation.T@d.site_xmat[actor.site].reshape(3,3)
                        settled=settled and (np.trace(relative)-1)/2>np.cos(.04)
                    if scene=='nut-assembly' and phase=='transfer':
                        relative=nut_rotation.T@d.xmat[b].reshape(3,3)
                        nut_stable=nut_stable+1 if distance<.003 and (np.trace(relative)-1)/2>np.cos(.03) else 0
                        settled=nut_stable>=10
                    advance=(settled and ticks>=8 and not holding)
                    if scene=='two-arm-handover' and case!='primary' and phase=='transfer':advance=advance and receiver_ticks>=20
                    if phase in ('close','release'):advance=ticks>=35
                    if phase=='verify':advance=ticks>=25
                    if advance:
                        if phase=='verify':break
                        phase=stages[stages.index(phase)+1];ticks=0
                    if ticks>=(600 if (scene=='two-arm-handover' and phase=='transfer') or
                                      (scene=='nut-assembly' and phase in ('transfer','lower')) else 250):break
                elif native_success:break
            result=dict(status='RECORDED',initial_state_sha256=initial,model_sha256=rec.geometry['model_sha256'],
                control_steps=len(actions),physics_steps=rec.count,simulation_time_s=float(d.time),
                task_success=bool(env._check_success()),native_task_success_any=native_success,
                task_score=float(reward),native_done=bool(done),intervention_executed=intervened,
                environment=metadata.environment,controller='OSC_POSE',control_freq_hz=20,
                camera='disabled for state-only acceptance; task camera assets unchanged',
                failure_reason=None if native_success else f'controller ended at {phase}')
            result['final_object_poses']=[dict(body=int(body),position_m=d.xpos[body].tolist(),
                rotation=d.xmat[body].reshape(3,3).tolist()) for body in roles['objects']]
            result['final_contact_pairs']=[]
            for contact_index,contact in enumerate(d.contact):
                wrench=np.zeros(6);mujoco.mj_contactForce(m,d,contact_index,wrench)
                if wrench[0]>.01:
                    result['final_contact_pairs'].append(dict(
                        geoms=[int(contact.geom1),int(contact.geom2)],
                        names=[m.geom(int(contact.geom1)).name,m.geom(int(contact.geom2)).name],
                        normal_force_n=float(wrench[0])))
            if scene=='two-arm-handover':result['final_native_grasp_state']=list(env._get_task_info())
            if scene=='wipe':
                near=[]
                for joint in env.robots[0]._ref_joint_indexes:
                    q=float(d.qpos[m.jnt_qposadr[joint]]);lo,hi=map(float,m.jnt_range[joint])
                    if lo!=hi and not lo+.1<q<hi-.1:
                        near.append(dict(joint=m.joint(joint).name,position_rad=q,
                            margin_rad=min(q-lo,hi-q),native_margin_rad=.1))
                contact_termination=bool(env.check_contact(env.robots[0].robot_model))
                result.update(wiped_markers=len(env.wiped_markers),total_markers=env.num_markers,
                              native_collision_count=env.collisions,native_force_excess_count=env.f_excess,
                              native_robot_contact=contact_termination,native_joint_limit_proximity=near)
                if done and not native_success:
                    result['failure_reason']='native robot contact' if contact_termination else 'native joint-limit proximity' if near else 'native termination'
        (folder/'actions.json').write_text(json.dumps(actions,separators=(',',':'))+'\n')
        return result
    finally:env.close()
